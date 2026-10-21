//! Rentals through their Hubs. A Hub is read to rent, end and list, to follow a booting rental
//! to its address, and to ask whether a rental that stays unreachable still exists; never per
//! run. Every rent and end is an ask recorded before it is made, under the key the Hub gets as
//! its idempotency key, so a Calcifer that stops mid-ask asks again under the same key. Only the
//! Hub's word (failed, released) ends a rental here; a rental is never replaced.
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::Duration,
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use platform::paths::Home;
use protocol::{
    calcifer::{
        Agreed, Ended, HubListing, HubProblem, Machine, MachineKind, MachineState, On, Quote,
        Reason, Rental, Rented, RunQuote, SubmitRun, Warning,
    },
    config,
    error::{Error, Kind},
    ids::RunId,
    machine as v1,
};
use ed25519_dalek::SigningKey;
use hub_client::{Client, Credential, DeviceKey, UserSession, types};
use sha2::{Digest as _, Sha256};
use tracing::warn;

use crate::{
    machines::{Machines, internal, lock, not_found, pin_of, write_key},
    store::{Ask, AskKind, AskState, Endpoint, MachineRow, RunRow},
};

/// The Hubs Calcifer rents from, and what each last listed.
pub(crate) struct Hubs {
    http: reqwest::Client,
    home: Home,
    pub(crate) configured: Vec<config::Hub>,
    listings: Mutex<HashMap<String, Listing>>,
    /// This computer's account at each Hub and where machines read it, for runs' access.
    pub(crate) facts: Mutex<HashMap<config::Origin, crate::access::Facts>>,
}

/// A Hub's last word: its listing, and every rental read since by id.
#[derive(Default)]
struct Listing {
    listed_at_ms: Option<i64>,
    rentals: HashMap<String, types::Rental>,
    problem: Option<HubProblem>,
    signed_out: bool,
}

impl Hubs {
    pub(crate) fn new(home: &Home, configured: Vec<config::Hub>) -> Self {
        Self {
            http: reqwest::Client::new(),
            home: home.clone(),
            configured,
            listings: Mutex::default(),
            facts: Mutex::default(),
        }
    }

    /// A client speaking as this computer's device key at the Hub.
    pub(crate) fn client(&self, origin: &str) -> Result<Client, Error> {
        let hub = config::Hub {
            name: None,
            origin: config::Origin::parse(origin).map_err(|_| signed_out(origin))?,
        };
        let key = DeviceKey::load(&hub.device_key(&self.home)).map_err(|_| signed_out(origin))?;
        let session = UserSession::new(key, Some(hub.bearer(&self.home)));
        Ok(Client::connect(
            origin,
            self.http.clone(),
            Credential::User(Arc::new(session)),
        ))
    }

    /// Remembers what a Hub said of one rental.
    fn seen(&self, origin: &str, rental: types::Rental) {
        lock(&self.listings)
            .entry(origin.to_owned())
            .or_default()
            .rentals
            .insert(rental.rental_id.clone(), rental);
    }

    fn rental(&self, origin: &str, id: &str) -> Option<types::Rental> {
        lock(&self.listings)
            .get(origin)
            .and_then(|listing| listing.rentals.get(id).cloned())
    }

    /// A rental's GPUs as its Hub last said them.
    pub(crate) fn gpus(&self, row: &MachineRow) -> Option<u32> {
        let rental = self.rental(&row.hub, row.rental_id.as_deref()?)?;
        u32::try_from(rental.accelerator_count).ok()
    }

    /// Adds the Hub's word to a recorded rental: its prices and times, and whether its Hub
    /// answered the last listing.
    pub(crate) fn describe(&self, machine: &mut Machine, asked: Option<&Ask>) {
        let listings = lock(&self.listings);
        let Some(listing) = listings.get(&machine.hub) else {
            return;
        };
        machine.unverified = listing.problem.is_some();
        if let Some(rental) = machine
            .rental_id
            .as_ref()
            .and_then(|id| listing.rentals.get(id))
        {
            let sku = asked
                .and_then(|ask| RentBody::of(ask).ok())
                .map(|body| body.request.sku);
            machine.rental = Some(view(rental, sku));
        }
    }

    /// Every Hub's last listing: when each answered, which did not, which this computer is not
    /// signed in to, and the live rentals of this account that `rows` does not record.
    pub(crate) fn listing(
        &self,
        rows: &[MachineRow],
    ) -> (Vec<HubListing>, Vec<HubProblem>, Vec<String>, Vec<Machine>) {
        let listings = lock(&self.listings);
        let (mut hubs, mut problems, mut signed_out, mut others) = (vec![], vec![], vec![], vec![]);
        for (origin, listing) in listings.iter() {
            if let Some(at) = listing.listed_at_ms {
                hubs.push(HubListing {
                    hub: origin.clone(),
                    listed_at_ms: at,
                });
            }
            problems.extend(listing.problem.clone());
            if listing.signed_out {
                signed_out.push(origin.clone());
            }
            for rental in listing.rentals.values() {
                let recorded = rows
                    .iter()
                    .any(|row| row.rental_id.as_deref() == Some(rental.rental_id.as_str()));
                if !recorded && !gone(&rental.state) {
                    others.push(other(origin, rental, listing.problem.is_some()));
                }
            }
        }
        others.sort_by(|a, b| a.name.cmp(&b.name));
        (hubs, problems, signed_out, others)
    }

    /// A live rental of this account that this controller did not rent, by name or rental id.
    pub(crate) fn other(&self, text: &str) -> Option<Machine> {
        let listings = lock(&self.listings);
        listings.iter().find_map(|(origin, listing)| {
            listing
                .rentals
                .values()
                .find(|rental| names(rental, text) && !gone(&rental.state))
                .map(|rental| other(origin, rental, listing.problem.is_some()))
        })
    }

    /// Every Hub the config names and every Hub a rental was asked of.
    fn origins(&self, rows: &[MachineRow]) -> Vec<String> {
        let mut origins: Vec<String> = self.configured.iter().map(|h| h.origin.to_string()).collect();
        for row in rows
            .iter()
            .filter(|r| r.kind == MachineKind::Rental && r.state != MachineState::Ended)
        {
            if !origins.contains(&row.hub) {
                origins.push(row.hub.clone());
            }
        }
        origins
    }
}

impl Machines {
    /// The Hub's price for renting `sku` at `hub` in `shape`, exactly as a rent would ask it.
    /// Nothing is recorded or kept.
    pub async fn quote(&self, hub: &str, sku: &str, shape: Shape) -> Result<Quote, Error> {
        let origin = config::Origin::parse(hub)?;
        let (request, ..) = request(sku, shape)?;
        Ok(quote_of(&self.quoted(&origin, &request).await?))
    }

    /// Rents a machine at `hub` under `key`: the same key again answers the same rental.
    pub async fn rent(
        self: &Arc<Self>,
        key: &str,
        hub: &str,
        sku: &str,
        shape: Shape,
    ) -> Result<Rented, Error> {
        let (ask, fresh) = self.ask_rent(key, hub, sku, shape, None).await?;
        let quote = quote_of(&RentBody::of(&ask)?.quote);
        let row = match ask.state {
            AskState::Done => {
                let name = ask.machine.as_deref().unwrap_or_default();
                self.store()
                    .find_machine(name)
                    .map_err(internal)?
                    .ok_or_else(|| not_found(name))?
            }
            AskState::Failed => {
                return Err(Error::new(
                    Kind::Conflict,
                    "rental_refused",
                    ask.reason.unwrap_or_default(),
                ));
            }
            // Asked apart from this call: a caller that stops waiting leaves the ask asked.
            AskState::Pending => {
                let (answer, answered) = tokio::sync::oneshot::channel();
                let me = self.clone();
                self.spawn(async move {
                    let _ = me.ask_hub(&ask, Some(answer)).await;
                });
                answered.await.map_err(|_| {
                    Error::new(
                        Kind::Unavailable,
                        "calcifer_stopping",
                        "Calcifer is stopping",
                    )
                })??
            }
        };
        Ok(Rented {
            machine: self.picture(&row)?,
            quote,
            replay: !fresh,
        })
    }

    /// Records a rent ask under `key` with the Hub's quote for its exact request, and the
    /// rental's key; an ask the key already names is answered, if it asked for the same. A rent
    /// the Hub will not quote records nothing.
    async fn ask_rent(
        &self,
        key: &str,
        hub: &str,
        sku: &str,
        shape: Shape,
        run: Option<RunId>,
    ) -> Result<(Ask, bool), Error> {
        let origin = config::Origin::parse(hub)?;
        let mut intent = serde_json::json!({"hub": origin, "sku": sku, "gpus": shape.gpus,
            "disk_gb": shape.disk_gb});
        // Only an image joins the intent, so a key asked before images keeps its digest.
        if let Some(image) = &shape.image {
            intent["image"] = image.as_str().into();
        }
        let digest = hex::encode(Sha256::digest(intent.to_string()));
        if let Some(asked) = self.store().ask(key).map_err(internal)? {
            if asked.digest != digest || asked.kind != AskKind::Rent {
                return Err(Error::new(
                    Kind::Conflict,
                    "idempotency_conflict",
                    format!("the key {key:?} already asked for another rental"),
                ));
            }
            return Ok((asked, false));
        }
        // An ask the Hub would refuse unasked, or one this computer cannot sign, records nothing.
        types::CreateRentalIdempotencyKey::try_from(key)
            .map_err(|e| Error::new(Kind::Usage, "idempotency_key_invalid", e.to_string()))?;
        let agreed = shape.agreed;
        let (request, alias, signer) = request(sku, shape)?;
        let quote = self.quoted(&origin, &request).await?;
        // Dearer than the renter agreed to a moment ago: nothing is rented.
        let now = quote_of(&quote).agreed();
        if let Some(agreed) = agreed.filter(|agreed| above(&now, agreed)) {
            return Err(Error::new(
                Kind::Conflict,
                "rental_price_rose",
                format!(
                    "{sku} now costs {}, above the {} agreed; nothing was rented",
                    rate(&now),
                    rate(&agreed)
                ),
            )
            .next(format!("cozy machine rent {sku}")));
        }
        write_key(&self.home.rental_key(&alias), &signer)?;
        let ask = Ask {
            key: key.to_owned(),
            kind: AskKind::Rent,
            hub: origin.into(),
            digest,
            body: serde_json::to_vec(&RentBody { request, quote }).expect("serializes"),
            machine: Some(alias),
            run,
            state: AskState::Pending,
            reason: None,
        };
        let (ask, fresh) = self.store().begin_ask(&ask).map_err(|e| match e {
            crate::store::StoreError::Conflict(_) => Error::new(
                Kind::Conflict,
                "idempotency_conflict",
                format!("the key {key:?} already asked for another rental"),
            ),
            e => internal(e),
        })?;
        Ok((ask, fresh))
    }

    /// The Hub's price for `request`; a refusal is the Hub's.
    async fn quoted(
        &self,
        origin: &config::Origin,
        request: &types::RentalRequest,
    ) -> Result<types::RentalQuote, Error> {
        let client = self.hubs.client(origin)?;
        match client.quote_rental(request).await {
            Ok(quote) => Ok(quote.into_inner()),
            Err(e) => Err(match asked(origin, e.into()) {
                Asked::Refused(e) | Asked::Unanswered(e) => e,
            }),
        }
    }

    /// Makes a recorded rent ask. The Hub's acceptance records the rental as a booting machine
    /// (no endpoint: its runs wait in the outbox) in the transaction that ends the ask; its
    /// refusal ends the ask and the run it was made for. A Hub that does not answer is asked
    /// again under the same key, paced; `told` hears the first answer, or that there was none.
    async fn ask_hub(
        self: &Arc<Self>,
        ask: &Ask,
        mut told: Option<tokio::sync::oneshot::Sender<Result<MachineRow, Error>>>,
    ) -> Result<MachineRow, Error> {
        let mut pause = Duration::ZERO;
        let mut tell = |answer: &Result<MachineRow, Error>| {
            if let Some(told) = told.take() {
                let _ = told.send(answer.clone());
            }
        };
        loop {
            match self.ask_once(ask).await {
                Err(Asked::Refused(error)) => {
                    let reason = format!("{}: {}", error.code, error.message);
                    self.store()
                        .settle_ask(&ask.key, Err(&reason))
                        .map_err(internal)?;
                    if let Some(run) = &ask.run {
                        let reason = Reason {
                            code: "rental_refused".into(),
                            message: reason,
                            ..Default::default()
                        };
                        self.calcifer.fail(run, reason).map_err(internal)?;
                    }
                    let refused = Err(error);
                    tell(&refused);
                    return refused;
                }
                Err(Asked::Unanswered(e)) => {
                    warn!(ask = ask.key, "asking {} again: {e}", ask.hub);
                    let key = &ask.key;
                    tell(&Err(e.remedy(format!(
                        "it is asked again under the key {key:?}; the same key answers its outcome"
                    ))));
                    pause = (pause * 2).clamp(Duration::from_secs(2), Duration::from_secs(60));
                    tokio::time::sleep(pause).await;
                }
                Ok(rental) => {
                    let alias = ask.machine.clone().unwrap_or_default();
                    let quoted = quote_of(&RentBody::of(ask)?.quote);
                    // Asked under its alias; the boot follower takes the Hub's name for it.
                    let row = MachineRow {
                        name: alias.clone(),
                        kind: MachineKind::Rental,
                        hub: ask.hub.clone(),
                        rental_id: Some(rental.rental_id.clone()),
                        endpoint: None,
                        key: self.home.rental_key(&alias),
                        state: MachineState::Booting,
                        served: false,
                        reason: None,
                    };
                    // The price shown is the price locked: an acceptance above the quote (the
                    // machine and its disk an hour, or traffic a GB) is released at once.
                    let locked = Agreed {
                        usd_micros_per_hour: hourly(&rental),
                        usd_micros_per_gb: rental.bandwidth_usd_micros_per_gb.unwrap_or_default(),
                    };
                    if above(&locked, &quoted.agreed()) {
                        let reason = format!(
                            "rental.rate_changed: {} locked {}, above the {} quoted; it is released",
                            ask.hub,
                            rate(&locked),
                            rate(&quoted.agreed())
                        );
                        let id = rental.rental_id.clone();
                        self.hubs.seen(&ask.hub, rental);
                        self.calcifer.put_machine(&row).map_err(internal)?;
                        self.store()
                            .settle_ask(&ask.key, Err(&reason))
                            .map_err(internal)?;
                        self.ask_end(&row.hub, &id, Some(row.name.clone()))?;
                        if let Some(run) = &ask.run {
                            let reason = Reason {
                                code: "rental_refused".into(),
                                message: reason.clone(),
                                ..Default::default()
                            };
                            self.calcifer.fail(run, reason).map_err(internal)?;
                        }
                        let refused =
                            Err(Error::new(Kind::Conflict, "rental_rate_changed", reason));
                        tell(&refused);
                        return refused;
                    }
                    self.hubs.seen(&ask.hub, rental);
                    self.store()
                        .settle_ask(&ask.key, Ok(Some(&row)))
                        .map_err(internal)?;
                    self.calcifer.machine_changed();
                    if let Some(run) = &ask.run {
                        self.calcifer.place(run, &row.name).map_err(internal)?;
                    }
                    self.follow_boot(row.clone());
                    self.placing.notify_one();
                    tell(&Ok(row.clone()));
                    return Ok(row);
                }
            }
        }
    }

    async fn ask_once(&self, ask: &Ask) -> Result<types::Rental, Asked> {
        let client = self.hubs.client(&ask.hub).map_err(Asked::Refused)?;
        let request = RentBody::of(ask).map_err(Asked::Refused)?.request;
        let key = types::CreateRentalIdempotencyKey::try_from(ask.key.as_str()).map_err(|e| {
            Asked::Refused(Error::new(
                Kind::Usage,
                "idempotency_key_invalid",
                e.to_string(),
            ))
        })?;
        match client.create_rental(&key, &request).await {
            Ok(rental) => Ok(rental.into_inner()),
            Err(e) => Err(asked(&ask.hub, e.into())),
        }
    }

    /// Rents a machine for an unplaced run (`--on new[:<sku>]`) under a key the run names, so a
    /// Calcifer that stops mid-ask asks again for the same rental. Without a product, the Hub
    /// places the run: the cheapest expected rental that holds it, named on the run.
    pub(crate) fn rent_for(self: &Arc<Self>, run: &RunRow, sku: Option<&str>, gpus: u32) {
        if !lock(&self.renting).insert(run.id.clone()) {
            return;
        }
        let (me, run, sku) = (self.clone(), run.clone(), sku.map(str::to_owned));
        self.spawn(async move {
            let key = format!("run-{}", run.id);
            let asked = async {
                if let Some(ask) = me.store().ask(&key).map_err(internal)? {
                    return Ok(ask);
                }
                let (sku, gpus) = match sku {
                    Some(sku) => (sku, (gpus > 0).then_some(gpus)),
                    None => me.placed(&run).await?,
                };
                let hub = run.hub.as_deref().unwrap_or_default();
                let shape = Shape {
                    gpus,
                    agreed: run.agreed,
                    ..Shape::default()
                };
                let asked = me.ask_rent(&key, hub, &sku, shape, Some(run.id.clone()));
                Ok(asked.await?.0)
            };
            let rented = match asked.await {
                Ok(ask) => match ask.state {
                    AskState::Pending => me.ask_hub(&ask, None).await.map(drop),
                    AskState::Done => {
                        let name = ask.machine.unwrap_or_default();
                        me.calcifer
                            .place(&run.id, &name)
                            .map(drop)
                            .map_err(internal)
                    }
                    AskState::Failed => Err(Error::new(
                        Kind::Conflict,
                        "rental_refused",
                        ask.reason.unwrap_or_default(),
                    )),
                },
                Err(error) => Err(error),
            };
            if let Err(error) = rented {
                let reason = Reason {
                    code: error.code,
                    message: error.message,
                    ..Default::default()
                };
                if let Err(e) = me.calcifer.fail(&run.id, reason) {
                    warn!(run = %run.id, "failing it: {e}");
                }
            }
            lock(&me.renting).remove(&run.id);
        });
    }

    /// Asks the run's Hub where to run it: the product to rent and its GPU count. The answer is
    /// kept on the run as a warning naming the product, its price and why.
    async fn placed(&self, run: &RunRow) -> Result<(String, Option<u32>), Error> {
        let hub = run.hub.as_deref().unwrap_or_default();
        let (sku, gpus, message) = self.placement(hub, &run.spec).await?;
        let warning = Warning {
            code: "placement".into(),
            message,
        };
        self.calcifer.warn(&run.id, &warning).map_err(internal)?;
        Ok((sku, gpus))
    }

    /// What `hub` places a run of `spec` on: the product, its GPU count, and why, in words.
    async fn placement(&self, hub: &str, spec: &v1::RunSpec) -> Result<(String, Option<u32>, String), Error> {
        let Some(v1::Source::Release(release)) = &spec.source else {
            return Err(Error::new(
                Kind::Usage,
                "placement_needs_release",
                "the Hub places a published release: run local code with --on <machine> or \
                 --on new:<sku>",
            ));
        };
        let request = types::RentalPlacementRequest {
            package: release.package.clone(),
            release: Some(release.release.clone()).filter(|r| !r.is_empty()),
            function: spec.entrypoint.clone(),
            bits: (spec.bits > 0)
                .then(|| types::RentalPlacementRequestBits::try_from(i64::from(spec.bits)))
                .transpose()
                .map_err(|e| Error::new(Kind::Validation, "bits_invalid", e.to_string()))?,
            gpus: std::num::NonZeroU64::new(spec.gpus.into()),
        };
        let client = self.hubs.client(hub)?;
        let placement = match client.place_rental(&request).await {
            Ok(placement) => placement.into_inner(),
            Err(e) => {
                return Err(match asked(hub, e.into()) {
                    Asked::Refused(e) | Asked::Unanswered(e) => e,
                });
            }
        };
        let Some(top) = placement.machines.first() else {
            let why: Vec<String> = placement
                .excluded
                .iter()
                .map(|e| format!("{} ×{}: {}", e.name, e.accelerator_count, e.reason))
                .collect();
            return Err(Error::new(
                Kind::Capacity,
                "placement_none_fits",
                format!(
                    "no product the Hub sells now holds this run: {}",
                    why.join("; ")
                ),
            ));
        };
        let message = format!(
            "the Hub placed it on {} ({}× {}, {} GB): ${:.2}/h, about ${:.2} expected, the \
             cheapest of {} products holding {} GB of {}-bit weights",
            top.name,
            top.accelerator_count,
            top.accelerator_model,
            top.vram_gb,
            (top.price_usd_micros_per_hour + top.storage_usd_micros_per_hour) as f64 / 1e6,
            top.expected_price.total_usd_micros as f64 / 1e6,
            placement.machines.len(),
            placement.weight_bytes / 1_000_000_000,
            placement.bits,
        );
        Ok((top.name.clone(), u32::try_from(top.accelerator_count).ok(), message))
    }

    /// What a `new` run would rent and its price, as `rent_for` would ask it: nothing is rented.
    pub async fn run_quote(&self, submit: &SubmitRun) -> Result<RunQuote, Error> {
        let On::New(sku) = &submit.on else {
            return Err(Error::new(Kind::Usage, "quote_needs_new", "only a run with `--on new` rents"));
        };
        let gpus = (submit.spec.gpus > 0).then_some(submit.spec.gpus);
        let (sku, gpus) = match sku {
            Some(sku) => (sku.clone(), gpus),
            None => {
                let (sku, gpus, _) = self.placement(&submit.hub, &submit.spec).await?;
                (sku, gpus)
            }
        };
        let quote = self.quote(&submit.hub, &sku, Shape { gpus, ..Shape::default() }).await?;
        Ok(RunQuote { sku, gpus, quote })
    }

    /// Asks again what a Calcifer that stopped had asked a Hub and not heard back. A run asked
    /// for is held from placement meanwhile.
    pub(crate) fn resume_asks(self: &Arc<Self>) {
        let pending = match self.store().pending_asks() {
            Ok(pending) => pending,
            Err(e) => return warn!("reading pending asks: {e}"),
        };
        for ask in pending {
            if let Some(run) = &ask.run {
                lock(&self.renting).insert(run.clone());
            }
            let me = self.clone();
            self.spawn(async move {
                if ask.kind == AskKind::End {
                    return me.finish_end(ask).await;
                }
                if let Err(e) = me.ask_hub(&ask, None).await {
                    warn!(ask = ask.key, "asked again: {e}");
                }
                if let Some(run) = &ask.run {
                    lock(&me.renting).remove(run);
                }
            });
        }
    }

    /// Follows a booting rental at its Hub until it has an address (then its runs are sent) or
    /// the Hub says it ended. The Hub's name for the pod replaces the one it was asked under
    /// until it serves.
    pub(crate) fn follow_boot(self: &Arc<Self>, row: MachineRow) {
        let Some(id) = row.rental_id.clone() else {
            return;
        };
        if !lock(&self.booting).insert(id.clone()) {
            return;
        }
        let me = self.clone();
        self.spawn(async move {
            me.boot(row).await;
            lock(&me.booting).remove(&id);
        });
    }

    async fn boot(self: &Arc<Self>, mut row: MachineRow) {
        let id = row.rental_id.clone().unwrap_or_default();
        let (mut pause, mut last) = (Duration::from_secs(2), String::new());
        loop {
            match self.store().machine(&row.name) {
                Ok(Some(now)) if now.state != MachineState::Ended && now.endpoint.is_none() => {
                    row.served = now.served;
                }
                _ => return,
            }
            match self.hubs.read(&row.hub, &id).await {
                Ok(rental) => {
                    if let Some(named) = hub_name(&rental)
                        && named != row.name
                        && !row.served
                    {
                        match self.calcifer.rename_machine(&row.name, &named) {
                            Ok(true) => {
                                let seen = lock(&self.seen).remove(&row.name);
                                lock(&self.seen).insert(named.clone(), seen.unwrap_or_default());
                                row.name = named;
                            }
                            Ok(false) => {}
                            Err(e) => {
                                warn!(machine = row.name, "taking the Hub's name {named}: {e}")
                            }
                        }
                    }
                    let boot = boot_phase(&rental);
                    lock(&self.seen).entry(row.name.clone()).or_default().boot = boot.clone();
                    self.calcifer.machine_changed();
                    if gone(&rental.state) {
                        return self.ended(&row.name, &cause(&rental));
                    }
                    if let Some(endpoint) = endpoint(&rental) {
                        row.endpoint = Some(endpoint);
                        row.state = MachineState::Ready;
                        if let Err(e) = self.calcifer.put_machine(&row) {
                            warn!(machine = row.name, "recording its address: {e}");
                            continue;
                        }
                        lock(&self.seen).entry(row.name.clone()).or_default().boot = None;
                        self.calcifer.machine_changed();
                        return self.watch(&row);
                    }
                    let now = format!("{} {boot:?}", rental.state);
                    pause = if now == last {
                        (pause * 3 / 2).min(Duration::from_secs(30))
                    } else {
                        Duration::from_secs(2)
                    };
                    last = now;
                }
                Err(e) => {
                    warn!(machine = row.name, "reading its boot: {e}");
                    pause = (pause * 2).min(Duration::from_secs(60));
                }
            }
            tokio::time::sleep(pause).await;
        }
    }

    /// Asks a rental's Hub whether it still exists; ends it here when the Hub says it does not.
    pub(crate) async fn hub_says_gone(&self, name: &str) -> bool {
        let Ok(Some(row)) = self.store().machine(name) else {
            return false;
        };
        let Some(id) = row.rental_id.as_deref() else {
            return false;
        };
        match self.hubs.read(&row.hub, id).await {
            Ok(rental) if gone(&rental.state) => {
                self.ended(name, &cause(&rental));
                true
            }
            Ok(_) => false,
            Err(e) => {
                warn!(machine = name, "asking its Hub: {e}");
                false
            }
        }
    }

    /// Lists every Hub's live rentals. A recorded rental its Hub no longer lists is read by id,
    /// and ends here only if the Hub says it ended: an absence is not an answer.
    pub(crate) async fn list_hubs(self: &Arc<Self>) {
        let rows = match self.store().machines() {
            Ok(rows) => rows,
            Err(e) => return warn!("reading the machines: {e}"),
        };
        let mut lists = tokio::task::JoinSet::new();
        for origin in self.hubs.origins(&rows) {
            let me = self.clone();
            lists.spawn(async move { me.list_hub(&origin).await });
        }
        lists.join_all().await;
        self.hubs.learn().await;
        self.calcifer.machine_changed();
    }

    async fn list_hub(&self, origin: &str) {
        let client = match self.hubs.client(origin) {
            Ok(client) => client,
            Err(_) => {
                let mut listings = lock(&self.hubs.listings);
                let listing = listings.entry(origin.to_owned()).or_default();
                (listing.signed_out, listing.problem) = (true, None);
                return;
            }
        };
        let listed = client
            .list_rentals(None, Some(types::ListRentalsState::Live))
            .await;
        let rentals = match listed {
            Ok(listing) => listing.into_inner().rentals,
            Err(e) => {
                let e = hub_client::Error::from(e);
                let mut listings = lock(&self.hubs.listings);
                let listing = listings.entry(origin.to_owned()).or_default();
                listing.signed_out = matches!(&e, hub_client::Error::Hub { status: 401, .. });
                let error = hub_refusal(origin, e);
                listing.problem = Some(HubProblem {
                    hub: origin.to_owned(),
                    code: error.cause.unwrap_or(error.code),
                    message: error.message,
                });
                return;
            }
        };
        let listed: Vec<String> = rentals.iter().map(|r| r.rental_id.clone()).collect();
        {
            let mut listings = lock(&self.hubs.listings);
            let listing = listings.entry(origin.to_owned()).or_default();
            listing.listed_at_ms = Some(crate::store::now_ms());
            (listing.signed_out, listing.problem) = (false, None);
            listing.rentals = rentals
                .into_iter()
                .map(|r| (r.rental_id.clone(), r))
                .collect();
        }
        let Ok(rows) = self.store().machines() else {
            return;
        };
        for row in rows {
            if row.kind == MachineKind::Rental
                && row.hub == origin
                && row.state != MachineState::Ended
                && row
                    .rental_id
                    .as_ref()
                    .is_some_and(|id| !listed.contains(id))
            {
                self.hub_says_gone(&row.name).await;
            }
        }
    }

    /// Asks the rental's Hub to end it; it is `ending` until the Hub affirms the pod gone.
    pub(crate) async fn end_rental(self: &Arc<Self>, row: MachineRow) -> Result<Ended, Error> {
        let id = row.rental_id.clone().unwrap_or_default();
        self.ask_end(&row.hub, &id, Some(row.name.clone()))?;
        Ok(Ended {
            machine: self.picture(&row)?,
            changed: true,
            abandoned: vec![],
        })
    }

    /// Ends a live rental of this account that this controller did not rent, by its name.
    pub(crate) async fn end_other(self: &Arc<Self>, text: &str) -> Result<Ended, Error> {
        if self.hubs.other(text).is_none() {
            self.list_hubs().await;
        }
        let mut machine = self.hubs.other(text).ok_or_else(|| not_found(text))?;
        let id = machine.rental_id.clone().unwrap_or_default();
        self.ask_end(&machine.hub, &id, None)?;
        machine.state = MachineState::Ending;
        Ok(Ended {
            machine,
            changed: true,
            abandoned: vec![],
        })
    }

    fn ask_end(
        self: &Arc<Self>,
        hub: &str,
        id: &str,
        machine: Option<String>,
    ) -> Result<(), Error> {
        let ask = Ask {
            key: format!("end-{id}"),
            kind: AskKind::End,
            hub: hub.to_owned(),
            digest: id.to_owned(),
            body: vec![],
            machine,
            run: None,
            state: AskState::Pending,
            reason: None,
        };
        let (ask, _) = self.store().begin_ask(&ask).map_err(internal)?;
        if ask.state == AskState::Failed {
            self.store().reopen_ask(&ask.key).map_err(internal)?;
        }
        // A pending end shows the machine `ending`.
        self.calcifer.machine_changed();
        let me = self.clone();
        self.spawn(async move { me.finish_end(ask).await });
        Ok(())
    }

    /// Releases the rental at its Hub, then reads it until the Hub says it ended. A Hub that
    /// does not know the rental proves nothing: the ask fails `release_unconfirmed` and the
    /// machine stays.
    async fn finish_end(self: Arc<Self>, ask: Ask) {
        let id = ask.digest.clone();
        let mut pause = Duration::ZERO;
        let settle = |outcome: Result<(), String>| {
            let settled = match &outcome {
                Ok(()) => self.store().settle_ask(&ask.key, Ok(None)),
                Err(reason) => self.store().settle_ask(&ask.key, Err(reason)),
            };
            if let Err(e) = settled {
                warn!(ask = ask.key, "{e}");
            }
            self.calcifer.machine_changed();
        };
        let mut released = false;
        loop {
            let client = match self.hubs.client(&ask.hub) {
                Ok(client) => client,
                Err(e) => return settle(Err(e.message)),
            };
            if !released {
                match client
                    .release_rental(&id)
                    .await
                    .map_err(hub_client::Error::from)
                {
                    Ok(_) => released = true,
                    Err(hub_client::Error::Hub { status: 404, .. }) => {
                        return settle(Err(format!(
                            "release_unconfirmed: {} does not know rental {id}; nothing was ended",
                            ask.hub
                        )));
                    }
                    Err(e) if e.refused() => released = true,
                    Err(e) => warn!(rental = id, "releasing it: {e}"),
                }
            }
            if released {
                match client.get_rental(&id).await {
                    Ok(rental) => {
                        let rental = rental.into_inner();
                        let gone_now = gone(&rental.state).then(|| cause(&rental));
                        self.hubs.seen(&ask.hub, rental);
                        if let Some(cause) = gone_now {
                            if let Some(name) = &ask.machine {
                                self.ended(name, &cause);
                            }
                            return settle(Ok(()));
                        }
                    }
                    Err(e) => warn!(
                        rental = id,
                        "reading its end: {}",
                        hub_client::Error::from(e)
                    ),
                }
            }
            pause = (pause * 2).clamp(Duration::from_secs(2), Duration::from_secs(30));
            tokio::time::sleep(pause).await;
        }
    }
}

impl Hubs {
    /// One rental as its Hub says it now.
    async fn read(&self, origin: &str, id: &str) -> Result<types::Rental, Error> {
        let client = self.client(origin)?;
        let rental = client
            .get_rental(id)
            .await
            .map_err(|e| hub_refusal(origin, e.into()))?
            .into_inner();
        self.seen(origin, rental.clone());
        Ok(rental)
    }
}

/// A rent ask as recorded: the exact request the Hub gets, and the quote it is held to.
#[derive(serde::Serialize, serde::Deserialize)]
struct RentBody {
    request: types::RentalRequest,
    quote: types::RentalQuote,
}

impl RentBody {
    fn of(ask: &Ask) -> Result<Self, Error> {
        serde_json::from_slice(&ask.body).map_err(|e| internal(format!("the recorded ask: {e}")))
    }
}

fn quote_of(quote: &types::RentalQuote) -> Quote {
    let expected = &quote.expected_price;
    Quote {
        gpu_usd_micros_per_hour: quote.price_usd_micros_per_hour,
        storage_usd_micros_per_hour: quote.storage_usd_micros_per_hour,
        bandwidth_usd_micros_per_gb: quote.bandwidth_usd_micros_per_gb,
        disk_gb: u32::try_from(quote.container_disk_gb).unwrap_or_default(),
        expected_hours: expected.hours,
        expected_download_gb: u32::try_from(expected.download_gb).unwrap_or_default(),
        expected_total_usd_micros: expected.total_usd_micros,
    }
}

/// What asking a Hub came to, short of an answer.
enum Asked {
    /// The Hub's refusal, which asking again would not change.
    Refused(Error),
    Unanswered(Error),
}

fn asked(origin: &str, e: hub_client::Error) -> Asked {
    match e.refused() {
        true => Asked::Refused(hub_refusal(origin, e)),
        false => Asked::Unanswered(hub_refusal(origin, e)),
    }
}

/// A Hub's refusal as the API answers it: the Hub's own code is the cause.
pub(crate) fn hub_refusal(origin: &str, e: hub_client::Error) -> Error {
    match e {
        hub_client::Error::Hub {
            status,
            code,
            message,
            remedy,
        } => {
            let error = Error::new(Kind::of_http(status), "hub_refused", format!("{origin}: {message}"))
                .cause(code);
            if remedy.is_empty() {
                error
            } else {
                error.remedy(remedy)
            }
        }
        e => Error::new(
            Kind::Unavailable,
            "hub_unavailable",
            format!("{origin}: {e}"),
        ),
    }
}

pub(crate) fn signed_out(origin: &str) -> Error {
    let current = config::load(config::Flags::default()).and_then(|loaded| loaded.config.current());
    let login = match current {
        Ok(hub) if config::Origin::parse(origin).is_ok_and(|o| o == hub.origin) => {
            "cozy login <email>".to_owned()
        }
        _ => format!("cozy login <email> --tensorhub {origin}"),
    };
    Error::new(
        Kind::Credential,
        "signed_out",
        format!("this computer is not signed in to {origin}"),
    )
    .next(login)
}

/// The Hub has proven the pod gone.
fn gone(state: &str) -> bool {
    matches!(state, "failed" | "released")
}

/// Why an ended rental ended: who released it, else its failure.
fn cause(rental: &types::Rental) -> String {
    rental
        .release_cause
        .clone()
        .or_else(|| rental.failure.as_ref().map(|f| f.code.clone()))
        .or_else(|| rental.detail.clone())
        .unwrap_or_else(|| format!("its Hub says it {}", rental.state))
}

/// The name the Hub gave the pod, when a machine can carry it.
fn hub_name(rental: &types::Rental) -> Option<String> {
    rental
        .machine_name
        .as_deref()
        .map(str::to_ascii_lowercase)
        .filter(|name| crate::machines::valid_name(name).is_ok())
}

/// A ready rental's address and pinned leaf, once the Hub publishes all of them.
fn endpoint(rental: &types::Rental) -> Option<Endpoint> {
    if rental.state != "ready" {
        return None;
    }
    let present = |field: &Option<String>| field.clone().filter(|v| !v.is_empty());
    Some(Endpoint {
        worker_id: present(&rental.worker_id)?,
        address: present(&rental.worker_address)?,
        pin: pin_of(rental.cert_pem.as_deref()?)?,
    })
}

/// A booting rental's phase, for people.
fn boot_phase(rental: &types::Rental) -> Option<String> {
    let boot = rental.boot.as_ref();
    let words: Vec<&str> = [
        Some(rental.state.as_str()),
        boot.and_then(|b| b.phase.as_deref()),
        boot.and_then(|b| b.activity.as_deref()),
    ]
    .into_iter()
    .flatten()
    .filter(|w| !w.is_empty())
    .collect();
    (!words.is_empty()).then(|| words.join(": "))
}

/// What the rental costs an hour, the machine and its disk: until a machine is bought the Hub's
/// `hourly_rate` is the machine's alone.
fn hourly(rental: &types::Rental) -> i64 {
    match rental.compute_usd_micros_per_hour {
        Some(compute) => compute + rental.storage_usd_micros_per_hour.unwrap_or_default(),
        None => rental.hourly_rate_usd_micros,
    }
}

fn view(rental: &types::Rental, sku: Option<String>) -> Rental {
    Rental {
        sku: sku.unwrap_or_else(|| rental.requested_accelerator_model.clone()),
        accelerator: rental.requested_accelerator_model.clone(),
        gpus: u32::try_from(rental.accelerator_count).unwrap_or_default(),
        usd_micros_per_hour: hourly(rental),
        spend_usd_micros: Some(rental.spend_usd_micros),
        spend_basis: Some(rental.spend_basis.clone()).filter(|b| !b.is_empty()),
        rented_at_ms: ms(&rental.created_at).unwrap_or_default(),
        ended_at_ms: rental.ended_at.as_deref().and_then(ms),
        disk_gb: u32::try_from(rental.container_disk_gb).unwrap_or_default(),
        failure: rental.failure.as_ref().map(|f| f.code.clone()),
        placement_note: rental.placement_note.clone().filter(|n| !n.is_empty()),
        provider_warning: rental.provider_warning.clone().filter(|w| !w.is_empty()),
        unreachable_since_ms: rental.unreachable_since.as_deref().and_then(ms),
        base_worker_image_digest: Some(rental.base_worker_image_digest.clone())
            .filter(|d| !d.is_empty()),
    }
}

/// The machine a rent asks for, beyond its SKU, and the most its renter agreed to pay.
#[derive(Default)]
pub struct Shape {
    pub gpus: Option<u32>,
    pub disk_gb: Option<u32>,
    pub image: Option<String>,
    pub agreed: Option<Agreed>,
}

/// The Hub's request for renting `sku` in `shape`, under a fresh alias and signing key.
fn request(sku: &str, shape: Shape) -> Result<(types::RentalRequest, String, SigningKey), Error> {
    let mut random = [0; 68];
    getrandom::fill(&mut random).expect("the OS provides randomness");
    let alias = format!("rent-{}", hex::encode(&random[..4]));
    let signer = SigningKey::from_bytes(random[36..].try_into().expect("32 bytes"));
    let request = types::RentalRequest {
        accelerator_count: shape.gpus.map(i64::from),
        container_disk_gb: shape.disk_gb.map(i64::from),
        creator_public_key: URL_SAFE_NO_PAD.encode(signer.verifying_key().to_bytes()),
        development: None,
        excluded_provider_machines: vec![],
        image: (shape.image.map(|image| image.try_into()).transpose()).map_err(
            |e: types::error::ConversionError| Error::new(Kind::Usage, "image_invalid", e.to_string()),
        )?,
        // The Hub holds a media plane's token hash; machines serve bytes over Read instead.
        media_token_sha256: hex::encode(Sha256::digest(&random[4..36])),
        name: alias.clone(),
        planned_source_bytes: None,
        provider: None,
        serving_models: vec![],
        sku: sku.to_owned(),
    };
    Ok((request, alias, signer))
}

/// Whether `now` asks more than `agreed`, an hour or a GB.
fn above(now: &Agreed, agreed: &Agreed) -> bool {
    now.usd_micros_per_hour > agreed.usd_micros_per_hour
        || now.usd_micros_per_gb > agreed.usd_micros_per_gb
}

fn rate(price: &Agreed) -> String {
    format!(
        "${:.2}/h and up to ${:.3}/GB",
        price.usd_micros_per_hour as f64 / 1e6,
        price.usd_micros_per_gb as f64 / 1e6
    )
}

/// An RFC 3339 time of the Hub's, in unix milliseconds.
fn ms(at: &str) -> Option<i64> {
    let at = time::OffsetDateTime::parse(at, &time::format_description::well_known::Rfc3339);
    at.ok()
        .map(|at| (at.unix_timestamp_nanos() / 1_000_000) as i64)
}

/// A live rental of this account that this controller did not rent.
fn other(origin: &str, rental: &types::Rental, unverified: bool) -> Machine {
    Machine {
        name: hub_name(rental).unwrap_or_else(|| rental.rental_id.clone()),
        kind: MachineKind::Rental,
        state: match rental.state.as_str() {
            "ready" | "degraded" => MachineState::Ready,
            "release_requested" => MachineState::Ending,
            _ => MachineState::Booting,
        },
        hub: origin.to_owned(),
        rental_id: Some(rental.rental_id.clone()),
        alias: None,
        address: rental.worker_address.clone(),
        fingerprint: rental
            .cert_pem
            .as_deref()
            .and_then(pin_of)
            .map(protocol::ids::Digest::from_bytes),
        reason: None,
        rental: Some(view(rental, None)),
        status: None,
        unreachable: None,
        boot: boot_phase(rental).filter(|_| rental.state != "ready"),
        running: 0,
        queued: 0,
        recorded: false,
        unverified,
    }
}

/// Whether `text` names the rental: its id, its Hub name, or the name it was asked under.
fn names(rental: &types::Rental, text: &str) -> bool {
    rental.rental_id == text
        || rental.name.eq_ignore_ascii_case(text)
        || rental
            .machine_name
            .as_deref()
            .is_some_and(|name| name.eq_ignore_ascii_case(text))
}
