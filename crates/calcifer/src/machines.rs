//! Machines: this computer's (`local`), rentals, and other computers' added by address.
//!
//! Each machine with an endpoint has one `Status` stream. Its phase, idle deadline, GPUs and
//! live runs are held here and answer `GET /v1/machines`; nothing is polled. A machine that
//! cannot be reached waits and is shown so: only its Hub's word, its own release, or its owner
//! ends it. Unpinned runs are placed here: `any` on a rental of this controller that fits,
//! `new:<sku>` on one rented for the run (`rentals`).
use std::{
    collections::{HashMap, HashSet},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Duration,
};

use axum::{
    Json,
    body::Bytes,
    extract::{Path as UrlPath, Query, State, rejection::QueryRejection},
    http::{HeaderMap, StatusCode, header},
    response::{IntoResponse, Response},
};
use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use cozy_platform::{durable, paths::Home};
use cozy_proto::{
    calcifer::{
        self as wire, Ended, Event, Keepalive, Machine, MachineKind, MachineList, MachineLog,
        MachineState, MachineStatus, NewMachine,
    },
    config,
    error::{Error, Kind},
    ids::{Digest, RunId},
    machine as v1,
};
use ed25519_dalek::{
    SigningKey,
    pkcs8::{DecodePrivateKey as _, EncodePrivateKey as _},
};
use prost::Message as _;
use serde::Deserialize;
use sha2::{Digest as _, Sha256};
use tokio::{
    sync::{Notify, broadcast::error::RecvError},
    task::{AbortHandle, JoinSet},
};
use tracing::warn;

use crate::{
    Calcifer,
    api::{Api, Refusal, parse},
    link::{self, Link, LinkError},
    rentals::Hubs,
    store::{AskKind, Endpoint, MachineRow, Store},
};

pub struct Machines {
    pub(crate) calcifer: Arc<Calcifer>,
    pub(crate) home: Home,
    pub(crate) hubs: Hubs,
    /// Model-source tokens every run carries (`ProviderAccess`), from the config.
    pub(crate) providers: Option<v1::ProviderAccess>,
    pub(crate) seen: Mutex<HashMap<String, Seen>>,
    watching: Mutex<HashMap<String, (Endpoint, AbortHandle)>>,
    /// Rentals whose boot is followed, by rental id.
    pub(crate) booting: Mutex<HashSet<String>>,
    /// Runs a rental is being asked for.
    pub(crate) renting: Mutex<HashSet<RunId>>,
    /// Told when a rental appears or is ready: `any` runs may fit it.
    pub(crate) placing: Notify,
    tasks: Mutex<JoinSet<()>>,
}

/// What Calcifer last heard of a machine, never stored.
#[derive(Default)]
pub(crate) struct Seen {
    frame: Option<v1::StatusFrame>,
    unreachable: Option<String>,
    /// A booting rental's phase, as its Hub says it.
    pub(crate) boot: Option<String>,
}

impl Machines {
    /// Watches every machine Calcifer holds, follows booting rentals, asks again what it asked
    /// a Hub and was not answered, and places unplaced runs. The config names the Hubs listed
    /// (a rental's own Hub is always read) and the model-source tokens runs carry.
    pub fn start(calcifer: Arc<Calcifer>, home: Home, config: &config::Config) -> Arc<Self> {
        let token = |secret: &Option<config::Secret>| {
            secret
                .as_ref()
                .map(|s| s.expose().to_owned())
                .unwrap_or_default()
        };
        let providers = v1::ProviderAccess {
            huggingface: token(&config.huggingface_token),
            civitai: token(&config.civitai_token),
        };
        let me = Arc::new(Self {
            hubs: Hubs::new(&home, config.hubs()),
            providers: (providers != v1::ProviderAccess::default()).then_some(providers),
            home,
            seen: Mutex::default(),
            watching: Mutex::default(),
            booting: Mutex::default(),
            renting: Mutex::default(),
            placing: Notify::new(),
            tasks: Mutex::default(),
            calcifer,
        });
        me.calcifer.serve_with(&me);
        match me.store().machines() {
            Ok(rows) => {
                for row in rows.iter().filter(|r| r.state != MachineState::Ended) {
                    match row.endpoint {
                        Some(_) => me.watch(row),
                        None if row.kind == MachineKind::Rental => me.follow_boot(row.clone()),
                        None => {}
                    }
                }
            }
            Err(e) => warn!("reading the machines: {e}"),
        }
        me.resume_asks();
        me.spawn(me.clone().place_runs());
        let lister = me.clone();
        me.spawn(async move { lister.list_hubs().await });
        me
    }

    /// The machine's latest Status, once one has arrived.
    pub fn frame(&self, name: &str) -> Option<v1::StatusFrame> {
        lock(&self.seen)
            .get(name)
            .and_then(|seen| seen.frame.clone())
    }

    /// Stops every task where it stands, as if the process ended.
    pub fn close(&self) {
        lock(&self.tasks).abort_all();
    }

    pub(crate) fn store(&self) -> &Store {
        self.calcifer.store()
    }

    pub(crate) fn spawn(&self, task: impl Future<Output = ()> + Send + 'static) -> AbortHandle {
        let mut tasks = lock(&self.tasks);
        while tasks.try_join_next().is_some() {}
        tasks.spawn(task)
    }

    /// Every machine: Calcifer's own, and the live rentals of this account it did not rent.
    /// `refresh` lists every Hub again first; otherwise the Hubs' last word is answered.
    pub async fn list(self: &Arc<Self>, refresh: bool) -> Result<MachineList, Error> {
        if refresh {
            self.list_hubs().await;
        }
        let rows = self.store().machines().map_err(internal)?;
        let mut machines = Vec::new();
        for row in rows.iter().filter(|r| r.state != MachineState::Ended) {
            machines.push(self.picture(row)?);
        }
        let (hubs, unanswered, signed_out, others) = self.hubs.listing(&rows);
        machines.extend(others);
        Ok(MachineList {
            machines,
            hubs,
            unanswered,
            signed_out,
        })
    }

    /// One machine by its name, former name or rental id, with its Status read now.
    pub async fn show(self: &Arc<Self>, text: &str) -> Result<Machine, Error> {
        let Some(row) = self.store().find_machine(text).map_err(internal)? else {
            return self.hubs.other(text).ok_or_else(|| not_found(text));
        };
        if row.state != MachineState::Ended
            && let Ok(Some(link)) = self.calcifer.link(&row.name)
        {
            match first(link.status(false)).await {
                Ok(frame) => self.observe(&row.name, frame),
                Err(e) => self.unreachable(&row.name, e.to_string()),
            }
        }
        self.picture(&row)
    }

    /// Adds another computer's machine: it must present the leaf `fingerprint` names and admit
    /// this controller's key before anything is recorded.
    pub async fn add(
        self: &Arc<Self>,
        name: &str,
        address: &str,
        fingerprint: &Digest,
    ) -> Result<Machine, Error> {
        valid_name(name)?;
        if let Some(row) = self.store().find_machine(name).map_err(internal)?
            && (row.state != MachineState::Ended || row.kind != MachineKind::Added)
        {
            return Err(Error::new(
                Kind::Conflict,
                "machine_name_taken",
                format!("{name} already names a machine"),
            )
            .next(format!("cozy machine show {name}")));
        }
        let unreachable = |e: LinkError| {
            Error::new(
                Kind::Unavailable,
                "machine_unreachable",
                format!("{address}: {e}"),
            )
        };
        let leaf: [u8; 32] =
            Sha256::digest(link::presented_leaf(address).await.map_err(unreachable)?).into();
        if &leaf != fingerprint.bytes() {
            return Err(Error::new(
                Kind::Credential,
                "fingerprint_mismatch",
                format!(
                    "{address} presents the TLS leaf {}, not {fingerprint}",
                    Digest::from_bytes(leaf)
                ),
            )
            .remedy("ask the machine's owner for its fingerprint again"));
        }
        let identity = Link::identity(address, leaf).await.map_err(unreachable)?;
        let key = self.owner_key()?;
        let row = MachineRow {
            name: name.into(),
            kind: MachineKind::Added,
            hub: String::new(),
            rental_id: None,
            endpoint: Some(Endpoint {
                worker_id: identity.worker_id,
                address: address.into(),
                pin: leaf,
            }),
            key: key.clone(),
            state: MachineState::Ready,
            served: true,
            reason: None,
        };
        let link =
            Link::new(row.endpoint.as_ref().expect("set"), signer(&key)?).map_err(unreachable)?;
        let frame = first(link.status(false)).await.map_err(|e| match e {
            LinkError::Unauthorized(_) => match self.controller_key() {
                Ok(public) => Error::new(
                    Kind::Credential,
                    "machine_unauthorized",
                    format!("{address} does not admit this controller's key"),
                )
                .remedy(format!("on that computer: cozy machine authorize {public}")),
                Err(e) => e,
            },
            e => unreachable(e),
        })?;
        self.calcifer.put_machine(&row).map_err(internal)?;
        self.observe(name, frame);
        self.watch(&row);
        self.picture(&row)
    }

    /// Records this computer's machine, which signs with this controller's owner key.
    pub fn put_local(self: &Arc<Self>, endpoint: Endpoint) -> Result<Machine, Error> {
        let row = MachineRow {
            name: "local".into(),
            kind: MachineKind::Local,
            hub: String::new(),
            rental_id: None,
            endpoint: Some(endpoint),
            key: self.owner_key()?,
            state: MachineState::Ready,
            served: true,
            reason: None,
        };
        self.calcifer.put_machine(&row).map_err(internal)?;
        self.watch(&row);
        self.picture(&row)
    }

    /// Ends a rental at its Hub (`ending` until the Hub affirms it), or removes an added machine
    /// from this controller. `local` is disabled, never ended.
    pub async fn end(self: &Arc<Self>, text: &str) -> Result<Ended, Error> {
        let Some(row) = self.store().find_machine(text).map_err(internal)? else {
            return self.end_other(text).await;
        };
        if row.state == MachineState::Ended {
            return Ok(Ended {
                machine: self.picture(&row)?,
                changed: false,
            });
        }
        match row.kind {
            MachineKind::Local => Err(Error::new(
                Kind::Conflict,
                "machine_local",
                "local is this computer's machine: it is disabled, not ended",
            )
            .next("cozy machine disable")),
            MachineKind::Added => {
                self.ended(&row.name, "removed from this controller");
                Ok(Ended {
                    machine: self.picture(&self.row(&row.name)?)?,
                    changed: true,
                })
            }
            MachineKind::Rental => self.end_rental(row).await,
        }
    }

    /// Starts a rental's 15-minute idle clock over, once the machine acknowledges it.
    pub async fn keepalive(&self, text: &str) -> Result<Keepalive, Error> {
        let (row, link) = self.linked(text)?;
        let frame = first(link.status(true))
            .await
            .map_err(|e| refusal(&row.name, e))?;
        let idle_deadline_ms = deadline(&frame);
        self.observe(&row.name, frame);
        Ok(Keepalive {
            acknowledged_at_ms: crate::store::now_ms(),
            idle_deadline_ms,
        })
    }

    /// The last `tail` bytes (0: all) of a log the machine keeps; empty when it keeps none.
    pub async fn logs(&self, text: &str, log: MachineLog, tail: u64) -> Result<Vec<u8>, Error> {
        let (row, link) = self.linked(text)?;
        let log = match log {
            MachineLog::Tensorfs => "tensorfs-transport",
            MachineLog::Memory => "memory",
        };
        match link.read_log(log, tail).await {
            Err(LinkError::NotFound) => Ok(vec![]),
            read => read.map_err(|e| refusal(&row.name, e)),
        }
    }

    pub(crate) fn row(&self, name: &str) -> Result<MachineRow, Error> {
        self.store()
            .machine(name)
            .map_err(internal)?
            .ok_or_else(|| not_found(name))
    }

    /// A machine and its connection, which it has only once it has an endpoint.
    fn linked(&self, text: &str) -> Result<(MachineRow, Link), Error> {
        let row = self
            .store()
            .find_machine(text)
            .map_err(internal)?
            .filter(|row| row.state != MachineState::Ended)
            .ok_or_else(|| not_found(text))?;
        match self.calcifer.link(&row.name) {
            Ok(Some(link)) => Ok((row, link)),
            Ok(None) => Err(Error::new(
                Kind::Unavailable,
                "machine_booting",
                format!("{} has no address yet", row.name),
            )),
            Err(e) => Err(Error::new(Kind::Unavailable, "machine_unreachable", e)),
        }
    }

    /// The machine as `GET /v1/machines` answers it.
    pub(crate) fn picture(&self, row: &MachineRow) -> Result<Machine, Error> {
        let store = self.store();
        let mut machine = Machine::from(row);
        machine.alias = store.alias(&row.name).map_err(internal)?;
        let load = store.machine_load().map_err(internal)?;
        (machine.running, machine.queued) = load.get(&row.name).copied().unwrap_or_default();
        let ending = store
            .pending_asks()
            .map_err(internal)?
            .into_iter()
            .any(|ask| {
                ask.kind == AskKind::End && ask.machine.as_deref() == Some(row.name.as_str())
            });
        if ending && row.state != MachineState::Ended {
            machine.state = MachineState::Ending;
        }
        if let Some(seen) = lock(&self.seen).get(&row.name) {
            machine.status = seen.frame.as_ref().map(status);
            machine.unreachable.clone_from(&seen.unreachable);
            machine.boot.clone_from(&seen.boot);
        }
        let asked = match row.kind {
            MachineKind::Rental => store.rent_ask(&row.name).map_err(internal)?,
            _ => None,
        };
        self.hubs.describe(&mut machine, asked.as_ref());
        Ok(machine)
    }

    /// Keeps one `Status` stream open to the machine while it has an endpoint and has not ended.
    pub(crate) fn watch(self: &Arc<Self>, row: &MachineRow) {
        let mut watching = lock(&self.watching);
        let endpoint = row
            .endpoint
            .clone()
            .filter(|_| row.state != MachineState::Ended);
        let current = watching.get(&row.name);
        if current.is_some_and(|(e, task)| Some(e) == endpoint.as_ref() && !task.is_finished()) {
            return;
        }
        if let Some((_, task)) = watching.remove(&row.name) {
            task.abort();
        }
        if let Some(endpoint) = endpoint {
            let task = self.spawn(self.clone().follow_status(row.name.clone(), row.kind));
            watching.insert(row.name.clone(), (endpoint, task));
        }
    }

    /// Records the machine as gone for good: its runs settle as the core rules, and it is no
    /// longer watched.
    pub(crate) fn ended(&self, name: &str, reason: &str) {
        if let Err(e) = self.calcifer.machine_ended(name, reason) {
            warn!(machine = name, "recording its end: {e}");
        }
        if let Some((_, task)) = lock(&self.watching).remove(name) {
            task.abort();
        }
        if let Some(seen) = lock(&self.seen).get_mut(name) {
            seen.boot = None;
        }
    }

    /// Follows the machine's Status for as long as it lives. A broken stream is opened again,
    /// paced; for a rental that stays unreachable its Hub is asked, paced apart, whether it
    /// still exists, which is the only word that ends it.
    async fn follow_status(self: Arc<Self>, name: String, kind: MachineKind) {
        let (mut pause, mut hub_pause) = (Duration::ZERO, Duration::ZERO);
        let mut hub_at = tokio::time::Instant::now();
        loop {
            match self.store().machine(&name) {
                Ok(Some(row)) if row.state != MachineState::Ended => {}
                _ => return,
            }
            let reason = match self.status_frames(&name, &mut pause).await {
                Ok(()) => {
                    self.ended(&name, "it released itself after its idle time");
                    return;
                }
                Err(reason) => reason,
            };
            self.unreachable(&name, reason);
            if pause.is_zero() {
                hub_pause = Duration::ZERO;
            }
            if kind == MachineKind::Rental && tokio::time::Instant::now() >= hub_at {
                hub_pause = (hub_pause * 2).clamp(Duration::from_secs(5), Duration::from_secs(60));
                hub_at = tokio::time::Instant::now() + hub_pause;
                if self.hub_says_gone(&name).await {
                    return;
                }
            }
            pause = (pause * 2).clamp(Duration::from_millis(500), Duration::from_secs(10));
            tokio::time::sleep(pause).await;
        }
    }

    /// Reads the machine's Status frames until it says it released itself (`Ok`) or the stream
    /// breaks. Any frame resets the pause.
    async fn status_frames(&self, name: &str, pause: &mut Duration) -> Result<(), String> {
        let link = self
            .calcifer
            .link(name)?
            .ok_or_else(|| "it has no address".to_string())?;
        let mut frames = link.status(false).await.map_err(|e| e.to_string())?;
        loop {
            match frames.message().await {
                Ok(Some(frame)) => {
                    *pause = Duration::ZERO;
                    let released = frame.phase == "releasing";
                    self.observe(name, frame);
                    if released {
                        return Ok(());
                    }
                }
                Ok(None) => return Err("the machine closed its Status stream".into()),
                Err(status) => return Err(link::classify(status).to_string()),
            }
        }
    }

    fn observe(&self, name: &str, frame: v1::StatusFrame) {
        self.calcifer.observe_warm(name, frame.warm.clone());
        let mut seen = lock(&self.seen);
        let seen = seen.entry(name.to_owned()).or_default();
        if frame.phase == "ready" && seen.frame.as_ref().is_none_or(|f| f.phase != "ready") {
            self.placing.notify_one();
        }
        seen.frame = Some(frame);
        seen.unreachable = None;
    }

    fn unreachable(&self, name: &str, reason: String) {
        lock(&self.seen)
            .entry(name.to_owned())
            .or_default()
            .unreachable = Some(reason);
    }

    /// Places each unplaced run, now, whenever one is created or returned to the outbox, and
    /// whenever a rental appears or is ready.
    async fn place_runs(self: Arc<Self>) {
        let mut events = self.calcifer.subscribe();
        loop {
            self.place_unplaced();
            loop {
                let event = tokio::select! {
                    event = events.recv() => event,
                    () = self.placing.notified() => break,
                };
                match event {
                    Ok((_, Event::Created { run, .. } | Event::Queued { run, .. }))
                        if run.machine.is_none() =>
                    {
                        break;
                    }
                    Ok(_) => {}
                    Err(RecvError::Lagged(_)) => break,
                    Err(RecvError::Closed) => return,
                }
            }
        }
    }

    fn place_unplaced(self: &Arc<Self>) {
        let unplaced = match self.store().unplaced() {
            Ok(unplaced) => unplaced,
            Err(e) => return warn!("reading unplaced runs: {e}"),
        };
        for (run, wants) in unplaced {
            let gpus = v1::RunSpec::decode(run.spec.as_slice()).map_or(0, |spec| spec.gpus);
            let placed = match wants.as_str() {
                // It waits for a rental that fits while none does.
                "any" => match self.fitting(gpus) {
                    Ok(Some(name)) => self.calcifer.place(&run.id, &name).map(drop),
                    Ok(None) => Ok(()),
                    Err(e) => Err(crate::Error::Invalid(e.message)),
                },
                // `new` rents what the Hub places it on; `new:<sku>` that product.
                wants => {
                    self.rent_for(&run, wants.strip_prefix("new:"), gpus);
                    Ok(())
                }
            };
            if let Err(e) = placed {
                warn!(run = %run.id, "placing it: {e}");
            }
        }
    }

    /// The least busy rental of this controller that fits a run of `gpus` GPUs (0: any), a
    /// ready one before a booting one: runs wait for a rental on its way, never buy around it.
    fn fitting(&self, gpus: u32) -> Result<Option<String>, Error> {
        let store = self.store();
        let load = store.machine_load().map_err(internal)?;
        let ending: HashSet<String> = store
            .pending_asks()
            .map_err(internal)?
            .into_iter()
            .filter(|ask| ask.kind == AskKind::End)
            .filter_map(|ask| ask.machine)
            .collect();
        let rows = store.machines().map_err(internal)?;
        let seen = lock(&self.seen);
        let count = |row: &MachineRow| match seen.get(&row.name).and_then(|s| s.frame.as_ref()) {
            Some(frame) if frame.phase == "ready" => Some(frame.gpus.len() as u32),
            _ => self.hubs.gpus(row),
        };
        Ok(rows
            .into_iter()
            .filter(|row| row.kind == MachineKind::Rental && row.state != MachineState::Ended)
            .filter(|row| !ending.contains(&row.name))
            .filter(|row| gpus == 0 || count(row).is_none_or(|n| n >= gpus))
            .min_by_key(|row| {
                let (running, queued) = load.get(&row.name).copied().unwrap_or_default();
                (row.endpoint.is_none(), running + queued, row.name.clone())
            })
            .map(|row| row.name))
    }

    /// This controller's signer for its own and added machines, minted on first use.
    pub(crate) fn owner_key(&self) -> Result<PathBuf, Error> {
        let path = self.home.owner_key();
        if !path.exists() {
            mint_key(&path)?;
        }
        Ok(path)
    }

    /// The owner key's public half, as `cozy machine authorize` takes it.
    pub fn controller_key(&self) -> Result<String, Error> {
        let key = signer(&self.owner_key()?)?;
        Ok(URL_SAFE_NO_PAD.encode(key.verifying_key().to_bytes()))
    }
}

#[derive(Deserialize)]
pub(crate) struct Refresh {
    #[serde(default)]
    refresh: bool,
}

pub(crate) async fn list(
    State(api): State<Arc<Api>>,
    query: Result<Query<Refresh>, QueryRejection>,
) -> Result<Json<MachineList>, Refusal> {
    let Query(query) = query.map_err(|e| Refusal(usage(e.body_text())))?;
    Ok(Json(
        api.machines.list(query.refresh).await.map_err(Refusal)?,
    ))
}

#[derive(Deserialize)]
pub(crate) struct Fresh {
    #[serde(default)]
    fresh: bool,
}

/// The machine; `?fresh=true` answers a Status it read now, or refuses.
pub(crate) async fn show(
    State(api): State<Arc<Api>>,
    UrlPath(name): UrlPath<String>,
    query: Result<Query<Fresh>, QueryRejection>,
) -> Result<Json<Machine>, Refusal> {
    let Query(query) = query.map_err(|e| Refusal(usage(e.body_text())))?;
    let machine = match query.fresh {
        true => fresh(&api.machines, &name).await,
        false => api.machines.show(&name).await,
    };
    Ok(Json(machine.map_err(Refusal)?))
}

/// One of Calcifer's machines with the Status frame it answered now, recorded as its latest:
/// one that cannot answer is `machine_unavailable`, never its last picture.
async fn fresh(machines: &Arc<Machines>, text: &str) -> Result<Machine, Error> {
    let row = machines.store().find_machine(text).map_err(internal)?;
    let row = row.ok_or_else(|| not_found(text))?;
    let unavailable = |why: String| {
        let message = format!("{} cannot answer now: {why}", row.name);
        Error::new(Kind::Unavailable, "machine_unavailable", message)
    };
    if row.state == MachineState::Ended {
        return Err(unavailable("it has ended".into()));
    }
    let link = machines.calcifer.link(&row.name).map_err(unavailable)?;
    let link = link.ok_or_else(|| unavailable("it has no address yet".into()))?;
    match first(link.status(false)).await {
        Ok(frame) => machines.observe(&row.name, frame),
        Err(e) => {
            machines.unreachable(&row.name, e.to_string());
            return Err(unavailable(e.to_string()));
        }
    }
    machines.picture(&row)
}

/// Rents one under the request's `Idempotency-Key` (202, or 200 when the key had), or adds
/// another computer's.
pub(crate) async fn create(
    State(api): State<Arc<Api>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, Refusal> {
    match parse::<NewMachine>(&body)? {
        NewMachine::Rent {
            sku,
            hub,
            gpus,
            disk_gb,
        } => {
            let key = headers
                .get("idempotency-key")
                .and_then(|v| v.to_str().ok())
                .map(str::trim)
                .filter(|key| !key.is_empty())
                .ok_or_else(|| {
                    Refusal(
                        Error::new(
                            Kind::Usage,
                            "idempotency_key_required",
                            "a rental is a paid ask: it needs an Idempotency-Key header",
                        )
                        .remedy("one key names one rental forever; retrying under it is safe"),
                    )
                })?;
            let rented = api
                .machines
                .rent(key, &hub, &sku, gpus, disk_gb)
                .await
                .map_err(Refusal)?;
            let status = match rented.replay {
                true => StatusCode::OK,
                false => StatusCode::ACCEPTED,
            };
            Ok((status, Json(rented)).into_response())
        }
        NewMachine::Add {
            name,
            address,
            fingerprint,
        } => {
            let added = api.machines.add(&name, &address, &fingerprint).await;
            Ok(Json(added.map_err(Refusal)?).into_response())
        }
    }
}

/// 202 while its Hub has not affirmed the end, 200 once it is over.
pub(crate) async fn end(
    State(api): State<Arc<Api>>,
    UrlPath(name): UrlPath<String>,
) -> Result<(StatusCode, Json<Ended>), Refusal> {
    let ended = api.machines.end(&name).await.map_err(Refusal)?;
    let status = match ended.machine.state {
        MachineState::Ending => StatusCode::ACCEPTED,
        _ => StatusCode::OK,
    };
    Ok((status, Json(ended)))
}

pub(crate) async fn keepalive(
    State(api): State<Arc<Api>>,
    UrlPath(name): UrlPath<String>,
) -> Result<Json<Keepalive>, Refusal> {
    Ok(Json(api.machines.keepalive(&name).await.map_err(Refusal)?))
}

#[derive(Deserialize)]
pub(crate) struct Tail {
    #[serde(default)]
    tail_bytes: u64,
}

pub(crate) async fn logs(
    State(api): State<Arc<Api>>,
    UrlPath((name, log)): UrlPath<(String, String)>,
    query: Result<Query<Tail>, QueryRejection>,
) -> Result<Response, Refusal> {
    let Query(query) = query.map_err(|e| Refusal(usage(e.body_text())))?;
    let log = match log.as_str() {
        "tensorfs" => MachineLog::Tensorfs,
        "memory" => MachineLog::Memory,
        other => {
            return Err(Refusal(Error::new(
                Kind::NotFound,
                "machine_log_unknown",
                format!("machines keep no log {other:?}: tensorfs or memory"),
            )));
        }
    };
    let bytes = api
        .machines
        .logs(&name, log, query.tail_bytes)
        .await
        .map_err(Refusal)?;
    Ok(([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], bytes).into_response())
}

fn usage(message: String) -> Error {
    Error::new(Kind::Usage, "query_invalid", message)
}

/// How to reach the TensorD whose `api-ready.json` is at `path`.
pub fn endpoint_of(path: &Path) -> Result<Endpoint, Error> {
    #[derive(serde::Deserialize)]
    struct Ready {
        address: String,
        worker_id: String,
        cert_pem: String,
    }
    let unreadable = |e: String| {
        Error::new(
            Kind::Unavailable,
            "machine_unreadable",
            format!("{}: {e}", path.display()),
        )
    };
    let bytes = std::fs::read(path).map_err(|e| unreadable(e.to_string()))?;
    let ready: Ready = serde_json::from_slice(&bytes).map_err(|e| unreadable(e.to_string()))?;
    Ok(Endpoint {
        worker_id: ready.worker_id,
        address: ready.address,
        pin: pin_of(&ready.cert_pem).ok_or_else(|| unreadable("no certificate".into()))?,
    })
}

/// The sha256 of a PEM certificate's DER.
pub(crate) fn pin_of(pem: &str) -> Option<[u8; 32]> {
    let body: String = pem
        .lines()
        .filter(|line| !line.starts_with("-----"))
        .collect();
    let der = base64::engine::general_purpose::STANDARD
        .decode(body.trim())
        .ok()
        .filter(|der| !der.is_empty())?;
    Some(Sha256::digest(der).into())
}

/// Merges a warm run's items into the machine's warm set, keyed by source and entrypoint: the
/// spec then names the whole set. Level 0 removes an item. The merged set is what the machine
/// holds once it takes the run, so the next warm run merges into it.
pub(crate) fn merge_warm(current: &mut Vec<v1::WarmItem>, spec: &mut v1::RunSpec) {
    if spec.kind != v1::RunKind::Warm as i32 {
        return;
    }
    let Some(set) = spec.set.as_mut() else {
        return;
    };
    for item in std::mem::take(&mut set.items) {
        current.retain(|held| (&held.source, &held.entrypoint) != (&item.source, &item.entrypoint));
        if item.level != v1::WarmLevel::Unspecified as i32 {
            current.push(item);
        }
    }
    set.items = current
        .iter()
        .map(|held| v1::WarmItem {
            holds: String::new(),
            held_back: String::new(),
            ..held.clone()
        })
        .collect();
    *current = set.items.clone();
}

fn status(frame: &v1::StatusFrame) -> MachineStatus {
    let level = |level: i32| {
        v1::WarmLevel::try_from(level)
            .map(|l| {
                l.as_str_name()
                    .trim_start_matches("WARM_LEVEL_")
                    .to_ascii_lowercase()
            })
            .unwrap_or_default()
    };
    MachineStatus {
        worker_id: frame.worker_id.clone(),
        version: frame.version.clone(),
        platform: frame.platform.clone(),
        phase: frame.phase.clone(),
        runtime: frame.runtime.clone(),
        tensorfs: frame.tensorfs.clone(),
        started_at_ms: frame.started_at_unix_ms,
        idle_deadline_ms: deadline(frame),
        gpus: frame
            .gpus
            .iter()
            .map(|gpu| wire::Gpu {
                index: gpu.index,
                name: gpu.name.clone(),
                memory_bytes: gpu.memory_bytes,
                driver: gpu.driver.clone(),
                family: Some(gpu.family.clone()).filter(|f| !f.is_empty()),
                memory_class: Some(gpu.memory_class.clone()).filter(|c| !c.is_empty()),
            })
            .collect(),
        disk_total_bytes: frame.disk.as_ref().map_or(0, |d| d.total_bytes),
        disk_free_bytes: frame.disk.as_ref().map_or(0, |d| d.free_bytes),
        live_runs: frame.runs.len() as u32,
        models: frame
            .models
            .iter()
            .map(|model| wire::Model {
                repository: model.repository.clone(),
                total_bytes: model.total_bytes,
                unique_bytes: model.unique_bytes,
                checkpoints: model
                    .checkpoints
                    .iter()
                    .map(|c| wire::Checkpoint {
                        release: c.release.clone(),
                        lane: c.lane.clone(),
                        manifest: c.manifest.clone(),
                        yanked: c.yanked,
                    })
                    .collect(),
            })
            .collect(),
        models_bytes: frame.models_bytes,
        environments: frame
            .environments
            .iter()
            .map(|env| wire::Environment {
                installation: env.installation.clone(),
                package: env.package.clone(),
                release: env.release.clone(),
                hub: env.hub.clone(),
                level: env.level.clone(),
                warning: env.warning.clone(),
            })
            .collect(),
        warm: frame
            .warm
            .iter()
            .map(|item| wire::WarmItem {
                source: match &item.source {
                    Some(v1::warm_item::Source::Release(r)) if r.release.is_empty() => {
                        r.package.clone()
                    }
                    Some(v1::warm_item::Source::Release(r)) => {
                        format!("{}@{}", r.package, r.release)
                    }
                    Some(v1::warm_item::Source::Installation(id)) => id.clone(),
                    None => String::new(),
                },
                entrypoint: item.entrypoint.clone(),
                level: level(item.level),
                holds: item.holds.clone(),
                held_back: item.held_back.clone(),
            })
            .collect(),
    }
}

fn deadline(frame: &v1::StatusFrame) -> Option<i64> {
    (frame.idle_deadline_unix_ms > 0).then_some(frame.idle_deadline_unix_ms)
}

/// The first frame of a Status stream.
async fn first(
    stream: impl Future<Output = Result<tonic::Streaming<v1::StatusFrame>, LinkError>>,
) -> Result<v1::StatusFrame, LinkError> {
    stream
        .await?
        .message()
        .await
        .map_err(link::classify)?
        .ok_or_else(|| LinkError::Unreachable("the machine sent no Status".into()))
}

/// A machine name: 1-32 of `a-z0-9-`, a letter first; `local`, `any` and `new` are taken.
pub(crate) fn valid_name(name: &str) -> Result<(), Error> {
    let mut chars = name.chars();
    let valid = name.len() <= 32
        && chars.next().is_some_and(|c| c.is_ascii_lowercase())
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
        && !["local", "any", "new"].contains(&name);
    if valid {
        Ok(())
    } else {
        Err(Error::new(
            Kind::Usage,
            "machine_name_invalid",
            format!(
                "{name:?} is not a machine name: 1-32 lowercase letters, digits and hyphens, a \
                 letter first, and not local, any or new"
            ),
        ))
    }
}

/// Mints an Ed25519 key at `path`.
pub(crate) fn mint_key(path: &Path) -> Result<SigningKey, Error> {
    let mut seed = [0; 32];
    getrandom::fill(&mut seed).expect("the OS provides randomness");
    let key = SigningKey::from_bytes(&seed);
    write_key(path, &key)?;
    Ok(key)
}

/// Keeps `key` at `path`: PKCS#8 PEM, owner-only.
pub(crate) fn write_key(path: &Path, key: &SigningKey) -> Result<(), Error> {
    let pem = key
        .to_pkcs8_pem(Default::default())
        .map_err(|e| internal(e.to_string()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    durable::private_dir(dir)
        .and_then(|()| durable::replace(path, pem.as_bytes(), 0o600))
        .map_err(|e| internal(e.to_string()))
}

pub(crate) fn signer(path: &Path) -> Result<SigningKey, Error> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|pem| SigningKey::from_pkcs8_pem(&pem).ok())
        .ok_or_else(|| {
            Error::new(
                Kind::Credential,
                "key_unreadable",
                format!("cannot read the signing key {}", path.display()),
            )
        })
}

/// A machine's refusal, as the API answers it.
fn refusal(name: &str, e: LinkError) -> Error {
    match e {
        LinkError::Refused { code, message } => Error::new(
            Kind::Conflict,
            "machine_refused",
            format!("{name}: {message}"),
        )
        .cause(code),
        LinkError::Unauthorized(message) => Error::new(
            Kind::Credential,
            "machine_unauthorized",
            format!("{name}: {message}"),
        ),
        LinkError::Released(message) => Error::new(
            Kind::Unavailable,
            "machine_released",
            format!("{name}: {message}"),
        ),
        LinkError::NotFound => Error::new(
            Kind::NotFound,
            "not_found",
            format!("{name} has no such thing"),
        ),
        LinkError::Unreachable(message) => Error::new(
            Kind::Unavailable,
            "machine_unreachable",
            format!("{name}: {message}"),
        ),
    }
}

pub(crate) fn not_found(text: &str) -> Error {
    Error::new(
        Kind::NotFound,
        "machine_not_found",
        format!("no machine is named {text:?}"),
    )
    .next("cozy machine list")
}

pub(crate) fn internal(e: impl ToString) -> Error {
    Error::new(Kind::Internal, "calcifer_store", e.to_string())
}

pub(crate) fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(package: &str, entrypoint: &str, level: v1::WarmLevel) -> v1::WarmItem {
        v1::WarmItem {
            source: Some(v1::warm_item::Source::Release(v1::Release {
                package: package.into(),
                release: String::new(),
            })),
            entrypoint: entrypoint.into(),
            level: level as i32,
            ..Default::default()
        }
    }

    fn warm(items: Vec<v1::WarmItem>) -> v1::RunSpec {
        v1::RunSpec {
            kind: v1::RunKind::Warm as i32,
            set: Some(v1::WarmSet { items }),
            ..Default::default()
        }
    }

    /// One item per warm run becomes the whole set the machine is sent, and the next warm run
    /// merges into that set even before the machine reports it.
    #[test]
    fn a_warm_runs_item_merges_into_the_machines_set() {
        let mut held = vec![v1::WarmItem {
            holds: "gpu".into(),
            ..item("a/sdxl", "generate", v1::WarmLevel::Gpu)
        }];
        let mut first = warm(vec![item("a/flux", "generate", v1::WarmLevel::Host)]);
        merge_warm(&mut held, &mut first);
        assert_eq!(
            first.set.unwrap().items,
            vec![
                item("a/sdxl", "generate", v1::WarmLevel::Gpu),
                item("a/flux", "generate", v1::WarmLevel::Host)
            ],
            "Status-only fields are not sent"
        );
        let mut second = warm(vec![
            item("a/sdxl", "generate", v1::WarmLevel::Unspecified),
            item("a/flux", "generate", v1::WarmLevel::Gpu),
        ]);
        merge_warm(&mut held, &mut second);
        assert_eq!(
            second.set.unwrap().items,
            vec![item("a/flux", "generate", v1::WarmLevel::Gpu)],
            "level 0 removes an item; the same source and entrypoint replaces it"
        );
        let mut call = v1::RunSpec::default();
        merge_warm(&mut held, &mut call);
        assert_eq!(call, v1::RunSpec::default(), "only warm runs are merged");
    }
}
