//! What a run's send carries for its Hub: where its machine reads the Hub (the Hub's own word,
//! kept per Hub: in a dev or CI Hub it is not the origin this computer dials) and, for a run that
//! publishes into its weights destination or reads one of its owner's checkpoints by digest,
//! this computer's user Bearer as its TensorFS token. The Bearer is added at the send, never to
//! the stored submission; each one the session mints after goes to every machine sent one, at the
//! session's own refresh point (`PUT /v1/bearer`). Signed out, a public run reads at its Hub's
//! origin.
use std::{
    sync::Weak,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use hub_client::{Bearer, Credential};
use protocol::{
    calcifer::Reason,
    config::Origin,
    error::{Error, Kind},
    ids::RunId,
    machine as v1,
};

use crate::{
    machines::{Machines, internal, lock},
    rentals::{Hubs, hub_refusal, signed_out},
    store::RunRow,
};

/// What a send carries besides the spec the run was submitted with.
#[derive(Default)]
pub(crate) struct Access {
    pub hub: Option<v1::HubAccess>,
    pub providers: Option<v1::ProviderAccess>,
    pub tensorfs: Option<v1::TensorFSOptions>,
    /// The account local code's org-relative references name: this computer's at the run's Hub.
    pub owner: Option<String>,
}

/// Why a run cannot be sent now.
pub(crate) enum Denied {
    /// For good: the run fails with this.
    Refused(Reason),
    /// The Hub did not answer: try again.
    Unreachable(String),
}

/// What a Hub says of this computer's account there, read once per device key: a login as
/// another account (a new key) reads it again.
#[derive(Clone)]
pub(crate) struct Facts {
    key: String,
    account: String,
    /// Where machines read the Hub.
    reads: String,
    /// Both where machines read it and its public origin: the configured Hub's other names.
    names: Vec<Origin>,
    /// The CA a machine trusts for the Hub when a private one signs it (a test Hub).
    ca_der: Vec<u8>,
}

impl Machines {
    /// The Hub access and provider tokens the send of `run` as `spec` carries.
    pub(crate) async fn access(&self, run: &RunRow, spec: &v1::RunSpec) -> Result<Access, Denied> {
        let providers = self.providers.clone();
        let origin = match &run.hub {
            Some(hub) => Some(self.hubs.resolve(hub).await),
            None => None,
        };
        let Some(origin) = origin.filter(|origin| {
            // Signed out, a public run reads at its Hub's origin (lib's finish_spec).
            self.hubs.client(origin).is_ok() || private(spec, None)
        }) else {
            return Ok(Access {
                providers,
                ..Default::default()
            });
        };
        let facts = self.hubs.facts(&origin).await?;
        let bearer = match needs_bearer(spec, &facts.account) {
            true => Some(self.hubs.bearer(&origin).await?),
            false => None,
        };
        if let (Some(machine), Some(bearer)) = (&run.machine, &bearer) {
            self.holds(machine, &origin, bearer);
        }
        let tensorfs = bearer.map(|bearer| tensorfs(&facts, &bearer));
        Ok(Access {
            hub: Some(v1::HubAccess {
                origin: facts.reads,
                ca_der: facts.ca_der,
            }),
            providers,
            tensorfs,
            owner: Some(facts.account),
        })
    }

    /// What a send of the stored `run` would carry to its Hub now; none when this computer is
    /// not signed in there and the run is public.
    pub async fn hub_access(&self, run: &RunId) -> Result<Option<v1::HubAccess>, Error> {
        let row = self.store().run(run.as_str()).map_err(internal)?;
        let row =
            row.ok_or_else(|| Error::new(Kind::NotFound, "run_not_found", run.to_string()))?;
        match self.access(&row, &row.spec).await {
            Ok(access) => Ok(access.hub),
            Err(Denied::Refused(reason)) => {
                Err(Error::new(Kind::Conflict, reason.code, reason.message))
            }
            Err(Denied::Unreachable(message)) => Err(Error::new(
                Kind::Unavailable,
                "hub_access_unavailable",
                message,
            )),
        }
    }
}

impl Machines {
    /// `machine` was sent `bearer`, this computer's at `origin`: it is given each one minted after.
    fn holds(&self, machine: &str, origin: &Origin, bearer: &Bearer) {
        let mut holders = lock(&self.bearers);
        let armed = holders.contains_key(origin);
        holders
            .entry(origin.clone())
            .or_default()
            .insert(machine.to_owned());
        drop(holders);
        if !armed {
            let granting = self.calcifer.granting.get().cloned().unwrap_or_default();
            self.spawn(refresh(granting, origin.clone(), bearer.clone()));
        }
    }

    /// Mints this computer's bearer at `origin` and gives it to each machine sent one that is
    /// still driven. None, and no machine is given any more, once none is left or none can be
    /// minted.
    async fn push(&self, origin: &Origin) -> Option<Bearer> {
        let minted = match self.hubs.facts(origin).await {
            Ok(facts) => self
                .hubs
                .bearer(origin)
                .await
                .ok()
                .map(|bearer| (tensorfs(&facts, &bearer), bearer)),
            Err(_) => None,
        };
        if let Some((options, _)) = &minted {
            let holders: Vec<String> = lock(&self.bearers)
                .get(origin)
                .into_iter()
                .flatten()
                .cloned()
                .collect();
            // An offline machine is not dialed; it gets the next one once it is back.
            for name in holders.into_iter().filter(|name| self.offline(name).is_none()) {
                match self.linked(&name) {
                    Ok((_, link)) => {
                        if let Err(error) = link.bearer(options).await {
                            tracing::warn!(machine = %name, %error, "giving the machine its Hub bearer");
                        }
                    }
                    Err(_) => {
                        lock(&self.bearers)
                            .entry(origin.clone())
                            .or_default()
                            .remove(&name);
                    }
                }
            }
        }
        let mut held = lock(&self.bearers);
        if minted.is_none() || held.get(origin).is_none_or(|machines| machines.is_empty()) {
            held.remove(origin);
            return None;
        }
        minted.map(|(_, bearer)| bearer)
    }
}

/// From the session's own refresh point of `held`, the bearer the machines were sent, each one
/// it mints at `origin` goes to them (`push`), until that stops.
async fn refresh(machines: Weak<Machines>, origin: Origin, held: Bearer) {
    let mut held = Some(held);
    while let Some(bearer) = held.take() {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        if bearer.refresh_at() <= now {
            tracing::warn!(%origin, "the Hub's bearers expire within the session's refresh margin");
            break;
        }
        tokio::time::sleep(Duration::from_secs(bearer.refresh_at() - now)).await;
        let Some(machines) = machines.upgrade() else {
            return;
        };
        held = machines.push(&origin).await;
        if held.is_none() {
            return;
        }
    }
    if let Some(machines) = machines.upgrade() {
        lock(&machines.bearers).remove(&origin);
    }
}

/// What a machine reads `facts`' Hub with under `bearer`.
fn tensorfs(facts: &Facts, bearer: &Bearer) -> v1::TensorFSOptions {
    v1::TensorFSOptions {
        endpoint: facts.reads.clone(),
        token: bearer.token.clone(),
        expires_unix_ms: i64::try_from(bearer.expires)
            .unwrap_or(i64::MAX)
            .saturating_mul(1000),
        ca_der: facts.ca_der.clone(),
    }
}

impl Hubs {
    /// This computer's live Bearer at `origin`, from the normal cached user session: held, kept,
    /// or minted again past its refresh point. The key stays on this computer.
    async fn bearer(&self, origin: &Origin) -> Result<Bearer, Denied> {
        let client = self.client(origin).map_err(|_| needs_sign_in(origin))?;
        let Credential::User(session) = client.credential() else {
            return Err(needs_sign_in(origin));
        };
        session
            .bearer(&client.anonymous())
            .await
            .map_err(|error| hub_denied(origin, error))
    }

    /// This computer's account at `origin` and where machines read it: read once per device
    /// key, so a login, logout or account switch is seen at the next run with no Hub read.
    pub(crate) async fn facts(&self, origin: &Origin) -> Result<Facts, Denied> {
        let client = self.client(origin).map_err(|_| needs_sign_in(origin))?;
        let Credential::User(session) = client.credential() else {
            return Err(needs_sign_in(origin));
        };
        let key = session.key().public_key();
        if let Some(facts) = lock(&self.facts)
            .get(origin)
            .filter(|facts| facts.key == key)
        {
            return Ok(facts.clone());
        }
        let hub = |error| hub_denied(origin, error);
        let account = client
            .get_current_account()
            .await
            .map_err(|e| hub(e.into()))?;
        let environment = client.get_execution_environment().await;
        let environment = environment
            .map_err(|e| hub(e.into()))?
            .into_inner()
            .environment;
        let names = [
            &environment.tensorhub_origin,
            &environment.tensorhub_public_origin,
        ];
        let facts = Facts {
            key,
            account: account.into_inner().name,
            names: names.iter().filter_map(|o| Origin::parse(o).ok()).collect(),
            reads: environment.tensorhub_origin,
            ca_der: environment
                .tensorhub_ca_der_b64url
                .and_then(|ca| URL_SAFE_NO_PAD.decode(ca.trim()).ok())
                .unwrap_or_default(),
        };
        lock(&self.facts).insert(origin.clone(), facts.clone());
        Ok(facts)
    }

    /// The configured Hub `origin` names: itself, or the one a machine reads or the public
    /// knows at `origin` (a dev Hub this computer dials on loopback is read elsewhere), as the
    /// facts read so far say.
    pub(crate) fn configured(&self, origin: &Origin) -> Option<Origin> {
        if self.configured.iter().any(|hub| &hub.origin == origin) {
            return Some(origin.clone());
        }
        let facts = lock(&self.facts);
        let named = self
            .configured
            .iter()
            .find(|hub| (facts.get(&hub.origin)).is_some_and(|facts| facts.names.contains(origin)));
        named.map(|hub| hub.origin.clone())
    }

    /// `origin`'s configured Hub, reading the facts of each configured Hub not read yet when the
    /// ones read so far do not name it; `origin` itself when none does.
    pub(crate) async fn resolve(&self, origin: &Origin) -> Origin {
        if let Some(hub) = self.configured(origin) {
            return hub;
        }
        self.learn().await;
        self.configured(origin).unwrap_or_else(|| origin.clone())
    }

    /// Reads the facts of each configured Hub this computer is signed in to and has not read.
    pub(crate) async fn learn(&self) {
        for hub in &self.configured {
            if !lock(&self.facts).contains_key(&hub.origin) {
                let _ = self.facts(&hub.origin).await;
            }
        }
    }

    /// An installation's Hub as a machine records it, as this computer names that Hub: the
    /// configured Hub's origin, else its one spelling; local code's stays empty.
    pub(crate) fn named(&self, hub: &str) -> String {
        match Origin::parse(hub) {
            Ok(origin) => self.configured(&origin).unwrap_or(origin).into(),
            Err(_) => hub.to_owned(),
        }
    }
}

/// Whether the send carries `owner`'s Bearer: the run reads one of `owner`'s checkpoints by digest
/// or publishes, and names no TensorFS options of its own (an explicit endpoint gets none).
fn needs_bearer(spec: &v1::RunSpec, owner: &str) -> bool {
    spec.tensorfs.is_none() && private(spec, Some(owner))
}

/// The existing outbound Hub context also names public model defaults resolved later by
/// TensorD. Copying its transport acquires no credential and makes no network call.
pub(crate) fn anonymous_tensorfs(spec: &v1::RunSpec) -> Option<v1::TensorFSOptions> {
    let hub = spec.hub.as_ref().filter(|hub| !hub.origin.is_empty())?;
    Some(v1::TensorFSOptions {
        endpoint: hub.origin.clone(),
        ca_der: hub.ca_der.clone(),
        ..Default::default()
    })
}

fn hub_denied(origin: &Origin, error: hub_client::Error) -> Denied {
    let refused = error.refused();
    let error = hub_refusal(origin, error);
    match refused {
        true => Denied::Refused(Reason {
            code: error.cause.unwrap_or(error.code),
            message: error.message,
            ..Default::default()
        }),
        false => unreachable(error.message),
    }
}

/// Whether the run does something at its Hub that is not public: publish into its weights
/// destination, or read a checkpoint by digest alone (a release's lanes are published), as a
/// model choice or a warm set member's. With `owner`, only `owner`'s checkpoints count.
fn private(spec: &v1::RunSpec, owner: Option<&str>) -> bool {
    let read = |model: &str, release: &str, manifest: &str| {
        by_digest(release, manifest)
            && owner.is_none_or(|owner| {
                (model.split_once('/')).is_some_and(|(org, _)| org.eq_ignore_ascii_case(owner))
            })
    };
    let members = spec.set.iter().flat_map(|set| &set.items);
    let mut choices = spec
        .models
        .iter()
        .chain(members.flat_map(|item| &item.models));
    !destination(spec).is_empty()
        || choices.any(|choice| {
            read(&choice.repository, &choice.release, &choice.manifest)
                || (choice.adapters.iter()).any(|a| read(&a.model, &a.release, &a.manifest))
        })
}

fn by_digest(release: &str, manifest: &str) -> bool {
    release.is_empty() && !manifest.is_empty()
}

/// The Hub model a run publishes into; empty for none or a `local/` one.
fn destination(spec: &v1::RunSpec) -> &str {
    let destination = spec.weights_destination.trim_start_matches("model://");
    match destination.starts_with("local/") {
        true => "",
        false => destination,
    }
}

fn needs_sign_in(origin: &str) -> Denied {
    let error: Error = signed_out(origin);
    refused(
        "hub_sign_in_required",
        format!(
            "this run publishes to or reads private models at {origin}, which needs this \
             computer's sign-in there: {}",
            error.message
        ),
    )
}

fn refused(code: &str, message: String) -> Denied {
    Denied::Refused(Reason {
        code: code.into(),
        message,
        ..Default::default()
    })
}

fn unreachable(message: String) -> Denied {
    Denied::Unreachable(message)
}
