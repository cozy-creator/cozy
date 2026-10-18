//! What a run carries to its Hub, minted at the send: where its machine reads the Hub (the Hub's
//! own word, kept per Hub: in a dev or CI Hub it is not the origin this computer dials), and, for
//! a run that publishes into its weights destination or reads one of its owner's checkpoints by
//! digest, the owner's capability for exactly that, signed offline with this computer's device
//! key and bound to the machine's TLS leaf (th-241). The machine trades it at the Hub's token
//! endpoint at its first private operation. The machine's spec digest leaves the capability out,
//! so a resend with a fresh one names the same run. Signed out, a public run reads at its Hub's
//! origin.
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use cozy_proto::{
    calcifer::Reason,
    config::{self, Origin},
    error::{Error, Kind},
    ids::RunId,
    machine as v1,
};
use hub_client::{DeviceKey, RunCapability};
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};

use crate::{
    link,
    machines::{Machines, internal, lock},
    rentals::{Hubs, hub_refusal, signed_out},
    store::RunRow,
};

/// A capability's life: a run past it reads only public content.
const CAPABILITY_LIFE: Duration = Duration::from_secs(24 * 3600);

/// What a send carries besides the spec the run was submitted with.
#[derive(Default)]
pub(crate) struct Access {
    pub hub: Option<v1::HubAccess>,
    pub providers: Option<v1::ProviderAccess>,
    /// The account local code's org-relative references name: this computer's at the run's Hub.
    pub owner: Option<String>,
}

/// Why a run cannot be sent now.
pub(crate) enum Denied {
    /// For good: the run fails with this.
    Refused(Reason),
    /// The Hub or the machine did not answer: try again.
    Unreachable(String),
}

/// What a Hub says of this computer's account there, read once per Calcifer.
#[derive(Clone)]
pub(crate) struct Facts {
    account: String,
    user_id: String,
    /// Where machines read the Hub, and its public origin: a capability's audience.
    reads: String,
    public: String,
    object_hosts: Vec<String>,
    /// The CA a machine trusts for the Hub when a private one signs it (a test Hub).
    ca_der: Vec<u8>,
    token_endpoint: String,
}

impl Machines {
    /// The Hub access and provider tokens the run's send to `machine` carries.
    pub(crate) async fn access(&self, run: &RunRow, machine: &str) -> Result<Access, Denied> {
        let providers = self.providers.clone();
        let spec = &run.spec;
        let Some(origin) = run.hub.as_ref().filter(|origin| {
            // Signed out, a public run reads at its Hub's origin (lib's finish_spec).
            self.device_key(origin).is_ok() || private(spec)
        }) else {
            return Ok(Access {
                hub: None,
                providers,
                owner: None,
            });
        };
        let facts = self.hubs.facts(origin).await?;
        let operations = match private(spec) {
            true => operations(&facts.account, spec),
            false => vec![],
        };
        let mut hub = v1::HubAccess {
            origin: facts.reads.clone(),
            object_hosts: facts.object_hosts.clone(),
            ca_der: facts.ca_der.clone(),
            ..Default::default()
        };
        if !operations.is_empty() {
            hub.capability = (self.capability(run, origin, machine, &facts, operations)).await?;
            hub.token_endpoint = facts.token_endpoint;
        }
        Ok(Access {
            hub: Some(hub),
            providers,
            owner: Some(facts.account.clone()),
        })
    }

    /// What a send of `run` to `machine` would carry to its Hub now, its capability minted
    /// fresh; none when this computer is not signed in there and the run is public.
    pub async fn hub_access(
        &self,
        run: &RunId,
        machine: &str,
    ) -> Result<Option<v1::HubAccess>, Error> {
        let row = self.store().run(run.as_str()).map_err(internal)?;
        let row =
            row.ok_or_else(|| Error::new(Kind::NotFound, "run_not_found", run.to_string()))?;
        match self.access(&row, machine).await {
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

    /// The owner's capability for `operations`, bound to the leaf `machine` presents.
    async fn capability(
        &self,
        run: &RunRow,
        origin: &Origin,
        machine: &str,
        facts: &Facts,
        operations: Vec<Value>,
    ) -> Result<String, Denied> {
        let row = self
            .store()
            .machine(machine)
            .map_err(|e| unreachable(e.to_string()))?;
        let endpoint = row
            .and_then(|row| row.endpoint)
            .ok_or_else(|| unreachable(format!("{machine} has no address")))?;
        let leaf = link::presented_leaf(&endpoint.address)
            .await
            .map_err(|e| unreachable(e.to_string()))?;
        if Sha256::digest(&leaf).as_slice() != endpoint.pin {
            return Err(unreachable(format!(
                "{machine} presented another TLS leaf than the one pinned"
            )));
        }
        let machine = hub_client::leaf_thumbprint(&leaf)
            .map_err(|e| refused("machine_key_unsupported", format!("{machine}: {e}")))?;
        let key = self.device_key(origin)?;
        Ok(key.sign_run_capability(&RunCapability {
            user_id: facts.user_id.clone(),
            hub: facts.public.clone(),
            machine,
            run: run.id.to_string(),
            operations,
            lifetime: CAPABILITY_LIFE,
        }))
    }

    fn device_key(&self, origin: &Origin) -> Result<DeviceKey, Denied> {
        let hub = config::Hub {
            name: None,
            origin: origin.clone(),
        };
        DeviceKey::load(&hub.device_key(&self.home)).map_err(|_| needs_sign_in(origin))
    }
}

impl Hubs {
    /// This computer's account at `origin`, where machines read it, and where they trade a
    /// capability: read once, then kept for the life of this Calcifer.
    pub(crate) async fn facts(&self, origin: &Origin) -> Result<Facts, Denied> {
        if let Some(facts) = lock(&self.facts).get(origin) {
            return Ok(facts.clone());
        }
        let client = self.client(origin).map_err(|_| needs_sign_in(origin))?;
        let hub = |e: hub_client::Error| match &e {
            hub_client::Error::Hub { status, .. } if *status < 500 => {
                let error = hub_refusal(origin, e);
                Denied::Refused(Reason {
                    code: error.cause.unwrap_or(error.code),
                    message: error.message,
                })
            }
            _ => unreachable(hub_refusal(origin, e).message),
        };
        let account = client
            .get_current_account()
            .await
            .map_err(|e| hub(e.into()))?;
        let environment = client.get_execution_environment().await;
        let environment = environment
            .map_err(|e| hub(e.into()))?
            .into_inner()
            .environment;
        let account = account.into_inner();
        let facts = Facts {
            account: account.name,
            user_id: account.user_id,
            reads: environment.tensorhub_origin,
            public: environment.tensorhub_public_origin,
            object_hosts: environment
                .tensorhub_object_storage_hosts
                .unwrap_or_default()
                .split(',')
                .map(str::trim)
                .filter(|host| !host.is_empty())
                .map(str::to_owned)
                .collect(),
            ca_der: environment
                .tensorhub_ca_der_b64url
                .and_then(|ca| URL_SAFE_NO_PAD.decode(ca.trim()).ok())
                .unwrap_or_default(),
            token_endpoint: client.token_endpoint().await.map_err(hub)?,
        };
        if facts.user_id.is_empty() {
            return Err(refused(
                "hub_account_incomplete",
                format!("{origin} names no user for account {}", facts.account),
            ));
        }
        lock(&self.facts).insert(origin.clone(), facts.clone());
        Ok(facts)
    }
}

/// Whether the run may do something at its Hub that is not public: publish, or read a
/// checkpoint by digest alone.
fn private(spec: &v1::RunSpec) -> bool {
    !destination(spec).is_empty()
        || spec.models.iter().any(|choice| {
            by_digest(&choice.release, &choice.manifest)
                || choice
                    .adapters
                    .iter()
                    .any(|adapter| by_digest(&adapter.release, &adapter.manifest))
        })
}

/// What the run does at its Hub that is not public (Go's `privateOperations`): read one of
/// `owner`'s checkpoints named by digest alone (a release's lanes are published), and publish
/// into its weights destination.
fn operations(owner: &str, spec: &v1::RunSpec) -> Vec<Value> {
    let mut operations = vec![];
    let mut read = |model: &str, release: &str, manifest: &str| {
        let owned = model
            .split_once('/')
            .is_some_and(|(org, _)| org.eq_ignore_ascii_case(owner));
        let read = json!({"type": "tensorhub_model_read", "model": model, "manifest": manifest});
        if owned && by_digest(release, manifest) && !operations.contains(&read) {
            operations.push(read);
        }
    };
    for choice in &spec.models {
        read(&choice.repository, &choice.release, &choice.manifest);
        for adapter in &choice.adapters {
            read(&adapter.model, &adapter.release, &adapter.manifest);
        }
    }
    let destination = destination(spec);
    if !destination.is_empty() {
        operations.push(json!({"type": "tensorhub_model_publish", "model": destination}));
    }
    operations
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
    })
}

fn unreachable(message: String) -> Denied {
    Denied::Unreachable(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(repository: &str, release: &str, manifest: &str) -> v1::ModelChoice {
        v1::ModelChoice {
            repository: repository.into(),
            release: release.into(),
            manifest: manifest.into(),
            ..Default::default()
        }
    }

    /// A capability names exactly what the run does privately: the owner's checkpoints read by
    /// digest (once each), and a publish into a Hub weights destination.
    #[test]
    fn a_run_names_only_its_private_operations() {
        let mut spec = v1::RunSpec {
            models: vec![
                choice("Fidika/sdxl", "", "sha256:aa"),
                choice("fidika/sdxl", "", "sha256:aa"),
                choice("fidika/flux", "2026.1", "sha256:bb"),
                choice("other/model", "", "sha256:cc"),
            ],
            weights_destination: "model://fidika/out".into(),
            ..Default::default()
        };
        spec.models[2].adapters.push(v1::Adapter {
            model: "fidika/lora".into(),
            manifest: "sha256:dd".into(),
            ..Default::default()
        });
        assert!(private(&spec));
        assert_eq!(
            operations("fidika", &spec),
            vec![
                json!({"type": "tensorhub_model_read", "model": "Fidika/sdxl", "manifest": "sha256:aa"}),
                json!({"type": "tensorhub_model_read", "model": "fidika/sdxl", "manifest": "sha256:aa"}),
                json!({"type": "tensorhub_model_read", "model": "fidika/lora", "manifest": "sha256:dd"}),
                json!({"type": "tensorhub_model_publish", "model": "fidika/out"}),
            ]
        );
        let public = v1::RunSpec {
            models: vec![choice("fidika/sdxl", "2026.1", "")],
            weights_destination: "local/out".into(),
            ..Default::default()
        };
        assert!(
            !private(&public),
            "a release and a local destination are public"
        );
    }
}
