//! What a run carries to its Hub, minted at the send: where its machine reads the Hub (the Hub's
//! own word, kept per Hub: in a dev or CI Hub it is not the origin this computer dials), and, for
//! a run that publishes into its weights destination or reads one of its owner's checkpoints by
//! digest, the owner's capability for exactly that, signed offline with this computer's device
//! key and bound to the machine's TLS leaf (th-241). The machine trades it at the Hub's token
//! endpoint at its first private operation. The machine's spec digest leaves the capability out,
//! so a resend with a fresh one names the same run. Signed out, a public run reads at its Hub's
//! origin.
//! Ordinary TensorFS access uses this computer's existing user Bearer at send time; it is never
//! added to the stored submission.
use std::time::Duration;

use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
use hub_client::{Credential, DeviceKey, RunCapability};
use protocol::{
    calcifer::Reason,
    config::{self, Origin},
    error::{Error, Kind},
    ids::RunId,
    machine as v1,
};
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
    pub tensorfs: Option<v1::TensorFSOptions>,
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
    /// Both, as a machine may record the Hub: the configured Hub's other names.
    names: Vec<Origin>,
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
        let origin = match &run.hub {
            Some(hub) => Some(self.hubs.resolve(hub).await),
            None => None,
        };
        let Some(origin) = origin.filter(|origin| {
            // Signed out, a public run reads at its Hub's origin (lib's finish_spec).
            self.device_key(origin).is_ok() || private(spec)
        }) else {
            return Ok(Access {
                hub: None,
                providers,
                tensorfs: None,
                owner: None,
            });
        };
        let origin = &origin;
        let facts = self.hubs.facts(origin).await?;
        let operations = match private(spec) {
            true => operations(&facts.account, spec),
            false => vec![],
        };
        let ordinary = ordinary_tensorfs_needed(spec, &operations);
        let mut hub = v1::HubAccess {
            origin: facts.reads.clone(),
            object_hosts: facts.object_hosts.clone(),
            ca_der: facts.ca_der.clone(),
            ..Default::default()
        };
        if !operations.is_empty() {
            hub.capability = (self.capability(run, origin, machine, &facts, operations)).await?;
            hub.token_endpoint = facts.token_endpoint.clone();
        }
        let tensorfs = match ordinary {
            true => Some(self.hubs.ordinary_tensorfs(origin, &facts).await?),
            false => None,
        };
        Ok(Access {
            hub: Some(hub),
            providers,
            tensorfs,
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
    /// Uses the normal cached user session. The key stays on this computer; only its ordinary
    /// Bearer and the Hub's existing machine-reachable endpoint/CA travel with this send.
    async fn ordinary_tensorfs(
        &self,
        origin: &Origin,
        facts: &Facts,
    ) -> Result<v1::TensorFSOptions, Denied> {
        let client = self.client(origin).map_err(|_| needs_sign_in(origin))?;
        let Credential::User(session) = client.credential() else {
            return Err(needs_sign_in(origin));
        };
        let token = session
            .bearer(&client.anonymous())
            .await
            .map_err(|error| hub_denied(origin, error))?;
        Ok(v1::TensorFSOptions {
            endpoint: facts.reads.clone(),
            token,
            ca_der: facts.ca_der.clone(),
        })
    }

    /// This computer's account at `origin`, where machines read it, and where they trade a
    /// capability: read once, then kept for the life of this Calcifer.
    pub(crate) async fn facts(&self, origin: &Origin) -> Result<Facts, Denied> {
        if let Some(facts) = lock(&self.facts).get(origin) {
            return Ok(facts.clone());
        }
        let client = self.client(origin).map_err(|_| needs_sign_in(origin))?;
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
        let account = account.into_inner();
        let names = [
            &environment.tensorhub_origin,
            &environment.tensorhub_public_origin,
        ];
        let facts = Facts {
            account: account.name,
            user_id: account.user_id,
            names: names.iter().filter_map(|o| Origin::parse(o).ok()).collect(),
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

/// The existing private-read/write decision also decides whether ordinary TensorFS needs this
/// Hub's owner credential. Explicit caller options remain authoritative.
fn ordinary_tensorfs_needed(spec: &v1::RunSpec, operations: &[Value]) -> bool {
    spec.tensorfs.is_none() && !operations.is_empty()
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
    match &error {
        hub_client::Error::Hub { status, .. } if *status < 500 => {
            let error = hub_refusal(origin, error);
            Denied::Refused(Reason {
                code: error.cause.unwrap_or(error.code),
                message: error.message,
            })
        }
        _ => unreachable(hub_refusal(origin, error).message),
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
        let read = json!({"type": "tensorhub_model_read", "model": model.to_ascii_lowercase(), "manifest": manifest});
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
        operations.push(
            json!({"type": "tensorhub_model_publish", "model": destination.to_ascii_lowercase()}),
        );
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
            weights_destination: "model://Fidika/Out".into(),
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
                json!({"type": "tensorhub_model_read", "model": "fidika/sdxl", "manifest": "sha256:aa"}),
                json!({"type": "tensorhub_model_read", "model": "fidika/lora", "manifest": "sha256:dd"}),
                json!({"type": "tensorhub_model_publish", "model": "fidika/out"}),
            ]
        );
        assert_eq!(spec.models[0].repository, "Fidika/sdxl");
        assert_eq!(spec.weights_destination, "model://Fidika/Out");
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

    #[test]
    fn ordinary_tensorfs_uses_existing_private_reads_and_writes_without_overriding_options() {
        let needs = |spec: &v1::RunSpec| ordinary_tensorfs_needed(spec, &operations("owner", spec));
        let mut read = v1::RunSpec {
            models: vec![choice("owner/private", "", "sha256:tree")],
            ..Default::default()
        };
        assert!(needs(&read));
        read.models = vec![choice("public/base", "1", "")];
        read.models[0].adapters.push(v1::Adapter {
            model: "owner/adapter".into(),
            manifest: "sha256:adapter".into(),
            ..Default::default()
        });
        assert!(needs(&read));
        let mut write = v1::RunSpec {
            weights_destination: "owner/output".into(),
            ..Default::default()
        };
        assert!(needs(&write));
        write.tensorfs = Some(v1::TensorFSOptions {
            endpoint: "https://explicit.example.test".into(),
            ..Default::default()
        });
        assert!(
            !needs(&write),
            "an explicit endpoint must not receive this Hub's credential"
        );
        let public = v1::RunSpec {
            models: vec![
                choice("public/base", "1", ""),
                choice("other/digest", "", "sha256:public"),
            ],
            weights_destination: "local/derived".into(),
            ..Default::default()
        };
        assert!(!needs(&public));
        assert!(
            !needs(&v1::RunSpec::default()),
            "local Writer calls need no Hub sign-in"
        );
    }

    #[test]
    fn ordinary_public_transport_keeps_the_selected_hub_without_a_bearer() {
        let hub = v1::HubAccess {
            origin: "https://selected.example.test".into(),
            ca_der: vec![4, 5],
            ..Default::default()
        };
        let map = |models| {
            anonymous_tensorfs(&v1::RunSpec {
                hub: Some(hub.clone()),
                models,
                ..Default::default()
            })
        };
        for models in [
            vec![],
            vec![choice("public/model", "1", "")],
            vec![v1::ModelChoice {
                parameter: "default".into(),
                ..Default::default()
            }],
            vec![choice("local/base", "", "")],
            vec![choice("", "", "sha256:local-header")],
            vec![v1::ModelChoice {
                source: "hf://owner/model".into(),
                ..Default::default()
            }],
        ] {
            let options = map(models).unwrap();
            assert_eq!(
                (options.endpoint, options.ca_der),
                (hub.origin.clone(), hub.ca_der.clone())
            );
            assert!(options.token.is_empty());
        }
        let mut local = choice("local/base", "", "");
        local.adapters.push(v1::Adapter {
            model: "public/adapter".into(),
            release: "1".into(),
            ..Default::default()
        });
        assert!(map(vec![local]).is_some());
        assert!(
            anonymous_tensorfs(&v1::RunSpec::default()).is_none(),
            "a no-Hub local run has no transport"
        );
        let warm = v1::RunSpec {
            hub: Some(hub),
            set: Some(v1::WarmSet {
                items: vec![v1::WarmItem {
                    models: vec![],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        };
        assert!(anonymous_tensorfs(&warm).unwrap().token.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ordinary_public_transport_reads_cold_bytes_from_the_selected_anonymous_endpoint() {
        use crate::{Calcifer, store::NewRun};
        use axum::{
            Router, extract::Path as Route, http::HeaderMap, response::IntoResponse, routing::get,
        };
        use std::sync::{Arc, Mutex};
        use tensorfs_core::{
            fetch::{
                Client, Config, Endpoint, Fetcher, ObjectStore, SourcePolicy, Want, upload_tree,
            },
            format::{Config as HeaderConfig, Dtype, Tree},
            store2::Store,
            writer::{Data, Source},
        };

        let temp = tempfile::tempdir().unwrap();
        let source = Store::open(temp.path().join("source")).unwrap();
        let data = vec![19u8; 16384];
        let mut writer = source.writer(None, None).unwrap();
        let config: HeaderConfig = serde_json::from_str("{}").unwrap();
        writer
            .component("model", "conformance.Fixture", config, None)
            .unwrap()
            .put(
                "weight",
                &Dtype::U8,
                &[data.len() as u64],
                Data::Plain(Source::Read(&mut &data[..])),
            )
            .unwrap();
        let header = writer.commit().unwrap();
        let mut tree = Tree::default();
        tree.set_variant("f32", header.clone());
        let tree = source.put_tree(&tree).unwrap();
        let client = Client::new(SourcePolicy {
            any_host: true,
            allow_local: true,
            max_redirects: 5,
            ..Default::default()
        });
        let files = temp.path().join("endpoint");
        let seed = Endpoint::new(&client, &format!("file://{}", files.display()), None).unwrap();
        upload_tree(&seed, "public/model", &tree, &source, Some("1"))
            .await
            .unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let observed = requests.clone();
        let app = Router::new().route(
            "/{*path}",
            get(move |Route(path): Route<String>, headers: HeaderMap| {
                let files = files.clone();
                let observed = observed.clone();
                async move {
                    assert!(
                        !headers.contains_key("authorization"),
                        "an anonymous read carried a credential"
                    );
                    observed.lock().unwrap().push(path.clone());
                    match std::fs::read(files.join(path)) {
                        Ok(bytes) => bytes.into_response(),
                        Err(_) => axum::http::StatusCode::NOT_FOUND.into_response(),
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let selected = format!("http://{}", listener.local_addr().unwrap());
        let serving = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let calcifer =
            Calcifer::open(&temp.path().join("calcifer"), &temp.path().join("outputs")).unwrap();
        let (row, _) = calcifer
            .store()
            .submit(
                "public",
                "d",
                NewRun {
                    id: RunId::mint([31; 12]),
                    kind: protocol::calcifer::Kind::Call,
                    target: "public/read".into(),
                    hub: Some(Origin::parse(&selected).unwrap()),
                    machine: None,
                    pinned: false,
                    spec: v1::RunSpec {
                        models: vec![choice("public/model", "1", "")],
                        ..Default::default()
                    },
                    out_dir: temp.path().join("outputs"),
                    warnings: vec![],
                    inputs: vec![],
                    wants: None,
                },
            )
            .unwrap();
        for models in [
            vec![],
            vec![choice("local/base", "", "")],
            vec![v1::ModelChoice {
                source: "hf://owner/model".into(),
                ..Default::default()
            }],
        ] {
            let mut local = row.spec.clone();
            local.models = models;
            calcifer.finish_spec(&row, &mut local, Access::default());
            assert!(local.tensorfs.unwrap().token.is_empty());
        }
        assert!(
            requests.lock().unwrap().is_empty(),
            "transport construction made a network call"
        );
        let mut sent = row.spec.clone();
        // Normal omitted CLI defaults have an empty model vector; actual resolution follows.
        sent.models.clear();
        calcifer.finish_spec(&row, &mut sent, Access::default());
        let options = sent.tensorfs.unwrap();
        assert_eq!(options.endpoint, selected);
        assert!(options.token.is_empty());
        let cold = Store::open(temp.path().join("cold-reader")).unwrap();
        assert!(
            !cold.has(&tree) && !cold.has(&header),
            "the read must begin without cached metadata"
        );
        let endpoint: Arc<dyn tensorfs_core::fetch::Origin> =
            Arc::new(Endpoint::new(&client, &options.endpoint, None).unwrap());
        let fetcher = Fetcher::new(
            Arc::new(cold.clone()) as Arc<dyn ObjectStore>,
            client,
            None,
            Config::default(),
        );
        let pulled = fetcher
            .pull(
                &endpoint,
                "public/model",
                Some("1"),
                Some("f32"),
                &Want::default(),
            )
            .await
            .unwrap();
        fetcher.settled().await;
        assert_eq!(pulled.header, header);
        assert_eq!(
            cold.checkpoint(&header)
                .unwrap()
                .read("model", "weight")
                .unwrap(),
            data
        );
        assert!(
            !requests.lock().unwrap().is_empty(),
            "a warm cache masked the selected endpoint"
        );
        serving.abort();
    }

    #[tokio::test]
    async fn an_ordinary_bearer_is_added_only_to_the_outbound_spec_after_restart() {
        use crate::{Calcifer, store::NewRun};
        use platform::paths::Home;

        let temp = tempfile::tempdir().unwrap();
        let home = Home::at(temp.path());
        let origin = Origin::parse("https://configured.example.test").unwrap();
        let hub = config::Hub {
            name: None,
            origin: origin.clone(),
        };
        platform::durable::private_dir(&home.keys()).unwrap();
        // A synthetic test sign-in, never a real user's key. The normal session must use its
        // live cached Bearer, so no network or alternate credential mechanism is involved.
        let key = serde_json::to_vec(&json!({
            "version": 1, "hub": origin.as_str(), "email": "test@example.test",
            "device_key_id": "ordinary-test-device", "private_key": URL_SAFE_NO_PAD.encode([17u8; 32])
        }))
        .unwrap();
        platform::durable::replace(&hub.device_key(&home), &key, 0o600).unwrap();
        let keep = |token: &str| {
            let bearer = hub_client::Bearer {
                token: token.into(),
                expires: u64::MAX,
                device_key_id: "ordinary-test-device".into(),
            };
            platform::durable::replace(
                &hub.bearer(&home),
                &serde_json::to_vec(&bearer).unwrap(),
                0o600,
            )
            .unwrap();
        };
        let facts = Facts {
            account: "owner".into(),
            user_id: "owner-id".into(),
            reads: "https://machine-reachable.example.test".into(),
            public: origin.to_string(),
            names: vec![],
            object_hosts: vec![],
            ca_der: vec![1, 2, 3],
            token_endpoint: String::new(),
        };
        let hubs = Hubs::new(&home, vec![hub.clone()]);
        let calcifer_home = temp.path().join("calcifer");
        let calcifer = Calcifer::open(&calcifer_home, &temp.path().join("outputs")).unwrap();
        let id = RunId::mint([19; 12]);
        let (row, _) = calcifer
            .store()
            .submit(
                "ordinary",
                "submission-digest",
                NewRun {
                    id: id.clone(),
                    kind: protocol::calcifer::Kind::Call,
                    target: "ordinary/writer".into(),
                    hub: Some(origin.clone()),
                    machine: None,
                    pinned: false,
                    spec: v1::RunSpec {
                        weights_destination: "owner/output".into(),
                        ..Default::default()
                    },
                    out_dir: temp.path().join("outputs"),
                    warnings: vec![],
                    inputs: vec![],
                    wants: None,
                },
            )
            .unwrap();
        keep("ordinary-cache-first");
        let first = match hubs.ordinary_tensorfs(&origin, &facts).await {
            Ok(value) => value,
            Err(_) => panic!("the normal cached user sign-in was not used"),
        };
        assert_eq!(first.endpoint, facts.reads);
        assert_eq!(first.ca_der, facts.ca_der);
        assert_eq!(first.token, "ordinary-cache-first");
        let mut sent = row.spec.clone();
        calcifer.finish_spec(
            &row,
            &mut sent,
            Access {
                tensorfs: Some(first),
                ..Default::default()
            },
        );
        assert_eq!(
            sent.tensorfs.as_ref().unwrap().token,
            "ordinary-cache-first"
        );
        assert!(!format!("{sent:?}").contains("ordinary-cache-first"));
        assert!(
            calcifer
                .store()
                .run(id.as_str())
                .unwrap()
                .unwrap()
                .spec
                .tensorfs
                .is_none()
        );
        drop(calcifer);

        let calcifer = Calcifer::open(&calcifer_home, &temp.path().join("outputs")).unwrap();
        let replay = calcifer.store().run(id.as_str()).unwrap().unwrap();
        assert!(replay.spec.tensorfs.is_none());
        keep("ordinary-cache-replacement");
        let next = match hubs.ordinary_tensorfs(&origin, &facts).await {
            Ok(value) => value,
            Err(_) => panic!("the replacement normal cached sign-in was not used"),
        };
        let mut resent = replay.spec.clone();
        calcifer.finish_spec(
            &replay,
            &mut resent,
            Access {
                tensorfs: Some(next.clone()),
                ..Default::default()
            },
        );
        assert_eq!(resent.tensorfs.unwrap().token, "ordinary-cache-replacement");
        let explicit = v1::TensorFSOptions {
            endpoint: "https://explicit.example.test".into(),
            ca_der: vec![4, 5],
            ..Default::default()
        };
        let mut custom = replay.spec.clone();
        custom.tensorfs = Some(explicit.clone());
        calcifer.finish_spec(
            &replay,
            &mut custom,
            Access {
                tensorfs: Some(next),
                ..Default::default()
            },
        );
        assert_eq!(custom.tensorfs, Some(explicit));
        drop(calcifer);
        let db = rusqlite::Connection::open(calcifer_home.join("state.db")).unwrap();
        let persisted: Vec<u8> = db
            .query_row("SELECT spec FROM runs WHERE id = ?1", [id.as_str()], |r| {
                r.get(0)
            })
            .unwrap();
        let persisted = String::from_utf8(persisted).unwrap();
        assert!(!persisted.contains("ordinary-cache-"));
        assert!(!persisted.contains("private_key"));
        assert!(
            serde_json::from_str::<v1::RunSpec>(&persisted)
                .unwrap()
                .tensorfs
                .is_none()
        );
    }
}
