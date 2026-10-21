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
            for name in holders {
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

#[cfg(test)]
mod tests {
    use super::*;
    use platform::paths::Home;
    use protocol::config;
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering::SeqCst},
    };

    /// A login as another account is the next run's owner, read once under its new key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_account_follows_the_device_key() {
        let reads = Arc::new(AtomicUsize::new(0));
        let counted = reads.clone();
        let account = move |headers: axum::http::HeaderMap| async move {
            counted.fetch_add(1, SeqCst);
            let bearer = headers["authorization"].to_str().unwrap();
            axum::Json(json!({"name": bearer.trim_start_matches("Bearer "), "user_id": "u"}))
        };
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        let environment =
            json!({"environment": {"TENSORHUB_ORIGIN": origin, "TENSORHUB_PUBLIC_ORIGIN": origin}});
        let app = axum::Router::new()
            .route("/v1/accounts/current", axum::routing::get(account))
            .route(
                "/v1/execution-environment",
                axum::routing::get(|| async move { axum::Json(environment) }),
            );
        tokio::spawn(async move { axum::serve(listener, app).await });
        let root = tempfile::tempdir().unwrap();
        let home = Home::at(root.path());
        let hub = sign_in(&home, &Origin::parse(&origin).unwrap());
        let hubs = Hubs::new(&home, vec![hub.clone()]);
        let owner = async || hubs.facts(&hub.origin).await.ok().unwrap().account;
        keep_bearer(&home, &hub, "first");
        assert_eq!(
            (owner().await, owner().await),
            ("first".into(), "first".into())
        );
        sign_in_as(&home, &hub.origin, 18);
        keep_bearer(&home, &hub, "second");
        assert_eq!(owner().await, "second");
        assert_eq!(reads.load(SeqCst), 2, "one account read per key");
    }

    fn choice(repository: &str, release: &str, manifest: &str) -> v1::ModelChoice {
        v1::ModelChoice {
            repository: repository.into(),
            release: release.into(),
            manifest: manifest.into(),
            ..Default::default()
        }
    }

    /// The Bearer goes with a run that publishes, or reads its owner's checkpoint by digest under
    /// any spelling: a model choice, an adapter or a warm set member. Never to an explicit endpoint.
    #[test]
    fn the_bearer_goes_only_with_the_owners_private_reads_and_publishes() {
        let needs = |spec: &v1::RunSpec| needs_bearer(spec, "owner");
        let read = v1::RunSpec {
            models: vec![choice("Owner/private", "", "sha256:tree")],
            ..Default::default()
        };
        assert!(needs(&read));
        let mut base = choice("public/base", "1", "");
        base.adapters.push(v1::Adapter {
            model: "owner/adapter".into(),
            manifest: "sha256:adapter".into(),
            ..Default::default()
        });
        let adapter = v1::RunSpec {
            models: vec![base],
            ..Default::default()
        };
        assert!(needs(&adapter));
        let warm = v1::RunSpec {
            kind: v1::RunKind::Warm,
            set: Some(v1::WarmSet {
                items: vec![v1::WarmItem {
                    models: vec![choice("owner/member", "", "sha256:member")],
                    ..Default::default()
                }],
            }),
            ..Default::default()
        };
        assert!(needs(&warm), "a warm set member named by hash");
        let mut write = v1::RunSpec {
            weights_destination: "model://Owner/output".into(),
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
            private(&public, None),
            "another's checkpoint by digest needs a sign-in"
        );
        let released = v1::RunSpec {
            models: vec![choice("owner/sdxl", "2026.1", "sha256:bb")],
            weights_destination: "local/out".into(),
            ..Default::default()
        };
        assert!(
            !private(&released, None),
            "a release and a local destination are public"
        );
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
        use tensorfs::{
            fetch::{
                Client, Config, Endpoint, Fetcher, ObjectStore, SourcePolicy, Want, upload_tree,
            },
            format::{Config as HeaderConfig, Dtype, Tree},
            store::Store,
            writer::{Data, Source},
        };

        let temp = tempfile::tempdir().unwrap();
        let source = Store::open(temp.path().join("source")).unwrap();
        let data = vec![19u8; 16384];
        let mut writer = source.writer(None, None).unwrap();
        let config: HeaderConfig = serde_json::from_str("{}").unwrap();
        writer
            .component("model", Default::default(), config, None)
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
        upload_tree(&seed, "public/model", &tree, &source, "1").await.unwrap();
        seed.update_repo("public/model", |old| {
            let mut repo = old.unwrap();
            repo.tag_mut("1").unwrap().public = true;
            Ok(Some(repo))
        })
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
        let endpoint: Arc<dyn tensorfs::fetch::Origin> =
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
        let hub = sign_in(&home, &origin);
        let keep = |token: &str| keep_bearer(&home, &hub, token);
        let facts = Facts {
            key: hub_client::DeviceKey::load(&hub.device_key(&home))
                .unwrap()
                .public_key(),
            account: "owner".into(),
            reads: "https://machine-reachable.example.test".into(),
            names: vec![],
            ca_der: vec![1, 2, 3],
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
        let first = match hubs.bearer(&origin).await {
            Ok(bearer) => tensorfs(&facts, &bearer),
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
        let next = match hubs.bearer(&origin).await {
            Ok(bearer) => tensorfs(&facts, &bearer),
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
    /// A synthetic test sign-in at `origin`, never a real user's key.
    fn sign_in(home: &platform::paths::Home, origin: &Origin) -> config::Hub {
        sign_in_as(home, origin, 17)
    }

    /// The sign-in of the device key made from `seed`.
    fn sign_in_as(home: &platform::paths::Home, origin: &Origin, seed: u8) -> config::Hub {
        let hub = config::Hub {
            name: None,
            origin: origin.clone(),
        };
        platform::durable::private_dir(&home.keys()).unwrap();
        let key = serde_json::to_vec(&json!({
            "version": 1, "hub": origin.as_str(), "email": "test@example.test",
            "device_key_id": "ordinary-test-device", "private_key": URL_SAFE_NO_PAD.encode([seed; 32])
        }))
        .unwrap();
        platform::durable::replace(&hub.device_key(home), &key, 0o600).unwrap();
        hub
    }

    /// Keeps `token` as the sign-in's live Bearer: the normal session uses it with no network.
    fn keep_bearer(home: &platform::paths::Home, hub: &config::Hub, token: &str) {
        let bearer = hub_client::Bearer {
            token: token.into(),
            expires: u64::MAX,
            device_key_id: "ordinary-test-device".into(),
        };
        let bearer = serde_json::to_vec(&bearer).unwrap();
        platform::durable::replace(&hub.bearer(home), &bearer, 0o600).unwrap();
    }

    /// A warm run carries its owner's Bearer when a member of the set it is sent names the owner's
    /// checkpoint by hash, a member the machine holds from an earlier send included.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_held_warm_member_named_by_hash_gets_the_bearer() {
        use crate::{Calcifer, machines::Machines, store::NewRun};
        use platform::paths::Home;

        let temp = tempfile::tempdir().unwrap();
        let home = Home::at(temp.path());
        // A closed loopback port: the send reads only what this computer holds.
        let origin = Origin::parse("http://127.0.0.1:1").unwrap();
        let hub = sign_in(&home, &origin);
        keep_bearer(&home, &hub, "warm-member-bearer");
        let calcifer =
            Calcifer::open(&temp.path().join("calcifer"), &temp.path().join("outputs")).unwrap();
        let config = config::Config {
            hubs: [("dev".to_owned(), origin.to_string())].into(),
            ..Default::default()
        };
        let key = hub_client::DeviceKey::load(&hub.device_key(&home)).unwrap();
        let machines = Machines::start(calcifer.clone(), home, &config);
        let facts = Facts {
            key: key.public_key(),
            account: "owner".into(),
            reads: origin.to_string(),
            names: vec![],
            ca_der: vec![],
        };
        lock(&machines.hubs.facts).insert(origin.clone(), facts);
        let item = |package: &str, models| v1::WarmItem {
            source: Some(v1::WarmSource::Release(v1::Release {
                package: package.into(),
                release: String::new(),
            })),
            entrypoint: "generate".into(),
            models,
            level: v1::WarmLevel::Gpu,
            ..Default::default()
        };
        let (mut row, _) = calcifer
            .store()
            .submit(
                "warm",
                "d",
                NewRun {
                    id: RunId::mint([23; 12]),
                    kind: protocol::calcifer::Kind::Warm,
                    target: "public/flux/generate".into(),
                    hub: Some(origin.clone()),
                    machine: None,
                    pinned: false,
                    spec: v1::RunSpec {
                        kind: v1::RunKind::Warm,
                        set: Some(v1::WarmSet {
                            items: vec![item("public/flux", vec![])],
                        }),
                        ..Default::default()
                    },
                    out_dir: temp.path().join("outputs"),
                    warnings: vec![],
                    inputs: vec![],
                    wants: None,
                },
            )
            .unwrap();
        // As placed on the machine "m".
        row.machine = Some("m".into());
        calcifer.observe_warm("m", vec![]);
        let alone = calcifer.sent_spec(&row).await.ok().unwrap();
        assert!(alone.tensorfs.unwrap().token.is_empty(), "a public set");
        let held = item("owner/sdxl", vec![choice("owner/sdxl", "", "sha256:held")]);
        calcifer.observe_warm("m", vec![held]);
        let sent = calcifer.sent_spec(&row).await.ok().unwrap();
        assert_eq!(sent.set.unwrap().items.len(), 2);
        assert_eq!(sent.tensorfs.unwrap().token, "warm-member-bearer");
        machines.close();
    }
}
