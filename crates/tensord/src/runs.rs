//! Run sources and preparation inside a run (`cozy.machine.v1` Run). A run is accepted at
//! once and prepares inside itself: its code installs, its models resolve at the run's Hub and
//! download, and each stage is the run's progress. Its owner's Hub bearer lives only in memory
//! (`hub::Bearer`), as do its provider tokens, for the run's life (its children prepare with
//! them); a restart before it completes ends the run FAILED (`journal::PREPARING`).
use crate::{
    api::domain,
    device_executor::Line,
    call_memo::{self, Memo},
    execution::Stoppable,
    catalog::HeldGeneration,
    gpu_service::{without_gpus, KeptMember, Level, Member, Plan, Pool},
    hub,
    journal::{CallMemo, Execution, Failure, InputFile, Installation, Outcome, ResultRecord},
    local_source::LocalSources,
    objects::{Objects, Refused},
    published::{Providers, Publisher, Request},
    service::{Call, Service},
};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use serde_json::{json, Value};
use crate::stop::Stop;
use tensorfs::format::ObjectRef;
use std::{
    collections::HashMap,
    io,
    sync::{Arc, Mutex},
};

mod model_folder;

pub enum Source {
    /// A published release by name (th-245); empty: the Hub's newest.
    Release { package: String, release: String },
    /// An installation this machine already holds for the signer.
    Installation(String),
    /// Unpublished code written with Write: its manifest's digest.
    Local(String),
    /// No code: a warm run that only makes its model choices (`cozy model upload`).
    Models,
}

/// A published release a run installs: its package and release.
type Published = (String, String);

/// One member of the warm set a warm run carries (`RunSpec.set`).
pub struct SetItem {
    pub source: Source,
    pub entrypoint: String,
    pub models: Vec<domain::ModelChoice>,
    /// As `Spec::bits`, for this member's slots.
    pub bits: u32,
    pub level: Level,
    /// The item as the caller sent it, for Status.
    pub sent: Vec<u8>,
}

pub struct Spec {
    /// Prepare only (`kind: warm`): install and download, then succeed.
    pub warm: bool,
    /// `kind: remove`: the caller's installations of the source leave the machine.
    pub remove: bool,
    /// A warm run's whole warm set for its caller, replacing the previous one.
    pub set: Option<Vec<SetItem>>,
    /// A child run's parent execution (a call through its parent's seam).
    pub parent: String,
    pub source: Source,
    pub entrypoint: String,
    pub input: Value,
    pub inputs: Vec<InputFile>,
    pub models: Vec<domain::ModelChoice>,
    pub attention_kernel: String,
    pub gpus: u32,
    /// The main model's weight precision (16, 8 or 4); 0, and every other slot, the lane rule.
    pub bits: u32,
    /// The Hub the run reads, under its owner's newest bearer there when its request carried one.
    pub hub: Option<hub::Source>,
    /// Provider tokens for source models, held in memory only.
    pub providers: Providers,
    pub tensorfs: Option<protocol::machine::TensorFSOptions>,
    /// `org/name`: the requester's repository for the run's weights outputs, or where a
    /// model-only warm run puts the source model it made.
    pub weights_destination: String,
    pub owner: String,
    /// The caller's catalog revision: held resolutions are reused only under it (th-245).
    pub catalog_revision: String,
    /// A child run of a callee: the other package's App its parent's environment holds.
    pub application: String,
    /// The spec's identity (its access excluded): a resubmitted id must carry the same.
    pub digest: String,
    /// The spec as sent, kept in the journal while it prepares so a restart prepares it again;
    /// empty for one that cannot (a child run, provider tokens, which live in memory only).
    pub journaled: Vec<u8>,
}

pub struct Runs {
    pub service: Arc<Service>,
    pub objects: Arc<Objects>,
    /// The one resolver of a run's code and models.
    pub publisher: Arc<Publisher>,
    pub local: Option<Arc<LocalSources>>,
    /// On a rental: its own Hub, where a run that names none reads anonymously.
    pub own_hub: Option<hub::Source>,
    /// Unfinished runs' preparation: their children prepare with it, and their weights are
    /// fetched and published with its access.
    pub contexts: Mutex<HashMap<String, RunContext>>,
}

/// What a run's children prepare with: its installation, Hub access, owner, catalog revision,
/// attention pin, bits and the model choices addressed to its callables (`<entrypoint>.models.<p>`).
#[derive(Clone, Default)]
pub struct RunContext {
    pub(crate) installation: String,
    pub(crate) hub: Option<hub::Source>,
    pub(crate) providers: Providers,
    pub(crate) tensorfs: Option<protocol::machine::TensorFSOptions>,
    pub(crate) weights_destination: String,
    pub(crate) owner: String,
    pub(crate) catalog_revision: String,
    pub(crate) attention_kernel: String,
    pub(crate) gpus: u32,
    pub(crate) bits: u32,
    pub(crate) models: Vec<domain::ModelChoice>,
    /// What its weights come under before its call's executor reads them (`Fetch::ahead`).
    pub(crate) ahead: Option<Arc<crate::access::Fetch>>,
}

/// A choice for the run's own slots: a bare parameter, or one under its own name.
fn own_choice(choice: &domain::ModelChoice, entrypoint: &str) -> bool {
    match choice.parameter.split_once(".models.") {
        None => true,
        Some((callable, _)) => callable == entrypoint,
    }
}

/// The slot of `package`'s `interface` a choice's parameter addresses: its full slot path
/// (`<callable>.models.<parameter>`), or that path behind the package's name, as the CLI names
/// another package's callable (`<org>/<name>/<callable>.models.<parameter>`).
fn addressed(parameter: &str, package: &str, interface: &Value) -> Option<String> {
    let path = parameter
        .strip_prefix(package)
        .and_then(|rest| rest.strip_prefix('/'))
        .unwrap_or(parameter);
    interface["entrypoints"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|entry| entry["models"].as_array().into_iter().flatten().any(|slot| slot["path"] == path))
        .then(|| path.to_string())
}

/// `installation`'s internal functions that construct a model: what a function loading none
/// itself calls.
fn internal_models(installation: &Installation) -> Result<Vec<String>, Refused> {
    let interface: Value = serde_json::from_slice(&installation.interface)
        .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
    let constructs = |entry: &Value| {
        let models = entry["models"].as_array().into_iter().flatten();
        models.into_iter().any(crate::gpu_service::constructed)
    };
    Ok(interface["entrypoints"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|entry| entry["internal"].as_bool() == Some(true) && constructs(entry))
        .filter_map(|entry| entry["name"].as_str().map(str::to_owned))
        .collect())
}

/// Whether `installation` declares a public entrypoint `name` for a warm set.
fn declares(installation: &Installation, name: &str) -> Result<bool, Refused> {
    let interface: Value = serde_json::from_slice(&installation.interface)
        .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
    let entry = interface["entrypoints"].as_array()
        .and_then(|rows| rows.iter().find(|row| row["name"] == name));
    if let Some(entry) = entry {
        public_root(entry)?;
    }
    Ok(entry.is_some())
}
fn public_root(entry: &Value) -> Result<(), Refused> {
    if entry["internal"].as_bool() == Some(true) {
        return Err(refused(
            "callable_internal",
            format!("{} is an internal package function", entry["name"].as_str().unwrap_or_default()),
        ));
    }
    Ok(())
}
fn refused(code: &'static str, message: impl Into<String>) -> Refused {
    Refused {
        code,
        message: message.into(),
    }
}

/// A named commit (tree) or variant header already held for an upload. Empty-repository choices
/// retain their legacy header meaning; local trees never fall back to a remote repository or its
/// newest release.
fn held_tree(store: &crate::storage::Store, choice: &domain::ModelChoice) -> Result<Option<crate::published::Named>, Refused> {
    let Some(manifest) = choice.manifest.as_ref().filter(|_| !choice.repository.is_empty()) else {
        return Ok(None);
    };
    let id = tensorfs::sha256::hex(&manifest.digest);
    let object = match crate::storage::held(store, &id) {
        Ok(object) => object,
        Err(error) if error.kind() == io::ErrorKind::NotFound && !choice.repository.starts_with("local/") => return Ok(None),
        Err(error) => return Err(refused("checkpoint_absent", format!("held checkpoint sha256:{id}: {error}"))),
    };
    let named = |lane, tree, header| crate::published::Named {
        repository: choice.repository.clone(), release: choice.release.clone(), lane, tree, header: Some(header),
    };
    let Some(parsed) = crate::storage::tree_or_header(store, &object).map_err(|error| refused("checkpoint_invalid", error.to_string()))? else {
        return Ok(Some(named(choice.lane.clone(), None, object)));
    };
    let (lane, header) = tensorfs::fetch::variant_of_tree(&parsed, (!choice.lane.is_empty()).then_some(choice.lane.as_str()))
        .map_err(|error| refused("checkpoint_variant_absent", error.to_string()))?;
    Ok(Some(named(lane, Some(object), header)))
}

impl Runs {
    /// The run this id already names, if its spec is `digest`; another spec is a conflict.
    pub(crate) fn existing(&self, actor: &str, id: &str, digest: &str) -> Result<Option<Execution>, Refused> {
        match self.service.engine.get_public(actor, id) {
            Ok(run) if run.canceled_before_acceptance && run.state == crate::journal::State::Canceled => Ok(Some(run)),
            Ok(run) if run.submission.as_ref().map(|s| s.invocation_digest.as_str()) == Some(digest) => Ok(Some(run)),
            Ok(_) => Err(refused("run_id_conflict", "this run id already names another run spec")),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Accepts the run (or answers the one this id already names) and starts its preparation.
    /// An accepted run is answered before its inputs are checked: they may be gone since.
    pub fn submit(self: &Arc<Self>, actor: &str, id: &str, spec: Spec) -> Result<Execution, Refused> {
        if let Some(run) = self.existing(actor, id, &spec.digest)? {
            return Ok(run);
        }
        if !spec.input.is_object() {
            return Err(refused(
                "invalid_request",
                "the payload is a JSON object of the function's parameters",
            ));
        }
        // Object sweeps stay out from validation through acceptance.
        let custody = self.objects.guard();
        let mut references = Vec::new();
        let unwritten = |input: &str| refused("input_unwritten", format!("input {input} was not written to this machine"));
        for input in &spec.inputs {
            let path = match self.objects.path(actor, &input.digest)? {
                Some((path, length)) if length == input.length => path,
                _ => return Err(unwritten(&input.input_id)),
            };
            references.push(ObjectRef {
                sha256: input.digest.trim_start_matches("sha256:").into(),
                length: input.length,
            });
            if input.media_type != crate::gpu_service::TREE_MEDIA {
                continue;
            }
            // A tree's files were written too: each its manifest names.
            let files = crate::trees::parse(&std::fs::read(path)?)
                .map_err(|e| refused("invalid_request", format!("input tree {}: {e}", input.input_id)))?;
            for (member, object) in files {
                if self.objects.path(actor, &format!("sha256:{}", object.sha256))?.is_none_or(|(_, held)| held != object.length) {
                    return Err(unwritten(&format!("{}/{member}", input.input_id)));
                }
                references.push(object);
            }
        }
        // A local source is refused here, not accepted to fail: the run takes custody of its
        // objects now, and a refused id is free for the client to write them and send again.
        if let Source::Local(digest) = &spec.source {
            let local = self.local.as_ref().ok_or_else(|| {
                refused("capability_unavailable", "this machine installs no local packages")
            })?;
            references.extend(local.written(actor, digest)?);
        }
        // A written file a warm run makes into a model (`object://sha256:<hex>/<file>`).
        for choice in &spec.models {
            // A folder is bound to one retained input tree, never an arbitrary store path.
            model_folder::input(&spec, choice)?;
            let Some(hex) = choice.source.strip_prefix("object://").and_then(|rest| rest.split('/').next()) else {
                continue;
            };
            let sha256 = hex.trim_start_matches("sha256:").to_string();
            match self.objects.path(actor, &format!("sha256:{sha256}"))? {
                Some((_, length)) => references.push(ObjectRef { sha256, length }),
                None => return Err(unwritten(&choice.source)),
            }
        }
        self.objects.retain(&references)?;
        let package = match &spec.source {
            Source::Release { package, .. } => package.clone(),
            Source::Installation(alias) => alias.clone(),
            Source::Local(digest) => format!("local:{digest}"),
            Source::Models => "models".into(),
        };
        let draft = crate::journal::Invocation {
            package,
            generation: String::new(),
            module: String::new(),
            entrypoint: spec.entrypoint.clone(),
            input: spec.input.clone(),
            attention_kernel: spec.attention_kernel.clone(),
            gpus: spec.gpus,
            bits: spec.bits,
            inputs: spec.inputs.clone(),
            parent: spec.parent.clone(),
            ..Default::default() // root and family: stamped at acceptance
        };
        let (record, new) = self
            .service
            .engine
            .accept_run_objects(actor, id, &spec.digest, draft, &references, &spec.journaled)
            .map_err(|e| match e.kind() {
                io::ErrorKind::AlreadyExists => refused("run_id_conflict", e.to_string()),
                _ => Refused::from(e),
            })?;
        drop(custody);
        if new {
            self.prepare_later(actor, &record.id, spec).map_err(Refused::from)?;
        }
        Ok(record)
    }

    /// Prepare accepted run `run` on a thread of its own, which ends the run unless it binds.
    pub(crate) fn prepare_later(self: &Arc<Self>, actor: &str, run: &str, spec: Spec) -> io::Result<()> {
        let (this, actor, run) = (self.clone(), actor.to_string(), run.to_string());
        let thread = std::thread::Builder::new().name(format!("prepare-{run}"));
        thread.spawn(move || {
            let stoppable = this.service.engine.stoppable(&run);
            // Nothing moving past its learned patience fails it `no_progress`, at once: a fetch
            // stuck in a read may never return to fail it, and its rental must start idling.
            let (done, finished) = std::sync::mpsc::channel::<()>();
            // One stall report per preparation, whichever side measured it.
            let reported = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let stalled = {
                let (service, reported) = (this.service.clone(), reported.clone());
                move || if !reported.swap(true, std::sync::atomic::Ordering::AcqRel) { service.stalled() }
            };
            let judge_stalled = stalled.clone();
            let (service, judged, stop) = (this.service.clone(), run.clone(), stoppable.stop.clone());
            let judge = std::thread::Builder::new().name(format!("judge-{run}")).spawn(move || {
                let waits = || service.engine.get(&judged).is_ok_and(|record| record.waiting.is_some());
                if let Some(verdict) = crate::process::judge_preparation(&stop, &finished, waits, crate::process::Liveness::default()) {
                    let failure = Failure::machine("no_progress", verdict);
                    judge_stalled();
                    if let Err(error) = service.engine.end_preparation(&judged, Outcome::Failed(failure)) {
                        tracing::warn!("run {judged}: its stalled preparation did not end: {error}");
                    }
                    stop.cancel();
                }
            });
            if let Err(error) = judge {
                tracing::warn!("run {run}: its preparation is not judged: {error}");
            }
            // A panic (a thread the container's task cap refused) ends the run FAILED: no
            // preparation outlives the thread that would end it.
            let prepared = std::panic::AssertUnwindSafe(|| this.prepare(&actor, &run, spec, &stoppable));
            let outcome = std::panic::catch_unwind(prepared).unwrap_or_else(|panic| {
                let text = panic.downcast_ref::<String>().cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|text| text.to_string()));
                Err(refused("preparation_failed", text.unwrap_or_else(|| "the preparation panicked".into())))
            });
            drop(done);
            let ended = match outcome {
                Ok(None) => Ok(()),
                Ok(Some(result)) => this.service.engine.end_preparation(&run, Outcome::Completed(result)).map(drop),
                Err(Refused { code, message }) => {
                    if code == "no_progress" {
                        stalled();
                    }
                    let origin = if code.starts_with("invalid") || code.ends_with("_absent") { 6 } else { 3 };
                    let failure = Failure { status: 3, cause: 7, origin, code: code.into(), message };
                    this.service.engine.end_preparation(&run, Outcome::Failed(failure)).map(drop)
                }
            };
            if let Err(error) = ended {
                tracing::warn!("run {run} preparation: {error}");
            }
        })?;
        Ok(())
    }

    /// A warm run's model choices made present when it names no entrypoint (`cozy package
    /// install`, `cozy model download`) or no code (`cozy model upload`, `cozy run upload`): a
    /// Hub checkpoint downloaded, a provider source or a written file (`object://`) made, a
    /// checkpoint this machine holds (a run's weights output, a `local/` alias) found. With a
    /// weights destination the one choice is put there: a `local/` alias held by name, or a
    /// repository at the run's TensorFS endpoint.
    fn warm_models(
        &self,
        actor: &str,
        id: &str,
        spec: &Spec,
        observe: &(dyn Fn(&str, u64, u64) + Sync),
        stoppable: &Stoppable,
    ) -> Result<Vec<Value>, Refused> {
        let stop = Some(&stoppable.stop);
        let destination = spec.weights_destination.trim_start_matches("model://");
        if !destination.is_empty() && spec.models.len() != 1 {
            return Err(refused("invalid_request", "a weights destination takes one model"));
        }
        // An upload goes to the run's TensorFS endpoint (`org/name@tag`: the tag moves to its
        // commit); TensorD publishes nothing itself.
        let upload = match (destination.is_empty() || destination.starts_with("local/"), &spec.tensorfs) {
            (true, _) => None,
            (false, Some(options)) => match destination.split_once('@') {
                Some((repository, tag)) if !tag.is_empty() => Some((options, repository, tag)),
                _ => return Err(refused("invalid_request", format!("upload destination {destination:?} names no tag (org/name@tag)"))),
            },
            (false, None) => return Err(refused("upload_endpoint_absent",
                format!("uploading to {destination} takes the run's TensorFS endpoint (`tensorfs` options), and it carries none"))),
        };
        if spec.models.is_empty() {
            return Ok(vec![]);
        }
        let publisher = &self.publisher;
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let hub_access = || {
            hub.as_ref().ok_or_else(|| {
                refused("hub_access_absent", "this needs the run's Hub access, and it carries none")
            })
        };
        let mut models = vec![];
        for choice in &spec.models {
            let folder = model_folder::prepare(self, actor, spec, choice, stoppable)?;
            let upload_store = folder.as_ref().map_or(publisher.store().as_ref(), |folder| &folder.store);
            // Each model's header, and the tree that names it when one does.
            let (header, tree, mut row) = if let Some(folder) = &folder {
                let row = json!({"parameter": choice.parameter, "source": choice.source,
                    "tree": folder.tree.id(), "variants": folder.variants});
                (folder.header.clone(), Some(folder.tree.clone()), row)
            } else if !choice.source.is_empty() {
                let made = publisher.make_source(&choice.source, &choice.profiles, &spec.providers, stop, observe)?;
                let row = json!({"parameter": choice.parameter, "source": choice.source,
                    "resolved": made.resolved, "profiles": made.profiles,
                    "repository": made.repository, "checkpoint": made.header.id()});
                let tree = crate::storage::local(publisher.store(), &made.repository, None, None)?.map(|local| local.tree);
                (made.header, tree, row)
            } else if let (Some(name), None) = (choice.repository.strip_prefix("local/"), &choice.manifest) {
                // A local alias: the model this machine's store holds under that name.
                let local = crate::storage::local(publisher.store(), &choice.repository, None, None)
                    .map_err(|e| refused("checkpoint_absent", e.to_string()))?
                    .filter(|local| crate::storage::present(publisher.store(), &local.header))
                    .ok_or_else(|| refused("checkpoint_absent", format!("this machine holds no local/{name}")))?;
                let row = json!({"parameter": choice.parameter,
                    "repository": choice.repository, "checkpoint": local.header.id()});
                (local.header, Some(local.tree), row)
            } else if choice.repository.is_empty() {
                // A checkpoint this machine already holds (a run's output), by its header.
                let header = choice.manifest.as_ref().ok_or_else(|| {
                    refused("invalid_request", format!("model choice {:?} names no model", choice.parameter))
                })?;
                let sha256 = tensorfs::sha256::hex(&header.digest);
                let held = crate::storage::held(publisher.store(), &sha256)
                    .map_err(|_| refused("checkpoint_absent", format!("this machine holds no checkpoint sha256:{sha256}")))?;
                let row = json!({"parameter": choice.parameter, "checkpoint": format!("sha256:{sha256}")});
                (held, None, row)
            } else if let Some(named) = if destination.is_empty() { None } else { held_tree(publisher.store(), choice)? } {
                let header = named.header().map_err(|(code, message)| refused(code, message))?.clone();
                let row = json!({"parameter": choice.parameter, "repository": named.repository,
                    "lane": named.lane, "tree": named.tree.as_ref().map(ObjectRef::id), "checkpoint": header.id()});
                (header, named.tree, row)
            } else {
                // A Hub checkpoint by name, or the owner's by hash; no lane is the one this
                // machine's GPUs would run at the run's bits (`precision`).
                let named = crate::published::choose(choice, &choice.parameter)
                    .map_err(|(code, message)| refused(code, message))?;
                let gpus = self.service.gpu().map(|gpu| gpu.gpus(gpu.width() as u32)).unwrap_or_default();
                let constraints = crate::precision::Constraints { gpus: gpus.capabilities.clone(),
                    bits: (spec.bits != 0).then_some(spec.bits), ..Default::default() };
                let ruled = named.lane.is_empty() && named.tree.is_none();
                let stage = format!("downloading {}", choice.repository);
                let (named, (hub, volume)) = publisher.download(&self.service, hub_access()?, named, &constraints, &spec.catalog_revision, stop, &|done, total| {
                    observe(&stage, done, total)
                })?;
                let header = named.header().map_err(|(code, message)| refused(code, message))?.clone();
                let read = publisher.store().read(&header).map_err(|e| e.to_string())
                    .and_then(|bytes| tensorfs::format::Header::from_cbor(&bytes).map_err(|e| e.to_string()))
                    .map_err(|e| refused("model_resolution_failed", e))?;
                let profile = crate::precision::Profile::of(&read);
                // The rule's reasons for what it took, as a run's (`cozy model download` on a CPU).
                if ruled {
                    let model = format!("{}@{}", named.repository, named.release);
                    for warning in constraints.warnings(&model, &profile) {
                        if let Err(error) = self.service.engine.append_warning(id, &warning) {
                            tracing::warn!("run {id}: {warning} ({error})");
                        }
                    }
                }
                let tree = named.tree.as_ref().map(ObjectRef::id);
                let row = json!({"parameter": choice.parameter, "repository": named.repository,
                    "release": named.release, "lane": named.lane, "precision": profile.family, "tree": tree,
                    "checkpoint": header.id(), "bytes": {"hub": hub, "volume": volume}});
                (header, None, row)
            };
            if let Some(name) = destination.strip_prefix("local/") {
                // A local alias (`cozy model download … local/name`): the model, held by name.
                crate::storage::set_local(publisher.store(), name, &header, &Default::default())
                    .map_err(|e| refused("local_alias_refused", e.to_string()))?;
                row["published"] = json!({"destination": destination, "checkpoint": header.id(), "converged": false});
            } else if let Some((options, repository, tag)) = upload {
                use tensorfs::fetch::{bearer, upload_tree_with, Client, Endpoint, SourcePolicy, UploadControl};
                // A tree goes up: the model's own, else one naming its header alone.
                let tree = match tree {
                    Some(tree) => tree,
                    None => crate::storage::tree_of(publisher.store(), &header)?,
                };
                let stage = format!("uploading to {destination}");
                let policy = SourcePolicy { any_host: true, allow_local: true, max_redirects: 8, ..Default::default() };
                let client = match options.ca_der.is_empty() {
                    true => Ok(Client::new(policy)),
                    false => Client::with_roots(policy, hub::pem(&options.ca_der).as_bytes()),
                }.map_err(|error| refused("upload_failed", error.to_string()))?;
                // The owner's newest bearer at the run's Hub, else the one the request carried.
                let token = match hub.as_ref().and_then(|hub| hub.bearer_at(&options.endpoint)) {
                    Some(bearer) => Some(bearer as Arc<dyn tensorfs::fetch::TokenProvider>),
                    None => (!options.token.is_empty()).then(|| bearer(options.token.clone())),
                };
                let anonymous = token.is_none();
                let mut endpoint = Endpoint::new(&client, &options.endpoint, token)
                    .map_err(|error| refused("upload_failed", error.to_string()))?;
                if anonymous {
                    endpoint = endpoint.anonymous();
                }
                let runtime = tensorfs::fetch::runtime().map_err(|error| refused("upload_failed", error.to_string()))?;
                let uploaded = runtime.block_on(upload_tree_with(&endpoint, repository, &tree, upload_store, tag, &[], &UploadControl {
                    cancelled: &|| stoppable.stop.is_cancelled(),
                    progress: &|done, total| observe(&stage, done, total),
                    commit: &|| stoppable.commit(),
                })).map_err(|error| match stoppable.stop.is_cancelled() {
                    true => refused(crate::published::CANCELED, "the upload was canceled before its ref update; admitted objects remain available"),
                    false => refused("upload_failed", error.to_string()),
                })?;
                row["published"] = json!({"destination": destination, "tree": uploaded.commit.id(), "tag": tag,
                    "checkpoint": header.id(), "converged": uploaded.converged});
            }
            models.push(row);
        }
        Ok(models)
    }

    /// The run's code and models made ready: Ok(None) once it is dispatchable, Ok(Some) for a
    /// warm run's result.
    fn prepare(&self, actor: &str, id: &str, spec: Spec, stoppable: &Stoppable) -> Result<Option<ResultRecord>, Refused> {
        let (engine, stop) = (self.service.engine.clone(), stoppable.stop.clone());
        let run = id.to_string();
        // Byte counts arrive per chunk: a new stage and a finished count show at once, bytes at
        // most 4 times a second.
        let last = std::sync::Mutex::new((String::new(), std::time::Instant::now()));
        let observe = Arc::new(move |stage: &str, done: u64, total: u64| {
            stop.at(stage, done, total);
            let mut last = last.lock().unwrap();
            if last.0 == stage && last.1.elapsed() < std::time::Duration::from_millis(250) && (total == 0 || done < total) {
                return;
            }
            *last = (stage.to_string(), std::time::Instant::now());
            let detail = json!({"stage": stage, "bytes_done": done, "bytes_total": total});
            let _ = engine.observe_progress(&run, 0, detail.to_string());
        });
        observe("preparing", 0, 0);
        let warm_result = |installed: Value, models: Vec<Value>| -> Result<Option<ResultRecord>, Refused> {
            let mut value = installed;
            value["models"] = Value::Array(models);
            Ok(Some(ResultRecord {
                value,
                artifacts: vec![],
                asset_bindings: vec![],
                adjustments: vec![],
            }))
        };
        if spec.remove {
            return self.remove(actor, &spec).map(Some);
        }
        if let Some(items) = &spec.set {
            return self.warm_set(actor, id, &spec, items, &observe, stoppable);
        }
        if let Source::Models = spec.source {
            let models = self.warm_models(actor, id, &spec, &*observe, stoppable)?;
            return warm_result(json!({}), models);
        }
        // A warm run naming no entrypoint installs its code, then makes its choices present.
        let install_only = spec.warm && spec.entrypoint.is_empty();
        let held = match &spec.source {
            Source::Release { .. } | Source::Models => None,
            Source::Installation(alias) => Some(
                self.service
                    .engine
                    .installation(actor, alias)?
                    .ok_or_else(|| {
                        refused(
                            "installation_absent",
                            "this machine holds no such installation for this signer",
                        )
                    })?,
            ),
            Source::Local(digest) => {
                let local = self.local.as_ref().ok_or_else(|| {
                    refused("capability_unavailable", "this machine installs no local packages")
                })?;
                let stage = |stage: &str, read: u64| observe(stage, read, 0);
                let installed = local.install(&self.service, actor, digest, Some(&stoppable.stop), id, &stage)?;
                self.supersede(actor, &installed)?;
                local.collect(actor, &installed.package);
                // A call's imports start as soon as its code is installed, as a release's do.
                if let (false, Some(gpu)) = (spec.warm, self.service.gpu()) {
                    gpu.prespawn(self.service.catalog.resolve(&installed.generation)?);
                }
                Some(installed)
            }
        };
        let held = match held {
            Some(installed) => Some(self.callee_view(installed, &spec.application)?),
            None => None,
        };
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let release = match &spec.source {
            Source::Release { package, release } => Some((package.clone(), release.clone())),
            _ => None,
        };
        // Its own slots' choices resolve here; the rest address its callables, whose calls
        // resolve them.
        let own: Vec<_> = match install_only {
            true => vec![],
            false => spec.models.iter().filter(|c| own_choice(c, &spec.entrypoint)).cloned().collect(),
        };
        let (installation, plan, precision) = self.resolve(
            actor,
            id,
            held,
            release,
            hub.clone(),
            &spec,
            &spec.entrypoint,
            &own,
            spec.bits,
            Some(Box::new({
                let observe = observe.clone();
                move |stage: &str, done: u64, total: u64| observe(stage, done, total)
            })),
            Some(&stoppable.stop),
        )?;
        let interface: Value = serde_json::from_slice(&installation.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let generation = self.service.catalog.resolve(&installation.generation)?;
        // A call's weights come from here on, in the Start pass's order, until its executor
        // asks for them itself; its imports start now unless they already did at install. A
        // warm run imports only when its level asks (`downloaded` never does).
        let mut ahead = None;
        if let (false, Some(gpu)) = (spec.warm, self.service.gpu()) {
            ahead = plan.as_ref().map(|plan| {
                let order = self.service.engine.get(id).map_or(u64::MAX, |record| record.root_seq());
                let fetch = crate::access::Fetch::new(self.publisher.machine().clone(), hub.clone(), order);
                match fetch.ahead(&plan.upfront()) {
                    Ok(()) => crate::access::meter(self.service.engine.clone(), id.to_string(), &fetch),
                    Err(error) => tracing::warn!("run {id}: its weights' early fetch: {error}"),
                }
                fetch
            });
            gpu.prespawn(generation.clone());
        }
        let fallback = (!generation.record.sdk_fallback.is_empty())
            .then(|| format!("{} {}: {}", installation.package, installation.release, generation.record.sdk_fallback));
        for warning in fallback.into_iter().chain(precision) {
            if let Err(error) = self.service.engine.append_warning(id, &warning) {
                tracing::warn!("run {id}: {warning} ({error})");
            }
        }
        let declared = interface["entrypoints"]
            .as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == spec.entrypoint.as_str()));
        if declared.is_none() && !install_only {
            return Err(refused(
                "invalid_entrypoint",
                format!("{} declares no entrypoint {:?}", installation.package, spec.entrypoint),
            ));
        }
        if spec.parent.is_empty() {
            if let Some(entry) = declared {
                public_root(entry)?;
            }
        }
        // A choice for another callable goes to the call that names it: one of this package's,
        // or a slot a callee App of its environment declares. One naming neither is a warning.
        for choice in spec.models.iter().filter(|c| !install_only && !own_choice(c, &spec.entrypoint)) {
            let declared = addressed(&choice.parameter, &installation.package, &interface).is_some()
                || generation.record.callees.iter().any(|callee| {
                    addressed(&choice.parameter, &callee.package, &callee.interface).is_some()
                });
            if !declared {
                let warning = format!(
                    "model choice {:?} names no model slot of this function or the packages it calls; it is ignored",
                    choice.parameter
                );
                if let Err(error) = self.service.engine.append_warning(id, &warning) {
                    tracing::warn!("run {id}: {warning} ({error})");
                }
            }
        }
        if spec.warm {
            let models = match install_only {
                true => self.warm_models(actor, id, &spec, &*observe, stoppable)?,
                false => vec![],
            };
            // A function's warm run ends holding what its constructions read (`downloaded`).
            let mut moved = (0, 0);
            if let Some(plan) = &plan {
                let fetch = crate::access::Fetch::new(self.publisher.machine().clone(), hub.clone(), u64::MAX);
                let report = |done, total| observe("downloading its weights", done, total);
                moved = fetch.download(&plan.upfront(), &stoppable.stop, &report).map_err(|e| match stoppable.stop.is_cancelled() {
                    true => refused(crate::published::CANCELED, "the run was canceled while it downloaded"),
                    false => {
                        let text = e.to_string();
                        refused(if text.starts_with("machine_disk_full") { "machine_disk_full" } else { "model_fetch_failed" }, text)
                    }
                })?;
            }
            // The installed release, its interface and the installation a call may name, in the
            // warm run's result.
            let installed = json!({"package": installation.package, "release": installation.release,
                "installation": installation.alias, "interface": interface,
                "bytes": {"hub": moved.0, "volume": moved.1}});
            return warm_result(installed, models);
        }
        let entry = declared.expect("checked above");
        let degree = match (&plan, self.service.pool()) {
            (Some(plan), _) => plan.degree,
            (None, Some(pool)) => pool
                .degree(entry, spec.gpus)
                .map_err(|e| refused("gpu_count_unavailable", e.to_string().replace("gpu_count_unavailable: ", "")))?,
            (None, None) => 0,
        };
        // A memoized callable's run is answered from the success this machine holds for its
        // signer under the same computation, and starts nothing. A writer (weights outputs,
        // published media) has an effect beyond its result: it always runs.
        let writer = entry["weights_outputs"].as_array().is_some_and(|rows| !rows.is_empty())
            || entry["publishes"].as_bool() == Some(true)
            || !spec.weights_destination.is_empty();
        let memo = match writer {
            false => call_memo::of(
                &generation.record,
                &spec.application,
                &entry["invocable"],
                &spec.input,
                &spec.inputs,
                (&spec.attention_kernel, spec.bits),
                call_memo::models(plan.as_ref(), &own),
            ),
            true => Memo::Off,
        };
        let (memo, answer) = match memo {
            Memo::Key(computation) => {
                let answer = self.objects.recall(actor, &computation, id)?;
                (CallMemo { memoized: answer.is_some(), computation, ..Default::default() }, answer)
            }
            Memo::Unmet(why) => (CallMemo { unmemoized: why, ..Default::default() }, None),
            Memo::Off => (CallMemo::default(), None),
        };
        // Its children prepare with its context, and its weights travel under its access.
        if answer.is_none() {
            let context = RunContext {
                installation: installation.alias.clone(),
                hub,
                providers: spec.providers.clone(),
                tensorfs: spec.tensorfs.clone(),
                weights_destination: spec.weights_destination.clone(),
                owner: spec.owner.clone(),
                catalog_revision: spec.catalog_revision.clone(),
                attention_kernel: spec.attention_kernel.clone(),
                gpus: spec.gpus,
                bits: spec.bits,
                models: spec.models.clone(),
                ahead,
            };
            self.contexts.lock().unwrap().insert(id.into(), context);
        }
        let bound = self.service.bind_prepared(
            id,
            &installation.generation,
            Call {
                entrypoint: spec.entrypoint,
                input: spec.input,
                attention_kernel: spec.attention_kernel,
                gpus: spec.gpus,
                bits: spec.bits,
                inputs: spec.inputs,
                degree,
                parent: spec.parent,
                application: spec.application,
            },
            plan.as_ref().map(|plan| plan.id.as_str()).unwrap_or_default(),
            memo,
            answer,
        );
        if bound.is_err() {
            self.end(id);
        }
        bound?;
        Ok(None)
    }

    /// A warm set member, built one way whether a warm run makes it or a restart brings it back
    /// (`restore_warm_sets`). Its weights come behind every run's, under `hub`: the run's access
    /// with its owner's newest bearer. Without one (no bearer survives a restart) they come from
    /// the rental's own Hub, else anonymously from its installation's Hub: a public release
    /// resumes, a private one waits for its owner's next run.
    fn member(&self, held: HeldGeneration, installation: &Installation, plan: Option<Plan>, level: Level,
        hub: Option<hub::Source>) -> Arc<Member> {
        let hub = hub.or_else(|| {
            let own = self.own_hub.clone().filter(|own| {
                installation.hub.is_empty() || hub::origin_key(&own.origin).as_deref() == Some(installation.hub.as_str())
            });
            own.or_else(|| hub::Source::new(&installation.hub, None, None).ok())
        });
        let fetch = plan.is_some().then(|| crate::access::Fetch::new(self.publisher.machine().clone(), hub, u64::MAX));
        Member::new(held, plan, level, fetch)
    }

    /// Each caller's warm set as the journal keeps it, built again (`member`) and brought up in
    /// the background: a restarted machine re-warms without its controller. A member whose
    /// installation is gone leaves the set.
    pub(crate) fn restore_warm_sets(&self) -> io::Result<()> {
        let Some(gpu) = self.service.gpu() else {
            return Ok(());
        };
        let kept = self.service.engine.with_journal(|journal| journal.warm_sets())?;
        let actors: std::collections::BTreeSet<&String> = kept.iter().map(|(actor, _)| actor).collect();
        for actor in actors {
            let all = kept.iter().filter(|(of, _)| of == actor);
            let (records, members): (Vec<String>, Vec<Arc<Member>>) = all
                .clone()
                .filter_map(|(_, record)| Some((record.clone(), self.restore_member(&gpu, actor, record)?)))
                .unzip();
            if records.len() < all.count() {
                self.service.engine.with_journal(|journal| journal.replace_warm_set(actor, &records))?;
            }
            gpu.set_members(actor, members);
        }
        self.service.rewarm(&gpu)
    }

    /// One kept member, with its plan as it was bound while the store still holds its weights.
    fn restore_member(&self, gpu: &Pool, actor: &str, record: &str) -> Option<Arc<Member>> {
        let kept: KeptMember = serde_json::from_str(record).ok()?;
        let installed = self.service.engine.installation(actor, &kept.installation).ok()??;
        let held = self.service.catalog.resolve(&installed.generation).ok()?;
        let bound = match kept.preparation.as_str() {
            "" => None,
            id => self.service.engine.preparation(actor, id).ok()?,
        };
        let plan = bound.and_then(|bound| gpu.plan(&bound).ok()).filter(|plan| gpu.holds(plan));
        Some(self.member(held, &installed, plan, Level::named(&kept.level)?, None))
    }

    /// A warm run's set: every member installed and its weights downloaded as a warm run of it
    /// alone does, the set kept in the journal, and one pass to bring the members up. The
    /// result lists each member's `level` and why it is lower than asked (`held_back`).
    fn warm_set(
        &self,
        actor: &str,
        id: &str,
        spec: &Spec,
        items: &[SetItem],
        observe: &Arc<impl Fn(&str, u64, u64) + Send + Sync + 'static>,
        stoppable: &Stoppable,
    ) -> Result<Option<ResultRecord>, Refused> {
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let (mut members, mut kept, mut rows) = (vec![], vec![], vec![]);
        for item in items {
            let (held, release) = match &item.source {
                Source::Installation(alias) => {
                    let held = self.service.engine.installation(actor, alias)?;
                    let absent = || refused("installation_absent", "this machine holds no such installation for this signer");
                    (Some(held.ok_or_else(absent)?), None)
                }
                Source::Release { package, release } => (None, Some((package.clone(), release.clone()))),
                _ => return Err(refused("invalid_request", "a warm set member names a release or an installation")),
            };
            // `installed` stops at the code: nothing of the function is resolved or downloaded.
            let (entrypoint, choices) = match item.level {
                Level::Installed => ("", &[][..]),
                _ => (item.entrypoint.as_str(), &item.models[..]),
            };
            let report = observe.clone();
            let (installation, plan, _) = self.resolve(
                actor,
                id,
                held,
                release,
                hub.clone(),
                spec,
                entrypoint,
                choices,
                item.bits,
                Some(Box::new(move |stage: &str, done: u64, total: u64| report(stage, done, total))),
                Some(&stoppable.stop),
            )?;
            if item.level > Level::Installed && !declares(&installation, &item.entrypoint)? {
                return Err(refused(
                    "invalid_entrypoint",
                    format!("{} declares no entrypoint {:?}", installation.package, item.entrypoint),
                ));
            }
            // A function that loads no model itself (an orchestrator) runs on the ones it calls.
            // The interface names no call graph, and internal functions exist only to be called:
            // it brings its package's internal functions' models, each a member of its own.
            let mut plans = vec![plan];
            if item.level > Level::Installed && plans[0].is_none() {
                for callee in internal_models(&installation)? {
                    let own: Vec<_> = item.models.iter().filter(|c| own_choice(c, &callee)).cloned().collect();
                    let report = observe.clone();
                    let (_, plan, _) = self.resolve(
                        actor,
                        id,
                        Some(installation.clone()),
                        None,
                        hub.clone(),
                        spec,
                        &callee,
                        &own,
                        item.bits,
                        Some(Box::new(move |stage: &str, done: u64, total: u64| report(stage, done, total))),
                        Some(&stoppable.stop),
                    )?;
                    plans.push(plan);
                }
            }
            rows.push((json!({"package": installation.package, "release": installation.release,
                "entrypoint": item.entrypoint}), plans.len()));
            for plan in plans {
                kept.push(serde_json::to_string(&KeptMember {
                    installation: installation.alias.clone(),
                    level: item.level.name().into(),
                    preparation: plan.as_ref().map(|plan| plan.id.clone()).unwrap_or_default(),
                    item: STANDARD.encode(&item.sent),
                })
                .map_err(io::Error::other)?);
                let held = self.service.catalog.resolve(&installation.generation)?;
                members.push(self.member(held, &installation, plan, item.level, hub.clone()));
            }
        }
        self.service
            .engine
            .with_journal(|journal| journal.replace_warm_set(actor, &kept))?;
        observe("warming", 0, 0);
        let holds = match self.service.gpu() {
            Some(gpu) => {
                gpu.set_members(actor, members);
                gpu.keep(&self.service.engine, actor);
                gpu.members(actor)
            }
            None => members.iter().map(|member| without_gpus(member.level)).collect(),
        };
        // An item's level is its lowest member's, and why that one is held back.
        let mut holds = holds.into_iter();
        let rows: Vec<Value> = rows
            .into_iter()
            .map(|(mut row, count)| {
                let lowest = holds.by_ref().take(count).min_by_key(|(level, _)| *level);
                if let Some((level, held_back)) = lowest {
                    row["level"] = level.name().into();
                    if !held_back.is_empty() {
                        row["held_back"] = held_back.into();
                    }
                }
                row
            })
            .collect();
        Ok(Some(ResultRecord {
            value: json!({"set": rows}),
            artifacts: vec![],
            asset_bindings: vec![],
            adjustments: vec![],
        }))
    }

    /// A remove run: the caller's installations its source names leave the journal, their
    /// warm set members and the GPU pool, and each environment no installation or unfinished
    /// run names any more is deleted unless a process still holds it (reclaim deletes it once
    /// that ends). A release names its package at the run's Hub (any Hub for local/ code, or
    /// for an installation recorded before installations named theirs).
    fn remove(&self, actor: &str, spec: &Spec) -> Result<ResultRecord, Refused> {
        let hub = spec.hub.clone().or_else(|| self.own_hub.clone());
        let origin = hub.and_then(|hub| hub::origin_key(&hub.origin)).unwrap_or_default();
        let named = |installed: &Installation| match &spec.source {
            Source::Installation(alias) => &installed.alias == alias,
            Source::Release { package, release } => {
                &installed.package == package
                    && (release.is_empty() || &installed.release == release)
                    && (package.starts_with("local/") || installed.hub.is_empty() || installed.hub == origin)
            }
            Source::Local(_) | Source::Models => false,
        };
        let (removed, reclaimed, retained) = self.uninstall(actor, named)?;
        if let Some(local) = &self.local {
            for gone in removed.iter().filter(|i| i.package.starts_with("local/")) {
                local.collect(actor, &gone.package);
            }
        }
        let rows: Vec<Value> = removed
            .iter()
            .map(|i| json!({"package": i.package, "release": i.release, "installation": i.alias, "hub": i.hub}))
            .collect();
        Ok(ResultRecord {
            value: json!({"removed": rows, "reclaimed_bytes": reclaimed, "retained": retained}),
            artifacts: vec![],
            asset_bindings: vec![],
            adjustments: vec![],
        })
    }

    /// `actor`'s installations `named` leave the journal, their warm set members and the GPU
    /// pool. Each environment no installation or unfinished run names any more is deleted,
    /// its idle executors ended first; one a process still holds is `retained` for reclaim.
    fn uninstall(
        &self,
        actor: &str,
        named: impl Fn(&Installation) -> bool,
    ) -> Result<(Vec<Installation>, u64, Vec<String>), Refused> {
        let engine = &self.service.engine;
        let mut removed = vec![];
        for installed in engine.installations(actor)?.into_iter().filter(|i| named(i)) {
            if let Some(gone) = engine.with_journal(|j| j.remove_installation(actor, &installed.alias))? {
                removed.push(gone);
            }
        }
        let generations: std::collections::BTreeSet<String> = removed.iter().map(|i| i.generation.clone()).collect();
        if let Some(gpu) = self.service.gpu() {
            gpu.drop_members(actor, &generations);
        }
        let bound = engine.with_journal(|j| j.bound_generations())?;
        let (mut reclaimed, mut retained) = (0u64, vec![]);
        for generation in generations.iter().filter(|g| !bound.contains(*g)) {
            if let Some(gpu) = self.service.gpu() {
                gpu.end_generation(generation);
            }
            match crate::reclaim::remove_generation(self.service.catalog.root(), generation)? {
                Some(bytes) => reclaimed += bytes,
                None => retained.push(generation.clone()),
            }
        }
        Ok((removed, reclaimed, retained))
    }

    /// A local package is one installation: the state just installed replaces its earlier
    /// ones, whose executors end and environments' records leave. One an unfinished run names
    /// stays until a later install finds it idle.
    fn supersede(&self, actor: &str, current: &Installation) -> Result<(), Refused> {
        let running = self.service.engine.with_journal(|j| j.running_generations())?;
        self.uninstall(actor, |prior| {
            prior.package == current.package
                && prior.alias != current.alias
                && prior.alias.starts_with("local-")
                && !running.contains(&prior.generation)
        })?;
        Ok(())
    }

    /// The run's installation, model plan and precision warnings, from the one resolver
    /// (`Publisher`): a release and a Hub-named model need the run's Hub access; held code and
    /// the models this machine holds (`local/` aliases, checkpoints by hash) do not. Unobserved,
    /// it is a hint's: its downloads go behind the ones calls wait on.
    #[allow(clippy::too_many_arguments)]
    fn resolve(
        &self,
        actor: &str,
        id: &str,
        held: Option<Installation>,
        release: Option<Published>,
        hub: Option<hub::Source>,
        spec: &Spec,
        entrypoint: &str,
        choices: &[domain::ModelChoice],
        bits: u32,
        observe: Option<crate::published::Observer>,
        stop: Option<&Stop>,
    ) -> Result<(Installation, Option<crate::gpu_service::Plan>, Vec<String>), Refused> {
        let (package, release) = release.unwrap_or_default();
        let request = Request {
            run: id.into(),
            source: hub,
            package,
            release,
            installed: held,
            owner: spec.owner.clone(),
            catalog_revision: spec.catalog_revision.clone(),
            providers: spec.providers.clone(),
            entrypoint: entrypoint.into(),
            choices: choices.to_vec(),
            gpus: spec.gpus,
            bits,
            imports: !spec.warm,
        };
        let prepared = self
            .publisher
            .prepare_now(&self.service, actor, &request, observe, stop)
            .map_err(|(code, message)| refused(code, message))?;
        Ok((prepared.installation, prepared.plan, prepared.warnings))
    }

    /// A child call of a running attempt: a run `<request>` under its parent's signer,
    /// idempotent on `intent`, preparing inside itself with its parent's context and the
    /// choices addressed to its callable.
    #[allow(clippy::too_many_arguments)]
    pub fn child(
        self: &Arc<Self>,
        parent: &Execution,
        request: &str,
        intent: &str,
        application: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
        passed: Vec<domain::ModelChoice>,
    ) -> Result<Execution, Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        // A call the same attempt made already is its run: nothing prepares.
        if let Some(existing) = self.existing(&actor, request, intent)? {
            return Ok(existing);
        }
        let spec =
            self.child_spec(&actor, &parent.id, application, entrypoint, input, inputs, intent, &passed)?;
        self.submit(&actor, request, spec)
    }

    /// The installation as a callee's child runs see it: the callee's package, release and
    /// interface (its weights and bindings are its own), in the caller's environment.
    fn callee_view(&self, installed: Installation, application: &str) -> Result<Installation, Refused> {
        if application.is_empty() {
            return Ok(installed);
        }
        let held = self.service.catalog.resolve(&installed.generation)?;
        if application == held.record.application {
            return Ok(installed);
        }
        let (package, release, interface) = held.record.app(application).ok_or_else(|| {
            refused("child_undeclared", format!("this environment holds no App {application}"))
        })?;
        Ok(Installation {
            package: package.into(),
            release: release.into(),
            interface: serde_json::to_vec(interface).map_err(io::Error::other)?,
            ..installed
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn child_spec(
        &self,
        actor: &str,
        parent: &str,
        application: &str,
        entrypoint: &str,
        input: Value,
        inputs: Vec<InputFile>,
        digest: &str,
        passed: &[domain::ModelChoice],
    ) -> Result<Spec, Refused> {
        let context = self.contexts.lock().unwrap().get(parent).cloned().ok_or_else(|| {
            refused("child_call_refused", "the caller's preparation context is gone")
        })?;
        let installed = self.service.engine.installation(actor, &context.installation)?
            .ok_or_else(|| refused("child_call_refused", "the caller's installation is gone"))?;
        let caller = installed.package.clone();
        let installed = self.callee_view(installed, application)?;
        let callee_interface: Value = serde_json::from_slice(&installed.interface)
            .map_err(|_| refused("package_interface_invalid", "held interface is corrupt"))?;
        let prefix = format!("{entrypoint}.");
        let mut spec = Spec {
            remove: false,
            warm: false,
            set: None,
            parent: parent.into(),
            source: Source::Installation(context.installation),
            entrypoint: entrypoint.into(),
            input,
            inputs,
            // A child takes the caller's choices addressed to a slot its own interface declares
            // (a callee's included), named by that slot's path; any other was a warning at the
            // caller's preparation.
            models: context
                .models
                .into_iter()
                .filter_map(|choice| {
                    let path = addressed(&choice.parameter, &installed.package, &callee_interface)
                        .filter(|path| path.starts_with(&prefix))?;
                    Some(domain::ModelChoice { parameter: path, ..choice })
                })
                .collect(),
            attention_kernel: context.attention_kernel,
            // The caller's count caps a child's group; the child's own declaration picks within it.
            gpus: context.gpus,
            // The asked lane follows the main model into the package's own children (long_form's
            // segments); another package's model (qwen-image-2 for references) takes its default.
            bits: if installed.package == caller { context.bits } else { 0 },
            hub: context.hub,
            providers: context.providers,
            tensorfs: context.tensorfs,
            weights_destination: String::new(),
            owner: context.owner,
            catalog_revision: context.catalog_revision,
            application: application.into(),
            digest: digest.into(),
            journaled: vec![],
        };
        // Both demand and prefetch overlay explicit artifacts on the same inherited
        // choices. Declaration and data-access validation stay in the existing resolver.
        for choice in passed {
            let path = addressed(&choice.parameter, &installed.package, &callee_interface)
                .filter(|path| path.starts_with(&prefix))
                .ok_or_else(|| refused("invalid_request", "model choice does not name a declared callee slot"))?;
            let mut choice = choice.clone();
            choice.parameter = path;
            spec.models.retain(|prior| prior.parameter != choice.parameter);
            spec.models.push(choice);
        }
        Ok(spec)
    }

    /// `model_prefetch`: an acknowledged hint about a future child call. TensorD resolves
    /// its models, downloading behind every download a call waits on, and asks the GPU pool
    /// to prewarm a Runtime construction when admitted.
    pub fn prefetch(
        self: &Arc<Self>, parent: &Execution, application: &str, entrypoint: &str,
        passed: Vec<domain::ModelChoice>,
    ) -> Result<(), Refused> {
        let actor = parent.submission.as_ref().map(|s| s.actor.clone()).unwrap_or_default();
        let spec = self.child_spec(&actor, &parent.id, application, entrypoint,
            json!({}), vec![], "", &passed)?;
        let Some(gpu) = self.service.gpu() else {
            return Ok(());
        };
        // Downloads may outlast the hint's GPU phase; the GPU load waits for its root's lane.
        let (root, family) = (parent.root_seq(), parent.family().map(String::from));
        let owner = parent.id.clone();
        let (runs, entrypoint) = (self.clone(), entrypoint.to_string());
        let application = application.to_string();
        let started = std::thread::Builder::new().name("child-prefetch".into()).spawn(move || {
            let prepared = (|| {
                    let Source::Installation(alias) = &spec.source else { unreachable!() };
                    let held = match runs.service.engine.installation(&actor, alias)? {
                        Some(installed) => Some(runs.callee_view(installed, &spec.application)?),
                        None => None,
                    };
                    let (hub, models) = (spec.hub.clone(), spec.models.clone());
                    runs.resolve(&actor, "", held, None, hub, &spec, &spec.entrypoint, &models, spec.bits, None, None)
                })();
            match prepared {
                Ok((installation, Some(plan), _)) if plan.degree > 0 && !plan.slots.is_empty() => {
                    match runs.service.catalog.resolve(&installation.generation)
                        .and_then(|held| held.application(&application)) {
                        Ok(held) => {
                            // Its weights come after its root's own, under the child's access.
                            let hub = spec.hub.clone().or_else(|| runs.own_hub.clone());
                            let fetch = crate::access::Fetch::new(runs.publisher.machine().clone(), hub, root);
                            gpu.prefetch(&runs.service.engine, held, plan, (root, family.clone()), owner.clone(), fetch)
                        }
                        Err(error) => tracing::warn!("prefetch of {entrypoint}: {error}"),
                    }
                }
                Ok(_) => (),
                Err(refusal) => {
                    tracing::warn!("prefetch of {entrypoint}: {}: {}", refusal.code, refusal.message)
                }
            }
        });
        if let Err(error) = started {
            tracing::warn!("prefetch of a child: {error}");
        }
        Ok(())
    }

    /// The fetches a run's preparation started, for its call to end once its executor asks.
    pub fn ahead(&self, id: &str) -> Option<Arc<crate::access::Fetch>> {
        self.contexts.lock().unwrap().get_mut(id).and_then(|context| context.ahead.take())
    }

    /// The run's TensorFS options for the Runtime `line` reaches, with its owner's newest bearer
    /// at its Hub, of which that executor hears each newer one.
    pub(crate) fn tensorfs(&self, id: &str, line: Line) -> Option<protocol::machine::TensorFSOptions> {
        let contexts = self.contexts.lock().unwrap();
        let context = contexts.get(id)?;
        let mut options = context.tensorfs.clone()?;
        if let Some(bearer) = context.hub.as_ref().and_then(|hub| hub.bearer_at(&options.endpoint)) {
            (options.token, options.expires_unix_ms) = bearer.given(line, &options.endpoint);
        }
        Some(options)
    }

    pub(crate) fn destination(&self, id: &str) -> String {
        self.contexts.lock().unwrap().get(id).map(|context| context.weights_destination.clone()).unwrap_or_default()
    }

    /// The Hub a run reads and writes: its own access, else the machine's.
    pub fn hub(&self, id: &str) -> Option<hub::Source> {
        let context = self.contexts.lock().unwrap().get(id).and_then(|c| c.hub.clone());
        context.or_else(|| self.own_hub.clone())
    }

    /// A run ended: its children prepare no more, and its access is forgotten.
    pub fn end(&self, id: &str) {
        self.contexts.lock().unwrap().remove(id);
    }
}
