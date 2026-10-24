//! Published packages and their models, prepared from a Hub by name (th-245) and reused: a
//! run whose installation and model resolution are held reads nothing from the Hub. Port of the
//! Python worker's release-root preparation (package set from the release's locked
//! requirements with uv; each model's lane by the lane rule (`precision`), then a TensorFS
//! download) with its `release_root_preparing` progress answers.
use crate::{
    api::domain,
    gpu_service::{Plan, Pool, ModelGrant},
    hub::{self, Catalog},
    journal::{Installation, Preparation},
    objects::Refused,
    precision,
    process::Supervised,
    service::Service,
};
use base64::Engine as _;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fs::{self, File, OpenOptions},
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    time::Instant,
};
use crate::{
    held_models::Pulled,
    origin::{self, priority, Machine},
    stop::Stop,
    storage::{self, Store},
};
use tensorfs::{
    err::Refusal,
    fetch::Progress as Counted,
    format::ObjectRef,
    sha256,
};
use crate::catalog::normalized;

/// How each published environment gets its Runtime SDK. Empty `requirements` keeps the
/// release's own locked SDK rows.
#[derive(Clone, Debug, Default)]
pub struct PackageSdk {
    pub uv: PathBuf,
    pub python: String,
    pub requirements: Vec<String>,
    pub find_links: Option<PathBuf>,
    /// The machine's description client (`tensord_client`), installed into each generation.
    pub client_wheel: Option<PathBuf>,
    /// An image's baked uv cache (expanded wheels, index records), linked rather than copied.
    pub seed_cache: Option<PathBuf>,
    /// On a local machine, its owner's uv cache and managed Pythons (what the owner's own and
    /// local-package installs use), so a release's Torch stack is fetched and kept once.
    pub owner_uv: Option<OwnerUv>,
    /// The development Runtime an update named by its wheel file (`0.24.6+e9baf7a4`): the
    /// operator's choice, which runs every package whatever its own bounds on the pair.
    pub runtime_build: String,
}

/// Where uv keeps its cache and managed Pythons for the owner whose environment `env` is: as
/// uv itself resolves them (its variables, then XDG, then the home directory's defaults).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OwnerUv {
    pub cache: PathBuf,
    pub python: PathBuf,
}

impl OwnerUv {
    pub fn of(env: &BTreeMap<String, String>) -> Option<Self> {
        let path = |name: &str| env.get(name).map(PathBuf::from).filter(|p| p.is_absolute());
        let home = path("HOME");
        let base = |xdg: &str, default: &str| path(xdg).or_else(|| home.as_ref().map(|h| h.join(default)));
        Some(Self {
            cache: path("UV_CACHE_DIR").or_else(|| Some(base("XDG_CACHE_HOME", ".cache")?.join("uv")))?,
            python: path("UV_PYTHON_INSTALL_DIR").or_else(|| Some(base("XDG_DATA_HOME", ".local/share")?.join("uv/python")))?,
        })
    }
}

impl PackageSdk {
    /// Executors use the selected Runtime/TensorFS pair (an update's, else the image's), so they
    /// speak this machine's protocol; without a complete pair, each release's locked SDK.
    pub fn selected(wheels: &Path, root: &Path, uv: PathBuf) -> Self {
        // An activation replaces sdk/current, not the immutable candidate directory. Freeze
        // that selection once: both cache identity and the installer's actual wheel inputs
        // must name the same build even when filenames/version labels repeat.
        let wheels = platform::durable::canonical(wheels).unwrap_or_else(|_| wheels.to_path_buf());
        let found: Vec<PathBuf> = fs::read_dir(&wheels)
            .into_iter()
            .flatten()
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|e| e == "whl"))
            .collect();
        let named = |prefix: &str| {
            found.iter().any(|w| w.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with(prefix)))
        };
        let complete = named("cozy_runtime-") && named("tensorfs-");
        let mut pair: Vec<String> = found.iter().map(|path| path.display().to_string()).collect();
        pair.sort();
        let seed = root.join("opt/cozy/dependency-seed/uv-cache");
        Self {
            uv,
            python: "3.12".into(),
            runtime_build: match complete {
                true => fs::read_to_string(wheels.join(crate::machine::update::RUNTIME_BUILD)).unwrap_or_default(),
                false => String::new(),
            },
            find_links: complete.then_some(wheels),
            requirements: if complete { pair } else { vec![] },
            client_wheel: None,
            seed_cache: seed.is_dir().then_some(seed),
            owner_uv: None,
        }
    }
}

pub struct Prepared {
    pub installation: Installation,
    pub plan: Option<Plan>,
    /// The lane rule's warnings for the run (`precision`), held with its resolution.
    pub warnings: Vec<String>,
}

pub struct Request {
    /// The run this prepares for: a run waiting on its install is told so by this id.
    pub run: String,
    /// The run's Hub: a release and a Hub-named model need it; held code and models do not.
    pub source: Option<hub::Source>,
    pub package: String,
    /// Empty: the Hub's newest, as it named it under `catalog_revision`.
    pub release: String,
    /// Unpublished code this machine already holds, in place of the release.
    pub installed: Option<Installation>,
    /// The account an unpublished package's org-relative model names belong to.
    pub owner: String,
    /// The caller's catalog revision: a held resolution is reused only under the same one.
    pub catalog_revision: String,
    /// Provider tokens for source models (memory only).
    pub providers: Providers,
    pub entrypoint: String,
    pub choices: Vec<domain::ModelChoice>,
    /// 0 selects the widest supported machine group; otherwise exactly this many GPUs.
    pub gpus: u32,
    /// The request's weight precision (16, 8 or 4); 0 lets the lane rule choose per slot.
    pub bits: u32,
    /// A call's: its imports start as soon as it installs. A warm run imports only when its
    /// level asks (`downloaded` never does).
    pub imports: bool,
}

impl Request {
    /// The run's Hub, for `what`: refused when the run carries no Hub access.
    fn hub(&self, what: &str) -> Result<&hub::Source, Failure> {
        let absent = || ("hub_access_absent", format!("{what} needs the run's Hub access, and this run carries none"));
        self.source.as_ref().ok_or_else(absent)
    }

    /// Its Hub's origin key; empty without one.
    fn origin(&self) -> String {
        self.source.as_ref().and_then(|source| hub::origin_key(&source.origin)).unwrap_or_default()
    }
}

/// A Hub checkpoint as a run names it (th-245): `repository` at `release` (empty: the newest) in
/// `lane` (empty: the lane rule's), or by hash (`tree` and no release): a commit (tree), or one
/// variant's header, which resolves to that header alone (no tree). A lane is a variant of its
/// release's tree. Resolved, it carries the release, lane and tree the Hub named and the lane's
/// header there, which is the checkpoint's id on this machine.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Named {
    pub repository: String,
    pub release: String,
    pub lane: String,
    /// The tree; a run's hash names one before it resolves (its length 0 when unsaid).
    pub tree: Option<ObjectRef>,
    /// The lane's cozytensors header in `tree`, once resolved.
    pub header: Option<ObjectRef>,
}

impl Named {
    /// Its tag's commit at the Hub (no tag: the default, the newest public version not yanked).
    fn at_hub(mut self, machine: &Machine, source: &hub::Source) -> Result<Self, Failure> {
        let (release, tree) = origin::release(machine, source, &self.repository, &self.release)
            .map_err(|e| refused("model_resolution_failed", e))?;
        (self.release, self.tree) = (release, Some(tree));
        Ok(self)
    }

    /// Its variant in its tree (TensorFS's one selection, header profiles only): its lane when
    /// it names one, else the one `constraints` take; and that variant's header.
    fn landed(mut self, machine: &Machine, source: &hub::Source, constraints: &precision::Constraints) -> Result<Self, Failure> {
        let failed = |e| refused("model_resolution_failed", e);
        let remote = origin::remote(machine, source).map_err(failed)?;
        let tree = machine.tree(&remote, self.tree()?, &Stop::default()).map_err(failed)?;
        if self.lane.is_empty() {
            let model = format!("{}@{}", self.repository, self.release);
            let selected = machine.select(&remote, &self.repository, &model, self.tree()?, &tree, constraints).map_err(failed)?;
            self.lane = selected.map_err(|e| ("model_resolution_failed", e.detail))?.variant;
        }
        let (path,header) = tree.variant(Some(&self.lane)).map_err(|e| ("model_resolution_failed",e.to_string()))?;
        self.lane = path.to_owned(); self.header = Some(header.clone());
        Ok(self)
    }

    pub(crate) fn tree(&self) -> Result<&ObjectRef, Failure> {
        self.tree.as_ref().ok_or_else(|| unresolved(self))
    }

    pub(crate) fn header(&self) -> Result<&ObjectRef, Failure> {
        self.header.as_ref().ok_or_else(|| unresolved(self))
    }
}

fn unresolved(named: &Named) -> Failure {
    ("model_resolution_failed", format!("{}@{} {} is not resolved", named.repository, named.release, named.lane))
}

/// The caller's provider tokens; empty reads public sources anonymously.
#[derive(Clone, Debug, Default)]
pub struct Providers {
    pub huggingface: String,
    pub civitai: String,
}

/// Where a preparation's stage and bytes are reported as they change.
pub type Observer = Box<dyn Fn(&str, u64, u64) + Send + Sync>;

#[derive(Default)]
struct Job {
    /// The stage its byte counts are reported under.
    stage: Mutex<String>,
    observer: Option<Observer>,
    /// Its run's stop: a cancel ends its downloads and installs.
    stop: Option<Stop>,
    /// Its run as its caller names it, and how that run is told what it waits for
    /// (`Engine::waiting`).
    caller: String,
    waiting: Option<Box<dyn Fn(Option<String>) + Send + Sync>>,
}
impl Job {
    fn set(&self, stage: String, moved: u64, total: u64) {
        if let Some(observe) = &self.observer {
            observe(&stage, moved, total);
        }
        *self.stage.lock().unwrap() = stage;
    }
    fn waiting(&self, reason: Option<String>) {
        if let Some(waiting) = &self.waiting {
            waiting(reason);
        }
    }
    fn stage(&self, stage: String) {
        self.set(stage, 0, 0);
    }
    fn bytes(&self, moved: u64, total: u64) {
        if let Some(observe) = &self.observer {
            observe(&self.stage.lock().unwrap(), moved, total);
        }
    }
}

type Failure = (&'static str, String);

/// A preparation its run's cancel stopped.
pub const CANCELED: &str = "preparation_canceled";
fn canceled() -> Failure {
    (CANCELED, "the run was canceled while it prepared".into())
}

/// What one environment generation is built from: the release's card (its exact pyproject and
/// uv.lock, its carried wheels, Python and interface), what its lock leaves to this machine, and
/// where the release lives.
#[derive(Clone, Copy)]
struct Environment<'a> {
    card: &'a Card,
    locked: &'a Locked,
    package: &'a str,
    release: &'a str,
    source: &'a hub::Source,
    /// What the person sees while it is built (`install_stage`), its steps counted after it.
    stage: &'a str,
}

pub struct Publisher {
    root: PathBuf,
    sdk: PackageSdk,
    machine: Arc<Machine>,
    store: Arc<Store>,
    /// One maker per source: a second asker waits, then finds it held (runs 5393/5394).
    making: Mutex<HashMap<String, Arc<Mutex<()>>>>,
    /// The run building each environment identity, which a second run waits for.
    building: Mutex<HashMap<String, String>>,
}

/// An identity's builder, forgotten however the build ends.
struct Building<'a>(&'a Mutex<HashMap<String, String>>, String);
impl Drop for Building<'_> {
    fn drop(&mut self) {
        self.0.lock().unwrap().remove(&self.1);
    }
}

impl Publisher {
    pub fn new(root: &Path, sdk: PackageSdk, machine: Arc<Machine>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        Ok(Arc::new(Self {
            root: root.to_path_buf(),
            sdk,
            store: machine.store().clone(),
            machine,
            making: Mutex::new(HashMap::new()),
            building: Mutex::new(HashMap::new()),
        }))
    }

    /// One run's preparation on the calling thread, reported through `observe`: a held
    /// installation and resolution answer at once.
    pub fn prepare_now(
        &self,
        service: &Arc<Service>,
        actor: &str,
        request: &Request,
        observe: Option<Observer>,
        stop: Option<&Stop>,
    ) -> Result<Prepared, Failure> {
        if let Ok(Some(prepared)) = self.held(service, actor, request) {
            return Ok(prepared);
        }
        let engine = service.engine.clone();
        let run = request.run.clone();
        let job = Job {
            observer: observe,
            stop: stop.cloned(),
            caller: service.engine.caller_id(&run),
            waiting: (!run.is_empty()).then(|| {
                Box::new(move |reason| {
                    if let Err(error) = engine.waiting(&run, reason) {
                        tracing::warn!("run {run}: its waiting reason was not shown: {error}");
                    }
                }) as Box<dyn Fn(Option<String>) + Send + Sync>
            }),
            ..Default::default()
        };
        self.work(service, actor, request, &job)
    }

    /// A slot's model made from its provider source (TensorFS `source_model`), kept as the
    /// local repository named by the source and its profiles.
    fn source_grant(
        &self,
        installation: &Installation,
        path: &str,
        choice: &domain::ModelChoice,
        providers: &Providers,
        job: &Job,
    ) -> Result<ModelGrant, Failure> {
        if !choice.repository.is_empty() || choice.manifest.is_some() {
            return Err((
                "model_override_invalid",
                format!("{path} names a source and a catalog checkpoint"),
            ));
        }
        let made = self.make(&choice.source, &choice.profiles, providers, job.stop.as_ref(), &|stage, moved, total| {
            job.set(format!("{stage} for {path}"), moved, total)
        })?;
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: made.repository,
            release: String::new(),
            lane: String::new(),
            manifest: made.header.id(),
            components: self.components(&made.header.id())?,
        })
    }

    /// The component names a held checkpoint's header declares.
    fn components(&self, header: &str) -> Result<Vec<String>, Failure> {
        let checkpoint = storage::checkpoint(&self.store, header).map_err(|e| ("checkpoint_absent", format!("{header}: {e}")))?;
        Ok(checkpoint.header().components().iter().map(|c| c.name.clone()).collect())
    }

    /// A slot's model this machine's store holds, by its header or its tree.
    fn held_grant(&self, installation: &Installation, path: &str, reference: &domain::Ref) -> Result<ModelGrant, Failure> {
        let named = format!("sha256:{}", sha256::hex(&reference.digest));
        let absent = |_| ("checkpoint_absent", format!("{path} names checkpoint {named}, which this machine does not hold"));
        let checkpoint = storage::checkpoint(&self.store, &named).map_err(absent)?;
        Ok(ModelGrant {
            package: installation.package.clone(),
            slot: path.to_string(),
            repository: String::new(),
            release: String::new(),
            lane: String::new(),
            components: checkpoint.header().components().iter().map(|c| c.name.clone()).collect(),
            manifest: checkpoint.id().id(),
        })
    }

    /// The store models are made and held in.
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    /// The machine's downloader.
    pub fn machine(&self) -> &Arc<Machine> {
        &self.machine
    }

    /// A provider source made into a local model (`make_source`), as a run's refusal.
    pub fn make_source(
        &self,
        source: &str,
        profiles: &[String],
        providers: &Providers,
        stop: Option<&Stop>,
        progress: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<Made, Refused> {
        self.make(source, profiles, providers, stop, progress)
            .map_err(|(code, message)| Refused { code, message })
    }

    /// A provider source made into a local model (`make_source`), one maker per source.
    fn make(&self, source: &str, profiles: &[String], providers: &Providers,
        stop: Option<&Stop>, progress: &(dyn Fn(&str, u64, u64) + Sync),
    ) -> Result<Made, Failure> {
        let mut sorted = profiles.to_vec();
        sorted.sort();
        let key = json!([source, sorted]).to_string();
        let gate = self.making.lock().unwrap().entry(key.clone()).or_default().clone();
        let made = {
            let _one = gate.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            make_source(&self.store, source, profiles, providers, stop.cloned(), progress)
        };
        let mut making = self.making.lock().unwrap();
        if Arc::strong_count(&gate) == 2 {
            making.remove(&key);
        }
        made
    }

    /// `named` from its Hub into the store, its ref recording the pull: its tree and header, then
    /// the `components` given (all when `None`, none when empty). Answers the bytes that came from the Hub and
    /// from the datacenter volume. Whatever is here moves nothing.
    fn pull(&self, source: &hub::Source, named: &Named, components: Option<&[String]>, stop: Option<&Stop>,
        bytes: &(dyn Fn(u64, u64) + Sync)) -> Result<(u64, u64), Failure> {
        let header = named.header()?;
        let remote = origin::remote(&self.machine, source).map_err(|e| refused("model_download_failed", e))?;
        let stop = stop.cloned().unwrap_or_default();
        let failed = |e: Refusal| match stop.is_cancelled() {
            true => canceled(),
            false => refused("model_download_failed", e),
        };
        let (started, subject) = (Instant::now(), format!("{}@{} {}", named.repository, named.release, named.lane));
        let failed = |e: Refusal| {
            storage::transport(&self.store, "pull-failed", &subject, &format!("origin={}: {e}", source.origin));
            failed(e)
        };
        let mut structure = self.machine.pull(&remote, header, Some(&[]), priority::NOW, None, &stop).map_err(failed)?;
        // A header by hash is recorded under a tree naming it alone.
        let tree = match &named.tree {
            Some(tree) => {
                let landed = self.machine.objects(&remote, std::slice::from_ref(tree), priority::NOW, &stop).map_err(failed)?;
                (structure.landed.origin, structure.landed.volume) = (structure.landed.origin + landed.origin, structure.landed.volume + landed.volume);
                tree.clone()
            }
            None => storage::tree_of(&self.store, header).map_err(io_failure)?,
        };
        let logged = |pulled: &tensorfs::fetch::Pulled| {
            let (fetched, reused) = (pulled.landed.origin + pulled.landed.volume, pulled.landed.local);
            let seconds = started.elapsed().as_secs_f64();
            let detail = format!("header={} tree={} objects={} fetched={fetched} reused={reused} seconds={seconds:.1} origin={}",
                header.id(), tree.id(), pulled.objects, source.origin);
            storage::transport(&self.store, "pull", &subject, &detail);
        };
        let pulled = Pulled { repository: named.repository.clone(), release: named.release.clone(), lane: named.lane.clone(), tree: tree.clone() };
        pulled.record(&self.store).map_err(io_failure)?;
        if components == Some(&[]) {
            logged(&structure);
            return Ok((structure.landed.origin, structure.landed.volume));
        }
        let progress = Arc::new(Counted::default());
        let pulled = origin::watched(&progress, bytes, || {
            self.machine.pull(&remote, header, components, priority::NOW, Some(progress.clone()), &stop)
        });
        let pulled = pulled.map_err(failed)?;
        logged(&pulled);
        let landed = pulled.landed;
        Ok((structure.landed.origin + landed.origin, structure.landed.volume + landed.volume))
    }

    /// `named` downloaded whole, as a request's weights are: what a caller asked for whole
    /// (`cozy model download`, an adapter).
    fn ensure(&self, source: &hub::Source, named: &Named, stop: Option<&Stop>,
        bytes: &(dyn Fn(u64, u64) + Sync)) -> Result<(u64, u64), Failure> {
        self.pull(source, named, None, stop, bytes)
    }

    /// The Hub checkpoint `header` is, as this machine pulled it (`Pulled`); none for one never
    /// pulled (local).
    fn pulled(&self, header: &ObjectRef) -> io::Result<Option<Named>> {
        Ok(Pulled::of(&self.store, header)?.map(|p| Named {
            repository: p.repository,
            release: p.release,
            lane: p.lane,
            tree: Some(p.tree),
            header: Some(header.clone()),
        }))
    }

    /// Each Hub checkpoint's header, landed at once: which of its chunks come, and when, its
    /// calls direct (`access`). A held header asks the Hub nothing.
    fn structure(&self, source: &hub::Source, checkpoints: &[Named], job: &Job) -> Result<(), Failure> {
        if checkpoints.is_empty() {
            return Ok(());
        }
        let names: Vec<String> = checkpoints.iter().map(|n| format!("{}@{} {}", n.repository, n.release, n.lane)).collect();
        job.stage(format!("resolving {}", names.join(" and ")));
        for named in checkpoints {
            self.pull(source, named, Some(&[]), job.stop.as_ref(), &|_, _| ())?;
        }
        Ok(())
    }

    /// The checkpoint `named` is: a tree by hash as given, a name as its release's tree at the
    /// Hub; then the header there of its lane, else of the first of `lanes` the tree has, else of
    /// its only lane. Journaled per catalog revision so a warm run, across restarts too,
    /// asks the Hub nothing (th-245).
    pub(crate) fn resolve(&self, service: &Service, source: &hub::Source, named: Named, constraints: &precision::Constraints, revision: &str) -> Result<Named, Failure> {
        if named.tree.is_some() {
            return measured(&self.machine, service, source, named, constraints);
        }
        let origin = hub::origin_key(&source.origin).unwrap_or_default();
        let name = json!(["profiles-v2", origin, named.repository, named.release, named.lane, constraints]).to_string();
        let held = service.engine.with_journal(|j| j.named(&name, revision)).map_err(io_failure)?;
        if let Some((release, lane, tree, header)) = held.and_then(|held| serde_json::from_str(&held).ok()) {
            return Ok(Named { release, lane, tree: Some(tree), header: Some(header), ..named });
        }
        let resolved = named.at_hub(&self.machine, source)?.landed(&self.machine, source, constraints)?;
        let answer = json!([resolved.release, resolved.lane, resolved.tree, resolved.header]).to_string();
        service.engine.with_journal(|j| j.bind_named(&name, revision, &answer)).map_err(io_failure)?;
        Ok(resolved)
    }

    /// A Hub checkpoint resolved (`resolve`) and downloaded whole into the store (a warm run's
    /// model choice: `cozy model download`); answers what the Hub named and the bytes that came
    /// from the Hub and from the datacenter volume.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn download(
        &self,
        service: &Service,
        source: &hub::Source,
        named: Named,
        constraints: &precision::Constraints,
        revision: &str,
        stop: Option<&Stop>,
        bytes: &(dyn Fn(u64, u64) + Sync),
    ) -> Result<(Named, (u64, u64)), Refused> {
        let failed = |(code, message)| Refused { code, message };
        let named = self.resolve(service, source, named, constraints, revision).map_err(failed)?;
        // A checkpoint this store already holds whole moves nothing and asks the Hub nothing.
        let moved = self.ensure(source, &named, stop, bytes).map_err(failed)?;
        Ok((named, moved))
    }

    /// The installation a request names. An unnamed release is the one the Hub named under
    /// the caller's catalog revision (`newest`), so a warm run asks the Hub nothing.
    fn alias(&self, service: &Service, request: &Request) -> io::Result<String> {
        if let Some(installed) = &request.installed {
            return Ok(installed.alias.clone());
        }
        let release = match (request.release.is_empty(), request.catalog_revision.is_empty()) {
            (true, false) => service.engine.with_journal(|j| j.named(&newest_key(request), &request.catalog_revision))?.unwrap_or_default(),
            _ => request.release.clone(),
        };
        Ok(self.release_alias(request, &release))
    }

    /// The Hub's newest release of the request's package, journaled per catalog revision as a
    /// model's name is (`resolve`): a moved revision asks once more. No revision names no catalog
    /// state, so that ask is the Hub's every time: kept, its first answer would stand forever.
    fn newest(&self, service: &Service, request: &Request, catalog: &Catalog, org: &str, name: &str) -> Result<String, Failure> {
        let (key, revision) = (newest_key(request), &request.catalog_revision);
        if revision.is_empty() {
            return newest_release(catalog, org, name);
        }
        if let Some(held) = service.engine.with_journal(|j| j.named(&key, revision)).map_err(io_failure)? {
            return Ok(held);
        }
        let release = newest_release(catalog, org, name)?;
        service.engine.with_journal(|j| j.bind_named(&key, revision, &release)).map_err(io_failure)?;
        Ok(release)
    }

    /// An earlier environment of this release here, which kept its card beside it (`CARD`): an
    /// exact release never changes, so its environment for another SDK reads nothing from the Hub.
    fn earlier(&self, service: &Service, actor: &str, request: &Request, release: &str) -> Option<PathBuf> {
        let origin = request.origin();
        service.engine.installations(actor).ok()?.into_iter()
            .filter(|i| i.package == request.package && i.release == release && i.hub == origin)
            .map(|i| service.catalog.root().join(&i.generation))
            .find(|dir| dir.join(CARD).is_file())
    }

    /// The Runtime and TensorFS versions an environment built here holds: this machine's own
    /// pair's (by its wheels' names), else the lock's.
    fn planned_pair(&self, locked: &Locked) -> BTreeMap<String, String> {
        let own = self.sdk.requirements.iter().filter_map(|wheel| {
            let mut parts = Path::new(wheel).file_name()?.to_str()?.split('-');
            Some((normalized(parts.next()?), parts.next()?.to_string()))
        });
        let pair: BTreeMap<String, String> = match self.sdk.requirements.is_empty() {
            true => locked.packages.clone(),
            false => own.collect(),
        };
        pair.into_iter().filter(|(name, _)| SDK.contains(&name.as_str())).collect()
    }

    /// The installation of `release` of the request's package at its Hub, by this machine's SDK.
    fn release_alias(&self, request: &Request, release: &str) -> String {
        let origin = request.origin();
        let sdk = format!(
            "{:?}{:?}{:?}",
            self.sdk.requirements, self.sdk.find_links, self.sdk.client_wheel
        );
        let key = format!("{origin}\0{}\0{release}\0{sdk}", request.package);
        format!("hub-{}", &sha256::hex_digest(key.as_bytes())[..32])
    }

    /// A held resolution's key: the request, under the machine's GPUs as the lane rule reads
    /// them (`gpus`), never a card's name.
    fn resolution_key(alias: &str, request: &Request, gpus: &str) -> String {
        let choices: Vec<_> = request
            .choices
            .iter()
            .map(|c| {
                json!([
                    c.parameter,
                    c.repository,
                    c.release,
                    c.lane,
                    c.manifest
                        .as_ref()
                        .map(|m| (sha256::hex(&m.digest), m.length)),
                    c.source,
                    c.profiles,
                    // Adapter order and scale are part of what the slot runs.
                    c.adapters
                        .iter()
                        .map(|a| [
                            &a.model,
                            &a.release,
                            &a.lane,
                            &a.manifest,
                            &a.component,
                            &a.source_component,
                            &a.scale,
                            &a.source
                        ])
                        .collect::<Vec<_>>()
                ])
            })
            .collect();
        let origin = request.origin();
        let package = request.installed.as_ref().map(|i| i.package.as_str()).unwrap_or(&request.package);
        let release = request.installed.as_ref().map(|i| i.release.as_str()).unwrap_or(&request.release);
        let key = json!({"installation":alias,"package":package,"release":release,"hub":origin,"owner":request.owner,"catalog":request.catalog_revision,"entrypoint":request.entrypoint,"choices":choices,"row":gpus,"gpus":request.gpus,"bits":request.bits});
        format!("hub-{}", sha256::hex_digest(key.to_string().as_bytes()))
    }

    /// The held installation and model plan, with no Hub read, or None.
    fn held(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
    ) -> io::Result<Option<Prepared>> {
        let held = match &request.installed {
            Some(installed) => Some(installed.clone()),
            None => service.engine.installation(actor, &self.alias(service, request)?)?,
        };
        let Some(installation) = held else {
            return Ok(None);
        };
        if service.catalog.resolve(&installation.generation).is_err() {
            return Ok(None);
        }
        if !declares_models(&installation, &request.entrypoint) {
            return Ok(Some(Prepared {
                installation,
                plan: None,
                warnings: vec![],
            }));
        }
        // An alias may now name another tree, or be gone. Resolve its current local metadata
        // before building a plan; the raw request key cannot prove a cached header is current.
        if local_aliases(&installation, request)? {
            return Ok(None);
        }
        let Some(gpu) = service.pool() else {
            return Ok(None);
        };
        let key = Self::resolution_key(&installation.alias, request, &machine_row(&gpu));
        let Some(id) = service.engine.with_journal(|j| j.resolution(actor, &key))? else {
            return Ok(None);
        };
        let Some(preparation) = service.engine.preparation(actor, &id)? else {
            return Ok(None);
        };
        let plan = gpu.plan(&preparation)?;
        // Its structure may have been reclaimed since; then the model resolves again. An
        // object reclaimed is fetched again by whatever reads it.
        if !gpu.holds(&plan) {
            return Ok(None);
        }
        let warnings = service.engine.with_journal(|j| j.named(&warnings_key(&key), &request.catalog_revision))?;
        Ok(Some(Prepared {
            installation,
            plan: Some(plan),
            warnings: warnings.and_then(|held| serde_json::from_str(&held).ok()).unwrap_or_default(),
        }))
    }

    fn work(
        &self,
        service: &Arc<Service>,
        actor: &str,
        request: &Request,
        job: &Job,
    ) -> Result<Prepared, Failure> {
        let held = match &request.installed {
            Some(installed) => Some(installed.clone()),
            None => service
                .engine
                .installation(actor, &self.alias(service, request).map_err(io_failure)?)
                .map_err(io_failure)?,
        };
        let installation = match held {
            Some(held) if service.catalog.resolve(&held.generation).is_ok() => held,
            _ if request.installed.is_some() => {
                return Err((
                    "release_root_installation_absent",
                    "this owner has not prepared the named installation".into(),
                ))
            }
            _ => self.install(service, actor, request, job)?,
        };
        if !declares_models(&installation, &request.entrypoint) {
            return Ok(Prepared {
                installation,
                plan: None,
                warnings: vec![],
            });
        }
        let gpu = service.pool().ok_or((
            "capability_unavailable",
            "this machine runs no functions".to_string(),
        ))?;
        let (plan, warnings) = self.model(service, &gpu, actor, &installation, request, job)?;
        Ok(Prepared {
            installation,
            plan: Some(plan),
            warnings,
        })
    }

    /// The request's release installed for this signer: its card and locked requirements as an
    /// earlier installation of it kept them, else read at its Hub by name (th-245); the Hub's
    /// newest release when it names none (`newest`). An installation of that release this
    /// signer holds already is the answer.
    fn install(
        &self,
        service: &Service,
        actor: &str,
        request: &Request,
        job: &Job,
    ) -> Result<Installation, Failure> {
        let (org, name) = request.package.split_once('/').ok_or((
            "release_root_invalid",
            "package must be org/name".to_string(),
        ))?;
        let source = request.hub("a release")?;
        let catalog = || Catalog::new(source).map_err(|e| ("catalog_read_failed", e.0));
        let release = match request.release.is_empty() {
            true => self.newest(service, request, &catalog()?, org, name)?,
            false => request.release.clone(),
        };
        let alias = self.release_alias(request, &release);
        if let Some(held) = service.engine.installation(actor, &alias).map_err(io_failure)? {
            if service.catalog.resolve(&held.generation).is_ok() {
                return Ok(held);
            }
        }
        // An earlier environment of this release (another SDK's): its card, kept beside it, and
        // the Runtime pair it holds. A kept card whose wheels the store no longer holds is read
        // again: its URLs expire.
        let earlier = self.earlier(service, actor, request, &release);
        let kept = earlier.as_ref().and_then(|dir| serde_json::from_slice::<Card>(&fs::read(dir.join(CARD)).ok()?).ok());
        let card = match kept.filter(|card| card.wheels.iter().all(|wheel| self.held_file(wheel))) {
            Some(card) => card,
            None => hub_card(&catalog()?, &request.package, &release, &self.sdk.python)?,
        };
        let locked = read_lock(&card.pyproject, &card.lock, source)?;
        let before = earlier.and_then(|dir| fs::read(dir.join("generation.json")).ok())
            .and_then(|raw| serde_json::from_slice::<crate::catalog::Generation>(&raw).ok())
            .map(|record| record.dependencies.into_iter().map(|d| (normalized(&d.name), d.version)).collect());
        let stage = install_stage(&request.package, &release, before.as_ref(), &self.planned_pair(&locked));
        let wheels: Vec<&str> = card.wheels.iter().map(|wheel| wheel.sha256.as_str()).collect();
        let identity = sha256::hex_digest(json!({"package":request.package,"release":release,"pyproject":card.pyproject,"lock":card.lock,"wheels":wheels,"python":card.python,"hub":hub::origin_key(&source.origin),"sdk":self.sdk.requirements,"links":self.sdk.find_links,"client":self.sdk.client_wheel}).to_string().as_bytes())[..32].to_string();
        let wanted = Environment {
            card: &card,
            locked: &locked,
            package: &request.package,
            release: &release,
            source,
            stage: &stage,
        };
        self.generation(service.catalog.root(), &identity, &wanted, job)?;
        keep_card(&service.catalog.root().join(&identity), &card);
        let held = service.catalog.resolve(&identity).map_err(io_failure)?;
        if let Some(gpu) = service.gpu() {
            // Its kernels compile, and a call's imports run, while its weights come.
            gpu.kernel_boot(held.clone());
            if request.imports {
                gpu.prespawn(held.clone());
            }
        }
        service
            .engine
            .bind_installation(Installation {
                actor: actor.into(),
                alias,
                generation: identity,
                package: request.package.clone(),
                release,
                interface: serde_json::to_vec(&held.record.interface)
                    .map_err(|e| io_failure(io::Error::other(e)))?,
                hub: request.origin(),
            })
            .map_err(io_failure)
    }

    /// Builds one immutable environment at its final path (its interpreter embeds the path);
    /// only the atomic `generation.json` makes it resolvable. Concurrent preparations of the
    /// same identity wait for one another.
    ///
    /// The release's own uv.lock is the install: `uv sync --frozen` takes every row this
    /// platform's resolution needs, by the lock's URLs and hashes, so a machine of any platform
    /// the lock covers installs it. The wheels the release's Hub hosts (its project wheel, its
    /// carried git dependencies, its org's packages) come through TensorFS's downloader instead
    /// and install from disk; the machine's own Runtime pair, where it has one, replaces the
    /// lock's.
    fn generation(
        &self,
        generations: &Path,
        identity: &str,
        wanted: &Environment,
        job: &Job,
    ) -> Result<(), Failure> {
        let Environment { card, locked, package, release, source, stage } = *wanted;
        let locks = generations.join(".locks");
        fs::create_dir_all(&locks).map_err(io_failure)?;
        let lock = File::create(locks.join(format!("{identity}.lock"))).map_err(io_failure)?;
        if lock.try_lock().is_err() {
            let holder = self.building.lock().unwrap().get(identity).cloned();
            job.waiting(Some(match holder {
                Some(run) => format!("waiting for run {run} to install {package} {release}"),
                None => format!("waiting for another run's install of {package} {release}"),
            }));
            lock.lock().map_err(io_failure)?;
            job.waiting(None);
        }
        self.building.lock().unwrap().insert(identity.into(), job.caller.clone());
        let _building = Building(&self.building, identity.into());
        let dir = generations.join(identity);
        if dir.join("generation.json").is_file() {
            return Ok(());
        }
        let _cache = self.uv_cache_hold().map_err(io_failure)?;
        if dir.exists() {
            fs::remove_dir_all(&dir).map_err(io_failure)?; // an interrupted, never-published build
        }
        let project = dir.join("project");
        fs::create_dir_all(&project).map_err(io_failure)?;
        fs::write(project.join("pyproject.toml"), &card.pyproject).map_err(io_failure)?;
        fs::write(project.join("uv.lock"), &card.lock).map_err(io_failure)?;
        let files: Vec<HubFile> = card.wheels.iter().chain(&locked.hub).cloned().collect();
        let own_sdk = !self.sdk.requirements.is_empty();
        let client = self.sdk.client_wheel.is_some();
        // One stage to the person, its steps counted: the Hub's files, the Python, the locked
        // packages, this machine's Runtime pair, the description.
        let steps = [!files.is_empty(), true, true, own_sdk, client].iter().filter(|s| **s).count();
        let step = std::cell::Cell::new(0);
        let next = || {
            step.set(step.get() + 1);
            job.stage(format!("{stage} {}/{steps}", step.get()));
        };
        if !files.is_empty() {
            next();
        }
        let wheels = self.fetch_hub(&files, source, &dir, job)?;
        let env = dir.join("env");
        let interpreter = platform::paths::venv_python(&env);
        let py = interpreter.to_string_lossy().to_string();
        next();
        self.uv(&["venv", "--no-project", "--no-config", "--python", &card.python, &env.to_string_lossy()], job.stop.as_ref())?;
        // The last install compiles the whole environment's bytecode: executors run `-I` from
        // an environment they may not write, so uncompiled modules compile again in every
        // executor (SDXL's imports: 10.7 s uncompiled, 4.5 s compiled; J/BREAKDOWN.md).
        let compile = |last: bool| if last { "--compile-bytecode" } else { "--no-compile-bytecode" };
        let hosted: Vec<String> = files.iter().map(HubFile::distribution).collect();
        let lock_sdk: Vec<&str> = SDK.into_iter().filter(|n| own_sdk && locked.packages.contains_key(*n)).collect();
        let sync = |last: bool| -> Result<(), Failure> {
            let project = project.to_string_lossy();
            // The release's own `[tool.uv]` settings hold (its build settings for a source-only
            // row): no `--no-config`.
            let mut args = vec!["sync", "--frozen", "--inexact", "--no-dev", "--no-install-project",
                "--project", &project, "--python", &py, compile(last)];
            let held = locked.git.iter().chain(&hosted).map(String::as_str).chain(lock_sdk.iter().copied());
            for name in held {
                args.extend(["--no-install-package", name]);
            }
            self.uv_observed(&args, &[("UV_PROJECT_ENVIRONMENT", env.as_os_str())], job.stop.as_ref(), &|read| job.bytes(read, 0))
                .map_err(|failure| platform(failure, package, release))
        };
        next();
        sync(wheels.is_empty() && !own_sdk && !client)?;
        if !wheels.is_empty() {
            let paths: Vec<String> = wheels.iter().map(|w| w.to_string_lossy().into_owned()).collect();
            let mut args = vec!["pip", "install", "--no-config", "--python", &py, compile(!own_sdk && !client), "--no-deps"];
            args.extend(paths.iter().map(String::as_str));
            self.uv(&args, job.stop.as_ref())?;
        }
        if own_sdk || client {
            let pins = self.export_pins(&project, &lock_sdk, job.stop.as_ref())?;
            fs::write(dir.join("constraints.txt"), pins).map_err(io_failure)?;
        }
        // The machine's own pair when it has one, whatever the lock's: a floor the release puts on
        // it that the pair misses refuses the run; other findings warn.
        let (sdk_choice, mut sdk_fallback) = (if own_sdk { "machine" } else { "locked" }, String::new());
        if own_sdk {
            next();
            sdk_fallback = self.install_sdk(&py, &dir, compile(!client), job.stop.as_ref()).map_err(|(code, detail)| {
                if code != "runtime_floor_unmet" {
                    return (code, detail);
                }
                let (pair, lock) = (self.planned_pair(locked), &locked.packages);
                let running = |name: &str| pair.get(name).map_or("none", String::as_str);
                let wanted = |name: &str| lock.get(name).map_or("<version>", String::as_str);
                (code, format!(
                    "{package} {release} needs {detail}, but this machine runs Runtime {} and TensorFS {}: update it first \
                     (cozy machine update <machine> --runtime {} --tensorfs {})",
                    running("cozy-runtime"), running("tensorfs"), wanted("cozy-runtime"), wanted("tensorfs"),
                ))
            })?;
            if !sdk_fallback.is_empty() {
                tracing::warn!("package {package} {release}: {sdk_fallback}");
                job.stage(format!("warning: {sdk_fallback}"));
            }
        }
        if let Some(wheel) = &self.sdk.client_wheel {
            next();
            let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
            let wheel = wheel.to_string_lossy().to_string();
            self.uv(&["pip", "install", "--no-config", "--python", &py, compile(true), "--constraints", &constraints, &wheel],
                job.stop.as_ref())?;
        }
        let interface = &card.interface;
        let application = protocol::interface::application(interface).ok_or((
            "package_prepare_interface_invalid",
            "the package interface names no application".to_string(),
        ))?;
        File::create(dir.join(".hold")).map_err(io_failure)?;
        // The Hub package each hosted wheel installs is its own name in the release's org, the
        // one org a Hub lock names; never read from the URL, whose shape is the Hub's to change
        // (th-245 did, and every callee became `local/<name>`: run 5234).
        let org = package.split_once('/').map_or("", |(org, _)| org);
        let hub_packages: BTreeMap<String, String> = hosted.iter().map(|name| (name.clone(), format!("{org}/{name}"))).collect();
        let (source_digest, callees) = match &self.sdk.client_wheel {
            Some(_) => describe_environment(&py, &locked.distribution, &hub_packages)?,
            None => (String::new(), vec![]),
        };
        let runtime_build = if sdk_choice == "machine" { self.sdk.runtime_build.as_str() } else { "" };
        let record = json!({"identity":identity,"package":locked.distribution,"version":release,"application":application,"python":interpreter,"dependencies":installed_sdk(&env),"interface":interface,"sdk":sdk_choice,"sdk_fallback":sdk_fallback,"runtime_build":runtime_build,"source_digest":source_digest,"callees":callees});
        let staged = dir.join(".generation.json.new");
        let mut file = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&staged)
            .map_err(io_failure)?;
        file.write_all(record.to_string().as_bytes())
            .and_then(|_| file.sync_all())
            .map_err(io_failure)?;
        fs::rename(&staged, dir.join("generation.json")).map_err(io_failure)?;
        platform::durable::sync_dir(&dir)
            .and_then(|()| platform::durable::sync_dir(generations))
            .map_err(|error| io_failure(error.into()))
    }

    /// The lock's exact versions as constraints (`name==version ; marker`), less `except`: what
    /// the machine's own Runtime pair and the client install against, so neither moves a
    /// locked package.
    fn export_pins(&self, project: &Path, except: &[&str], stop: Option<&Stop>) -> Result<String, Failure> {
        let out = project.join("pins.txt");
        let (project, path) = (project.to_string_lossy(), out.to_string_lossy());
        let mut args = vec!["export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes", "--no-annotate", "--no-header",
            "--format", "requirements.txt", "--project", &project, "--output-file", &path];
        for name in except {
            args.extend(["--no-emit-package", name]);
        }
        self.uv(&args, stop)?;
        let text = fs::read_to_string(&out).map_err(io_failure)?;
        Ok(text.lines().filter(|l| l.contains("==") && !l.trim_start().starts_with('#')).map(|l| format!("{l}\n")).collect())
    }

    /// The release's Hub-hosted wheels through TensorFS's downloader instead of uv's: 1 MiB
    /// ranged parts side by side, a late one asked again, verified by sha256 into the store.
    /// uv asks one GET per wheel and waits out every stall of it (run 5107, oczy: the install
    /// sat in two ~147 s rounds behind the Hub's 302 to R2, while TensorFS moved 51 GiB from
    /// the same R2 in 91 s). Answers each wheel's path, under its own file name.
    fn fetch_hub(&self, files: &[HubFile], source: &hub::Source, dir: &Path, job: &Job) -> Result<Vec<PathBuf>, Failure> {
        if files.is_empty() {
            return Ok(vec![]);
        }
        let wheels = dir.join("wheels");
        fs::create_dir_all(&wheels).map_err(io_failure)?;
        // Bytes as each file lands, of all of them: a person sees an install that crawls.
        let total: u64 = files.iter().map(|file| file.length).sum();
        let moved = std::sync::atomic::AtomicU64::new(0);
        std::thread::scope(|scope| {
            let handles: Vec<_> = files
                .iter()
                .map(|file| {
                    scope.spawn(|| -> Result<PathBuf, Failure> {
                        let path = self.hub_wheel(file, source, &wheels, job.stop.as_ref())?;
                        let landed = moved.fetch_add(file.length, std::sync::atomic::Ordering::Relaxed);
                        job.bytes(landed + file.length, total);
                        Ok(path)
                    })
                })
                .collect();
            // Each file is a fetch, judged by what it moves: this thread waits on them.
            let _asking = tensorfs::fetch::asking::begin();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap_or_else(|_| Err(io_failure(io::Error::other("a package file download panicked")))))
                .collect()
        })
    }

    /// Whether the store holds a Hub file (its URL expires; its bytes do not).
    fn held_file(&self, file: &HubFile) -> bool {
        file.length > 0 && storage::present(&self.store, &ObjectRef { sha256: file.sha256.clone(), length: file.length })
    }

    fn hub_wheel(&self, file: &HubFile, source: &hub::Source, wheels: &Path, stop: Option<&Stop>) -> Result<PathBuf, Failure> {
        let stop = stop.cloned().unwrap_or_default();
        let trust = source.ca_der.as_deref().map(|der| hub::pem(der).into_bytes());
        let length = (file.length > 0).then_some(file.length);
        let blob = crate::machine::pypi::object_at(&self.machine, &file.url, &file.sha256, length, trust, &stop)
            .map_err(|e| match stop.is_cancelled() {
                true => canceled(),
                // Named by its file: a plan's URL is presigned, its query a credential.
                false => ("package_file_download_failed", format!("{}: {}", file.name, e.to_string().replace(&file.url, &file.name))),
            })?;
        let path = wheels.join(&file.name);
        fs::hard_link(&blob, &path)
            .or_else(|_| fs::copy(&blob, &path).map(drop))
            .map_err(io_failure)?;
        Ok(path)
    }

    /// This machine's own uv cache, which `reclaim` manages; None when an image's seeded cache
    /// serves installs (environments symlink into it, so nothing there is ever removed) or the
    /// owner's does (uv and its owner manage that one).
    pub fn uv_cache(&self) -> Option<PathBuf> {
        (self.sdk.seed_cache.is_none() && self.sdk.owner_uv.is_none()).then(|| self.root.join("uv-cache"))
    }

    /// Held shared through one environment build, so `reclaim` removes nothing from the cache
    /// between uv's own per-command holds either.
    fn uv_cache_hold(&self) -> io::Result<Option<File>> {
        let Some(cache) = self.uv_cache() else {
            return Ok(None);
        };
        fs::create_dir_all(&cache)?;
        let lock = File::options()
            .create(true)
            .truncate(false)
            .write(true)
            .open(cache.join(".lock"))?;
        lock.lock_shared()?;
        Ok(Some(lock))
    }

    /// This machine's own Runtime and TensorFS over a release's locked requirements. Every other
    /// locked version stays exact: the lock's pins constrain the install, and a pin the pair's own
    /// requirements reject keeps the lock's version while the pair goes in without its
    /// dependencies. A bound the release puts on the pair that it misses (a floor: Fibo's
    /// `cozy-runtime>=0.30.16` for #1700's BatchNorm fix) refuses, `runtime_floor_unmet`: never a
    /// run on the wrong Runtime with a warning. What else uv's check finds that the pair adds is
    /// the run's warning. Returns that warning; empty when there is none.
    fn install_sdk(&self, py: &str, dir: &Path, compile: &str, stop: Option<&Stop>) -> Result<String, Failure> {
        let constraints = dir.join("constraints.txt").to_string_lossy().to_string();
        let links = self.sdk.find_links.as_ref().map(|p| p.to_string_lossy().to_string());
        let install = |how: &[&str]| {
            let mut args = vec!["pip", "install", "--no-config", "--python", py, compile];
            args.extend(how);
            if let Some(links) = &links {
                args.extend(["--find-links", links]);
            }
            args.extend(self.sdk.requirements.iter().map(String::as_str));
            self.uv(&args, stop)
        };
        // What the release's own closure reports already stands (NVIDIA's aarch64 cusparseLt
        // wheel is tagged `sbsa`, so uv flags it on every Linux arm64 machine).
        let before = self.check(py, stop)?;
        let mut warnings = vec![];
        if let Err((code, detail)) = install(&["--constraints", &constraints]) {
            if code == CANCELED {
                return Err((code, detail));
            }
            install(&["--no-deps"])?;
            warnings.push(format!(
                "this machine's Runtime and TensorFS want versions the release's lock does not hold, which it keeps ({})",
                crate::triage::ends(&detail, 1000)
            ));
        }
        let added: Vec<String> = self.check(py, stop)?.difference(&before).cloned().collect();
        let floors: Vec<&str> = (added.iter().map(String::as_str))
            .filter(|line| SDK.iter().any(|name| line.contains(&format!("requires `{name}"))))
            .filter_map(|line| line.split('`').nth(3))
            .collect();
        if !floors.is_empty() {
            return Err(("runtime_floor_unmet", floors.join(" and ")));
        }
        if !added.is_empty() {
            warnings.push(format!("uv pip check: {}", added.join("; ")));
        }
        let build = &self.sdk.runtime_build;
        if !warnings.is_empty() && !build.is_empty() {
            warnings.push(format!("this machine's development Runtime {build} (`--runtime <path>`) runs it regardless"));
        }
        Ok(warnings.join("; "))
    }

    fn uv(&self, args: &[&str], stop: Option<&Stop>) -> Result<(), Failure> {
        self.uv_observed(args, &[], stop, &|_| ())
    }

    /// `uv pip check`'s findings in an environment, one line each (`The package …`).
    fn check(&self, py: &str, stop: Option<&Stop>) -> Result<BTreeSet<String>, Failure> {
        match self.uv(&["pip", "check", "--no-config", "--python", py], stop) {
            Ok(()) => Ok(BTreeSet::new()),
            Err((code, detail)) => {
                let found: BTreeSet<String> = detail.lines().map(str::trim).filter(|l| l.starts_with("The package ")).map(String::from).collect();
                if found.is_empty() { Err((code, detail)) } else { Ok(found) }
            }
        }
    }

    /// `uv` with `read` told, about once a second, how many bytes the process has written so far
    /// (its downloads and the wheels it unpacks), so a long install shows it moves. Its
    /// process group is killed when `stop` is canceled, or when its CPU and bytes stop moving
    /// past its measured patience (`process::supervise`).
    fn uv_observed(&self, args: &[&str], env: &[(&str, &std::ffi::OsStr)], stop: Option<&Stop>,
        read: &(dyn Fn(u64) + Sync)) -> Result<(), Failure> {
        crate::machine::client::ensure_uv(&self.sdk.uv).map_err(|e| ("uv_unavailable", e.to_string()))?;
        let mut command = Command::new(&self.sdk.uv);
        command.args(args).env_clear().envs(env.iter().copied()).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::piped());
        platform::process::own_group(&mut command);
        // Windows spells its names in any case (`Path`), and resolves no host name for a
        // process without `SystemRoot`; TEMP and TMP are its temporary directory.
        const KEPT: [&str; 7] = ["PATH", "SSL_CERT_FILE", "SSL_CERT_DIR", "LANG", "SYSTEMROOT", "TEMP", "TMP"];
        for (name, value) in protocol::config::environment() {
            if KEPT.iter().any(|kept| kept.eq_ignore_ascii_case(name)) {
                command.env(name, value);
            }
        }
        let owner = self.sdk.owner_uv.as_ref();
        command
            .env("HOME", self.root.join("home"))
            .env("UV_PYTHON_INSTALL_DIR", owner.map_or_else(|| self.root.join("python"), |o| o.python.clone()));
        // As the Python worker: an image's seeded cache, symlinked so overlayfs never copies
        // the baked Torch payload into each environment; else the owner's, else this machine's.
        match (&self.sdk.seed_cache, owner) {
            (Some(seed), _) => command
                .env("UV_CACHE_DIR", seed)
                .env("UV_LINK_MODE", "symlink"),
            (None, Some(owner)) => command.env("UV_CACHE_DIR", &owner.cache),
            (None, None) => command.env("UV_CACHE_DIR", self.root.join("uv-cache")),
        };
        let mut child = command.spawn().map_err(|e| {
            (
                "package_installation_uv_absent",
                format!("cannot run uv: {e}"),
            )
        })?;
        let mut stderr = child.stderr.take().expect("piped uv stderr");
        let (ended, stderr) = std::thread::scope(|scope| {
            let drained = scope.spawn(move || {
                let mut kept = Vec::new();
                let _ = stderr.read_to_end(&mut kept);
                kept
            });
            let ended = crate::process::supervise(&mut child, stop, "uv", read);
            (ended, drained.join().unwrap_or_default())
        });
        let step = args[..2.min(args.len())].join(" ");
        match ended.map_err(|e| ("package_installation_uv_failed", format!("uv: {e}")))? {
            Supervised::Exited(status) if status.success() => Ok(()),
            Supervised::Exited(_) => Err((
                "package_installation_uv_failed",
                format!("uv {step}:\n{}", crate::api::install::stderr_tail(&stderr)),
            )),
            Supervised::Canceled => Err(canceled()),
            Supervised::Wedged(verdict) => Err(("no_progress", format!("uv {step}: {verdict}"))),
        }
    }

    /// The run's model grants and its precision warnings. Each slot takes the run's choice, else
    /// its declared default (also under a choice of adapters alone); a Hub one is exact as its
    /// choice names it (th-241), else in the first lane of the rule's order its release
    /// publishes (`precision`). The request's GPU
    /// count chooses the group independently of the weight lane; 0 uses the widest group every
    /// slot supports on the machine. A slot with neither is refused.
    fn model(
        &self,
        service: &Service,
        gpu: &Pool,
        actor: &str,
        installation: &Installation,
        request: &Request,
        job: &Job,
    ) -> Result<(Plan, Vec<String>), Failure> {
        let interface: Value = serde_json::from_slice(&installation.interface).map_err(|_| {
            (
                "package_prepare_interface_invalid",
                "held interface is corrupt".to_string(),
            )
        })?;
        let slots = model_slots(&interface, &request.entrypoint)
            .filter(|slots| !slots.is_empty())
            .ok_or((
                "package_prepare_interface_invalid",
                "declared model slot absent".to_string(),
            ))?;
        let degree = crate::gpu_service::group_degree(&slots, request.gpus, gpu.width());
        let gpus = gpu.gpus(degree);
        let main = main_model(&slots);
        // One slot's grant, its Hub checkpoint and its warnings.
        let pick = |slot: &Value| -> Result<(ModelGrant, Option<Named>, Vec<String>), Failure> {
            let path = slot.get("path").and_then(Value::as_str).unwrap_or_default().to_string();
            let parameter = path.rsplit('.').next().unwrap_or_default().to_string();
            job.stage(format!("choosing the model for {path}"));
            let asked = request.choices.iter().find(|c| c.parameter == path || c.parameter == parameter)
                .filter(|c| !c.repository.is_empty() || c.manifest.is_some() || !c.source.is_empty());
            let declared = default_choice(slot, &path);
            let choice = asked.or(declared.as_ref()).ok_or_else(|| choice_absent(&path, &installation.package))?;
            if !choice.source.is_empty() {
                return Ok((self.source_grant(installation, &path, choice, &request.providers, job)?, None, vec![]));
            }
            if let (Some(reference), true) = (
                choice.manifest.as_ref().filter(|m| m.digest.len() == 32),
                choice.repository.is_empty(),
            ) {
                // A model passed by value (a ModelArtifact a run here produced): held here.
                return Ok((self.held_grant(installation, &path, reference)?, None, vec![]));
            }
            if choice.repository.starts_with("local/") && choice.manifest.is_none() {
                let reference = match choice.release.is_empty() {
                    true => choice.repository.clone(),
                    false => format!("{}@{}", choice.repository, choice.release),
                };
                let (release, lane) = ((!choice.release.is_empty()).then_some(choice.release.as_str()), (!choice.lane.is_empty()).then_some(choice.lane.as_str()));
                let local = storage::local(&self.store, &choice.repository, release, lane)
                    .and_then(|local| local.ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "this machine holds no such model")))
                    .map_err(|error| ("model_resolution_failed", format!("{reference}: {error}")))?;
                let header = local.header.id();
                return Ok((ModelGrant {
                    package: installation.package.clone(), slot: path, repository: choice.repository.clone(),
                    release: local.release, lane: local.variant,
                    components: self.components(&header)?, manifest: header,
                }, None, vec![]));
            }
            let named = choose(choice, &path)?;
            let tolerances = precision::Slot::of(slot);
            // The asked lane is the main model's; every other slot takes the no-flag rule.
            let asked = if Some(slot) == main { request.bits } else { 0 };
            let ruled = named.lane.is_empty() && named.tree.is_none();
            let constraints = tolerances.constraints(&gpus, asked);
            let named = self.resolve(service, request.hub(&format!("the model for {path}"))?, named, &constraints, &request.catalog_revision)?;
            let model = format!("{}@{}", named.repository, named.release);
            let warnings = if ruled {
                let remote = origin::remote(&self.machine, request.hub("selected header")?).map_err(|e| refused("model_resolution_failed",e))?;
                self.machine.objects(&remote, std::slice::from_ref(named.header()?), priority::NOW, &Stop::default()).map_err(|e| refused("model_resolution_failed",e))?;
                let header = tensorfs::format::Header::from_cbor(&self.store.read(named.header()?).map_err(|e| ("model_resolution_failed",e.to_string()))?).map_err(|e| ("model_resolution_failed",e.to_string()))?;
                // Raw weights are data and ask no lane; else the rule's own reasons, as it selected.
                if tolerances.raw { vec![] } else { constraints.warnings(&model, &precision::Profile::of(&header)) }
            } else { vec![] };
            let grant = ModelGrant {
                package: installation.package.clone(),
                slot: path,
                repository: named.repository.clone(),
                release: named.release.clone(),
                lane: named.lane.clone(),
                manifest: named.header()?.id(),
                components: vec![],
            };
            Ok((grant, (!named.repository.starts_with("local/")).then_some(named), warnings))
        };
        // Each checkpoint lands as its structure: what of it comes, and when, its calls direct.
        // Only a Hub names a checkpoint here (`pick` refused it without the run's access).
        let land = |picked: &[(ModelGrant, Option<Named>, Vec<String>)]| {
            let named = distinct(picked.iter().filter_map(|(_, named, _)| named.as_ref()));
            match named.is_empty() {
                true => Ok(()),
                false => self.structure(request.hub("a Hub model")?, &named, job),
            }
        };
        let picked: Vec<_> = slots.iter().map(pick).collect::<Result<_, _>>()?;
        land(&picked)?;
        let (mut grants, mut warnings) = (vec![], vec![]);
        for (grant, _, said) in picked {
            grants.push(grant);
            warnings.extend(said);
        }
        for grant in grants.iter_mut().filter(|g| g.components.is_empty() && !g.repository.is_empty()) {
            grant.components = self.components(&grant.manifest)?;
        }
        for grant in &mut grants {
            let parameter = grant
                .slot
                .rsplit('.')
                .next()
                .unwrap_or_default()
                .to_string();
            if let Some(choice) = request
                .choices
                .iter()
                .find(|c| c.parameter == grant.slot || c.parameter == parameter)
            {
                // An adapter is selected for the run's GPUs by the no-flag rule: `--lane` is the main model's.
                let constraints = precision::Constraints { gpus: gpus.capabilities.clone(), ..Default::default() };
                let resolve = |source: &hub::Source, named| self.resolve(service, source, named, &constraints, &request.catalog_revision);
                apply_adapters(self, request.source.as_ref(), &resolve, choice, grant, &request.providers, job)?;
            }
        }
        job.stage(format!("preparing {}", request.package));
        let plan = gpu
            .prepare_root(
                installation,
                &request.entrypoint,
                &request.choices,
                &grants,
                request.gpus,
            )
            .map_err(|e| ("model_preparation_failed", e.to_string()))?;
        service
            .engine
            .bind_preparation(Preparation {
                actor: actor.into(),
                id: plan.id.clone(),
                installation: installation.alias.clone(),
                document: serde_json::to_vec(&plan).map_err(|e| io_failure(io::Error::other(e)))?,
            })
            .map_err(io_failure)?;
        let key = Self::resolution_key(&installation.alias, request, &machine_row(gpu));
        let said = Value::from(warnings.clone()).to_string();
        service
            .engine
            .with_journal(|j| {
                j.bind_resolution(actor, &key, &installation.package, &plan.id)?;
                j.bind_named(&warnings_key(&key), &request.catalog_revision, &said)
            })
            .map_err(io_failure)?;
        Ok((plan, warnings))
    }
}

/// A function's main model, the one a run's `--lane` filters (owner, 10-22): its first slot
/// constructed up front, neither raw weights nor on demand. Its LoRAs and auxiliary models take
/// the no-flag rule.
fn main_model(slots: &[Value]) -> Option<&Value> {
    slots.iter().find(|slot| crate::gpu_service::constructed(slot) && slot["on_demand"] != true)
}

/// Each checkpoint once.
fn distinct<'a>(named: impl Iterator<Item = &'a Named>) -> Vec<Named> {
    let mut checkpoints = Vec::<Named>::new();
    for named in named {
        if !checkpoints.iter().any(|c| c.header == named.header) {
            checkpoints.push(named.clone());
        }
    }
    checkpoints
}

/// The machine's GPUs as a held resolution is keyed by: how many, and what they run.
fn machine_row(gpu: &Pool) -> String {
    format!("{}x{:?}", gpu.width(), gpu.gpus(gpu.width() as u32))
}

/// Where a held resolution's precision warnings are journaled.
fn warnings_key(resolution: &str) -> String {
    json!(["warnings", resolution]).to_string()
}

/// A Hub model slot's checkpoint as its choice names it (th-245): the owner's by hash, else
/// `repository` at `release` in the choice's lane; no lane is the lane rule's (`resolve`).
pub(crate) fn choose(choice: &domain::ModelChoice, path: &str) -> Result<Named, Failure> {
    if choice.repository.is_empty() {
        return Err(("model_choice_absent", format!("{path}'s choice names no model repository")));
    }
    let tree = choice.manifest.as_ref().map(|m| ObjectRef { sha256: sha256::hex(&m.digest), length: m.length });
    Ok(Named { repository: choice.repository.clone(), release: choice.release.clone(), lane: choice.lane.clone(), tree, header: None })
}

/// A provider source made into a local model.
#[derive(Clone, Debug)]
pub struct Made {
    /// `local/source-<hash>`, named by the source's pinned spelling and converter.
    pub repository: String,
    pub header: ObjectRef,
    /// The pinned source the provider resolved.
    pub resolved: String,
    pub profiles: Vec<String>,
}

/// `source` imported into a local model (normalized at ingest, design §6), or the one already
/// made from its pinned spelling: a moving revision is read again, a pinned one is reused.
fn make_source(
    store: &Store,
    source: &str,
    profiles: &[String],
    providers: &Providers,
    cancellation: Option<Stop>,
    progress: &(dyn Fn(&str, u64, u64) + Sync),
) -> Result<Made, Failure> {
    let bearer = |token: &str| if token.is_empty() { String::new() } else { format!("bearer {token}") };
    let access = tensorfs::import::Access {
        huggingface_token: bearer(&providers.huggingface),
        civitai_token: bearer(&providers.civitai),
        ..Default::default()
    };
    let [converter] = match profiles {
        [] => [None],
        [one] => [Some(one.as_str())],
        _ => return Err(("model_source_failed", "a source is converted by one table".into())),
    };
    let listed = tensorfs::import::remote::parse(source)
        .and_then(|spec| tensorfs::import::remote::list(&spec, &access))
        .map_err(|e| refused("model_source_failed", e))?;
    let identity = serde_json_canonicalizer::to_vec(&json!(["cozy.machine-source-model/2", listed.pinned, converter]))
        .map_err(|e| io_failure(io::Error::other(e)))?;
    let name = format!("source-{}", &sha256::hex_digest(&identity)[..40]);
    let made = |header| Made {
        repository: format!("local/{name}"),
        header,
        resolved: listed.pinned.clone(),
        profiles: profiles.to_vec(),
    };
    let held = crate::storage::local(store, &format!("local/{name}"), None, None).map_err(io_failure)?;
    if let Some(local) = held.filter(|local| crate::storage::present(store, &local.header)) {
        crate::storage::transport(store, "import", source, &format!("header={} reused", local.header.id()));
        return Ok(made(local.header));
    }
    let stop = || cancellation.as_ref().is_some_and(Stop::is_cancelled);
    // What the import downloaded, for its transport line.
    let downloaded = std::sync::atomic::AtomicU64::new(0);
    let counted = |stage: &str, done: u64, total: u64| {
        if stage == "downloading" {
            downloaded.fetch_max(total, std::sync::atomic::Ordering::Relaxed);
        }
        progress(stage, done, total)
    };
    let request = tensorfs::import::Request {
        sources: &[source.to_string()],
        converter,
        source_variant: None,
        access: &access,
        tmp: &store.root().join("tmp"),
        progress: &counted,
        stop: Some(&stop),
    };
    let started = Instant::now();
    let (header, files) = store.import(&request, Some(&name)).map_err(|e| {
        crate::storage::transport(store, "import-failed", source, &e.to_string());
        match stop() {
            true => canceled(),
            false => refused("model_source_failed", e),
        }
    })?;
    let closure = store.closure(&header).unwrap_or_default();
    let detail = format!(
        "header={} objects={} bytes={} downloaded={} seconds={:.1}",
        header.id(),
        closure.len(),
        closure.iter().map(|o| o.length).sum::<u64>(),
        downloaded.load(std::sync::atomic::Ordering::Relaxed),
        started.elapsed().as_secs_f64()
    );
    crate::storage::transport(store, "import", source, &detail);
    crate::storage::set_local(store, &name, &header, &files).map_err(io_failure)?;
    Ok(made(header))
}

/// A TensorFS refusal as a run's reason; a download the disk cannot fit is its own.
fn refused(code: &'static str, refusal: tensorfs::err::Refusal) -> Failure {
    match refusal.code {
        tensorfs::err::Code::CAPACITY_EXHAUSTED => ("machine_disk_full", refusal.to_string()),
        // The owner's bearer was refused (expired, revoked); without one a private object is absent.
        tensorfs::err::Code::CREDENTIAL_REQUIRED => ("hub_credential_refused", refusal.detail),
        _ => (code, refusal.to_string()),
    }
}

/// A slot's caller adapters applied to its resolved base: each adapter downloaded, then one
/// adapter view composed (`adapter::compose`); the grant then names the view.
fn apply_adapters(
    publisher: &Publisher,
    source: Option<&hub::Source>,
    resolve: &dyn Fn(&hub::Source, Named) -> Result<Named, Failure>,
    choice: &domain::ModelChoice,
    grant: &mut ModelGrant,
    providers: &Providers,
    job: &Job,
) -> Result<(), Failure> {
    if choice.adapters.is_empty() {
        return Ok(());
    }
    let store = &publisher.store;
    job.stage(format!("preparing model adapters for {}", grant.slot));
    let object = |header: &str| {
        storage::held(store, header).map_err(|e| ("model_download_failed", format!("checkpoint {header} is not held: {e}")))
    };
    let mut selections = vec![];
    for adapter in &choice.adapters {
        let header = if adapter.source.is_empty() {
            // By name, resolved at the Hub, or the owner's by hash (th-245).
            if adapter.model.is_empty() {
                return Err(("adapter_absent", format!("an adapter of {} names no model", grant.slot)));
            }
            let tree = (!adapter.manifest.is_empty())
                .then(|| ObjectRef { sha256: adapter.manifest.trim_start_matches("sha256:").into(), length: 0 });
            let source = source.ok_or_else(|| ("hub_access_absent",
                format!("an adapter of {} names a Hub model, and this run carries no Hub access", grant.slot)))?;
            let named = resolve(source, Named {
                repository: adapter.model.clone(),
                release: adapter.release.clone(),
                lane: adapter.lane.clone(),
                tree,
                header: None,
            })?;
            // An adapter is small and its view composes from all of it: it comes whole.
            publisher.ensure(source, &named, job.stop.as_ref(), &|d, t| job.bytes(d, t))?;
            named.header()?.id()
        } else {
            // A provider-source LoRA (civitai://, hf://) is made here, normalized at ingest.
            let made = publisher.make(&adapter.source, &adapter.profiles, providers, job.stop.as_ref(), &|stage, moved, total| {
                job.set(format!("{stage} for an adapter of {}", grant.slot), moved, total)
            })?;
            made.header.id()
        };
        let strength = if adapter.scale.is_empty() {
            1.0
        } else {
            adapter.scale.parse::<f64>().map_err(|_| {
                (
                    "model_override_invalid",
                    format!("adapter scale {:?} is not a decimal", adapter.scale),
                )
            })?
        };
        selections.push(crate::adapter::Selection {
            checkpoint: object(&header)?,
            component: adapter.component.clone(),
            source_component: adapter.source_component.clone(),
            strength,
        });
    }
    // The view re-packs the base's small parts: its chunks come first (its load reads them anyway).
    let base = object(&grant.manifest)?;
    if let (Some(pulled), Some(source)) = (publisher.pulled(&base).map_err(io_failure)?, source) {
        publisher.ensure(source, &pulled, job.stop.as_ref(), &|d, t| job.bytes(d, t))?;
    }
    // The view is the store's to keep from here: the call that reads it holds it open.
    let composed = crate::adapter::compose(store, &base, &selections)
        .map_err(|e| ("model_adapter_refused", e.to_string()))?;
    grant.repository = composed.name;
    grant.manifest = composed.checkpoint.id().id();
    Ok(())
}

/// A catalog read the Hub answered 404 to is the thing named absent there; any other failure
/// is the read's, naming what it read and the Hub's whole answer (run 14446 said only "internal:
/// unexpected internal failure").
/// The Hub's own 404 names an absence; one without its envelope (`http_404`: a tunnel or proxy
/// in front of it) names nothing (tfs-101).
fn absent(refusal: hub_client::Error, code: &'static str, read: &str, named: impl FnOnce() -> String) -> Failure {
    let failed = |why: String| ("catalog_read_failed", format!("reading {read}: {why}"));
    match refusal {
        hub_client::Error::Hub { status: 404, code: hub, .. } if !hub.starts_with("http_") => (code, named()),
        hub_client::Error::Hub { status, code, message, remedy } => failed(match remedy.is_empty() {
            true => format!("the Hub answered {status} {code}: {message}"),
            false => format!("the Hub answered {status} {code}: {message} ({remedy})"),
        }),
        other => failed(other.to_string()),
    }
}

/// What a person sees while a release's environment is built here: an install, named the first
/// run of the release on this machine when no environment of it is here yet. The Runtime and
/// TensorFS are named only when one moves from the version an earlier environment of the
/// release holds (an update of this machine's pair): linking the pair in is part of the install.
fn install_stage(package: &str, release: &str, before: Option<&BTreeMap<String, String>>, after: &BTreeMap<String, String>) -> String {
    let Some(before) = before else {
        return format!("installing {package} {release} on this machine (first run of this release)");
    };
    let moved: Vec<String> = after.iter().filter_map(|(name, version)| {
        let was = before.get(name).filter(|was| *was != version)?;
        let label = if name == "cozy-runtime" { "Runtime" } else { "TensorFS" };
        Some(format!("{label} {was} → {version}"))
    }).collect();
    match moved.is_empty() {
        true => format!("installing {package} {release} on this machine"),
        false => format!("updating {package} {release} on this machine: {}", moved.join(", ")),
    }
}

/// The journal name of an unnamed release of the request's package at its Hub.
fn newest_key(request: &Request) -> String {
    json!(["newest", request.origin(), request.package]).to_string()
}

/// A checkpoint named by its hash, with what a hash alone does not say: its length (the Hub's) and
/// its format; a tree (a commit) resolves to its lane's header, a header is the checkpoint itself.
/// Read once under the run's access (its request's bearer for the owner's) and kept, since a hash
/// never names other bytes.
fn measured(machine: &Machine, service: &Service, source: &hub::Source, named: Named, constraints: &precision::Constraints) -> Result<Named, Failure> {
    if named.header.is_some() {
        return Ok(named);
    }
    let tree = named.tree()?.clone();
    let key = json!(["profiles-v2", hub::origin_key(&source.origin).unwrap_or_default(), named.repository, tree.id(), named.lane, constraints]).to_string();
    let held = service.engine.with_journal(|j| j.named(&key, "")).map_err(io_failure)?;
    if let Some((tree, lane, header)) = held.and_then(|held| serde_json::from_str(&held).ok()) {
        return Ok(Named { tree, lane, header: Some(header), ..named });
    }
    let failed = |e| refused("model_resolution_failed", e);
    let object = match tree.length {
        0 => origin::object(machine, source, &tree.sha256).map_err(failed)?,
        _ => tree,
    };
    let remote = origin::remote(machine, source).map_err(failed)?;
    machine.objects(&remote, std::slice::from_ref(&object), priority::NOW, &Stop::default()).map_err(failed)?;
    let commit = storage::tree_or_header(machine.store(), &object).map_err(|e| ("model_resolution_failed", e.to_string()))?.is_some();
    let named = match commit {
        true => Named { tree: Some(object), ..named }.landed(machine, source, constraints)?,
        false => Named { tree: None, header: Some(object), ..named },
    };
    let answer = json!([named.tree, named.lane, named.header]).to_string();
    service.engine.with_journal(|j| j.bind_named(&key, "", &answer)).map_err(io_failure)?;
    Ok(named)
}

/// What installing a release reads of it, from its install plan: its interface and Python, its
/// exact pyproject and uv.lock, and the wheels its Hub hosts for it (its project wheel and carried
/// dependencies). Kept beside each environment built from it (`CARD`).
#[derive(serde::Deserialize, serde::Serialize)]
struct Card {
    interface: Value,
    python: String,
    pyproject: String,
    lock: String,
    wheels: Vec<HubFile>,
}
const CARD: &str = "release.json";

/// `release`'s install plan, read at its Hub.
fn hub_card(catalog: &Catalog, package: &str, release: &str, python: &str) -> Result<Card, Failure> {
    let (org, name) = package.split_once('/').unwrap_or_default();
    let read = format!("the install plan of {package}@{release} at {}", catalog.origin());
    let plan = catalog.install_plan(org, name, release, None).map_err(|e| absent(e, "release_absent", &read, || {
        format!("{package} has no release {release} at {}", catalog.origin())
    }))?;
    if plan.release != release {
        return Err(("catalog_read_failed", format!("{package}@{release} named another release")));
    }
    let text = |what: &str, document: &hub_client::types::ExactDocument| {
        base64::engine::general_purpose::STANDARD.decode(&document.canonical_bytes).ok().and_then(|bytes| String::from_utf8(bytes).ok())
            .ok_or(("package_prepare_plan_invalid", format!("the release's {what} is not base64 UTF-8")))
    };
    let interface = serde_json::from_str::<Value>(&text("interface", &plan.package_interface)?).ok().filter(Value::is_object).ok_or((
        "package_prepare_interface_missing",
        "the release carries no package interface".to_string(),
    ))?;
    let wheels = plan.downloads.iter().map(|file| {
        HubFile::new(&file.url, &file.digest, &file.path, file.length.try_into().unwrap_or(0))
            .ok_or(("package_prepare_plan_invalid", format!("the release's plan names an unusable file {}", file.path)))
    });
    Ok(Card {
        interface,
        python: plan.python_version.clone().filter(|v| !v.is_empty()).unwrap_or_else(|| python.to_string()),
        pyproject: text("pyproject.toml", &plan.pyproject)?,
        lock: text("uv.lock", &plan.uv_lock)?,
        wheels: wheels.collect::<Result<_, _>>()?,
    })
}

/// `card` kept beside the environment at `generation` for the release's later environments.
/// A cache, written by rename and never synced: one lost is read from the Hub again.
fn keep_card(generation: &Path, card: &Card) {
    let staged = generation.join(format!(".{CARD}.{}", uuid::Uuid::new_v4().simple()));
    let kept = serde_json::to_vec(card).map_err(io::Error::other)
        .and_then(|bytes| fs::write(&staged, bytes))
        .and_then(|()| fs::rename(&staged, generation.join(CARD)));
    if let Err(error) = kept {
        tracing::warn!("release card {}: {error}", generation.display());
    }
}

/// The newest release of `org/name` at the Hub (a run naming none), by the one version ordering
/// (`newest_by`): the newest final release, else the newest pre-release; yanked ones never.
fn newest_release(catalog: &Catalog, org: &str, name: &str) -> Result<String, Failure> {
    let card = catalog
        .package(org, name)
        .map_err(|e| absent(e, "package_absent", &format!("the releases of {org}/{name} at {}", catalog.origin()),
            || format!("{org}/{name} is not published at {}", catalog.origin())))?;
    let open = card.releases.iter().filter(|row| row.yanked != Some(true) && row.yanked_at.as_deref().is_none_or(str::is_empty));
    tensorfs::format::newest_by(open, |row| row.release.as_str())
        .map(|row| row.release.clone())
        .ok_or(("release_absent", format!("{org}/{name} has no release that is not yanked")))
}

/// A wheel the run's Hub hosts, downloaded by its digest (`fetch_hub`) and installed from disk.
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
struct HubFile {
    url: String,
    sha256: String,
    name: String,
    /// 0 when its source does not say: then asked of the file's host.
    length: u64,
}

impl HubFile {
    fn new(url: &str, sha256: &str, name: &str, length: u64) -> Option<Self> {
        let sha256 = sha256.strip_prefix("sha256:").unwrap_or(sha256);
        let plain = |c: char| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-' | '+');
        (name.ends_with(".whl")
            && !name.starts_with('.')
            && name.chars().all(plain)
            && sha256.len() == 64
            && sha256.chars().all(|c| c.is_ascii_hexdigit()))
        .then(|| Self { url: url.into(), sha256: sha256.to_ascii_lowercase(), name: name.into(), length })
    }

    /// The distribution the wheel installs, from its file name (`<name>-<version>-...whl`).
    fn distribution(&self) -> String {
        normalized(self.name.split('-').next().unwrap_or_default())
    }
}

/// One file named by its digest, into this machine's store: a download that stops moving
/// fails on its measured stillness, `ledger`'s stop ends it, and what landed stays for the
/// next ask. Answers the blob's path.
fn io_failure(error: io::Error) -> Failure {
    ("release_root_preparation_failed", error.to_string())
}

/// A slot's declared default as a choice (its `default`: `[org/]model@release`, or a local
/// model `local/<name>`, its newest): what a run that names no model for the slot gets, in the
/// lane rule's lane. A model with no org is first-party, the root's (xs-071).
pub(crate) fn default_choice(slot: &Value, path: &str) -> Option<domain::ModelChoice> {
    let default = slot.get("default")?.as_str()?;
    let (model, release) = match default.split_once('@') {
        Some(named) => named,
        None => (default.starts_with("local/").then_some(default)?, ""),
    };
    let repository = match model.contains('/') {
        true => model.into(),
        false => format!("{}/{model}", protocol::names::ROOT),
    };
    Some(domain::ModelChoice { parameter: path.into(), repository, release: release.into(), ..Default::default() })
}

fn local_aliases(installation: &Installation, request: &Request) -> io::Result<bool> {
    let interface: Value = serde_json::from_slice(&installation.interface).map_err(io::Error::other)?;
    Ok(model_slots(&interface, &request.entrypoint).unwrap_or_default().iter().any(|slot| {
        let path = slot["path"].as_str().unwrap_or_default();
        let parameter = path.rsplit('.').next().unwrap_or_default();
        let asked = request.choices.iter().find(|c| c.parameter == path || c.parameter == parameter)
            .filter(|c| !c.repository.is_empty() || c.manifest.is_some() || !c.source.is_empty());
        let declared = default_choice(slot, path);
        asked.or(declared.as_ref()).is_some_and(|choice| choice.source.is_empty()
            && choice.manifest.is_none() && choice.repository.starts_with("local/"))
    }))
}

/// A slot with neither a choice nor a declared default.
pub(crate) fn choice_absent(path: &str, package: &str) -> Failure {
    ("model_choice_absent", format!("the run names no model for {path}, and {package} declares no default for it"))
}

fn model_slots(interface: &Value, entrypoint: &str) -> Option<Vec<Value>> {
    let entry = protocol::interface::entrypoint(interface, entrypoint)?;
    entry["models"].is_array().then(|| protocol::interface::models(entry).to_vec())
}

fn declares_models(installation: &Installation, entrypoint: &str) -> bool {
    serde_json::from_slice(&installation.interface)
        .ok()
        .and_then(|i: Value| model_slots(&i, entrypoint))
        .is_some_and(|m| !m.is_empty())
}

/// What a release's uv.lock leaves to this machine: its git-sourced packages (installed from the
/// release's carried wheels), the rows the run's Hub hosts (each one wheel, moved by TensorFS),
/// and every package it names.
#[derive(Debug, Default)]
struct Locked {
    distribution: String,
    git: Vec<String>,
    hub: Vec<HubFile>,
    /// Each package it names, at its version.
    packages: BTreeMap<String, String>,
}

/// The Runtime SDK pair: the machine's own, where it has one.
const SDK: [&str; 2] = ["cozy-runtime", "tensorfs"];

/// The environment's root digest and the other Apps it holds, described inside it by the
/// machine's client without importing them (`runtime_describe`). A failed description fails
/// this preparation; it never publishes an environment with silently missing callees.
pub fn describe_environment(
    python: &str,
    root: &str,
    packages: &BTreeMap<String, String>,
) -> Result<(String, Vec<crate::catalog::Callee>), Failure> {
    #[derive(serde::Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum Reply {
        DescribedEnvironment { source_digest: String, callees: Vec<crate::catalog::Callee> },
        DescribeFailed { code: String, detail: String },
    }
    // -S keeps authored .pth startup code out; only the venv's own site-packages are added.
    const BOOTSTRAP: &str = "import pathlib,runpy,sys;\
        root=pathlib.Path(sys.executable).absolute().parent.parent;\
        sys.path[:0]=[str(p) for p in (*(root/'lib').glob('python*/site-packages'),root/'Lib'/'site-packages') if p.is_dir()];\
        runpy.run_module('tensord_client.runtime_describe',run_name='__main__')";
    let described = Command::new(python)
        .args(["-I", "-S", "-c", BOOTSTRAP])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .and_then(|mut child| {
            let request = json!({"kind": "describe_environment", "root": root, "packages": packages}).to_string();
            child.stdin.take().expect("piped").write_all(request.as_bytes())?;
            child.wait_with_output()
        });
    let output = described.map_err(io_failure)?;
    let reply = serde_json::from_slice::<Reply>(&output.stdout).map_err(|error| {
        ("package_prepare_description_failed", format!("environment description failed: {error}"))
    })?;
    match reply {
        Reply::DescribedEnvironment { source_digest, callees } if output.status.success() => Ok((source_digest, callees)),
        Reply::DescribeFailed { code, detail } => Err(("package_prepare_description_failed", format!("{code}: {detail}"))),
        _ => Err(("package_prepare_description_failed", "environment description did not complete".into())),
    }
}

/// The Runtime and TensorFS an environment holds, by their installed distribution names.
fn installed_sdk(env: &Path) -> Vec<Value> {
    let mut found: Vec<Value> = platform::paths::venv_site_packages(env)
        .into_iter()
        .flat_map(|site| fs::read_dir(site).into_iter().flatten().flatten())
        .filter_map(|entry| {
            let name = entry.file_name().into_string().ok()?;
            let (distribution, version) = name.strip_suffix(".dist-info")?.split_once('-')?;
            let distribution = normalized(distribution);
            SDK.contains(&distribution.as_str()).then(|| json!({"name": distribution, "version": version}))
        })
        .collect();
    found.sort_by_key(|d| d["name"].as_str().unwrap_or_default().to_string());
    found
}

/// The release's pyproject and uv.lock read for what `generation` hands uv and what it moves
/// itself. A Hub row with more than one wheel (one per platform) is uv's to choose.
fn read_lock(pyproject: &str, lock: &str, source: &hub::Source) -> Result<Locked, Failure> {
    #[derive(serde::Deserialize)]
    struct Lock {
        #[serde(default)]
        package: Vec<Row>,
        #[serde(default, rename = "supported-markers")]
        supported: Vec<String>,
    }
    #[derive(serde::Deserialize)]
    struct Row {
        name: String,
        #[serde(default)]
        version: String,
        #[serde(default)]
        source: BTreeMap<String, toml::Value>,
        #[serde(default)]
        wheels: Vec<Wheel>,
    }
    #[derive(serde::Deserialize)]
    struct Wheel {
        url: Option<String>,
        hash: Option<String>,
        #[serde(default)]
        size: u64,
    }
    #[derive(serde::Deserialize)]
    struct Pyproject {
        project: Project,
    }
    #[derive(serde::Deserialize)]
    struct Project {
        name: String,
    }
    let invalid = |what: &str, e: toml::de::Error| ("package_prepare_lock_invalid", format!("the release's {what} does not parse: {e}"));
    let project = toml::from_str::<Pyproject>(pyproject).map_err(|e| invalid("pyproject.toml", e))?.project.name;
    let lock = toml::from_str::<Lock>(lock).map_err(|e| invalid("uv.lock", e))?;
    // Refused before anything is fetched: a Mac or Windows computer told at once where it runs.
    if let Some(there) = elsewhere(&lock.supported, uv_platform()) {
        let here = format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH);
        return Err(("release_platform_unsupported",
            format!("this release runs on {there}, not this machine ({here}): `--on new` rents a machine for it")));
    }
    let rows = lock.package;
    let mut locked = Locked { distribution: project, ..Default::default() };
    for row in rows {
        let name = normalized(&row.name);
        if row.source.contains_key("git") {
            locked.git.push(name.clone());
        } else if let [Wheel { url: Some(url), hash: Some(hash), size }] = row.wheels.as_slice() {
            let file = url.rsplit('/').next().unwrap_or_default();
            if let Some(file) = HubFile::new(url, hash, file, *size).filter(|_| hub::publishes(source, url)) {
                locked.hub.push(file);
            }
        }
        locked.packages.insert(name, row.version);
    }
    Ok(locked)
}

/// This machine's `sys_platform`, as a lock's markers spell it.
fn uv_platform() -> &'static str {
    match std::env::consts::OS {
        "macos" => "darwin",
        "windows" => "win32",
        os => os,
    }
}

/// Where a lock's `supported-markers` say it runs, when none of them is `here`'s OS: `Linux
/// x86_64, Linux aarch64`. A lock without them, or one naming no OS, runs anywhere uv resolves it.
fn elsewhere(markers: &[String], here: &str) -> Option<String> {
    let value = |marker: &str, key: &str| Some(marker.split(&format!("{key} == '")).nth(1)?.split('\'').next()?.to_owned());
    let platforms: Vec<Option<String>> = markers.iter().map(|m| value(m, "sys_platform")).collect();
    if platforms.is_empty() || platforms.iter().any(|os| os.as_deref().is_none_or(|os| os == here)) {
        return None;
    }
    let named = |marker: &String| {
        let os = match value(marker, "sys_platform").as_deref() {
            Some("linux") => "Linux".to_owned(),
            Some("darwin") => "macOS".to_owned(),
            Some("win32") => "Windows".to_owned(),
            os => os.unwrap_or_default().to_owned(),
        };
        value(marker, "platform_machine").map_or(os.clone(), |arch| format!("{os} {arch}"))
    };
    Some(markers.iter().map(named).collect::<Vec<_>>().join(", "))
}

/// A lock this machine's platform is outside of, in uv's words: its author's to fix, by adding
/// the platform to `[tool.uv] environments` and locking again.
fn platform(failure: Failure, package: &str, release: &str) -> Failure {
    const OUTSIDE: [&str; 3] = [
        "not compatible with the lockfile's supported environments",
        "wheel for the current platform",
        "has no wheels with a matching platform tag",
    ];
    match failure {
        (code, detail) if code == "package_installation_uv_failed" && OUTSIDE.iter().any(|s| detail.contains(s)) => (
            "release_platform_unsupported",
            format!("{package}@{release} is not locked for this machine ({}/{}): {detail}", std::env::consts::OS, std::env::consts::ARCH),
        ),
        failure => failure,
    }
}
