//! The pool runs every call on a retained executor of its plan: deviceless (degree 0), on one
//! GPU, or on a group of K (rank 0 forms followers). It supplies weight sources and host
//! layouts and each GPU's ledger; Runtime constructs models and varena in each executor decides
//! its bytes (design memory-varena §1). TensorD admits starts, reaps and ends stalled processes.
//! Engine owns acceptance and run custody.
use cozy_platform::process::{Birth, Exact};
use crate::{
    catalog::HeldGeneration,
    device_executor::{
        self, channel_lost, Answer, Binding, Cancellation, Construction, DeviceCommand,
        DeviceExecutor, ExecutorConfig, Forked, Frame, Kind, ModelLoad, Services,
    },
    execution::Engine,
    gpu_allocation::Claims,
    deposits::Deposits,
    journal::{
        AssetBinding, Execution, Failure, Outcome, Preparation, State,
    },
    launch_identity::Seal,
    memory::{GpuLedger, MemoryConfig},
    model_sources::{ModelSources, SelectedManifest},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::Ordering,
        Arc, Condvar, Mutex, Weak,
    },
    time::Instant,
};
use crate::storage::Store;

fn threads() -> u32 {
    crate::launch_identity::DEFAULT_THREADS
}
fn yes() -> bool {
    true
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelGrant {
    pub package: String,
    pub slot: String,
    pub repository: String,
    pub release: String,
    pub lane: String,
    pub manifest: String,
    pub components: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct PublishedPackage {
    pub package: String,
    pub distribution: String,
    pub release: String,
    pub generation: String,
}

/// How the machine shares its host and GPUs.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct HostOptions {
    /// `shared` (the default): a guest that gives memory back whenever anything on the host
    /// stalls on it; `dedicated` (a rented pod's image or grant): only when this cgroup does.
    #[serde(default)]
    pub mode: cozy_platform::contain::host_pressure::HostMode,
}

/// Root-sealed configuration, never peer-controlled paths or environment logic switches.
#[derive(Clone, Debug, Deserialize)]
pub struct GpuConfig {
    #[serde(default)]
    pub identity: Option<crate::launch_identity::LaunchIdentity>,
    /// The GPU envelope, `"0"` or `"0,1,..."`: a degree-K plan occupies K contiguous devices.
    pub devices: String,
    /// The receipt's inventory (`nvidia-smi`): each GPU's compute capability and memory, matched
    /// to envelope entries by index or UUID, for the lane rule (`precision`).
    #[serde(default)]
    pub inventory: Vec<crate::machine::receipt::Gpu>,
    /// Load's device limit where NVML cannot read the device total.
    pub authorized_device_limit_bytes: Option<u64>,
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Explicitly configured locations; the executor seal is imposed on top of them.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// The machine's build toolchain (`cozy machine toolchain install`): once it holds nvcc,
    /// executors started after find it through CUDA_HOME and PATH.
    #[serde(default)]
    pub toolchain: Option<PathBuf>,
    /// `OMP_NUM_THREADS` the seal imposes. torch's allocator is the Runtime's, set in code.
    #[serde(default = "threads")]
    pub threads: u32,
    /// Explicit authority for already verified cached catalog bytes, for runs without a
    /// Hub. Published runs resolve under the owner's Hub access instead; a cache hit alone
    /// grants no private model.
    #[serde(default)]
    pub models: Vec<ModelGrant>,
    #[serde(default)]
    pub packages: Vec<PublishedPackage>,
    #[serde(default)]
    pub host: HostOptions,
    /// Keep one import-only executor per installed GPU generation and fork executors from
    /// it (`fork/1`), so a new executor's imports are already done. It costs that process's
    /// host memory (no device memory); false spawns every executor, which imports again.
    #[serde(default = "yes")]
    pub prespawn: bool,
    /// The image's Runtime interpreter and the executors' Runtime wheel: the machine compiles
    /// its kernels with them at start, before any package is installed (`image_kernel_boot`).
    #[serde(default)]
    pub image_kernels: Option<ImageKernels>,
}
#[derive(Clone, Debug, Deserialize)]
pub struct ImageKernels {
    pub python: PathBuf,
    pub runtime_wheel: PathBuf,
}
impl GpuConfig {
    pub fn load(path: &Path) -> io::Result<Self> {
        let config: Self = serde_json::from_reader(File::open(path)?).map_err(io::Error::other)?;
        let envelope = config.envelope();
        let unique: std::collections::BTreeSet<_> = envelope.iter().collect();
        if unique.len() != envelope.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "GPU scope names one device or a list of distinct devices",
            ));
        }
        if config
            .environment
            .keys()
            .any(|key| key.contains("TOKEN") || key.contains("SECRET") || key.contains("PASSWORD"))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "device executor environment cannot carry credentials",
            ));
        }
        Ok(config)
    }
    /// A machine without GPUs: deviceless executors only, with a minimal environment whose
    /// home is `home`.
    pub fn deviceless(home: &Path) -> io::Result<Self> {
        fs::create_dir_all(home)?;
        serde_json::from_value(serde_json::json!({"devices": "",
            "environment": {"PATH": "/usr/local/bin:/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": home}}))
            .map_err(io::Error::other)
    }
    /// The configured devices in order (`CUDA_VISIBLE_DEVICES` entries); none on a CPU machine.
    pub fn envelope(&self) -> Vec<String> {
        self.devices
            .split(',')
            .map(|d| d.trim().to_string())
            .filter(|d| !d.is_empty())
            .collect()
    }
}

/// One declared model slot of a plan and its bound checkpoint.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PlanSlot {
    pub binding: Binding,
    /// The Hub lane its checkpoint was resolved in (`run show`); empty for one held here.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub lane: String,
    /// Native selected encoded source bytes; not a GPU-fit or full-memory charge.
    pub selected_encoded_bytes: u64,
    /// Built and fetched when the function first uses it (`on_demand`): nothing of it is
    /// fetched or warmed before then.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub on_demand: bool,
}

/// One `Weights[M]` argument: the checkpoint a call reads raw (never constructed), and the
/// Hub checkpoint it was resolved as (none for one this machine holds).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlanSource {
    pub parameter: String,
    pub manifest: String,
    pub length: u64,
    #[serde(default)]
    pub repository: String,
    #[serde(default)]
    pub release: String,
    #[serde(default)]
    pub lane: String,
}

/// A prepared call: its construction (the package's generation, App, entrypoint, constructed
/// models and degree) and the checkpoints it reads raw. Not who submitted: every submitter of
/// the same construction is served by its executors, one per GPU lane (several deviceless), a
/// request at a time each. What an actor may run is its own journal rows.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Plan {
    /// The preparation: the construction and its sources.
    pub id: String,
    /// Its executors' key: the construction alone, so calls reading other sources share them.
    #[serde(default)]
    pub construction: String,
    pub installation: String,
    pub generation: String,
    pub entrypoint: String,
    /// Every constructed model slot the entrypoint declares, in declaration order.
    pub slots: Vec<PlanSlot>,
    #[serde(default)]
    pub sources: Vec<PlanSource>,
    /// GPUs it runs on (a group's are the first `degree` envelope devices); 0: deviceless.
    pub degree: u32,
}
impl Plan {
    /// A call that constructs nothing: deviceless, or one GPU (`degree`).
    pub fn weightless(generation: &str, application: &str, entrypoint: &str, degree: u32) -> Self {
        let semantic = serde_json::json!({"generation": generation, "application": application,
            "entrypoint": entrypoint, "slots": [], "degree": degree});
        let id = format!("plan-{}", tensorfs_core::sha256::hex_digest(&crate::boundary_json::exact(&semantic)));
        Self {
            construction: id.clone(),
            id,
            installation: String::new(),
            generation: generation.into(),
            entrypoint: entrypoint.into(),
            slots: vec![],
            sources: vec![],
            degree,
        }
    }
    /// The plan a journaled preparation binds.
    pub fn of(preparation: &Preparation) -> io::Result<Self> {
        let mut plan: Plan =
            serde_json::from_slice(&preparation.document).map_err(io::Error::other)?;
        if plan.id != preparation.id || plan.installation != preparation.installation {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "preparation differs from owned journal binding",
            ));
        }
        if plan.construction.is_empty() {
            plan.construction = plan.id.clone();
        }
        Ok(plan)
    }
    /// Every checkpoint it reads, constructed or raw: the store keeps them while it may run.
    pub fn manifests(&self) -> Vec<String> {
        let constructed = self.slots.iter().map(|slot| slot.binding.snapshot.clone());
        constructed.chain(self.sources.iter().map(|source| source.manifest.clone())).collect()
    }
    pub fn selections(&self) -> Vec<SelectedManifest> {
        merged(&self.slots)
    }
    /// Each model it runs as the machine resolved it, constructed or raw (`run show`).
    pub fn models(&self) -> Vec<serde_json::Value> {
        let row = |slot: &str, model: (&str, &str), lane: &str, manifest: &str| {
            serde_json::json!({"slot": slot, "repository": model.0, "release": model.1, "lane": lane, "manifest": manifest})
        };
        let constructed = self.slots.iter().map(|s| {
            let model = s.binding.model.rsplit_once('@').unwrap_or_default();
            row(&s.binding.model_parameter_name, model, &s.lane, &s.binding.snapshot)
        });
        let raw = self.sources.iter().map(|s| row(&s.parameter, (&s.repository, &s.release), &s.lane, &s.manifest));
        constructed.chain(raw).collect()
    }
    /// What a call reads before its function runs: every slot's components but an on-demand
    /// one's, which Runtime fetches when a stage first uses it.
    pub fn upfront(&self) -> Vec<(String, Vec<String>)> {
        merged(self.slots.iter().filter(|slot| !slot.on_demand)).into_iter().map(|s| (s.manifest, s.components)).collect()
    }
}

/// `slots`' checkpoints, one row per manifest with every component a slot of it reads.
fn merged<'a>(slots: impl IntoIterator<Item = &'a PlanSlot>) -> Vec<SelectedManifest> {
    let mut rows: Vec<SelectedManifest> = vec![];
    for slot in slots {
        match rows.iter_mut().find(|r| r.manifest == slot.binding.snapshot) {
            Some(row) => {
                for c in &slot.binding.components {
                    if !row.components.contains(c) {
                        row.components.push(c.clone());
                    }
                }
            }
            None => rows.push(SelectedManifest {
                manifest: slot.binding.snapshot.clone(),
                components: slot.binding.components.clone(),
            }),
        }
    }
    rows
}

/// A model row of an interface entry that is constructed: not a `Weights[M]` argument.
pub fn constructed(model: &serde_json::Value) -> bool {
    model.get("weights").and_then(serde_json::Value::as_bool) != Some(true)
}

/// The widest group every constructed slot declares (`sequence_parallel.degrees`, one always)
/// within `available` GPUs and, when one is asked for, `wanted`; Runtime `machine_lanes.widths`.
/// A count asked for is a cap, never a refusal: a caller's `--gpus 2` gives its one-GPU children one.
pub fn group_degree(models: &[serde_json::Value], wanted: u32, available: usize) -> u32 {
    let cap = match wanted {
        0 => available,
        wanted => available.min(wanted as usize),
    };
    let mut common: std::collections::BTreeSet<u32> = (1..=64).collect();
    for model in models.iter().filter(|model| constructed(model)) {
        let mut declared = std::collections::BTreeSet::from([1u32]);
        if let Some(rows) = model
            .pointer("/sequence_parallel/degrees")
            .and_then(serde_json::Value::as_array)
        {
            declared.extend(rows.iter().filter_map(|d| d.as_u64()).map(|d| d as u32));
        }
        common = common.intersection(&declared).copied().collect();
    }
    common.into_iter().filter(|d| *d as usize <= cap).max().unwrap_or(1)
}

/// How far a function is prepared on this machine: a warm set member's `level`, and what
/// Status says a member or an installation `holds`. Each level includes the ones before it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Level {
    Installed,
    Downloaded,
    /// A process with its imports done: no device, no VRAM.
    Imported,
    /// An executor with the model constructed, its weights in the host tier.
    Host,
    /// Its weights on the GPU.
    Gpu,
}
impl Level {
    pub const ALL: [Level; 5] = [
        Level::Installed,
        Level::Downloaded,
        Level::Imported,
        Level::Host,
        Level::Gpu,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Level::Installed => "installed",
            Level::Downloaded => "downloaded",
            Level::Imported => "imported",
            Level::Host => "host",
            Level::Gpu => "gpu",
        }
    }
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|level| level.name() == name)
    }
}

/// A warm set member as the journal keeps it: enough to bring it back after a restart, and
/// to show it in Status as its caller sent it.
#[derive(Serialize, Deserialize)]
pub struct KeptMember {
    pub installation: String,
    pub level: String,
    /// Its function's bound preparation; empty when it binds no model on a GPU.
    pub preparation: String,
    /// The `WarmItem` the caller sent, base64.
    pub item: String,
}

/// Why a member stays at `installed`: nothing of it loads on a GPU.
const UNBOUND: &str = "its function binds no model on a GPU, or the store let its weights go";
/// What a member holds on a machine with no GPU: its code (a function that binds a model is
/// refused there).
pub fn without_gpus(asked: Level) -> (Level, &'static str) {
    (Level::Installed, if asked > Level::Installed { UNBOUND } else { "" })
}

/// One member of a caller's warm set, as the pool keeps it.
pub struct Member {
    pub held: HeldGeneration,
    /// The models its function binds. None: it binds none on a GPU, or the store let them go.
    pub plan: Option<Plan>,
    /// The level the controller asked for.
    pub level: Level,
    /// Why it holds less than `level`, from the last pass over the set.
    held_back: Mutex<&'static str>,
    /// Its weights' fetches, under the warm run's access, for as long as it is in the set.
    fetch: Option<Arc<crate::access::Fetch>>,
}
impl Member {
    pub fn new(held: HeldGeneration, plan: Option<Plan>, level: Level, fetch: Option<Arc<crate::access::Fetch>>) -> Arc<Self> {
        Arc::new(Self {
            held,
            plan,
            level,
            held_back: Mutex::new(""),
            fetch,
        })
    }
}

struct Session {
    plan: String,
    /// Its lane's first GPU (envelope index), for groups as well as one-GPU replicas.
    start: usize,
    generation: String,
    /// The App it serves: its environment's root, or a callee's App in it.
    application: String,
    /// GPUs of its group; its followers (rank 1 first) once it started, held by pidfd so a
    /// call that fails can name the GPU whose process ended first.
    degree: u32,
    followers: Vec<cozy_platform::process::Exact>,
    loaded: bool,
    executor: DeviceExecutor,
    /// When it last came back idle: idle executors end least recently used first.
    used: Instant,
    /// Its selected models' headers and assets, served on `model_source`.
    sources: Arc<ModelSources>,
    /// This executor in the host tier and the layouts it may adopt.
    launch: Launch,
    /// It has served a call: its next one is not its first.
    invoked: bool,
}

/// How a session's executor came to exist, for the load record.
#[derive(Clone, Debug, Serialize)]
struct Launch {
    /// `fork` (from the generation's import-only executor) or `spawn`.
    mode: &'static str,
    /// From the launch decision to the executor's Hello.
    ms: f64,
    /// Its Start: wall time and the executor's own legs.
    start_ms: f64,
    start: Vec<(String, f64)>,
}

/// A generation's import-only executor that executors fork from (`fork/1`). Its children
/// die with it, so it ends only with the pool or, childless, to give its host memory back.
#[derive(Default)]
struct Zygote {
    state: Mutex<ZygoteState>,
    changed: Condvar,
    /// Executors forked from it that may still live.
    children: Mutex<Vec<Birth>>,
    /// Its last fork (or its start): childless parents end least recently used first.
    used: Mutex<Option<Instant>>,
}
#[derive(Default)]
enum ZygoteState {
    #[default]
    Starting,
    Ready(Box<DeviceExecutor>),
    /// Executors of this generation spawn: its Runtime does not fork, or the parent ended.
    Off(String),
    /// Its import-only start failed: this launch spawns, the next one starts a parent again
    /// (a slow or broken first import is not a reason to stop forking for the machine's life).
    Failed(String),
}
impl Zygote {
    fn set(&self, state: ZygoteState) {
        if let ZygoteState::Off(reason) | ZygoteState::Failed(reason) = &state {
            tracing::info!("executor prespawn off: {reason}");
        }
        *self.state.lock().unwrap() = state;
        self.changed.notify_all();
    }
    /// Wait while the parent imports.
    fn wait_started(&self) {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, ZygoteState::Starting) {
            state = self.changed.wait(state).unwrap();
        }
    }
    /// Fork an executor, waiting while the parent imports. A refusal or an ended parent
    /// gives the configuration back for a spawn.
    fn fork(
        &self,
        config: ExecutorConfig,
        on_birth: impl FnOnce(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Forked> {
        let mut state = self.state.lock().unwrap();
        while matches!(*state, ZygoteState::Starting) {
            state = self.changed.wait(state).unwrap();
        }
        let ZygoteState::Ready(parent) = &mut *state else {
            return Ok(Forked::Refused(
                Box::new(config),
                "no import-only executor".into(),
            ));
        };
        let forked = parent.fork(config, on_birth)?;
        match &forked {
            Forked::Ready(child) => {
                self.children.lock().unwrap().push(child.birth.clone());
                *self.used.lock().unwrap() = Some(Instant::now());
            }
            Forked::Lost(..) => {
                // Its children ended with it; the pool replaces it (`new_session`).
                *state = ZygoteState::Off(ENDED.into());
            }
            Forked::Refused(..) => {}
        }
        Ok(forked)
    }
    fn failed(&self) -> bool {
        matches!(*self.state.lock().unwrap(), ZygoteState::Failed(_))
    }
    fn ready(&self) -> bool {
        matches!(*self.state.lock().unwrap(), ZygoteState::Ready(_))
    }
    /// Ready and with no live child: ending it ends nothing else.
    fn childless(&self) -> bool {
        if !matches!(*self.state.lock().unwrap(), ZygoteState::Ready(_)) {
            return false;
        }
        let mut children = self.children.lock().unwrap();
        children.retain(|birth| !birth.ended().unwrap_or(true));
        children.is_empty()
    }
    /// Its private host bytes (PSS less shared memory), 0 when not running.
    fn private_bytes(&self) -> u64 {
        match &*self.state.lock().unwrap() {
            ZygoteState::Ready(parent) => cozy_platform::contain::host_memory::process(parent.birth.pid)
                .map_or(0, |memory| memory.pss.saturating_sub(memory.pss_shmem)),
            _ => 0,
        }
    }
}

/// A parent that died (killed, out of memory): the pool starts a new one.
const ENDED: &str = "import-only executor ended";

/// What a parent costs before any is measured: an import-only SDXL or Anima executor is
/// 0.78 GiB RSS (J/RESULTS.md).
const UNMEASURED_PARENT: u64 = 1 << 30;

/// The `sessions` key of `plan`'s executor whose lane starts at GPU `start`.
fn session_key(plan: &str, start: usize) -> String {
    match start {
        0 => plan.into(),
        _ => format!("{plan}@{start}"),
    }
}

/// An App of an environment: `(generation, application)`. Executors are keyed by it.
type App = (String, String);

fn parent_app(held: &HeldGeneration) -> App {
    (held.record.identity.clone(), held.record.application.clone())
}

/// The manifests live executors read. A download's GC must not evict them: an evicted file a
/// session still reads frees no disk and breaks the session's next load.
#[derive(Default)]
struct Serving {
    next: std::sync::atomic::AtomicU64,
    held: Mutex<BTreeMap<u64, Vec<String>>>,
}
/// One executor's entry in `Serving`, removed when its process has exited.
struct ServingHold {
    serving: Arc<Serving>,
    id: u64,
}
impl Drop for ServingHold {
    fn drop(&mut self) {
        self.serving.held.lock().unwrap().remove(&self.id);
    }
}

pub struct Pool {
    root: PathBuf,
    serving: Arc<Serving>,
    /// Scopes JIT caches to this machine run; earlier runs' scopes are removed at start.
    incarnation: String,
    config: GpuConfig,
    store: Arc<Store>,
    /// This machine's memory log, in its store's `logs`.
    notes: crate::memory::MemoryLog,
    /// Each root's lane and each GPU's call (`gpu_allocation`).
    allocations: Arc<crate::gpu_allocation::Allocations>,
    /// Idle GPU executors by `session_key`: a call takes its own out for its run. One per
    /// construction and lane; the memory policy decides which keep weights mapped.
    sessions: Mutex<BTreeMap<String, Session>>,
    /// Idle deviceless executors, least recently used first: any number per construction, as
    /// many as ran at once. Host pressure ends them, before any GPU executor.
    deviceless: Mutex<Vec<Session>>,
    /// Set (under `sessions`) once the pool stops: an executor coming back then is ended.
    stopped: std::sync::atomic::AtomicBool,
    /// Live executor lane leaders, including running sessions removed from `sessions`.
    leaders: Mutex<std::collections::BTreeSet<(String, usize)>>,
    /// Every live GPU executor process (leaders and followers) by pid: its plan, lane and
    /// handle, so a process the ledger names stalled can be ended and a row read as a plan's.
    tenants: Mutex<BTreeMap<u32, Tenant>>,
    /// Per generation, the import-only executor sessions fork from. Declared after
    /// `sessions`: executors end before the parent they were forked from.
    zygotes: Mutex<BTreeMap<(String, String), Arc<Zygote>>>,
    /// Each generation's kernel boot of this machine run: its process while it runs.
    kernel_boots: Mutex<BTreeMap<String, Option<cozy_platform::process::Exact>>>,
    /// Each caller's warm set, in the order it sent it.
    members: Mutex<BTreeMap<String, Vec<Arc<Member>>>>,
    /// One pass over the warm sets at a time.
    keeping: Mutex<()>,
    /// Each live executor's (generation, level) by plan: what Status reads while a call holds
    /// `sessions`.
    levels: Mutex<BTreeMap<String, (App, Level)>>,
    /// Each envelope GPU and its ledger, in envelope order.
    devices: Vec<Device>,
    /// Host layouts executors deposited, granted to the next executor of the same weights.
    deposits: Arc<Deposits>,
    /// Shared mode's room for the host's other programs, and the stalls the watcher saw: the
    /// count at the last bring-back pass says whether one came since (`host_pressure`).
    reserve: Mutex<cozy_platform::contain::host_pressure::Reserve>,
    stalls: std::sync::atomic::AtomicU64,
    stalls_seen: std::sync::atomic::AtomicU64,
    /// Memoized Model methods' results, shared by every run on this machine.
    memo: Arc<crate::memo::Memo>,
    // Drop session/resource custody before ending the actual spawning thread.
    launcher: crate::child_launcher::ChildLauncher,
}
struct Device {
    entry: String,
    /// Its ledger; None where neither NVML nor the configuration says its memory.
    memory: Option<GpuLedger>,
}

/// One GPU executor process: its plan and lane, and a handle that ends exactly it.
struct Tenant {
    plan: String,
    start: usize,
    exact: Exact,
}

pub(crate) struct Permit {
    pool: Arc<Pool>,
    engine: Weak<Engine>,
    token: Option<crate::gpu_allocation::Permit>,
}
/// A deviceless root's allocation (Slurm-style), held while its call runs: its children's GPU
/// steps run in its lane.
pub(crate) struct RootAllocation {
    pool: Arc<Pool>,
    engine: Weak<Engine>,
    hold: crate::gpu_allocation::Hold,
}
impl Drop for RootAllocation {
    fn drop(&mut self) {
        let Some(engine) = self.engine.upgrade() else { return };
        if let Some(Some(lane)) = self.pool.allocations.bound(self.hold.root()) {
            self.pool.note_gpus(&engine, self.hold.root(), "release", lane, None);
        }
        engine.notify_activity();
    }
}
impl Drop for Permit {
    fn drop(&mut self) {
        drop(self.token.take());
        if let Some(engine) = self.engine.upgrade() {
            engine.notify_activity();
        }
    }
}
pub(crate) struct WakeOnExit(pub(crate) Weak<Engine>);
impl Drop for WakeOnExit {
    fn drop(&mut self) {
        if let Some(engine) = self.0.upgrade() {
            engine.notify_activity();
        }
    }
}

/// Whether an accepted call waits in the queue for a GPU: any, or of `root` or an older root.
fn gpu_request_queued(engine: &Engine, root: Option<u64>) -> bool {
    engine.ready().is_ok_and(|ready| ready.iter().any(|record| {
        record.invocation.degree > 0 && root.is_none_or(|root| record.root_seq() <= root)
    }))
}

impl Pool {
    pub fn new(root: &Path, config: GpuConfig, store: Arc<Store>) -> io::Result<Arc<Self>> {
        fs::create_dir_all(root)?;
        let notes = crate::memory::MemoryLog::at(store.root().join("logs"));
        if let Some(identity) = config.identity {
            identity.traverse(root)?;
            identity.traverse(
                root.parent()
                    .ok_or_else(|| io::Error::other("GPU root has no owned state parent"))?,
            )?;
        }
        // Chunk grants hold one descriptor per object.
        cozy_platform::process::raise_fd_limit();
        let envelope = config.envelope();
        let incarnation = uuid::Uuid::new_v4().simple().to_string();
        crate::launch_identity::remove_stale_jit(root, &incarnation);
        remove_old_executor_roots(root);
        let shared = config.host.mode == cozy_platform::contain::host_pressure::HostMode::Shared;
        let mut devices = vec![];
        for entry in &envelope {
            let capacity = config
                .inventory
                .iter()
                .find(|g| g.device_index.to_string() == *entry || g.device_uuid == *entry)
                .map(|g| g.memory_bytes)
                .or(config.authorized_device_limit_bytes);
            let memory = GpuLedger::start(entry, shared, config.memory.floor_bytes, capacity, Default::default(), notes.clone())
                .unwrap_or_else(|error| {
                    tracing::warn!("memory: GPU {entry} has no ledger: {error}");
                    None
                });
            devices.push(Device { entry: entry.clone(), memory });
        }
        let pool = Arc::new(Self {
            launcher: crate::child_launcher::ChildLauncher::new()?,
            root: root.to_path_buf(),
            serving: Arc::default(),
            devices,
            incarnation,
            config,
            store,
            notes,
            allocations: crate::gpu_allocation::Allocations::new(envelope.len()),
            sessions: Mutex::new(BTreeMap::new()),
            deviceless: Mutex::new(Vec::new()),
            stopped: Default::default(),
            leaders: Mutex::default(),
            tenants: Mutex::default(),
            zygotes: Mutex::new(BTreeMap::new()),
            kernel_boots: Mutex::new(BTreeMap::new()),
            members: Mutex::new(BTreeMap::new()),
            keeping: Mutex::new(()),
            levels: Mutex::new(BTreeMap::new()),
            deposits: Arc::default(),
            reserve: Mutex::default(),
            stalls: Default::default(),
            stalls_seen: Default::default(),
            memo: Arc::new(crate::memo::Memo::open(root.join("stage-memo"))?),
        });
        for device in &pool.devices {
            let weak = Arc::downgrade(&pool);
            let end = move |pid| if let Some(pool) = weak.upgrade() { pool.end_stalled(pid) };
            device.memory.iter().for_each(|ledger| ledger.on_stalled(Box::new(end.clone())));
        }
        pool.watch_host();
        Ok(pool)
    }
    /// The host-pressure watcher (`host_pressure`): one thread blocked in poll() until the
    /// kernel reports a stall, then a rung given back when the feedback rule says so.
    fn watch_host(self: &Arc<Self>) {
        let mode = self.config.host.mode;
        let pressure = match cozy_platform::contain::host_pressure::Pressure::arm(mode) {
            Ok(pressure) => pressure,
            Err(error) => {
                tracing::warn!("host pressure: no PSI trigger ({error}); memory goes back only as calls plan");
                return;
            }
        };
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("host-pressure".into())
            .spawn(move || {
                let mut feedback = cozy_platform::contain::host_pressure::Feedback::new(
                    pressure.sample().map(|sample| sample.total()).unwrap_or(0),
                    std::time::Instant::now(),
                );
                while pressure.wait().is_ok() {
                    let Some(pool) = pool.upgrade() else { break };
                    let Ok(sample) = pressure.sample() else { break };
                    let Some(give) = feedback.observe(sample, std::time::Instant::now()) else { continue };
                    pool.stalls.fetch_add(1, Ordering::AcqRel);
                    pool.host_room_now();
                    let gave = give && pool.give_back_one();
                    if give && !gave {
                        feedback.exhausted();
                    }
                    pool.notes.note(serde_json::json!({"event": "host_pressure",
                        "mode": format!("{mode:?}"), "stalled_us": sample.stalled_us(),
                        "limit_hits": sample.limit_hits(), "gave": gave,
                        "stopped_at": feedback.stopped_at()}));
                }
            });
        if let Err(error) = started {
            tracing::warn!("host pressure: {error}");
        }
    }
    /// Host bytes discretionary holdings may take now (a warm member, an import ahead): what
    /// the host has before its tightest limit, less, in shared mode, what its other programs
    /// used recently beyond their use now (`host_pressure::Reserve`). None: unreadable.
    fn host_room_now(&self) -> Option<u64> {
        let host = cozy_platform::contain::host_memory::read();
        let available = u64::try_from(host.available).ok()?;
        if self.config.host.mode == cozy_platform::contain::host_pressure::HostMode::Dedicated {
            return Some(available);
        }
        let ours = cozy_platform::contain::host_memory::tree_private(std::process::id());
        let total = u64::try_from(cozy_platform::contain::host_memory::total()).ok()?;
        let others = cozy_platform::contain::host_pressure::others(&host, total, ours)?;
        Some(available.saturating_sub(self.reserve.lock().unwrap().observe(others)))
    }
    /// A bring-back pass: with no stall since the last one, the reserve forgets half.
    fn quiet_pass(&self) {
        let stalls = self.stalls.load(Ordering::Acquire);
        if self.stalls_seen.swap(stalls, Ordering::AcqRel) != stalls {
            return;
        }
        let host = cozy_platform::contain::host_memory::read();
        let ours = cozy_platform::contain::host_memory::tree_private(std::process::id());
        let others = u64::try_from(cozy_platform::contain::host_memory::total())
            .ok()
            .and_then(|total| cozy_platform::contain::host_pressure::others(&host, total, ours));
        if let Some(others) = others {
            self.reserve.lock().unwrap().quiet(others);
        }
    }
    /// One rung back to the host, in the agreed order: sealed layouts no tenant holds, an
    /// import-only parent with no live child, an idle deviceless executor, then an idle
    /// executor (outside every warm set first) on a GPU no call holds. Whether anything went
    /// back.
    pub fn give_back_one(&self) -> bool {
        self.deposits.release_one()
            || self.end_idle_parent(false) > 0
            || self.end_idle_deviceless(|_| true)
            || self.end_lru_idle("", false, 0..0)
    }
    /// End the least recently used idle deviceless executor `which` picks; whether one ended.
    fn end_idle_deviceless(&self, which: impl Fn(&Session) -> bool) -> bool {
        let victim = {
            let mut idle = self.deviceless.lock().unwrap();
            idle.iter().position(&which).map(|at| idle.remove(at))
        };
        let Some(session) = victim else {
            return false;
        };
        let (plan, start) = (session.plan.clone(), session.start);
        if let Err(error) = session.executor.shutdown() {
            tracing::warn!("ending idle deviceless executor of {plan}: {error}");
        }
        self.ended(&plan, start, 0);
        true
    }
    /// End one idle executor other than `plan`'s, the first GPU's least recently used first:
    /// on the caller's own GPUs (`own`), or on a GPU whose call slot is free meanwhile.
    fn end_lru_idle(&self, plan: &str, spare: bool, own: std::ops::Range<usize>) -> bool {
        (0..self.devices.len()).any(|device| {
            let held = match own.contains(&device) {
                true => None,
                false => match self.allocations.upkeep(device..device + 1) {
                    Some(permit) => Some(permit),
                    None => return false,
                },
            };
            let ended = self.end_idle_on(device, plan, spare);
            drop(held);
            ended
        })
    }
    /// End the least recently used idle executor on GPU `device` other than `plan`'s: outside
    /// every warm set first, and never a member's when `spare`. The caller holds its call slot.
    fn end_idle_on(&self, device: usize, plan: &str, spare: bool) -> bool {
        let members: std::collections::BTreeSet<String> = {
            let sets = self.members.lock().unwrap();
            sets.values().flatten().filter_map(|m| m.plan.as_ref().map(|p| p.construction.clone())).collect()
        };
        let victim = {
            let sessions = self.sessions.lock().unwrap();
            let on = |s: &Session| (s.start..s.start + s.degree as usize).contains(&device);
            sessions
                .iter()
                .filter(|(_, s)| on(s) && s.plan != plan && !(spare && members.contains(&s.plan)))
                .min_by_key(|(_, s)| (members.contains(&s.plan), s.used))
                .map(|(key, _)| key.clone())
        };
        victim.is_some_and(|key| self.end_session(&key))
    }
    /// End idle executor `key` and observe its exit: what it held goes back. Whether it was idle.
    fn end_session(&self, key: &str) -> bool {
        let Some(session) = self.sessions.lock().unwrap().remove(key) else {
            return false;
        };
        let (plan, start, degree) = (session.plan.clone(), session.start, session.degree);
        self.notes.note(serde_json::json!({"event": "end_idle", "plan": plan, "gpus": start..start + degree as usize}));
        if let Err(error) = session.executor.shutdown() {
            tracing::warn!("memory: ending idle executor {key}: {error}");
        }
        self.ended(&plan, start, degree);
        true
    }
    /// A process the ledger names stalled (asked for bytes, its progress still): it ends now.
    /// An idle one leaves the pool; a running one's call fails as its executor ends.
    fn end_stalled(&self, pid: u32) {
        let Some(tenant) = self.tenants.lock().unwrap().get(&pid).map(|t| (t.plan.clone(), t.start, t.exact.kill())) else {
            return tracing::warn!("memory: stalled pid {pid} is no executor of this machine");
        };
        let (plan, start, killed) = tenant;
        self.notes.note(serde_json::json!({"event": "end_stalled", "pid": pid, "plan": plan,
            "killed": killed.as_ref().map_or_else(|e| e.to_string(), |_| "yes".into())}));
        let key = session_key(&plan, start);
        let idle = self.sessions.lock().unwrap().remove(&key);
        if let Some(session) = idle {
            let (degree, executor) = (session.degree, session.executor);
            if let Err(error) = executor.terminate() {
                tracing::warn!("memory: ending stalled executor {key}: {error}");
            }
            self.ended(&plan, start, degree);
        }
    }
    pub fn memo(&self) -> &crate::memo::Memo {
        &self.memo
    }
    pub fn config(&self) -> &GpuConfig {
        &self.config
    }
    /// The manifests this pool's live executors read (`Serving`).
    pub fn serving(&self) -> Vec<String> {
        let held = self.serving.held.lock().unwrap();
        let mut manifests: Vec<String> = held.values().flatten().cloned().collect();
        manifests.sort();
        manifests.dedup();
        manifests
    }
    /// The kernel store; live executors and kernel boots hold their namespace's lock.
    pub fn kernel_caches(&self) -> crate::reclaim::KernelCaches {
        crate::reclaim::KernelCaches {
            root: self.root.join("kernels"),
        }
    }
    /// GPUs this machine may group.
    pub fn width(&self) -> usize {
        self.devices.len()
    }
    /// The first `degree` GPUs as the lane rule reads them; one the inventory does not name
    /// runs no native format.
    pub fn gpus(&self, degree: u32) -> crate::precision::Gpus {
        let ccs: Vec<&str> = self.devices.iter().take(degree as usize)
            .map(|d| self.inventory(&d.entry).map_or("", |g| g.compute_cap.as_str())).collect();
        crate::precision::Gpus::of(&ccs)
    }
    /// The lane rule's budget on the first `degree` GPUs (design §2.1): the smallest one's memory
    /// less a quarter of it for activations. None when the inventory names no memory.
    pub fn budget(&self, degree: u32) -> Option<u64> {
        let lane = &self.devices[..self.devices.len().min(degree as usize)];
        let memory = lane.iter().map(|d| self.inventory(&d.entry).map(|g| g.memory_bytes)).min()??;
        Some(memory - memory / 4)
    }
    fn inventory(&self, entry: &str) -> Option<&crate::machine::receipt::Gpu> {
        self.config.inventory.iter().find(|g| g.device_index.to_string() == entry || g.device_uuid == entry)
    }

    /// The `degree` GPUs from envelope index `start`: a group's are the first ones; none for a
    /// deviceless call.
    fn lane(&self, start: usize, degree: u32) -> io::Result<&[Device]> {
        if degree == 0 {
            return Ok(&[]);
        }
        self.devices.get(start..start + degree as usize).ok_or_else(|| self.unavailable(degree))
    }
    fn unavailable(&self, wanted: u32) -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            format!("gpu_count_unavailable: {wanted} GPUs asked of a machine with {}", self.devices.len()),
        )
    }
    fn lane_devices(&self, start: usize, degree: u32) -> io::Result<String> {
        Ok(self
            .lane(start, degree)?
            .iter()
            .map(|d| d.entry.as_str())
            .collect::<Vec<_>>()
            .join(","))
    }
    /// `plan`'s executor on the lane from `start` is gone, its processes with it: each GPU of
    /// the lane reads NVML now, so their rows are reaped and their bytes returned (which wakes
    /// the dispatch).
    fn ended(&self, plan: &str, start: usize, degree: u32) {
        let key = session_key(plan, start);
        self.levels.lock().unwrap().remove(&key);
        self.leaders.lock().unwrap().remove(&(plan.to_owned(), start));
        self.tenants.lock().unwrap().retain(|_, t| !(t.plan == plan && t.start == start));
        for ledger in self.lane(start, degree).unwrap_or_default().iter().filter_map(|d| d.memory.as_ref()) {
            ledger.now();
        }
    }
    /// A loaded executor's level from its leader's ledger row: `gpu` while its weights are
    /// resident there, else `host`.
    fn loaded_level(&self, start: usize, pid: u32) -> Level {
        match self.devices.get(start).and_then(|d| d.memory.as_ref()).is_some_and(|m| m.resident(pid)) {
            true => Level::Gpu,
            false => Level::Host,
        }
    }
    /// The engine's dispatch wakes whenever bytes free on a GPU: a queued request asks `fits`
    /// again then, and never polls.
    pub fn wake_on_freed(&self, engine: &Arc<Engine>) {
        for ledger in self.devices.iter().filter_map(|d| d.memory.as_ref()) {
            let engine = Arc::downgrade(engine);
            ledger.on_freed(Box::new(move || engine.upgrade().iter().for_each(|e| e.notify_activity())));
        }
    }

    /// Replace `actor`'s warm set; `keep` then brings its members up. A member dropped from
    /// it becomes an ordinary tenant: it stays as it is until its memory is needed.
    pub fn set_members(&self, actor: &str, members: Vec<Arc<Member>>) {
        let mut sets = self.members.lock().unwrap();
        match members.is_empty() {
            true => drop(sets.remove(actor)),
            false => drop(sets.insert(actor.into(), members)),
        }
    }
    /// `actor`'s warm set without the members of removed `generations`.
    pub fn drop_members(&self, actor: &str, generations: &std::collections::BTreeSet<String>) {
        let mut kept = self.warm_set(actor);
        let before = kept.len();
        kept.retain(|member| !generations.contains(&member.held.record.identity));
        if kept.len() != before {
            self.set_members(actor, kept);
        }
    }
    /// A removed generation leaves memory now: each idle executor of it ends while its lane's
    /// call slots are free, then its import-only parent once childless, then the GPU weights no
    /// installed generation read. One in a call ends with the call's tenancy, as any idle
    /// executor does.
    pub fn end_generation(&self, identity: &str) {
        let lanes: Vec<(String, usize, u32)> = {
            let sessions = self.sessions.lock().unwrap();
            let idle = sessions.values().filter(|s| s.generation == identity);
            idle.map(|s| (s.plan.clone(), s.start, s.degree)).collect()
        };
        for (plan, start, degree) in lanes {
            let Some(permit) = self.allocations.upkeep(start..start + degree as usize) else {
                continue;
            };
            self.end_session(&session_key(&plan, start));
            drop(permit);
        }
        while self.end_idle_deviceless(|session| session.generation == identity) {}
        let parents: Vec<Arc<Zygote>> = {
            let mut zygotes = self.zygotes.lock().unwrap();
            let ended: Vec<App> = zygotes
                .iter()
                .filter(|((generation, _), zygote)| generation == identity && zygote.childless())
                .map(|(key, _)| key.clone())
                .collect();
            ended.into_iter().filter_map(|key| zygotes.remove(&key)).collect()
        };
        for zygote in parents {
            if let ZygoteState::Ready(parent) = std::mem::take(&mut *zygote.state.lock().unwrap()) {
                if let Err(error) = parent.shutdown() {
                    tracing::warn!("ending a removed package's import-only executor: {error}");
                }
            }
        }
    }
    /// One pass over `actor`'s warm set: each member is brought as far toward its level as
    /// free room and the room of idle tenants outside every warm set allow. It never takes
    /// from another member or a running call, and keeps the reason when it stops short.
    pub fn keep(self: &Arc<Self>, engine: &Arc<Engine>, actor: &str) {
        let _pass = self.keeping.lock().unwrap();
        self.quiet_pass();
        for member in self.warm_set(actor) {
            // From `downloaded` up, what its constructions read comes behind every call's.
            if let (Some(plan), Some(fetch), true) = (&member.plan, &member.fetch, member.level >= Level::Downloaded) {
                if let Err(error) = fetch.keep(&plan.upfront()) {
                    tracing::warn!("warm set: downloading {}: {error}", plan.construction);
                }
            }
            let held_back = match (&member.plan, member.level) {
                (_, Level::Installed) => "",
                (None, _) => UNBOUND,
                (Some(plan), Level::Downloaded) if !self.downloaded(plan) => "its weights are downloading",
                (_, Level::Downloaded) => "",
                (_, Level::Imported) => self.import(&member.held),
                (Some(plan), Level::Host | Level::Gpu) => match self.import(&member.held) {
                    "" => self.load_member(engine, &member, plan),
                    held_back => held_back,
                },
            };
            *member.held_back.lock().unwrap() = held_back;
        }
    }
    /// Every warm set's pass, off the caller's thread: after a call ends, when room may
    /// have come back for a member a request pushed down.
    pub fn keep_all(self: &Arc<Self>, engine: &Arc<Engine>) {
        let actors: Vec<String> = self.members.lock().unwrap().keys().cloned().collect();
        if actors.is_empty() {
            return;
        }
        let (pool, engine) = (self.clone(), engine.clone());
        let started = std::thread::Builder::new()
            .name("warm-set".into())
            .spawn(move || actors.iter().for_each(|actor| pool.keep(&engine, actor)));
        if let Err(error) = started {
            tracing::warn!("warm set: {error}");
        }
    }
    /// A member's executor loaded when the call slot is free: empty once it is, else why not.
    fn load_member(self: &Arc<Self>, engine: &Arc<Engine>, member: &Member, plan: &Plan) -> &'static str {
        let (held, level) = (&member.held, member.level);
        if self.level(&plan.construction).is_some_and(|holds| holds >= level) {
            return "";
        }
        let Some(_permit) = self.permit(engine, None, None, 0..plan.degree as usize, &[]) else {
            return "a request holds the GPU; it loads when that ends";
        };
        // A request queued for the GPU goes first; the member comes back after it.
        if gpu_request_queued(engine, None) {
            return "a request is queued; it loads after";
        }
        let asked = Instant::now();
        let mut result = self.prewarm(engine, held, plan.clone(), 0, true, member.fetch.clone());
        if let (Ok("loaded" | "already loaded"), Level::Gpu | Level::Host) = (&result, level) {
            result = self.map_member(plan, level);
        }
        self.settle_lane(&plan.construction, 0, plan.degree);
        self.note_prewarm(&plan.construction, Default::default(), asked.elapsed(), &result);
        match result {
            Ok("loaded" | "already loaded" | "mapped") => "",
            Ok(held_back) => held_back,
            Err(error) => {
                tracing::warn!("warm set: loading {}: {error}", plan.construction);
                "its load failed (the machine's log says why)"
            }
        }
    }
    /// Warming a loaded member: its executor loads its models at `level` (`gpu`, or `host` in
    /// RAM) and varena fills them: "mapped", or why they stay lower.
    fn map_member(&self, plan: &Plan, level: Level) -> io::Result<&'static str> {
        let holds = self.levels.lock().unwrap().get(&plan.construction).map(|(_, level)| *level);
        if holds == Some(level) {
            return Ok("mapped");
        }
        if plan.degree != 1 {
            return Ok("a group of GPUs maps its weights at its first call");
        }
        let Some(mut session) = self.sessions.lock().unwrap().remove(&plan.construction) else {
            return Ok("its executor ended");
        };
        let reply = session.executor.command(
            &DeviceCommand::Map {
                construction: plan.construction.clone(),
                level: if level == Level::Gpu { "gpu" } else { "ram" },
            },
            &mut device_executor::Baseline,
        );
        let complete = reply.as_ref().is_ok_and(|reply| reply.ok && reply.complete.unwrap_or(true));
        self.keep_idle(session);
        let reply = reply?;
        if !reply.ok {
            tracing::warn!("warm set: mapping {}: {} {}", plan.construction, reply.code, reply.detail);
            return Ok("its weights could not be mapped (the machine's log says why)");
        }
        match complete {
            true => Ok("mapped"),
            false => Ok("part of its weights stay in host memory: the GPU has no room for them all"),
        }
    }
    /// What each of `actor`'s members holds now, and why that is lower than it asked.
    pub fn members(&self, actor: &str) -> Vec<(Level, &'static str)> {
        let holds = |member: &Arc<Member>| {
            let (holds, held_back) = (self.holding(member), *member.held_back.lock().unwrap());
            match (holds < member.level, holds, held_back) {
                (false, ..) => (holds, ""),
                (true, _, "") => (holds, "a request took its room; it returns when room does"),
                (true, ..) => (holds, held_back),
            }
        };
        self.warm_set(actor).iter().map(holds).collect()
    }
    /// The highest level an executor of `plan` holds, on any GPU.
    fn level(&self, plan: &str) -> Option<Level> {
        let levels = self.levels.lock().unwrap();
        let of_plan = levels.iter().filter(|(key, _)| key.split('@').next() == Some(plan));
        of_plan.map(|(_, (_, level))| *level).max()
    }
    fn warm_set(&self, actor: &str) -> Vec<Arc<Member>> {
        let sets = self.members.lock().unwrap();
        sets.get(actor).cloned().unwrap_or_default()
    }
    fn holding(&self, member: &Member) -> Level {
        let executor = member.plan.as_ref().and_then(|plan| self.level(&plan.construction));
        let imported = || {
            let zygotes = self.zygotes.lock().unwrap();
            zygotes.get(&parent_app(&member.held)).cloned()
        };
        match (executor, &member.plan) {
            (Some(level), _) => level,
            _ if imported().is_some_and(|parent| parent.ready()) => Level::Imported,
            (None, Some(plan)) if self.downloaded(plan) => Level::Downloaded,
            _ => Level::Installed,
        }
    }
    /// Keep an executor idle for its next call, noting what a loaded one holds for Status.
    /// Once the pool stopped it ends instead.
    fn keep_idle(&self, session: Session) {
        let key = session_key(&session.plan, session.start);
        if session.degree == 0 {
            let mut idle = self.deviceless.lock().unwrap();
            if !self.stopped.load(Ordering::Acquire) {
                return idle.push(session);
            }
            drop(idle);
            if let Err(error) = session.executor.shutdown() {
                tracing::warn!("ending executor {key} after the pool stopped: {error}");
            }
            return;
        }
        let mut sessions = self.sessions.lock().unwrap();
        if self.stopped.load(Ordering::Acquire) {
            drop(sessions);
            let (plan, start, degree) = (session.plan.clone(), session.start, session.degree);
            if let Err(error) = session.executor.shutdown() {
                tracing::warn!("ending executor {key} after the pool stopped: {error}");
            }
            return self.ended(&plan, start, degree);
        }
        let mut session = session;
        session.used = Instant::now();
        if session.loaded {
            let level = self.loaded_level(session.start, session.executor.birth.pid);
            let noted = ((session.generation.clone(), session.application.clone()), level);
            self.levels.lock().unwrap().insert(key.clone(), noted);
        }
        sessions.insert(key, session);
    }
    /// A lane's turn is over: an executor not kept idle is gone.
    fn settle_lane(&self, plan: &str, start: usize, degree: u32) {
        if !self.sessions.lock().unwrap().contains_key(&session_key(plan, start)) {
            self.ended(plan, start, degree);
        }
    }
    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }
    /// Whether the store holds the structure of every model `plan` binds.
    pub fn holds(&self, plan: &Plan) -> bool {
        plan.selections().iter().all(|selected| crate::storage::held(&self.store, &selected.manifest).is_ok())
    }
    /// Every object its constructions read up front is here: `downloaded`.
    pub fn downloaded(&self, plan: &Plan) -> bool {
        plan.upfront().iter().all(|(manifest, components)| crate::access::whole(&self.store, manifest, components))
    }
    /// The highest level each App of each generation holds now (keyed `(generation,
    /// application)`): an import-only parent, or an executor. A callee's executors in a
    /// caller's environment are its own, never the caller's.
    pub fn levels(&self) -> BTreeMap<App, Level> {
        let parents: Vec<_> = self.zygotes.lock().unwrap().clone().into_iter().collect();
        let mut highest: BTreeMap<App, Level> = parents
            .into_iter()
            .filter(|(_, parent)| parent.ready())
            .map(|(app, _)| (app, Level::Imported))
            .collect();
        for (app, level) in self.levels.lock().unwrap().values() {
            let held = highest.entry(app.clone()).or_insert(*level);
            *held = (*held).max(*level);
        }
        highest
    }
    /// `held`'s import-only parent, started here when the host has room for it: empty once
    /// it is up, else why it is not.
    fn import(&self, held: &HeldGeneration) -> &'static str {
        let known = self.zygotes.lock().unwrap().contains_key(&parent_app(held));
        if !known && !self.parent_fits(held) {
            return "no host memory free for its imports";
        }
        let Some((parent, start)) = self.zygote(held) else {
            return "this machine imports nothing ahead for it (no fork, or it binds no model)";
        };
        if start {
            parent.set(self.import_only(held));
        }
        parent.wait_started();
        match (parent.ready(), parent.failed()) {
            (true, _) => "",
            (_, false) => "its Runtime forks no executors: nothing imports ahead of a request",
            (_, true) => {
                // The next pass imports again; a request that needs it starts one itself.
                self.forget_parent(held, &parent);
                "its imports failed (the machine's log says why)"
            }
        }
    }
    /// Room for a request of `plan` on every GPU of the lane, in device order (so two groups
    /// never wait on each other), at `shape` once it is known (`spawn`: it starts a process too):
    /// while it does not fit and no request runs there, the bytes of exited executors are waited
    /// for while they come back, then idle executors end, least recently used first (never
    /// `plan`'s, nor a warm member's when `spare`). Whether it fits everywhere; a request starts
    /// either way, and varena in its executor makes what room it can.
    fn make_room(&self, plan: &str, start: usize, degree: u32, shape: Option<&str>, spawn: bool, spare: bool) -> io::Result<bool> {
        self.reap();
        let mut fit = true;
        for (gpu, device) in (start..).zip(self.lane(start, degree)?) {
            let Some(ledger) = &device.memory else { continue };
            loop {
                ledger.settle();
                if ledger.fits(plan, shape, spawn) {
                    break;
                }
                if ledger.running() || !self.end_idle_on(gpu, plan, spare) {
                    self.notes.note(serde_json::json!({"event": "short", "plan": plan, "gpu": device.entry,
                        "shape": shape, "running": ledger.running()}));
                    fit = false;
                    break;
                }
            }
        }
        Ok(fit)
    }
    /// Idle executors that exited leave the pool.
    fn reap(&self) {
        let gone: Vec<Session> = {
            let mut sessions = self.sessions.lock().unwrap();
            let ended = |s: &Session| s.executor.birth.ended().unwrap_or(false);
            let keys: Vec<String> = sessions.iter().filter(|(_, s)| ended(s)).map(|(key, _)| key.clone()).collect();
            keys.iter().filter_map(|key| sessions.remove(key)).collect()
        };
        for session in gone {
            self.ended(&session.plan, session.start, session.degree);
        }
    }
    pub fn source_facts(&self, selections: &[SelectedManifest]) -> io::Result<ModelSources> {
        ModelSources::open_shared(self.store.clone(), selections)
    }
    pub fn published_installation(
        &self,
        service: &crate::service::Service,
        actor: &str,
        package: &str,
        release: &str,
    ) -> io::Result<Option<crate::journal::Installation>> {
        let Some(selected) = self
            .config
            .packages
            .iter()
            .find(|p| p.package == package && p.release == release)
        else {
            return Ok(None);
        };
        let held = service.catalog.resolve(&selected.generation)?;
        if held.record.package != selected.distribution || held.record.version != release {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "published catalog mapping differs from actual held package",
            ));
        }
        let alias = format!(
            "published-{}",
            tensorfs_core::sha256::hex_digest(
                format!("{package}\0{release}\0{}", selected.generation).as_bytes()
            )
        );
        service
            .engine
            .bind_installation(crate::journal::Installation {
                actor: actor.into(),
                alias,
                generation: selected.generation.clone(),
                package: package.into(),
                release: release.into(),
                interface: serde_json::to_vec(&held.record.interface)?,
                hub: String::new(),
            })
            .map(Some)
    }

    /// A call's degree as its declaration and this machine set it: 0 for `accelerator=False`
    /// or a machine with no GPU, else its constructed slots' group, or the GPUs a call that
    /// constructs nothing asks for (one by default); more than the machine has is refused.
    pub fn degree(&self, entry: &serde_json::Value, wanted: u32) -> io::Result<u32> {
        let models = entry.get("models").and_then(serde_json::Value::as_array).cloned().unwrap_or_default();
        let constructed: Vec<_> = models.into_iter().filter(constructed).collect();
        Ok(match entry.get("accelerator").and_then(serde_json::Value::as_bool) {
            Some(false) => 0,
            _ if self.devices.is_empty() => 0,
            _ if constructed.is_empty() && wanted as usize > self.devices.len() => {
                return Err(self.unavailable(wanted))
            }
            // It constructs nothing (a script): the GPUs it asks for, held by its one process.
            _ if constructed.is_empty() => wanted.max(1),
            _ => group_degree(&constructed, wanted, self.devices.len()),
        })
    }

    /// Derive each model slot's binding from the installed SDK's real static interface and
    /// the models the Hub resolved under the owner's access (`resolved`, one per slot), or
    /// configured cached byte authority: a constructed slot becomes the plan's construction, a
    /// `Weights[M]` one its source. `degree` 0 takes the widest group every constructed slot
    /// declares on this machine. No model code executes during description/preparation.
    pub fn prepare_root(
        &self,
        installed: &crate::journal::Installation,
        entrypoint: &str,
        choices: &[crate::api::domain::ModelChoice],
        resolved: &[ModelGrant],
        degree: u32,
    ) -> io::Result<Plan> {
        let invalid = |detail: &str| io::Error::new(io::ErrorKind::InvalidData, detail.to_string());
        let interface: serde_json::Value =
            serde_json::from_slice(&installed.interface).map_err(io::Error::other)?;
        let entry = interface
            .get("entrypoints")
            .and_then(serde_json::Value::as_array)
            .and_then(|rows| {
                rows.iter().find(|row| {
                    row.get("name").and_then(serde_json::Value::as_str) == Some(entrypoint)
                })
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "declared entrypoint absent"))?;
        let application = interface
            .get("application")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| invalid("declared application absent"))?;
        let models = entry.get("models").and_then(serde_json::Value::as_array).cloned().unwrap_or_default();
        let prefix = format!("{entrypoint}.models.");
        let slot_of = |model: &serde_json::Value| -> io::Result<(String, String)> {
            let path = model
                .get("path")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("declared model path absent"))?;
            let parameter = path
                .strip_prefix(&prefix)
                .filter(|v| !v.is_empty() && !v.contains('.'))
                .ok_or_else(|| invalid("declared model path is not a root parameter"))?;
            Ok((path.to_string(), parameter.to_string()))
        };
        let slots = models.iter().map(slot_of).collect::<io::Result<Vec<_>>>()?;
        for choice in choices {
            if !slots.iter().any(|(path, parameter)| {
                choice.parameter == *path || choice.parameter == *parameter
            }) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "model choice does not name a declared root slot",
                ));
            }
            // Hub-resolved grants already carry a slot's adapters (as its adapter view);
            // configured cached authority has no resolution step to apply them.
            if (!choice.source.is_empty() || !choice.profiles.is_empty() || !choice.adapters.is_empty())
                && resolved.is_empty()
            {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "provider sources and adapters need the run's preparation (configured grants take neither)",
                ));
            }
        }
        let degree = self.degree(entry, degree)?;
        let (mut planned, mut sources) = (vec![], vec![]);
        for (model, (path, parameter)) in models.iter().zip(&slots) {
            let class = model
                .get("class")
                .and_then(serde_json::Value::as_str)
                .ok_or_else(|| invalid("declared model class absent"))?;
            let chosen = choices
                .iter()
                .find(|c| c.parameter == *path || c.parameter == *parameter);
            // A model passed by value (a ModelArtifact a run here produced) is held here.
            let passed = chosen
                .filter(|c| resolved.is_empty() && c.repository.is_empty())
                .and_then(|c| c.manifest.as_ref())
                .map(|reference| self.held_grant(&installed.package, path, reference))
                .transpose()?;
            let authority: Vec<&ModelGrant> = match (&passed, resolved.is_empty()) {
                (Some(passed), _) => vec![passed],
                (None, true) => self.config.models.iter().collect(),
                (None, false) => resolved.iter().collect(),
            };
            let grants: Vec<_> = authority
                .into_iter()
                .filter(|grant| {
                    grant.package == installed.package
                        && grant.slot == *path
                        // A resolved grant was resolved from the choice itself.
                        && (!resolved.is_empty() || chosen.is_none_or(|choice| {
                            (choice.repository.is_empty() || choice.repository == grant.repository)
                                && (choice.release.is_empty() || choice.release == grant.release)
                                && (choice.lane.is_empty() || choice.lane == grant.lane)
                                && choice.manifest.as_ref().is_none_or(|reference| {
                                    tensorfs_core::sha256::hex(&reference.digest)
                                        == grant.manifest.trim_start_matches("sha256:")
                                })
                        }))
                })
                .collect();
            if grants.len() != 1 {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("cached model selection for {path} has no unique configured authority"),
                ));
            }
            let grant = grants[0];
            if !constructed(model) {
                let length = crate::storage::held(&self.store, &grant.manifest)
                    .map_err(|_| invalid(&format!("the checkpoint of {path} is not held here")))?
                    .length;
                if chosen.and_then(|c| c.manifest.as_ref()).is_some_and(|r| r.length != 0 && r.length != length) {
                    return Err(io::Error::new(io::ErrorKind::InvalidInput, "selected manifest length differs from cached bytes"));
                }
                sources.push(PlanSource {
                    parameter: parameter.clone(),
                    manifest: grant.manifest.clone(),
                    length,
                    repository: grant.repository.clone(),
                    release: grant.release.clone(),
                    lane: grant.lane.clone(),
                });
                continue;
            }
            let mut wanted = std::collections::BTreeSet::new();
            // The components this callable declares it uses, else every one its methods use.
            if let Some(declared) = model.get("components").and_then(serde_json::Value::as_array) {
                for component in declared {
                    wanted.insert(
                        component
                            .as_str()
                            .ok_or_else(|| invalid("declared component is not a name"))?
                            .to_string(),
                    );
                }
            } else if let Some(use_map) = model
                .get("component_use")
                .and_then(serde_json::Value::as_object)
            {
                for components in use_map.values() {
                    for component in components
                        .as_array()
                        .ok_or_else(|| invalid("declared component use is not an array"))?
                    {
                        wanted.insert(
                            component
                                .as_str()
                                .ok_or_else(|| invalid("declared component is not a name"))?
                                .to_string(),
                        );
                    }
                }
            }
            let components: Vec<_> = grant
                .components
                .iter()
                .filter(|c| wanted.is_empty() || wanted.contains(*c))
                .cloned()
                .collect();
            if components.is_empty() || wanted.iter().any(|c| !components.contains(c)) {
                return Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    format!("declared components of {path} exceed authorized cached model scope"),
                ));
            }
            let sources = self.source_facts(&[SelectedManifest {
                manifest: grant.manifest.clone(),
                components: components.clone(),
            }])?;
            let (_, selected_encoded_bytes, length) = sources.selected_facts(&grant.manifest)?;
            if chosen
                .and_then(|c| c.manifest.as_ref())
                .is_some_and(|r| r.length != 0 && r.length != length)
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "selected manifest length differs from cached bytes",
                ));
            }
            let binding = Binding {
                application: application.into(),
                model_class: class.into(),
                model_binding_path: path.clone(),
                model_parameter_name: parameter.clone(),
                model_parameter_names: vec![parameter.clone()],
                component: components[0].clone(),
                components: components.clone(),
                snapshots: components
                    .into_iter()
                    .map(|c| (c, grant.manifest.clone()))
                    .collect(),
                snapshot: grant.manifest.clone(),
                release: format!("{}@{}", installed.package, installed.release),
                package: installed.package.clone(),
                model: format!("{}@{}", grant.repository, grant.release),
                ..Default::default()
            };
            planned.push(PlanSlot {
                binding,
                lane: grant.lane.clone(),
                selected_encoded_bytes,
                on_demand: model.get("on_demand").and_then(serde_json::Value::as_bool) == Some(true),
            });
        }
        let digest = |value: serde_json::Value| -> io::Result<String> {
            let canonical = serde_json_canonicalizer::to_vec(&value).map_err(io::Error::other)?;
            Ok(tensorfs_core::sha256::hex_digest(&canonical))
        };
        let construction = format!("gpu-{}", digest(serde_json::json!({"generation": installed.generation,
            "application": application, "entrypoint": entrypoint, "slots": planned, "degree": degree}))?);
        let id = match sources.is_empty() {
            true => construction.clone(),
            false => format!("plan-{}", digest(serde_json::json!({"construction": construction, "sources": sources}))?),
        };
        Ok(Plan {
            id,
            construction,
            installation: installed.alias.clone(),
            generation: installed.generation.clone(),
            entrypoint: entrypoint.into(),
            slots: planned,
            sources,
            degree,
        })
    }
    pub fn plan(&self, preparation: &Preparation) -> io::Result<Plan> {
        Plan::of(preparation)
    }
    /// A slot's checkpoint this machine's store holds, by its exact manifest.
    fn held_grant(&self, package: &str, path: &str, reference: &crate::api::domain::Ref) -> io::Result<ModelGrant> {
        let sha = tensorfs_core::sha256::hex(&reference.digest);
        let absent = |why: String| io::Error::new(io::ErrorKind::NotFound, format!("checkpoint_absent: {path} names sha256:{sha}: {why}"));
        let object = tensorfs_core::format::ObjectRef { sha256: sha.clone(), length: reference.length };
        let checkpoint = self.store.checkpoint(&object).map_err(|e| absent(e.to_string()))?;
        Ok(ModelGrant {
            package: package.into(),
            slot: path.into(),
            repository: String::new(),
            release: String::new(),
            lane: String::new(),
            manifest: format!("sha256:{sha}"),
            components: checkpoint.header().components().iter().map(|c| c.name.clone()).collect(),
        })
    }
    fn permit(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        root: Option<u64>,
        family: Option<&str>,
        devices: std::ops::Range<usize>,
        claims: &Claims,
    ) -> Option<Permit> {
        let token = self.allocations.call(root, family, devices, claims)?;
        Some(Permit { pool: self.clone(), engine: Arc::downgrade(engine), token: Some(token) })
    }
    /// A deviceless root's allocation, `width` GPUs, held while its call runs. None for a
    /// root with no GPU work.
    pub(crate) fn root(self: &Arc<Self>, engine: &Arc<Engine>, record: &Execution, width: usize) -> Option<RootAllocation> {
        (width > 0).then(|| RootAllocation {
            hold: self.allocations.register(record.root_seq(), width),
            pool: self.clone(),
            engine: Arc::downgrade(engine),
        })
    }
    /// `Context.release_gpus()` of a root: its lane ends now; its next step queues again at
    /// its original place.
    pub(crate) fn release(&self, engine: &Engine, root: u64) {
        if let Some(lane) = self.allocations.release(root) {
            self.note_gpus(engine, root, "release", lane, None);
        }
    }
    /// `root`'s lane, bound now when free; None while it waits. Each transition is logged on
    /// the root run as a `gpu` entry: a grant, a wait naming its current blocker, and a lane
    /// that ended for it as its younger holder's release behind it.
    pub(crate) fn root_lane(&self, engine: &Engine, root: u64, claims: &mut Claims) -> Option<std::ops::Range<usize>> {
        let (lane, yielded) = self.allocations.lane(root, claims);
        for (holder, lane) in yielded {
            self.note_gpus(engine, holder, "release", lane, Some(root));
        }
        match lane {
            crate::gpu_allocation::Lane::Held { lane, new } => {
                if new {
                    self.note_gpus(engine, root, "grant", lane.clone(), None);
                }
                Some(lane)
            }
            crate::gpu_allocation::Lane::Waiting { behind, new } => {
                // A root not started yet just stays queued.
                if new && engine.get(&root.to_string()).is_ok_and(|record| matches!(record.state, State::Starting | State::Running)) {
                    self.note_gpus(engine, root, "wait", 0..0, behind);
                }
                None
            }
        }
    }
    /// A root's allocation changed (`grant`, `wait` behind `behind`, `release`): one `gpu`
    /// entry on its log, with its envelope GPUs and width.
    fn note_gpus(&self, engine: &Engine, root: u64, state: &str, gpus: std::ops::Range<usize>, behind: Option<u64>) {
        let request = |seq: u64| engine.get(&seq.to_string()).ok()
            .and_then(|record| record.submission.map(|s| s.request_id)).unwrap_or_default();
        let width = self.allocations.width(root);
        let body = serde_json::json!({"gpu": {"state": state, "gpus": gpus.collect::<Vec<_>>(),
            "width": width, "behind": behind.map(request).unwrap_or_default()}});
        if let Err(error) = engine.append_gpu(&root.to_string(), &body) {
            tracing::warn!("gpu allocation entry: {error}");
        }
    }
    /// The GPUs with an executor of `plan` now, by envelope index (a lane's first).
    fn hosting(&self, plan: &str) -> Vec<usize> {
        let leaders = self.leaders.lock().unwrap();
        leaders.iter().filter(|(p, _)| p == plan).map(|(_, start)| *start).collect()
    }
    /// Whether a replica of `plan` fits on GPU `device` now (`fits`, its widest measured shape,
    /// a new process). Without a ledger, when no executor is there.
    fn fits(&self, plan: &str, device: usize) -> bool {
        match &self.devices[device].memory {
            Some(ledger) => ledger.fits(plan, None, true),
            None => !self.leaders.lock().unwrap().iter().any(|(_, start)| *start == device),
        }
    }
    /// Where `plan` runs now, with its GPUs' call slots: inside its root's `lane` when it has
    /// one, else (a standalone call) on GPUs no root holds or older root claims. A group takes
    /// a free contiguous lane; one GPU, in order: a free one with its idle executor, a free one
    /// where its learned call fits beside the tenants there (a replica when it runs elsewhere
    /// too), or, when it runs nowhere or inside its root's lane, any free one. None: it waits,
    /// and a standalone call claims the GPUs it waits for from later requests.
    fn place(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        plan: &Plan,
        (root, family): (u64, Option<&str>),
        lane: Option<std::ops::Range<usize>>,
        claims: &mut Claims,
    ) -> Option<(usize, Permit)> {
        let range = lane.clone().unwrap_or(0..self.devices.len());
        let degree = plan.degree as usize;
        let take = |start: usize, claims: &Claims| {
            self.permit(engine, Some(root), family, start..start + degree, claims).map(|permit| (start, permit))
        };
        if degree > 1 {
            let placed = (range.start..=range.end.saturating_sub(degree)).find_map(|start| take(start, claims));
            if placed.is_none() && lane.is_none() {
                self.allocations.claim(root, degree, claims);
            }
            return placed;
        }
        let hosts: Vec<usize> = self.hosting(&plan.construction).into_iter().filter(|d| range.contains(d)).collect();
        let placed = hosts
            .iter()
            .find_map(|device| take(*device, claims))
            .or_else(|| {
                let fit = |device: &usize| !hosts.contains(device) && self.fits(&plan.construction, *device);
                range.clone().filter(fit).find_map(|device| take(device, claims))
            })
            .or_else(|| {
                (hosts.is_empty() || lane.is_some()).then(|| range.clone().find_map(|device| take(device, claims))).flatten()
            });
        match &placed {
            None if lane.is_some() => {}
            None if hosts.is_empty() => claims.iter_mut().for_each(|claim| { claim.get_or_insert(root); }),
            None => hosts.iter().for_each(|device| { claims[*device].get_or_insert(root); }),
            Some((device, _)) if !hosts.contains(device) => {
                self.notes.note(serde_json::json!({"event": "place", "plan": plan.construction,
                    "gpu": self.devices[*device].entry, "running_on": hosts.iter()
                        .map(|d| &self.devices[*d].entry).collect::<Vec<_>>()}));
            }
            Some(_) => {}
        }
        placed
    }
    /// Run `record` on an executor of its plan, or leave it queued (false). A deviceless root
    /// registers its lane for its children's GPU steps while it runs; a GPU call runs on the
    /// GPUs `place` gives it, inside its root's lane when that root holds one.
    pub fn dispatch(
        self: &Arc<Self>,
        engine: &Arc<Engine>,
        record: &Execution,
        held: HeldGeneration,
        plan: Plan,
        claims: &mut Claims,
        calls: Option<Arc<crate::calls::Calls>>,
    ) -> io::Result<bool> {
        let root = record.root_seq();
        if plan.degree == 0 {
            let width = match record.invocation.gpus {
                0 => held.record.widest_group(self.width()) as usize,
                gpus => gpus as usize,
            };
            let root = (record.invocation.parent.is_empty() && self.width() > 0)
                .then(|| self.root(engine, record, width))
                .flatten();
            let pool = self.clone();
            return engine.dispatch_managed(&record.id, move |engine, id| {
                let result = pool.run(&engine, &id, held, plan, 0, calls.as_ref());
                if let Err(error) = &result {
                    settle(&engine, &id, error)?;
                }
                drop(root);
                if let Some(calls) = &calls {
                    calls.ended(&id);
                }
                result
            });
        }
        let lane = match self.allocations.registered(root) {
            true => match self.root_lane(engine, root, claims) {
                Some(lane) => Some(lane),
                None => return Ok(false),
            },
            false => None,
        };
        let Some((start, permit)) = self.place(engine, &plan, (root, record.family()), lane, claims) else {
            return Ok(false);
        };
        engine.dispatch_managed(&record.id, move |engine, id| {
            let pool = permit.pool.clone();
            let result = pool.run(&engine, &id, held, plan, start, calls.as_ref());
            if let Err(error) = &result {
                settle(&engine, &id, error)?;
            }
            if let Some(calls) = &calls {
                calls.ended(&id);
            }
            // The call slot is free: a warm set member this call pushed down may come back.
            drop(permit);
            pool.keep_all(&engine);
            result
        })
    }
    /// One line per Load in `loads.jsonl`: the executor's load facts, the host tier's, and
    /// the machine's and executor's RSS/PSS.
    fn record_load(
        &self,
        plan: &str,
        gpus: &str,
        took: std::time::Duration,
        loaded: &Frame,
        executor: u32,
        launch: &Launch,
    ) -> io::Result<()> {
        use cozy_platform::contain::host_memory::{process, ProcessMemory};
        #[derive(Serialize)]
        struct Line<'a> {
            plan: &'a str,
            gpus: &'a str,
            load_ms: f64,
            facts: &'a Option<device_executor::LoadFacts>,
            machine: ProcessMemory,
            executor: ProcessMemory,
            launch: &'a Launch,
        }
        let line = Line {
            plan,
            gpus,
            load_ms: took.as_secs_f64() * 1e3,
            facts: &loaded.facts,
            machine: process(std::process::id()).unwrap_or_default(),
            executor: process(executor).unwrap_or_default(),
            launch,
        };
        let mut bytes = serde_json::to_vec(&line)?;
        bytes.push(b'\n');
        fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("loads.jsonl"))?
            .write_all(&bytes)
    }
    /// One line per call in `invokes.jsonl`: wall time, whether it was the executor's first,
    /// and the executor's weight-movement facts and metrics.
    fn record_invoke(&self, plan: &str, gpus: &str, first: bool, took: std::time::Duration, reply: &Frame) {
        let line = serde_json::json!({
            "plan": plan,
            "gpus": gpus,
            "first": first,
            "invoke_ms": took.as_secs_f64() * 1e3,
            "memory": reply.memory,
            "metrics": reply.metrics,
        });
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("invokes.jsonl"))
            .and_then(|mut file| file.write_all(format!("{line}\n").as_bytes()));
        if let Err(error) = written {
            tracing::warn!("invokes.jsonl: {error}");
        }
    }

    /// A running parent's hint that it will call `plan` next (`model_prefetch`, H3 long-form's
    /// `prefetch(motion_segment)`): load it in the background where it fits beside the tenants
    /// already there (it never makes room), without waiting for the parent's own run to end.
    /// It takes the GPU's call slot like any call, so a child call in flight finishes first, and
    /// loads only inside its root's lane: once bound, until released.
    pub fn prefetch(self: &Arc<Self>, engine: &Arc<Engine>, held: HeldGeneration, plan: Plan, (root, family): (u64, Option<String>), owner: String, fetch: Arc<crate::access::Fetch>) {
        let (pool, engine) = (self.clone(), engine.clone());
        let started = std::thread::Builder::new()
            .name("executor-prefetch".into())
            .spawn(move || {
                let active = || pool.allocations.registered(root) && engine.get(&owner).is_ok_and(|record| {
                    record.state == State::Running && record.cancel_actor.is_none()
                });
                if !active() { return; }
                let asked = Instant::now();
                if let Some((zygote, start)) = pool.zygote(&held) {
                    if start {
                        zygote.set(pool.import_only(&held));
                    }
                    zygote.wait_started();
                }
                let (key, degree) = (plan.construction.clone(), plan.degree);
                if !pool.hosting(&key).is_empty() {
                    return pool.note_prewarm(&key, asked.elapsed(), Default::default(), &Ok("already loaded"));
                }
                // Queued requests of its root and older roots go first; a released lane ends
                // the hint.
                let (start, _permit) = loop {
                    if !active() { return; }
                    match pool.allocations.bound(root) {
                        Some(None) => return,
                        Some(Some(lane)) if !gpu_request_queued(&engine, Some(root)) => {
                            let mut claims = vec![None; pool.width()];
                            if let Some(placed) = pool.place(&engine, &plan, (root, family.as_deref()), Some(lane), &mut claims) {
                                break placed;
                            }
                        }
                        _ => {}
                    }
                    std::thread::sleep(std::time::Duration::from_millis(100));
                };
                if !active() { return; }
                let waited = asked.elapsed();
                let result = pool.prewarm(&engine, &held, plan, start, false, Some(fetch));
                pool.settle_lane(&key, start, degree);
                pool.note_prewarm(&key, waited, asked.elapsed() - waited, &result);
            });
        if let Err(error) = started {
            tracing::warn!("executor prefetch: {error}");
        }
    }

    /// One line per prewarm in `prewarm.jsonl`: what it did and how long it waited first.
    fn note_prewarm(
        &self,
        plan: &str,
        waited: std::time::Duration,
        took: std::time::Duration,
        result: &io::Result<&'static str>,
    ) {
        let line = serde_json::json!({
            "plan": plan,
            "outcome": result.as_ref().map_or_else(|e| format!("failed: {e}"), |o| o.to_string()),
            "waited_ms": waited.as_secs_f64() * 1e3,
            "took_ms": took.as_secs_f64() * 1e3,
        });
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("prewarm.jsonl"))
            .and_then(|mut file| file.write_all(format!("{line}\n").as_bytes()));
        if let Err(error) = written {
            tracing::warn!("prewarm.jsonl: {error}");
        }
    }

    /// Load `plan` on the lane from `start`, whose call slots the caller holds.
    fn prewarm(
        &self,
        engine: &Arc<Engine>,
        held: &HeldGeneration,
        plan: Plan,
        start: usize,
        member: bool,
        fetch: Option<Arc<crate::access::Fetch>>,
    ) -> io::Result<&'static str> {
        if self.sessions.lock().unwrap().contains_key(&session_key(&plan.construction, start)) {
            return Ok("already loaded");
        }
        // A warm set member may end idle executors outside every set; a prefetch only starts
        // where it fits. Neither touches a member or a running call.
        let fits = match member {
            true => self.make_room(&plan.construction, start, plan.degree, None, true, true)?,
            false => self.lane(start, plan.degree)?.iter().all(|d| d.memory.as_ref().is_none_or(|m| m.fits(&plan.construction, None, true))),
        };
        if !fits {
            return Ok("no GPU room beside the warm set and the running call");
        }
        let mut session = self.new_session(engine, held, &plan, start, |_, _| Ok(()))?;
        match self.call(engine, "", held, plan, &mut session, true, None, fetch) {
            Ok(_) => {
                self.keep_idle(session);
                Ok("loaded")
            }
            // A prewarm is no run: there is no triage to keep.
            Err(error) => Err(ended_with(engine, "", error, session.executor)),
        }
    }

    pub fn stop(&self) -> io::Result<()> {
        let sessions = {
            let mut sessions = self.sessions.lock().unwrap();
            self.stopped.store(true, Ordering::Release);
            std::mem::take(&mut *sessions)
        };
        let deviceless = std::mem::take(&mut *self.deviceless.lock().unwrap());
        for session in sessions.into_values().chain(deviceless) {
            let (plan, start, degree) = (session.plan.clone(), session.start, session.degree);
            session.executor.shutdown()?;
            self.ended(&plan, start, degree);
        }
        for zygote in std::mem::take(&mut *self.zygotes.lock().unwrap()).into_values() {
            if let ZygoteState::Ready(parent) = std::mem::take(&mut *zygote.state.lock().unwrap()) {
                parent.shutdown()?;
            }
        }
        let boots = std::mem::take(&mut *self.kernel_boots.lock().unwrap());
        for boot in boots.into_values().flatten() {
            boot.kill()?;
        }
        Ok(())
    }

    /// Compile a generation's kernels for every card of the envelope in the background
    /// (machine start, after an install, so beside the model download): the Runtime's
    /// `machine_kernels` under the generation's own seal, so each key is its executors' and
    /// a kernel already in the store is not built again. Once per generation and machine
    /// run. It opens no device; its builds are niced and share the host. An executor that
    /// needs a kernel still compiling waits for it.
    pub fn kernel_boot(self: &Arc<Self>, held: HeldGeneration) {
        if binds_models(&held.record.interface) {
            self.boot_kernels(held.record.identity, held.record.python, None);
        }
    }

    /// The same compile at machine start, before any install: the image's Runtime
    /// interpreter (its torch and toolkit, which package environments install from the same
    /// seed) running the executors' Runtime, unpacked from its wheel, so its keys are theirs.
    pub fn image_kernel_boot(self: &Arc<Self>) {
        let Some(image) = self.config.image_kernels.clone() else {
            return;
        };
        match unpack_runtime(&image.runtime_wheel, &self.root.join("kernel-runtime")) {
            Ok(runtime) => self.boot_kernels("image".into(), image.python, Some(runtime)),
            Err(error) => tracing::warn!("kernel boot: image Runtime: {error}"),
        }
    }

    fn boot_kernels(self: &Arc<Self>, generation: String, python: PathBuf, runtime: Option<PathBuf>) {
        {
            let mut boots = self.kernel_boots.lock().unwrap();
            if boots.contains_key(&generation) {
                return;
            }
            boots.insert(generation.clone(), None);
        }
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("kernel-boot".into())
            .spawn(move || {
                let began = Instant::now();
                let Some(launched) = pool
                    .upgrade()
                    .map(|pool| pool.launch_kernel_boot(&generation, &python, runtime.as_deref()))
                else {
                    return;
                };
                // Ended by its exit; killed only on a measured wedge of its process group.
                let ended = launched.and_then(|(mut child, boot, log)| {
                    let liveness = crate::process::Liveness::default();
                    let reaped = crate::process::reap(&boot, Some(&mut child), liveness)?;
                    Ok((reaped, log))
                });
                if let Some(pool) = pool.upgrade() {
                    if let Some(boot) = pool.kernel_boots.lock().unwrap().get_mut(&generation) {
                        *boot = None;
                    }
                    pool.note_kernel_boot(&generation, began.elapsed(), ended);
                }
            });
        if let Err(error) = started {
            tracing::warn!("kernel boot: {error}");
        }
    }

    fn launch_kernel_boot(
        &self,
        generation: &str,
        python: &Path,
        runtime: Option<&Path>,
    ) -> io::Result<(std::process::Child, cozy_platform::process::Exact, PathBuf)> {
        let devices: Vec<_> = self.devices.iter().map(|d| d.entry.as_str()).collect();
        let seal = Seal::prepare(
            &self.root,
            self.config.identity,
            &self.incarnation,
            generation,
            &devices.join(","),
        )?;
        let log = self.root.join(format!("kernel-boot.{generation}.log"));
        let output = File::create(&log)?;
        let scope = Arc::new(cozy_platform::contain::Scope::create(&cozy_platform::contain::namespace(
            self.root.parent().unwrap_or(&self.root),
        ))?);
        let launch = || {
            let mut command =
                crate::launch_identity::trampoline(python, self.config.identity, Some(&scope))?;
            match runtime {
                None => command.args(["-I", "-m", "cozy_runtime.internal.machine_kernels"]),
                Some(runtime) => command.args(["-I", "-c", RUNTIME_FIRST]).arg(runtime),
            };
            command
                .env_clear()
                .envs(seal.environment(&self.config.environment))
                .envs(scope.environment())
                .stdin(std::process::Stdio::null())
                .stdout(output.try_clone()?)
                .stderr(output.try_clone()?);
            crate::launch_identity::inherit(&mut command, &[seal.kernel_hold.as_ref()]);
            let child = self.launcher.spawn(command)?;
            let birth = Birth::of(child.id())?;
            let boot = cozy_platform::process::Exact::open(&birth)?
                .ok_or_else(|| io::Error::other("launched kernel boot has no exact birth"))?;
            io::Result::Ok((child, boot))
        };
        match launch() {
            Ok((child, boot)) => {
                let boot = boot.with_scope(Some(scope));
                let mut boots = self.kernel_boots.lock().unwrap();
                boots.insert(generation.to_string(), Some(boot.try_clone()?));
                Ok((child, boot, log))
            }
            Err(error) => {
                let _ = scope.end();
                Err(error)
            }
        }
    }

    /// One line per kernel boot in `kernel-boot.jsonl`: how it ended and what each kernel
    /// said last (`kernel-boot.<generation>.log` holds its progress too).
    fn note_kernel_boot(
        &self,
        generation: &str,
        took: std::time::Duration,
        ended: io::Result<(crate::process::Reaped, PathBuf)>,
    ) {
        let mut line = serde_json::json!({
            "generation": generation,
            "took_ms": took.as_secs_f64() * 1e3,
        });
        match ended {
            Ok((reaped, log)) => {
                let kernels: Vec<serde_json::Value> = fs::read_to_string(log)
                    .unwrap_or_default()
                    .lines()
                    .filter_map(|row| serde_json::from_str::<serde_json::Value>(row).ok())
                    .filter(|row| row.is_object() && row["state"] != "compiling")
                    .collect();
                line["status"] = reaped.status.to_string().into();
                line["killed"] = reaped.killed.into();
                line["stragglers"] = reaped.stragglers.into();
                line["kernels"] = kernels.into();
            }
            Err(error) => line["failed"] = error.to_string().into(),
        }
        tracing::info!("kernel boot: {line}");
        let written = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.root.join("kernel-boot.jsonl"))
            .and_then(|mut file| file.write_all(format!("{line}\n").as_bytes()));
        if let Err(error) = written {
            tracing::warn!("kernel-boot.jsonl: {error}");
        }
    }

    /// Start a generation's import-only executor in the background for a call being prepared,
    /// so its imports overlap the rest of the preparation.
    pub fn prespawn(self: &Arc<Self>, held: HeldGeneration) {
        if self.zygotes.lock().unwrap().contains_key(&parent_app(&held)) {
            return;
        }
        if !self.parent_fits(&held) {
            tracing::info!(
                "executor prespawn skipped for {}: no host room",
                held.record.identity
            );
            return;
        }
        let Some((zygote, true)) = self.zygote(&held) else {
            return;
        };
        *zygote.used.lock().unwrap() = Some(Instant::now());
        let pool = Arc::downgrade(self);
        let started = std::thread::Builder::new()
            .name("executor-prespawn".into())
            .spawn(move || {
                let state = match pool.upgrade() {
                    Some(pool) => pool.import_only(&held),
                    None => ZygoteState::Off("the GPU pool ended".into()),
                };
                zygote.set(state);
            });
        if let Err(error) = started {
            tracing::warn!("executor prespawn: {error}");
        }
    }

    /// Whether the host has room for `held`'s parent beside what it holds now: the largest
    /// private bytes a live parent measures. A prespawn never makes room; a request that needs
    /// a parent always gets one.
    fn parent_fits(&self, _held: &HeldGeneration) -> bool {
        let parents: Vec<Arc<Zygote>> = self.zygotes.lock().unwrap().values().cloned().collect();
        let need = parents.iter().map(|parent| parent.private_bytes()).max().filter(|b| *b > 0);
        self.host_room_now().is_none_or(|room| room >= need.unwrap_or(UNMEASURED_PARENT))
    }

    /// Drop a dead parent, if it is still the generation's: the next launch starts another.
    fn forget_parent(&self, held: &HeldGeneration, dead: &Arc<Zygote>) {
        let mut zygotes = self.zygotes.lock().unwrap();
        let key = parent_app(held);
        if zygotes
            .get(&key)
            .is_some_and(|current| Arc::ptr_eq(current, dead))
        {
            zygotes.remove(&key);
        }
    }

    /// End a parent with no live child, for host room: one of a generation outside every
    /// warm set before a member's (`spare`: never a member's), least recently used first.
    /// Returns the bytes it held privately (at least 1 when one ended unmeasured), 0 when
    /// there is none to end.
    fn end_idle_parent(&self, spare: bool) -> u64 {
        let kept: std::collections::BTreeSet<(String, String)> = {
            let sets = self.members.lock().unwrap();
            let members = sets.values().flatten().filter(|member| member.level >= Level::Imported);
            members.map(|member| parent_app(&member.held)).collect()
        };
        let victim = {
            let mut zygotes = self.zygotes.lock().unwrap();
            let chosen = zygotes
                .iter()
                .filter(|(generation, zygote)| zygote.childless() && !(spare && kept.contains(*generation)))
                .min_by_key(|(generation, zygote)| (kept.contains(*generation), *zygote.used.lock().unwrap()))
                .map(|(generation, _)| generation.clone());
            chosen.and_then(|generation| zygotes.remove(&generation))
        };
        let Some(zygote) = victim else {
            return 0;
        };
        let freed = zygote.private_bytes();
        if let ZygoteState::Ready(parent) = std::mem::take(&mut *zygote.state.lock().unwrap()) {
            if let Err(error) = parent.shutdown() {
                tracing::warn!("ending an idle import-only executor: {error}");
            }
        }
        freed.max(1)
    }

    /// The generation's import-only executor, and whether the caller must start it. None
    /// when this pool spawns every executor (no GPU, or no prespawn) or the generation
    /// constructs no model.
    fn zygote(&self, held: &HeldGeneration) -> Option<(Arc<Zygote>, bool)> {
        let stopped = self.stopped.load(Ordering::Acquire);
        if stopped || !self.config.prespawn || self.devices.is_empty() || !binds_models(&held.record.interface) {
            return None;
        }
        let mut zygotes = self.zygotes.lock().unwrap();
        let key = parent_app(held);
        if let Some(zygote) = zygotes.get(&key) {
            return Some((zygote.clone(), false));
        }
        let zygote = Arc::new(Zygote::default());
        zygotes.insert(key, zygote.clone());
        Some((zygote, true))
    }

    /// Spawn an executor of `held` and import torch, the Runtime and the package with no
    /// device (`Start.import_only`), to fork executors from.
    fn import_only(&self, held: &HeldGeneration) -> ZygoteState {
        let started = (|| {
            let (root, socket, directory) = self.executor_endpoint()?;
            let mut executor = DeviceExecutor::spawn_owned(
                self.executor_config(held, root, socket, 0, self.devices.len().min(1) as u32)?,
                &self.launcher,
                |_, _| Ok(()),
            )?;
            executor.retain_until_exit(directory);
            let interface = self.interface_file(&executor, held)?;
            let reply = executor.command(
                &DeviceCommand::Start {
                    devices: self.lane_devices(0, self.devices.len().min(1) as u32)?,
                    application: held.record.application.clone(),
                    call_interfaces: crate::calls::call_interfaces(held, &interface),
                    constructions: vec![],
                    package_interface: interface,
                    sequence_parallel_degree: 1,
                    import_only: true,
                    ledgers: 0,
                    memory_mode: "",
                },
                &mut device_executor::Baseline,
            )?;
            if !reply.ok {
                executor.shutdown()?;
                return Ok(ZygoteState::Failed(format!(
                    "import-only start refused: {}: {}",
                    reply.code, reply.detail
                )));
            }
            io::Result::Ok(ZygoteState::Ready(Box::new(executor)))
        })();
        started
            .unwrap_or_else(|error| ZygoteState::Failed(format!("import-only executor: {error}")))
    }

    /// A new executor root and its socket path. The socket is named through this process's
    /// descriptor of the root (Linux `sun_path` limit), or inside it for another identity.
    fn executor_endpoint(&self) -> io::Result<(PathBuf, PathBuf, File)> {
        let root = self.root.join(uuid::Uuid::new_v4().simple().to_string());
        fs::create_dir(&root)?;
        let directory = cozy_platform::durable::open_dir(&root)?;
        let socket = if self.config.identity.is_some() {
            // Another UID cannot traverse this owner's /proc/fd magic link.
            // A deliberately short owned state root is required for this operation.
            let socket = root.join("executor");
            if socket.as_os_str().len() > 107 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "configured identity needs a short owned executor socket path (Linux sun_path)",
                ));
            }
            socket
        } else {
            // Linux names it through this process's descriptor of the root; elsewhere it is the
            // root's own, or a short private one when that is too long.
            #[cfg(target_os = "linux")]
            let socket = PathBuf::from(format!(
                "/proc/{}/fd/{}/executor",
                std::process::id(),
                std::os::fd::AsRawFd::as_raw_fd(&directory)
            ));
            #[cfg(not(target_os = "linux"))]
            let socket = crate::device_executor::socket_path(&root, "executor")?;
            socket
        };
        Ok((root, socket, directory))
    }

    fn executor_config(
        &self,
        held: &HeldGeneration,
        root: PathBuf,
        socket: PathBuf,
        start: usize,
        degree: u32,
    ) -> io::Result<ExecutorConfig> {
        let mut seal = Seal::prepare(
            &self.root,
            self.config.identity,
            &self.incarnation,
            &held.record.identity,
            &self.lane_devices(start, degree)?,
        )?;
        // A group's NCCL seal (NVLS off, peer memory only over NVLink).
        seal.group = degree > 1;
        seal.threads = self.config.threads;
        Ok(ExecutorConfig {
            python: held.record.python.clone(),
            root,
            socket,
            environment: match &self.config.toolchain {
                Some(dir) => crate::machine::toolchain::environment(&self.config.environment, dir, ""),
                None => self.config.environment.clone(),
            },
            seal,
            generation_hold: Some(held.retention()),
            identity: self.config.identity,
            cgroup_namespace: Some(cozy_platform::contain::namespace(
                self.root.parent().unwrap_or(&self.root),
            )),
        })
    }

    /// The installed package interface, written where the executor (its identity) reads it.
    fn interface_file(
        &self,
        executor: &DeviceExecutor,
        held: &HeldGeneration,
    ) -> io::Result<PathBuf> {
        let path = executor.root_path().join("package-interface.json");
        fs::write(&path, serde_json::to_vec(&held.record.interface)?)?;
        if let Some(identity) = self.config.identity {
            identity.readable(&path)?;
        }
        Ok(path)
    }
    fn run(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
        plan: Plan,
        start: usize,
        calls: Option<&Arc<crate::calls::Calls>>,
    ) -> io::Result<()> {
        let (key, degree) = (plan.construction.clone(), plan.degree);
        let result = self.run_on(engine, id, held, plan, start, calls);
        // Broken/failed exchanges close the owner channel. Sources and the journal
        // reservation survive until the exact receiver has exited.
        self.settle_lane(&key, start, degree);
        result
    }

    /// One call on the lane from `start` (none, deviceless): an idle executor of its plan there
    /// leaves the pool for the call (or one starts), and comes back once it served it.
    fn run_on(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: HeldGeneration,
        plan: Plan,
        start: usize,
        calls: Option<&Arc<crate::calls::Calls>>,
    ) -> io::Result<()> {
        // A cancel ends the call's children with it.
        let stops = |cancel: Cancellation| {
            let (request, calls) = (id.to_string(), calls.map(Arc::downgrade));
            Arc::new(move || {
                let canceled = cancel.cancel(&request);
                if let Some(calls) = calls.as_ref().and_then(std::sync::Weak::upgrade) {
                    calls.stop_children(&request);
                }
                canceled
            })
        };
        let (key, degree) = (session_key(&plan.construction, start), plan.degree);
        let retained = match degree {
            0 => {
                let mut idle = self.deviceless.lock().unwrap();
                idle.retain(|session| !session.executor.birth.ended().unwrap_or(true));
                let newest = idle.iter().rposition(|session| session.plan == plan.construction);
                newest.map(|at| idle.remove(at))
            }
            _ => {
                self.reap();
                self.sessions.lock().unwrap().remove(&key)
            }
        };
        let session = match retained {
            Some(session) => {
                engine.register_managed(
                    id,
                    session.executor.birth.clone(),
                    stops(session.executor.cancellation()),
                )?;
                session
            }
            None => {
                // Room for its process and its widest measured request before it exists.
                self.make_room(&plan.construction, start, degree, None, true, false)?;
                let on_birth = |birth: &Birth, cancel: &Cancellation| {
                    engine.register_managed(id, birth.clone(), stops(cancel.clone()))
                };
                match self.new_session(engine, &held, &plan, start, on_birth) {
                    Err(error) if error.kind() == io::ErrorKind::Unsupported => {
                        engine.finish(id, Outcome::Failed(Failure::machine("executor_unsupported", error.to_string())))?;
                        return Ok(());
                    }
                    session => session?,
                }
            }
        };
        // A development Runtime build is named by its build, not only its version.
        let hello = &session.executor.hello;
        let build = &held.record.runtime_build;
        let runtime = match build.split_once('+') {
            Some((version, _)) if version == hello.runtime_version => build.clone(),
            _ => hello.runtime_version.clone(),
        };
        let facts = crate::journal::ExecutorFacts::observed(session.executor.birth.pid, runtime, hello.tensorfs_version.clone());
        if !engine.authorize_managed(id, Some(facts))? {
            engine.finish_stopped(id)?;
            self.keep_idle(session);
            return Ok(());
        }
        let fresh = plan.clone();
        let error = {
            let mut session = session;
            let fetch = calls.map(|calls| engine.get(id).map(|record| calls.fetch(&record, &plan))).transpose()?;
            match self.call(engine, id, &held, plan, &mut session, false, calls, fetch) {
                Ok(true) => {
                    self.keep_idle(session);
                    return Ok(());
                }
                // Not reusable: gone (exit observed) before its context is released.
                Ok(false) => return session.executor.terminate().map(drop),
                Err(error) if undelivered(&error) => ending(error, session.executor),
                Err(error) => return Err(ended_with(engine, id, error, session.executor)),
            }
        }; // the rest of the ended session (sources, grants) goes here
        // Never started: once that exit is observed, this dispatch runs it on a fresh executor
        // (cold, so a second loss is FAILED). An unproven exit stays charged.
        if !undelivered(&error) {
            return Err(error);
        }
        tracing::warn!("execution {id}: {error}; starting it on a fresh executor");
        self.ended(&fresh.construction, start, fresh.degree);
        if !engine.redeliver(id)? {
            return Ok(()); // canceled meanwhile
        }
        self.run_on(engine, id, held, fresh, start, calls)
    }

    /// Launch an executor for `plan` (fork from its generation's import-only executor, else
    /// spawn) and open its selected metadata sources and CPU-buffer grants.
    fn new_session(
        &self,
        engine: &Arc<Engine>,
        held: &HeldGeneration,
        plan: &Plan,
        start: usize,
        on_birth: impl Fn(&Birth, &Cancellation) -> io::Result<()>,
    ) -> io::Result<Session> {
        let launched = Instant::now();
        self.leaders.lock().unwrap().insert((plan.construction.clone(), start));
        let (root, socket, directory) = self.executor_endpoint()?;
        let config = self.executor_config(held, root, socket, start, plan.degree)?;
        // A forked child takes its lane and seal with its environment, so a group forks from
        // the same import-only parent (sealed to the first GPU) as a single GPU does.
        let mut config = Some(config);
        let mut forked = None;
        // A parent found dead is replaced at once: importing a new one costs this executor
        // what a spawn would, and the next ones fork again.
        for _ in 0..2 {
            let Some((zygote, start)) = self.zygote(held) else {
                break;
            };
            if start {
                zygote.set(self.import_only(held));
            }
            match zygote.fork(config.take().expect("one config per launch"), &on_birth)? {
                Forked::Ready(executor) => {
                    forked = Some(*executor);
                    break;
                }
                Forked::Refused(returned, reason) => {
                    tracing::warn!("executor fork refused, spawning: {reason}");
                    config = Some(*returned);
                    if zygote.failed() {
                        self.forget_parent(held, &zygote);
                    }
                    break;
                }
                Forked::Lost(returned, reason) => {
                    tracing::warn!("import-only executor lost ({reason}); starting another");
                    config = Some(*returned);
                    self.forget_parent(held, &zygote);
                }
            }
        }
        let (mut executor, mode) = match forked {
            Some(executor) => (executor, "fork"),
            None => (
                DeviceExecutor::spawn_owned(
                    config.expect("a refused fork returns its config"),
                    &self.launcher,
                    &on_birth,
                )?,
                "spawn",
            ),
        };
        let launch = Launch {
            mode,
            ms: launched.elapsed().as_secs_f64() * 1e3,
            start_ms: 0.0,
            start: vec![],
        };
        executor.retain_until_exit(directory);
        executor.retain_until_exit(WakeOnExit(Arc::downgrade(engine)));
        executor.keep_memo(self.memo.clone());
        if plan.degree > 0 {
            let exact = Exact::open(&executor.birth)?.ok_or_else(|| io::Error::other("executor exited before its start"))?;
            let tenant = Tenant { plan: plan.construction.clone(), start, exact };
            self.tenants.lock().unwrap().insert(executor.birth.pid, tenant);
        }
        let selections = plan.selections();
        let id = self.serving.next.fetch_add(1, Ordering::Relaxed);
        self.serving.held.lock().unwrap().insert(
            id,
            selections.iter().map(|s| s.manifest.clone()).collect(),
        );
        // Kept until the process is seen to exit, not until its session is dropped: an
        // executor whose exit is unproven may still read these files.
        executor.retain_until_exit(ServingHold {
            serving: self.serving.clone(),
            id,
        });
        let sources = Arc::new(ModelSources::open_shared(self.store.clone(), &selections)?);
        Ok(Session {
            plan: plan.construction.clone(),
            start,
            generation: plan.generation.clone(),
            application: held.record.application.clone(),
            degree: plan.degree,
            followers: vec![],
            loaded: false,
            executor,
            used: Instant::now(),
            sources,
            launch,
            invoked: false,
        })
    }

    /// Load (once) and invoke (unless `load_only`: a prewarm, no request). Ok(false): the
    /// executor must not be reused.
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        engine: &Arc<Engine>,
        id: &str,
        held: &HeldGeneration,
        plan: Plan,
        session: &mut Session,
        load_only: bool,
        calls: Option<&Arc<crate::calls::Calls>>,
        fetch: Option<Arc<crate::access::Fetch>>,
    ) -> io::Result<bool> {
        let (start, degree) = (session.start, plan.degree);
        let construction = plan.construction.clone();
        let gpus = self.lane(start, degree)?;
        let _active: Vec<_> = gpus.iter().filter_map(|d| d.memory.as_ref()).map(GpuLedger::active).collect();
        let retained = session.loaded;
        if retained {
            session.executor.begin_request();
        }
        // A group runs one rank per GPU; a plan that constructs nothing holds its GPUs in one
        // process (the Runtime refuses a group without constructed models).
        let ranks = if plan.slots.is_empty() { 1 } else { degree.max(1) };
        let lane = self.lane_devices(start, degree)?;
        let mut callbacks = Callbacks {
            engine,
            id,
            completed: 0,
            deposits: &self.deposits,
            sources: &session.sources,
            pool: self,
            store: &self.store,
            spool: None,
            caller: None,
            weights: None,
            fetch: fetch.as_ref(),
            raw: &plan.sources,
        };
        if !session.loaded {
            let interface_path = self.interface_file(&session.executor, held)?;
            let starting = Instant::now();
            // Each GPU's ledger and its eventfd follow the Start, in lane order: every rank claims
            // its own row.
            let grants = match gpus.iter().all(|d| d.memory.is_some()) {
                true => gpus.iter().filter_map(|d| d.memory.as_ref()).map(GpuLedger::grant).collect::<io::Result<Vec<_>>>()?,
                false => vec![],
            };
            let ledgers: Vec<File> = grants.into_iter().flatten().collect();
            // Rank 0 spawns and forms every follower inside this command; its watch meters
            // the whole group's work, so formation ends only on measured lack of progress.
            let started = command_ok(session.executor.command_with(
                &DeviceCommand::Start {
                    devices: lane.clone(),
                    application: held.record.application.clone(),
                    package_interface: interface_path.clone(),
                    sequence_parallel_degree: ranks,
                    import_only: false,
                    call_interfaces: crate::calls::call_interfaces(held, &interface_path),
                    constructions: constructions(&plan),
                    ledgers: (ledgers.len() / 2) as u32,
                    memory_mode: match self.config.host.mode {
                        cozy_platform::contain::host_pressure::HostMode::Shared => "shared",
                        cozy_platform::contain::host_pressure::HostMode::Dedicated => "dedicated",
                    },
                },
                &ledgers,
                &mut callbacks,
            )?)?;
            session.launch.start_ms = starting.elapsed().as_secs_f64() * 1e3;
            session.launch.start =
                serde_json::from_value(started.stages.clone()).unwrap_or_default();
            if started.follower_pids.len() + 1 != ranks as usize {
                return Err(io::Error::other(Refused {
                    code: "group_unformed".into(),
                    detail: format!(
                        "a {degree}-GPU start reported followers {:?}",
                        started.follower_pids
                    ),
                }));
            }
            // Only processes in rank 0's own group are taken as its followers: each is its own
            // GPU's tenant, named in that GPU's ledger by its own pid.
            let members = cozy_platform::process::group_members(&session.executor.birth);
            session.followers = started
                .follower_pids
                .iter()
                .filter_map(|pid| members.iter().find(|m| m.pid == *pid))
                .filter_map(|birth| Exact::open(birth).ok().flatten())
                .collect();
            for follower in &session.followers {
                let tenant = Tenant { plan: construction.clone(), start, exact: follower.try_clone()? };
                self.tenants.lock().unwrap().insert(follower.birth().pid, tenant);
            }
            let load = match plan.slots.first() {
                // A call that constructs nothing: its weightless construction.
                None => DeviceCommand::Load {
                    construction: construction.clone(),
                    devices: lane.clone(),
                    sequence_parallel_degree: ranks,
                    binding: Box::new(Binding {
                        application: held.record.application.clone(),
                        package_interface: interface_path.to_string_lossy().into_owned(),
                        ..Binding::default()
                    }),
                    models: vec![],
                    attention_pin: String::new(),
                },
                Some(first) => {
                    let store = self.store.root().to_string_lossy().into_owned();
                    let load = |slot: &PlanSlot| {
                        let mut binding = slot.binding.clone();
                        binding.package_interface = interface_path.to_string_lossy().into();
                        binding.store = store.clone();
                        ModelLoad { binding }
                    };
                    let first = load(first);
                    // The one-model form names its model alone.
                    let models = match plan.slots.len() {
                        1 => vec![],
                        _ => plan.slots.iter().map(load).collect(),
                    };
                    DeviceCommand::Load {
                        construction: construction.clone(),
                        devices: lane.clone(),
                        sequence_parallel_degree: ranks,
                        binding: Box::new(first.binding),
                        models,
                        // The request it loads for names a kernel: its build is imported first.
                        attention_pin: match load_only {
                            true => String::new(),
                            false => engine.get(id)?.invocation.attention_kernel,
                        },
                    }
                }
            };
            let started = std::time::Instant::now();
            let loaded = command_ok(session.executor.command(&load, &mut callbacks)?)?;
            self.record_load(
                &construction,
                &lane,
                started.elapsed(),
                &loaded,
                session.executor.birth.pid,
                &session.launch,
            )?;
            command_ok(session.executor.command(
                &DeviceCommand::Activate {
                    construction: construction.clone(),
                },
                &mut callbacks,
            )?)?;
            session.loaded = true;
        }
        if load_only {
            return Ok(true);
        }
        let record = engine.get(id)?;
        let prepared = session.executor.command(
            &DeviceCommand::PrepareRequest {
                request_id: id.into(),
                construction: construction.clone(),
                entrypoint: plan.entrypoint.clone(),
                payload: record.invocation.input.clone(),
                attention_kernel: record.invocation.attention_kernel.clone(),
                bits: record.invocation.bits,
                input_metadata: record
                    .invocation
                    .inputs
                    .iter()
                    .map(|i| (i.input_id.clone(), serde_json::json!({"input_id":i.input_id,"media_type":i.media_type,"digest":i.digest,"length":i.length,"order":i.order})))
                    .collect(),
            },
            &mut callbacks,
        );
        // A retained executor that went away while idle: PrepareRequest enters no handler,
        // so the attempt never started (see `run_on`).
        let prepared = prepared.map_err(|error| match retained && channel_lost(&error) {
            true => io::Error::other(Undelivered(format!(
                "retained executor gone before the request reached it: {error}"
            ))),
            false => error,
        })?;
        // Its pin needs another build of a kernel than this executor imported: the request
        // never started, so a fresh executor, loaded with the pin, runs it.
        if retained && !prepared.ok && prepared.code == "attention_kernel_needs_executor" {
            return Err(io::Error::other(Undelivered(prepared.detail.clone())));
        }
        if !prepared.ok
            && !matches!(
                prepared.code.as_str(),
                "executor_not_ready" | "poisoned_generation"
            )
        {
            // Canceled while it waited for its attention kernel's compile (Runtime 0.23.4).
            if prepared.terminal == "canceled" && engine.get(id)?.cancel_actor.is_some() {
                engine.finish(id, Outcome::Canceled)?;
                return Ok(true);
            }
            // A pre-entry refusal leaves the executor Ready; the run ends with its reason.
            let terminal = if prepared.terminal.is_empty() {
                "refused"
            } else {
                &prepared.terminal
            };
            let failure =
                Failure::executor(terminal, &prepared.origin, &prepared.code, &prepared.detail);
            keep_triage(
                engine,
                id,
                &session.executor,
                &device_executor::Outcome {
                    terminal: terminal.into(),
                    origin: prepared.origin.clone(),
                    code: prepared.code.clone(),
                    message: prepared.detail.clone(),
                    traceback: prepared.traceback.clone(),
                },
            );
            engine.finish(id, Outcome::Failed(failure))?;
            return Ok(true);
        }
        let prepared = command_ok(prepared)?;
        // Its measured peak is recorded under its shape: room for it, and the key named to varena.
        let shape = shape(&prepared.features);
        self.make_room(&construction, start, degree, Some(&shape), false, false)?;
        let _dispatched = Dispatched::new(gpus, id, &format!("{construction}|{shape}"));
        let spool = if let Some(identity) = self.config.identity {
            // Keep Journal/results/admin paths private to the core. A separate peer-owned
            // spool lives only inside this executor's already authorized output directory.
            let spool = session.executor.root_path().join(format!("output-{id}"));
            fs::create_dir(&spool)?;
            identity.own(&spool)?;
            spool
        } else {
            engine.staging(id)?
        };
        let _spool = crate::execution::Spool(spool.clone());
        let inputs = stage_inputs(
            &self.store,
            self.config.identity,
            &spool,
            &record.invocation.inputs,
        )?;
        callbacks.spool = Some(spool.clone());
        let document = held.record.app(&record.invocation.module)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "installed output App is absent"))?.2;
        // The attempt's calls reach their records; its sources and outputs, its weights grant.
        let interface = session.executor.root_path().join("package-interface.json");
        let caller = calls.map(|calls| calls.caller(&record, held, document, &interface, &spool).0);
        callbacks.caller = caller.as_ref();
        let grant = match (calls, &fetch) {
            (Some(calls), Some(fetch)) => calls.grant(engine, &record, &plan, document, &spool, fetch)?,
            _ => None,
        };
        callbacks.weights = calls.zip(grant.as_ref()).map(|(calls, grant)| (calls.weights(), grant));
        session.executor.waits = caller.as_ref().map(|caller| caller.waits(engine));
        let invoked = Instant::now();
        let first = !session.invoked;
        session.invoked = true;
        let reply = session.executor.command(
            &DeviceCommand::Invoke {
                max_output_bytes: crate::output_capacity::for_callable(document, "entrypoints", &plan.entrypoint)?,
                request_id: id.into(),
                construction: construction.clone(),
                entrypoint: plan.entrypoint.clone(),
                spool: spool.clone(),
                deadline_s: None,
                attention_kernel: record.invocation.attention_kernel.clone(),
                bits: record.invocation.bits,
                inputs: inputs.inputs,
                trees: inputs.trees,
                sources: plan.sources.iter().map(|s| (s.parameter.clone(), s.manifest.clone())).collect(),
                weights: grant.as_ref().map(|grant| grant.outputs.clone()).unwrap_or_default(),
            },
            &mut callbacks,
        );
        session.executor.waits = None;
        let reply = reply?;
        self.record_invoke(&construction, &lane, first, invoked.elapsed(), &reply);
        record_measurements(engine, id, &plan, &reply);
        if reply.attention_applied {
            if let Err(error) = engine.apply_attention(id) {
                tracing::warn!("run {id}: applied attention pin not recorded: {error}");
            }
        }
        // Rank 0 answered, so a follower that ended during the call is the group's first fault.
        let lost = self.lost_followers(start, degree, &session.followers);
        if !reply.ok || !reply.quiescent || !reply.poisoned.is_empty() || !lost.is_empty() {
            // The author's failure keeps its traceback and origin in triage; the executor's
            // end would record only "executor_ended".
            if let Some(outcome) = reply.outcome.as_ref().filter(|o| !o.traceback.is_empty()) {
                keep_triage(engine, id, &session.executor, outcome);
            }
            // The run ends once the executor is gone; its own reason travels with it, typed.
            return Err(io::Error::other(unsettled(&reply, &lost)));
        }
        let outcome = reply
            .outcome
            .as_ref()
            .ok_or_else(|| io::Error::other("device outcome absent"))?;
        match outcome.terminal.as_str() {
            "succeeded" => {
                let result = (|| {
                    let (value, bindings) =
                        device_executor::postprocess(&session.executor.codec(), &spool, &reply)?;
                    let adjusted = device_executor::adjustments(id, &reply);
                    engine.managed_result(id, &spool, value, output_bindings(bindings), adjusted)
                })();
                if let Err(error) = result {
                    // The invocation is already quiescent; a codec/custody
                    // failure cannot cause authored work to be dispatched again.
                    engine.finish(
                        id,
                        Outcome::Failed(
                            crate::journal::Failure::custody(&error.to_string()),
                        ),
                    )?;
                }
            }
            "canceled" if engine.get(id)?.cancel_actor.is_some() => {
                engine.finish(id, Outcome::Canceled)?;
            }
            _ => {
                keep_triage(engine, id, &session.executor, outcome);
                engine.finish(
                    id,
                    Outcome::Failed(
                        crate::journal::Failure::executor(
                            &outcome.terminal,
                            &outcome.origin,
                            &outcome.code,
                            &outcome.message,
                        ),
                    ),
                )?;
            }
        }
        Ok(true)
    }

    /// Followers whose process has ended, as people name their GPUs ("GPU 1's process 3224
    /// ended"); empty while the whole group lives.
    fn lost_followers(&self, start: usize, degree: u32, followers: &[cozy_platform::process::Exact]) -> String {
        let lane = self.lane(start, degree).unwrap_or_default();
        followers
            .iter()
            .enumerate()
            .filter(|(_, follower)| follower.ended())
            .map(|(index, follower)| {
                let gpu = lane.get(index + 1).map_or("?", |d| d.entry.as_str());
                format!("GPU {gpu}'s process {} ended", follower.birth().pid)
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// `machine_kernels` from the Runtime tree named by the first argument, ahead of the
/// interpreter's own Runtime; its builders inherit that path.
const RUNTIME_FIRST: &str = "import runpy, sys; sys.path.insert(0, sys.argv.pop(1)); \
    runpy.run_module('cozy_runtime.internal.machine_kernels', run_name='__main__')";

/// A Runtime wheel's importable tree, unpacked once per wheel digest under `into`.
fn unpack_runtime(wheel: &Path, into: &Path) -> io::Result<PathBuf> {
    let bytes = fs::read(wheel)?;
    let tree = into.join(&tensorfs_core::sha256::hex_digest(&bytes)[..32]);
    if tree.is_dir() {
        return Ok(tree);
    }
    let staging = into.join(format!(".{}", uuid::Uuid::new_v4().simple()));
    let mut archive = zip::ZipArchive::new(io::Cursor::new(bytes)).map_err(io::Error::other)?;
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index).map_err(io::Error::other)?;
        // Scripts and data (the bundled machine) are not imported.
        let Some(name) = entry.enclosed_name().filter(|name| {
            !name.iter().next().is_some_and(|top| top.to_string_lossy().ends_with(".data"))
        }) else {
            continue;
        };
        let path = staging.join(name);
        if entry.is_dir() {
            fs::create_dir_all(&path)?;
            continue;
        }
        fs::create_dir_all(path.parent().unwrap_or(&staging))?;
        io::copy(&mut entry, &mut File::create(&path)?)?;
    }
    match fs::rename(&staging, &tree) {
        Err(_) if tree.is_dir() => fs::remove_dir_all(&staging).map(|()| tree),
        renamed => renamed.map(|()| tree),
    }
}

/// Whether any entrypoint of an installed interface constructs a model.
fn binds_models(interface: &serde_json::Value) -> bool {
    interface
        .get("entrypoints")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|entries| {
            entries.iter().any(|entry| {
                entry
                    .get("models")
                    .and_then(serde_json::Value::as_array)
                    .is_some_and(|models| models.iter().any(constructed))
            })
        })
}

/// An error ends its executor; the exact exit is observed before the run settles, and a
/// kill's measurement joins the reason.
/// A request's shape key: its normalized features (PrepareRequest's), hashed to 16 hex digits.
fn shape(features: &BTreeMap<String, serde_json::Value>) -> String {
    let canonical = crate::boundary_json::exact(&serde_json::json!(features));
    tensorfs_core::sha256::hex_digest(&canonical)[..16].to_string()
}

/// A request named to varena on every GPU of its lane (`<package>|<shape>`, the key its peak is
/// recorded under), until it ends.
struct Dispatched<'a> {
    gpus: &'a [Device],
    id: String,
}
impl<'a> Dispatched<'a> {
    fn new(gpus: &'a [Device], id: &str, key: &str) -> Self {
        for ledger in gpus.iter().filter_map(|d| d.memory.as_ref()) {
            if let Err(error) = ledger.dispatch(id, key) {
                tracing::warn!("memory: request {id} runs unrecorded: {error}");
            }
        }
        Self { gpus, id: id.into() }
    }
}
impl Drop for Dispatched<'_> {
    fn drop(&mut self) {
        self.gpus.iter().filter_map(|d| d.memory.as_ref()).for_each(|ledger| ledger.undispatch(&self.id));
    }
}

fn ended_with(engine: &Engine, id: &str, error: io::Error, executor: DeviceExecutor) -> io::Error {
    let (pid, stderr_tail) = (executor.birth.pid, executor.stderr_tail());
    let error = ending(error, executor);
    keep_lifecycle_triage(engine, id, pid, &error.to_string(), &stderr_tail);
    error
}

fn ending(error: io::Error, executor: DeviceExecutor) -> io::Error {
    match executor.terminate() {
        Ok(ended) => with_end(error, &ended),
        Err(unproven) => io::Error::other(format!("{error}; executor exit unproven: {unproven}")),
    }
}

/// A kill's measurement joins the reason, and so does the exit of an executor that ended on
/// its own mid-command: its signal or exit code is then the only cause there is.
fn with_end(error: io::Error, ended: &device_executor::Ended) -> io::Error {
    let why = match (&ended.killed, channel_lost(&error)) {
        (Some(killed), _) => killed.clone(),
        (None, true) => format!("the executor ended on its own ({})", ended.status),
        (None, false) => return error,
    };
    match refused(&error) {
        Some(refusal) => io::Error::other(Refused {
            code: refusal.code.clone(),
            detail: format!("{}; {why}", refusal.detail),
        }),
        None if undelivered(&error) => io::Error::other(Undelivered(format!("{error}; {why}"))),
        None => io::Error::other(format!("{error}; {why}")),
    }
}

/// Why an invoke ends its executor: the call's outcome, else the executor's own refusal (an
/// older Runtime answers `ok: false` when the attempt raised outside its handler), never "".
fn unsettled(reply: &Frame, lost: &str) -> Refused {
    let (code, message) = match &reply.outcome {
        Some(o) if !o.code.is_empty() => (o.code.as_str(), o.message.as_str()),
        _ if !reply.ok && !reply.code.is_empty() => (reply.code.as_str(), reply.detail.as_str()),
        _ => ("executor_poisoned", "the executor named no reason"),
    };
    let (code, message) = match lost.is_empty() {
        true => (code.to_string(), message.to_string()),
        false => (
            "group_broken".into(),
            format!("{lost} during the call, first; then {code}: {message}"),
        ),
    };
    let detail = match reply.poisoned.as_str() {
        "" => format!("{message}; the executor did not settle quiescent"),
        poisoned => format!("{message}; the executor did not settle quiescent (poisoned: {poisoned})"),
    };
    Refused { code, detail }
}

/// Every error ends the run once its executor is gone: FAILED with the reason, or CANCELED
/// when a cancel was journaled. Nothing returns to the queue to fail again.
pub(crate) fn settle(engine: &Arc<Engine>, id: &str, error: &io::Error) -> io::Result<()> {
    let record = engine.get(id)?;
    if record.state.terminal() {
        return Ok(());
    }
    let ended = record
        .process
        .as_ref()
        .map(Birth::ended)
        .transpose()?
        .unwrap_or(true);
    if !ended {
        return Ok(()); // exit unproven: the reservation stays charged and nonterminal
    }
    if let Some(ended) = error
        .get_ref()
        .and_then(|inner| inner.downcast_ref::<device_executor::EndedBeforeStart>())
    {
        let pid = record.process.as_ref().map_or(0, |birth| birth.pid);
        keep_lifecycle_triage(engine, id, pid, &ended.to_string(), &ended.stderr_tail);
    }
    let outcome = if record.cancel_actor.is_some() {
        // Canceled: the reason its executor ended (it ignored the cancel and was killed, say)
        // is the run's last log line.
        if let Err(log) = engine.append_log(id, "warn", &format!("canceled; its executor ended: {error}")) {
            tracing::warn!("run {id}: {log}");
        }
        Outcome::Canceled
    } else if let Some(refusal) = refused(error) {
        // The executor's own reason, typed: a group's first fault names its GPU.
        Outcome::Failed(
            Failure::executor("failed", "runtime", &refusal.code, &refusal.detail),
        )
    } else if record.state == State::Starting {
        Outcome::Failed(
            Failure::abandoned(&format!("device executor did not start: {error}")),
        )
    } else {
        Outcome::Failed(Failure::abandoned(&format!("device executor ended: {error}")))
    };
    engine.finish(id, outcome).map(drop)
}

/// Executor roots keep each ended executor's logs for a day; every executor of an earlier
/// machine run has ended before this pool exists.
pub(crate) fn remove_old_executor_roots(root: &Path) {
    const KEEP: std::time::Duration = std::time::Duration::from_secs(24 * 60 * 60);
    let Ok(entries) = fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let executor = name.len() == 32
            && name
                .to_string_lossy()
                .bytes()
                .all(|b| b.is_ascii_hexdigit());
        let old = entry
            .metadata()
            .and_then(|m| m.modified())
            .is_ok_and(|modified| modified.elapsed().is_ok_and(|age| age > KEEP));
        if executor && old {
            if let Err(error) = fs::remove_dir_all(entry.path()) {
                tracing::warn!("old executor root {}: {error}", entry.path().display());
            }
        }
    }
}

/// A tree input's media type: its digest names a tree manifest (`entries` of files by path).
pub const TREE_MEDIA: &str = "application/vnd.cozy.tree-manifest";

/// A call's inputs in its spool: file inputs by field path as read-only copies, and input
/// trees by reference (`Tree` fields) as read-only directories with their manifest digest.
pub(crate) struct Staged {
    pub inputs: BTreeMap<String, serde_json::Value>,
    pub trees: BTreeMap<String, (PathBuf, String)>,
}

pub(crate) fn stage_inputs(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    spool: &Path,
    inputs: &[crate::journal::InputFile],
) -> io::Result<Staged> {
    let mut staged = Staged {
        inputs: BTreeMap::new(),
        trees: BTreeMap::new(),
    };
    if inputs.is_empty() {
        return Ok(staged);
    }
    let directory = spool.join("inputs");
    fs::create_dir(&directory)?;
    for (position, input) in inputs.iter().enumerate() {
        let field: String = input
            .input_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-') {
                    c
                } else {
                    '_'
                }
            })
            .take(96)
            .collect();
        let local = directory.join(format!("{position:03}-{field}"));
        if input.media_type == TREE_MEDIA {
            let manifest = held_bytes(store, &input.digest, input.length)?;
            materialize_tree(store, identity, &manifest, &local)?;
            // A payload names a tree by its field's ref or by its manifest digest.
            staged.trees.insert(input.digest.clone(), (local.clone(), input.digest.clone()));
            staged.trees.insert(input.input_id.clone(), (local, input.digest.clone()));
            continue;
        }
        copy_held(store, identity, &input.digest, input.length, &local)?;
        staged.inputs.insert(
            input.input_id.clone(),
            serde_json::json!({"local":local,"media_type":input.media_type,"digest":input.digest,"length":input.length,"order":input.order,"file_state":null}),
        );
    }
    if let Some(identity) = identity {
        identity.readable(&directory)?;
    }
    Ok(staged)
}

/// One held object copied read-only to `local`, exactly `length` bytes.
fn copy_held(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    digest: &str,
    length: u64,
    local: &Path,
) -> io::Result<()> {
    let object = tensorfs_core::format::ObjectRef { sha256: digest.trim_start_matches("sha256:").into(), length };
    let mut source = store.file(&object).map_err(crate::storage::io_error)?;
    let mut copy = cozy_platform::durable::Mode::mode(
        fs::OpenOptions::new().write(true).create_new(true),
        0o444,
    )
    .open(local)?;
    // A scratch copy the attempt reads: its spool goes with the attempt, so it is never synced.
    if std::io::copy(&mut source, &mut copy)? != length {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "an input's held bytes changed length",
        ));
    }
    if let Some(identity) = identity {
        identity.readable(local)?;
    }
    Ok(())
}

fn held_bytes(store: &Store, digest: &str, length: u64) -> io::Result<Vec<u8>> {
    let object = tensorfs_core::format::ObjectRef { sha256: digest.trim_start_matches("sha256:").into(), length };
    store.read(&object).map_err(crate::storage::io_error)
}

/// A tree manifest's files under `root`: each a relative path below it, each a held object.
fn materialize_tree(
    store: &Store,
    identity: Option<crate::launch_identity::LaunchIdentity>,
    manifest: &[u8],
    root: &Path,
) -> io::Result<()> {
    let invalid = |why: &str| io::Error::new(io::ErrorKind::InvalidData, format!("tree manifest: {why}"));
    let document = crate::trees::parse(manifest).map_err(|_| invalid("not a tree of files"))?;
    fs::create_dir(root)?;
    for (path, object) in &document {
        let (sha, length) = (&object.sha256, object.length);
        let relative = Path::new(path);
        let local = relative.components().all(|c| matches!(c, std::path::Component::Normal(_)));
        if !local || path.contains(['\\', '\0']) {
            return Err(invalid("a path leaves its tree"));
        }
        let target = root.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        copy_held(store, identity, &format!("sha256:{sha}"), length, &target)?;
    }
    if let Some(identity) = identity {
        for entry in walk_dirs(root)? {
            identity.readable(&entry)?;
        }
    }
    Ok(())
}

pub(crate) fn walk_dirs(root: &Path) -> io::Result<Vec<PathBuf>> {
    let mut found = vec![root.to_path_buf()];
    let mut index = 0;
    while index < found.len() {
        for entry in fs::read_dir(&found[index])? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                found.push(entry.path());
            }
        }
        index += 1;
    }
    Ok(found)
}

/// The failed attempt's triage bundle, kept before the run settles: the executor's own terminal
/// and traceback, and its stderr tail.
/// A lifecycle failure's bundle, when the executor wrote no terminal of its own: the reason
/// (with any kill measurement) and the end of its stderr. A canceled run keeps none.
fn keep_lifecycle_triage(engine: &Engine, id: &str, pid: u32, reason: &str, stderr_tail: &str) {
    let Ok(record) = engine.get(id) else {
        return;
    };
    if record.cancel_actor.is_some() || engine.triage(id).ok().flatten().is_some() {
        return;
    }
    let request = record
        .submission
        .as_ref()
        .map_or_else(|| id.to_string(), |s| s.request_id.clone());
    engine.record_triage(
        id,
        &crate::triage::Facts {
            request_id: &request,
            attempt: record.attempt,
            terminal: "failed",
            origin: "machine",
            code: "executor_ended",
            message: reason,
            traceback: "",
            executor_pid: pid,
            stderr_tail,
        },
    );
}

pub(crate) fn keep_triage(
    engine: &Engine,
    id: &str,
    executor: &DeviceExecutor,
    outcome: &device_executor::Outcome,
) {
    let record = engine.get(id).ok();
    let request = record
        .as_ref()
        .and_then(|record| record.submission.as_ref())
        .map_or_else(|| id.to_string(), |s| s.request_id.clone());
    engine.record_triage(
        id,
        &crate::triage::Facts {
            request_id: &request,
            attempt: record.map_or(0, |record| record.attempt),
            terminal: &outcome.terminal,
            origin: &outcome.origin,
            code: &outcome.code,
            message: &outcome.message,
            traceback: &outcome.traceback,
            executor_pid: executor.birth.pid,
            stderr_tail: &executor.stderr_tail(),
        },
    );
}

pub(crate) fn output_bindings(bindings: Vec<device_executor::AssetBinding>) -> Vec<AssetBinding> {
    bindings
        .into_iter()
        .map(|binding| AssetBinding {
            relative_path: binding.name,
            asset_ref: binding.asset_ref,
            media_type: binding.media_type,
            length: binding.length,
        })
        .collect()
}

/// An executor's refusal of a lifecycle command (start, load, activate), in its own words:
/// a group's names the GPU whose fault came first (Runtime `RankGroup` latches it).
#[derive(Debug)]
struct Refused {
    code: String,
    detail: String,
}
impl std::fmt::Display for Refused {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.code, self.detail)
    }
}
impl std::error::Error for Refused {}
/// The request never reached a handler; see `call`.
#[derive(Debug)]
struct Undelivered(String);
impl std::fmt::Display for Undelivered {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}
impl std::error::Error for Undelivered {}
fn undelivered(error: &io::Error) -> bool {
    error.get_ref().is_some_and(|inner| inner.is::<Undelivered>())
}

fn refused(error: &io::Error) -> Option<&Refused> {
    error.get_ref()?.downcast_ref::<Refused>()
}

/// What the call's executor measured, kept as the run's (`run show`): its stage and step
/// tracks, its ranks' execution records, its kernel observations (the attention and GEMM
/// kernels that served, Sol's dense and sparse calls) and the models it ran, as resolved.
pub(crate) fn record_measurements(engine: &Engine, id: &str, plan: &Plan, reply: &Frame) {
    let software = engine.get(id).ok().and_then(|run| run.executor.map(|executor| executor.software(run.attempt)));
    let measurements = measured(plan, reply, software, crate::memory::call::CallMemory::of(reply));
    let recorded = serde_json::to_vec(&measurements)
        .map_err(io::Error::other)
        .and_then(|bytes| engine.with_journal(|journal| journal.record_measurements(id, &bytes)));
    if let Err(error) = recorded {
        tracing::warn!("run {id}: measurements not kept: {error}");
    }
}


fn measured<S: Serialize, M: Serialize>(plan: &Plan, reply: &Frame, software: S, memory: M) -> serde_json::Value {
    let kernels: Vec<&serde_json::Value> = (reply.observations.as_array().into_iter().flatten())
        .filter(|row| row["name"].as_str().is_some_and(|name| name.starts_with("attention.") || name.starts_with("gemm.")))
        .collect();
    serde_json::json!({"attribution": reply.attribution, "software": software, "execution": reply.execution,
        "observations": kernels, "memory": memory, "models": plan.models()})
}

pub(crate) fn command_ok(frame: Frame) -> io::Result<Frame> {
    if frame.ok {
        Ok(frame)
    } else {
        Err(io::Error::other(Refused {
            code: frame.code,
            detail: frame.detail,
        }))
    }
}
/// Each constructed slot's checkpoint and components: what a Start schedules first.
fn constructions(plan: &Plan) -> Vec<Construction> {
    let slot = |slot: &PlanSlot| Construction {
        path: slot.binding.model_binding_path.clone(),
        manifest: slot.binding.snapshot.clone(),
        components: slot.binding.components.clone(),
    };
    plan.slots.iter().map(slot).collect()
}

/// A call's `fetch_channel` answers: a fresh channel over its checkpoints held in part.
struct Ahead<'a>(Option<&'a crate::access::Fetch>);
impl Services for Ahead<'_> {
    fn request(&mut self, frame: &Frame, descriptor: Option<File>) -> io::Result<(Answer, Option<File>)> {
        drop(descriptor);
        let mut answer = Answer::unavailable(frame.seq);
        let (Kind::FetchChannel, Some(fetch)) = (frame.kind, self.0) else {
            return Ok((answer, None));
        };
        Ok(match fetch.channel() {
            Ok(channel) => {
                (answer.ok, answer.descriptor) = (true, true);
                answer.code.clear();
                answer.detail.clear();
                (answer, Some(channel))
            }
            Err(error) => {
                answer.code = "fetch_channel_refused".into();
                answer.detail = error.to_string();
                (answer, None)
            }
        })
    }
}

struct Callbacks<'a> {
    engine: &'a Arc<Engine>,
    id: &'a str,
    completed: u64,
    deposits: &'a Deposits,
    sources: &'a Arc<ModelSources>,
    pool: &'a Pool,
    store: &'a Store,
    /// The invoking request's spool, where its published assets' bytes are.
    spool: Option<PathBuf>,
    /// The invoking request as a caller: its child-call frames.
    caller: Option<&'a crate::calls::Caller>,
    /// Its weights broker and grant: its `Weights[M]` sources and weights outputs.
    weights: Option<(&'a crate::weights::Weights, &'a crate::weights::Grant)>,
    /// The call's fetches: its readers wait on them, and its executor asks for channels.
    fetch: Option<&'a Arc<crate::access::Fetch>>,
    /// Its `Weights[M]` sources, whose headers and assets `model_source` answers too.
    raw: &'a [crate::gpu_service::PlanSource],
}
impl Callbacks<'_> {
    /// The weight set's chunk files, once its objects landed, and their digest in `answer`.
    fn chunks(&self, manifest: &str, answer: &mut Answer) -> io::Result<Vec<File>> {
        let objects = self.sources.chunks(manifest)?;
        if let Some(fetch) = self.fetch {
            fetch.wait(manifest, &objects)?;
        }
        answer.objects_sha256 = crate::deposits::objects_digest(&objects);
        self.sources.files(&objects)
    }
}
impl Services for Callbacks<'_> {
    /// A weight set's bytes: the sealed host layout an executor deposited for its manifest and
    /// layout, else every chunk file of its selected components (TensorD never fills).
    fn read_grant(&mut self, frame: &Frame, plan: Option<File>) -> io::Result<(Answer, Vec<File>)> {
        drop(plan);
        let mut answer = Answer::unavailable(frame.seq);
        let files = match self.deposits.grant(&frame.manifest, &frame.layout) {
            Some(layout) => {
                answer.held = true;
                Ok(vec![layout])
            }
            None => self.chunks(&frame.manifest, &mut answer),
        };
        self.pool.notes.note(serde_json::json!({"event": "read_grant", "manifest": frame.manifest,
            "layout": frame.layout, "held": answer.held, "files": files.as_ref().map_or(0, Vec::len)}));
        Ok(match files {
            Ok(files) => {
                (answer.ok, answer.descriptors) = (true, files.len() as u64);
                answer.code.clear();
                answer.detail.clear();
                (answer, files)
            }
            Err(error) => {
                answer.code = "read_grant_refused".into();
                answer.detail = error.to_string();
                (answer, Vec::new())
            }
        })
    }
    fn progress(&mut self, frame: &Frame) {
        crate::calls::observe(self.engine, self.id, &mut self.completed, frame, self.caller);
    }
    fn request(
        &mut self,
        frame: &Frame,
        descriptor: Option<File>,
    ) -> io::Result<(Answer, Option<File>)> {
        if frame.kind == Kind::FetchChannel {
            return Ahead(self.fetch.map(|fetch| &**fetch)).request(frame, descriptor);
        }
        if frame.kind == Kind::Deposit {
            // A layout varena sealed in the executor, kept for the next executor of its weights.
            let layout = descriptor.ok_or_else(|| io::Error::other("a deposit omitted its layout"))?;
            let kept = self.deposits.deposit(&frame.manifest, &frame.layout, layout);
            self.pool.notes.note(serde_json::json!({"event": "deposit", "manifest": frame.manifest,
                "layout": frame.layout, "kept": kept.as_ref().map_or_else(ToString::to_string, |()| "yes".into())}));
            return Ok(match kept {
                Ok(()) => (Answer::ok(frame.seq), None),
                Err(error) => (Answer::refused(frame.seq, "deposit_refused", error.to_string()), None),
            });
        }
        if frame.kind == Kind::ModelSource {
            drop(descriptor);
            // A refusal is an answer: that executor fails its load with the reason.
            // A `Weights[M]` source's header and assets too, for Runtime to plan its reads.
            let id = |m: &str| m.trim_start_matches("sha256:").to_string();
            let served = match self.raw.iter().any(|s| id(&s.manifest) == id(&frame.manifest)) {
                true => {
                    let selected = [SelectedManifest { manifest: frame.manifest.clone(), components: vec![] }];
                    ModelSources::open_shared(self.pool.store.clone(), &selected).and_then(|raw| raw.serve(frame))
                }
                false => self.sources.serve(frame),
            };
            return Ok(match served {
                Ok((answer, file)) => (answer, Some(file)),
                Err(error) => {
                    let mut answer = Answer::unavailable(frame.seq);
                    answer.code = "model_source_refused".into();
                    answer.detail = error.to_string();
                    (answer, None)
                }
            });
        }
        if frame.kind == Kind::Publish {
            drop(descriptor);
            let answer = match &self.spool {
                Some(spool) => {
                    crate::products::publish(self.store, self.engine, self.id, spool, frame)
                }
                None => Answer::unavailable(frame.seq),
            };
            return Ok((answer, None));
        }
        if frame.kind == Kind::WeightsWriter {
            drop(descriptor);
            let Some((weights, grant)) = self.weights else {
                return Ok((Answer::unavailable(frame.seq), None));
            };
            let (mut answer, channel, adopted) = weights.answer(grant, frame);
            for upload in grant.take_uploads() {
                crate::calls::record_upload(self.engine, self.id, &upload);
            }
            if let Some(adopted) = adopted {
                if let Err(error) = crate::products::record_manifest(self.engine, self.id, &adopted.output, &adopted.manifest) {
                    answer = Answer::refused(frame.seq, "publish_refused", error.to_string());
                }
            }
            return Ok((answer, channel));
        }
        drop(descriptor);
        match self.caller {
            Some(caller) => caller.seam(frame),
            None => Ok((Answer::unavailable(frame.seq), None)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[cfg(unix)]
    use std::os::unix::process::ExitStatusExt;

    /// Runs 5313/5318/5319 ended `executor_poisoned: ; the executor did not settle quiescent
    /// (poisoned: )`: the Runtime's tail re-raised a sticky CUDA fault and answered `ok: false`.
    #[test]
    fn an_unsettled_invoke_names_its_reason() {
        let reason = |reply: serde_json::Value, lost: &str| {
            let refused = unsettled(&serde_json::from_value(reply).unwrap(), lost);
            format!("{}: {}", refused.code, refused.detail)
        };
        let fault = "AcceleratorError: CUDA error: an illegal memory access was encountered";
        assert_eq!(
            reason(json!({"reply": "invoke", "ok": false, "code": "unhandled_exception", "detail": fault}), ""),
            format!("unhandled_exception: {fault}; the executor did not settle quiescent")
        );
        let outcome = json!({"terminal": "failed", "code": "unhandled_exception", "message": format!("{fault} in denoise")});
        assert_eq!(
            reason(json!({"ok": true, "quiescent": false, "poisoned": "unhandled_exception in denoise", "outcome": outcome}), ""),
            format!("unhandled_exception: {fault} in denoise; the executor did not settle quiescent (poisoned: unhandled_exception in denoise)")
        );
        assert_eq!(
            reason(json!({"ok": true, "quiescent": false}), "GPU 1's process 7 ended"),
            "group_broken: GPU 1's process 7 ended during the call, first; then executor_poisoned: \
             the executor named no reason; the executor did not settle quiescent"
        );
    }

    /// What `run show` reads of a call: the models it ran as the machine resolved them (run
    /// 10236's background removal: `paul/birefnet@0.1.0 fp16`), constructed or raw, and its
    /// attention and GEMM kernels (the attention row is that run's segment call's); nothing
    /// else it observed.
    #[test]
    fn a_calls_measurements_name_its_resolved_models_and_its_kernels() {
        let binding = Binding {
            model_parameter_name: "model".into(),
            model: "paul/birefnet@0.1.0".into(),
            snapshot: "sha256:aa".into(),
            ..Default::default()
        };
        let source = PlanSource {
            parameter: "lora".into(),
            manifest: "sha256:bb".into(),
            length: 9,
            repository: "paul/wushu".into(),
            release: "1.0.0".into(),
            lane: "bf16".into(),
        };
        let plan = Plan {
            id: "p".into(),
            construction: "c".into(),
            installation: "i".into(),
            generation: "g".into(),
            entrypoint: "remove_background".into(),
            slots: vec![PlanSlot { binding, lane: "fp16".into(), selected_encoded_bytes: 0, on_demand: false }],
            sources: vec![source],
            degree: 1,
        };
        let reply: Frame = serde_json::from_value(json!({"reply": "invoke", "ok": true, "observations": [
            {"kind": "confession", "name": "attention.selected", "value": "kitchen-sol-producer-fast-shared-qkv",
                "fields": {"component": "base_model/ref2va_dit", "dense_reference": "kitchen-int8", "pin": "", "sites": 52}},
            {"kind": "confession", "name": "gemm.selected", "value": "cutlass-sm80-int8",
                "fields": {"component": "base_model/ref2va_dit", "format": "int8", "sites": 416}},
            {"kind": "confession", "name": "weights.mode", "value": "resident", "fields": {"stage": "denoise"}}]}))
        .unwrap();
        let measured = measured(&plan, &reply, None::<()>, ());
        assert_eq!(measured["models"], json!([
            {"slot": "model", "repository": "paul/birefnet", "release": "0.1.0", "lane": "fp16", "manifest": "sha256:aa"},
            {"slot": "lora", "repository": "paul/wushu", "release": "1.0.0", "lane": "bf16", "manifest": "sha256:bb"}]));
        let kept: Vec<&str> = measured["observations"].as_array().unwrap().iter().map(|o| o["name"].as_str().unwrap()).collect();
        assert_eq!(kept, ["attention.selected", "gemm.selected"]);
    }

    #[cfg(unix)]
    #[test]
    fn an_executor_that_ended_on_its_own_names_its_exit() {
        let ended = |status: i32| device_executor::Ended {
            status: std::process::ExitStatus::from_raw(status),
            killed: None,
            stragglers: 0,
        };
        let eof = || io::Error::new(io::ErrorKind::UnexpectedEof, "stock executor EOF before reply");
        let error = with_end(eof(), &ended(libc::SIGSEGV));
        assert_eq!(
            error.to_string(),
            "stock executor EOF before reply; the executor ended on its own (signal: 11 (SIGSEGV))"
        );
        // An executor ended by the machine after a refusal adds nothing to the refusal.
        let refusal = || io::Error::other(Refused { code: "x".into(), detail: "y".into() });
        assert_eq!(with_end(refusal(), &ended(libc::SIGKILL)).to_string(), "x: y");
    }

    #[test]
    fn the_image_boot_runs_machine_kernels_from_the_executors_runtime_wheel() {
        let dir = std::env::temp_dir().join(format!("kernel-runtime-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        let wheel = dir.join("cozy_runtime-9.9.9-py3-none-any.whl");
        let mut zip = zip::ZipWriter::new(File::create(&wheel).unwrap());
        let options = zip::write::SimpleFileOptions::default();
        for (name, body) in [
            ("cozy_runtime/__init__.py", ""),
            ("cozy_runtime/internal/__init__.py", ""),
            ("cozy_runtime/internal/machine_kernels.py", "import sys; print(__name__, sys.path[0])"),
            ("cozy_runtime-9.9.9.data/scripts/tensord", "not imported"),
        ] {
            zip.start_file(name, options).unwrap();
            zip.write_all(body.as_bytes()).unwrap();
        }
        zip.finish().unwrap();
        let tree = unpack_runtime(&wheel, &dir.join("trees")).unwrap();
        assert_eq!(unpack_runtime(&wheel, &dir.join("trees")).unwrap(), tree);
        assert!(!tree.join("cozy_runtime-9.9.9.data").exists());
        let ran = std::process::Command::new("python3")
            .args(["-I", "-c", RUNTIME_FIRST])
            .arg(&tree)
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&ran.stdout).trim(), format!("__main__ {}", tree.display()));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn a_group_is_the_widest_degree_every_slot_declares_on_this_machine() {
        let h3 = [
            json!({"path":"e.models.base_model","sequence_parallel":{"degrees":[2,4,7,8]}}),
            json!({"path":"e.models.turbo_lora","sequence_parallel":{"degrees":[2,4,7,8]}}),
        ];
        assert_eq!(group_degree(&h3, 0, 1), 1);
        assert_eq!(group_degree(&h3, 0, 2), 2);
        assert_eq!(group_degree(&h3, 0, 6), 4);
        assert_eq!(group_degree(&h3, 1, 4), 1);
        assert_eq!(group_degree(&h3, 2, 4), 2);
        // A count no slot declares, or wider than the machine, caps the group: never a refusal.
        assert_eq!(group_degree(&h3, 3, 4), 2);
        assert_eq!(group_degree(&h3, 4, 2), 2);
        // A slot without a declaration runs on one GPU only, so the group is one.
        let mixed = [h3[0].clone(), json!({"path":"e.models.vae"})];
        assert_eq!(group_degree(&mixed, 0, 4), 1);
        assert_eq!(group_degree(&[json!({"path":"e.models.vae"})], 2, 2), 1);
    }

    /// An integrated GPU (GB10) answers NVML's memory info with NotSupported. Its ledger still
    /// has a capacity (the host's memory), without which executors share nothing (run 10322).
    /// Under a fake `libnvidia-ml.so.1` that answers so, in a child of this test.
    #[cfg(target_os = "linux")]
    #[test]
    fn a_gpu_whose_memory_nvml_does_not_support_has_the_hosts_memory_as_its_ledgers() {
        let dir = std::env::temp_dir().join(format!("fake-nvml-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join("nvml.c"), FAKE_NVML).unwrap();
        let built = std::process::Command::new("cc")
            .args(["-shared", "-fPIC", "-o"]).arg(dir.join("libnvidia-ml.so.1")).arg(dir.join("nvml.c"))
            .status().unwrap();
        assert!(built.success());
        let child = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "gpu_service::tests::an_integrated_gpus_ledger_is_the_hosts_memory", "--ignored"])
            .env("LD_LIBRARY_PATH", &dir)
            .output().unwrap();
        let out = String::from_utf8_lossy(&child.stdout);
        assert!(child.status.success() && out.contains("1 passed"), "{out}{}", String::from_utf8_lossy(&child.stderr));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    #[ignore = "a_gpu_whose_memory_nvml_does_not_support_has_the_hosts_memory_as_its_ledgers runs it under its fake NVML"]
    fn an_integrated_gpus_ledger_is_the_hosts_memory() {
        let root = std::env::temp_dir().join(format!("gpu-integrated-{}", uuid::Uuid::new_v4()));
        let config: GpuConfig = serde_json::from_value(json!({"devices": "0",
            "authorized_device_limit_bytes": null, "prespawn": false})).unwrap();
        let store = crate::storage::open(&root.join("store")).unwrap();
        let pool = Pool::new(&root.join("gpu"), config, store).unwrap();
        let host = cozy_platform::contain::host_memory::total();
        assert!(host > 0);
        assert_eq!(pool.devices[0].memory.as_ref().map(|ledger| ledger.status().unwrap().capacity), Some(host as u64));
        let _ = fs::remove_dir_all(root);
    }

    /// NVML for one integrated GPU: every call the memory sampler makes, and memory info
    /// NotSupported (3), as GB10's driver answers.
    #[cfg(target_os = "linux")]
    const FAKE_NVML: &str = r#"
#include <stdio.h>
static int gpu;
int nvmlInit_v2(void) { return 0; }
int nvmlInit_v3(unsigned flags) { return 0; }
int nvmlShutdown(void) { return 0; }
int nvmlDeviceGetCount_v2(unsigned *n) { *n = 1; return 0; }
int nvmlDeviceGetHandleByIndex_v2(unsigned i, void **d) { if (i) return 2; *d = &gpu; return 0; }
int nvmlDeviceGetHandleByUUID(const char *u, void **d) { *d = &gpu; return 0; }
int nvmlDeviceGetIndex(void *d, unsigned *i) { *i = 0; return 0; }
int nvmlDeviceGetUUID(void *d, char *b, unsigned n) { snprintf(b, n, "GPU-integrated-test"); return 0; }
int nvmlDeviceGetName(void *d, char *b, unsigned n) { snprintf(b, n, "NVIDIA GB10"); return 0; }
int nvmlSystemGetDriverVersion(char *b, unsigned n) { snprintf(b, n, "595.84"); return 0; }
int nvmlDeviceGetMemoryInfo(void *d, void *m) { return 3; }
int nvmlDeviceGetMemoryInfo_v2(void *d, void *m) { return 3; }
int nvmlDeviceGetDisplayActive(void *d, int *s) { *s = 0; return 0; }
"#;

    /// A fresh pool over `width` GPUs NVML does not know (this host's own stay untouched):
    /// every GPU has room.
    fn pool_of(width: usize) -> (PathBuf, Arc<Pool>, Arc<Engine>) {
        let root = std::env::temp_dir().join(format!("gpu-place-{}", uuid::Uuid::new_v4()));
        let devices: Vec<_> = (0..width).map(|gpu| format!("GPU-test-{gpu}")).collect();
        let config: GpuConfig = serde_json::from_value(json!({"devices": devices.join(","),
            "authorized_device_limit_bytes": null, "prespawn": false})).unwrap();
        let store = crate::storage::open(&root.join("store")).unwrap();
        let pool = Pool::new(&root.join("gpu"), config, store).unwrap();
        let engine = Engine::open(&root.join("execution")).unwrap();
        (root, pool, engine)
    }
    fn plan(id: &str, degree: u32) -> Plan {
        Plan {
            id: id.into(),
            construction: id.into(),
            installation: "i".into(),
            generation: "g".into(),
            entrypoint: "e".into(),
            slots: vec![],
            sources: vec![],
            degree,
        }
    }

    #[test]
    fn calls_of_one_plan_spread_over_free_gpus_and_a_waiting_group_keeps_its_place() {
        let (root, pool, engine) = pool_of(4);
        let (image, film) = (plan("image", 1), plan("film", 2));
        let place = |plan: &Plan, seq: u64, claims: &mut [Option<u64>]| pool.place(&engine, plan, (seq, None), None, claims);
        let serving = |gpu: usize| pool.leaders.lock().unwrap().insert(("image".into(), gpu));
        let mut claims = [None; 4];
        let (gpu, first) = place(&image, 1, &mut claims).unwrap();
        assert_eq!(gpu, 0);
        serving(0);
        // GPU 0 serves it: the next call starts a replica on the next free GPU.
        let (gpu, second) = place(&image, 2, &mut claims).unwrap();
        assert_eq!(gpu, 1);
        serving(1);
        // A full-machine group waits and keeps later requests off its GPUs.
        assert!(place(&plan("full", 4), 3, &mut claims).is_none());
        assert_eq!(claims, [Some(3); 4]);
        drop(first);
        assert!(place(&image, 4, &mut claims).is_none(), "the full-machine group has priority");
        let (gpu, third) = place(&image, 4, &mut [None; 4]).unwrap();
        assert_eq!(gpu, 0);
        serving(2);
        // The next pass: an idle executor of the plan comes before a new replica.
        drop((second, third));
        let mut claims = [None; 4];
        let held: Vec<_> = (5..8).map(|seq| place(&image, seq, &mut claims).unwrap()).collect();
        assert_eq!(held.iter().map(|(gpu, _)| *gpu).collect::<Vec<_>>(), [0, 1, 2]);
        assert_eq!(place(&image, 8, &mut claims).unwrap().0, 3);
        // Every GPU of its lane free at once: the group runs.
        drop(held);
        assert_eq!(place(&film, 9, &mut [None; 4]).unwrap().0, 0);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    #[test]
    fn independent_two_gpu_requests_use_disjoint_lanes() {
        let (root, pool, engine) = pool_of(4);
        let group = plan("two", 2);
        let place = |seq: u64| pool.place(&engine, &group, (seq, None), None, &mut [None; 4]);
        let (first_gpu, first) = place(1).unwrap();
        let (second_gpu, second) = place(2).unwrap();
        assert_eq!((first_gpu, second_gpu), (0, 2));
        assert!(place(3).is_none());
        drop(first);
        assert_eq!(place(3).unwrap().0, 0);
        drop(second);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    /// A deviceless root accepted as request `key`, ready to dispatch.
    fn root_job(engine: &Engine, key: &str, gpus: u32) -> Execution {
        let invocation = crate::journal::Invocation {
            package: "p".into(), generation: "0".repeat(32), module: "p:app".into(),
            entrypoint: "run".into(), gpus, ..Default::default()
        };
        let (accepted, _) = engine.accept_run("actor", key, key, invocation.clone()).unwrap();
        engine.bind_prepared(&accepted.id, invocation, "", Default::default(), None).unwrap()
    }
    /// The root's `gpu` log entries: (state, gpus, behind).
    fn allocation_log(engine: &Engine, id: &str) -> Vec<(String, Vec<u64>, String)> {
        engine.logs(id).unwrap().iter().filter_map(|stored| {
            let entry: serde_json::Value = serde_json::from_slice(&stored.product).ok()?;
            let gpu = entry.get("gpu")?;
            Some((gpu["state"].as_str()?.into(), gpu["gpus"].as_array()?.iter().filter_map(|g| g.as_u64()).collect(),
                gpu["behind"].as_str()?.into()))
        }).collect()
    }

    fn running_roots(engine: &Arc<Engine>, keys: [&str; 3]) -> [Execution; 3] {
        keys.map(|key| {
            let job = root_job(engine, key, 0);
            engine.dispatch_managed(&job.id, |_, _| Ok(())).unwrap();
            job
        })
    }
    fn request(record: &Execution) -> String {
        record.submission.as_ref().map(|s| s.request_id.clone()).unwrap_or_default()
    }

    #[test]
    fn a_root_keeps_its_lane_between_steps_and_later_roots_wait_in_acceptance_order() {
        let (root, pool, engine) = pool_of(4);
        let [first, second, _] = running_roots(&engine, ["first", "second", "unused"]);
        let first_allocation = pool.root(&engine, &first, 4).unwrap();
        let second_allocation = pool.root(&engine, &second, 2).unwrap();
        let (a, b) = (first.root_seq(), second.root_seq());
        // Both ready: the older root binds, the younger waits.
        let mut claims = [None; 4];
        let lane = pool.root_lane(&engine, a, &mut claims).unwrap();
        assert_eq!(lane, 0..4);
        assert!(pool.root_lane(&engine, b, &mut claims).is_none());
        // A reference on one GPU, then a group over the lane: nobody else runs in the gap.
        let (gpu, reference) = pool.place(&engine, &plan("reference", 1), (a, None), Some(lane.clone()), &mut [None; 4]).unwrap();
        assert_eq!(gpu, 0);
        drop(reference);
        assert!(pool.root_lane(&engine, b, &mut [None; 4]).is_none());
        assert!(pool.place(&engine, &plan("other", 1), (9, None), None, &mut [None; 4]).is_none(), "nor a standalone call");
        let (_, segment) = pool.place(&engine, &plan("segment", 4), (a, None), Some(lane), &mut [None; 4]).unwrap();
        // Release with a segment in flight: the younger root binds once it returns.
        pool.release(&engine, a);
        pool.release(&engine, a);
        assert!(pool.root_lane(&engine, b, &mut [None; 4]).is_none());
        drop(segment);
        let lane = pool.root_lane(&engine, b, &mut [None; 4]).unwrap();
        assert_eq!(lane, 0..2);
        // The released root comes back ahead of every later root but never displaces the
        // holder: it waits for the lane, the holder's steps go on, a later standalone call
        // may not take the free half.
        let mut claims = [None; 4];
        assert!(pool.root_lane(&engine, a, &mut claims).is_none());
        let (_, step) = pool.place(&engine, &plan("image", 1), (b, None), Some(lane), &mut claims).unwrap();
        assert!(pool.place(&engine, &plan("other", 1), (9, None), None, &mut claims).is_none());
        drop((step, second_allocation));
        assert_eq!(pool.root_lane(&engine, a, &mut [None; 4]), Some(0..4));
        drop(first_allocation);
        assert_eq!(allocation_log(&engine, &first.id), [
            ("grant".into(), vec![0, 1, 2, 3], String::new()), ("release".into(), vec![0, 1, 2, 3], String::new()),
            ("wait".into(), vec![], request(&second)), ("grant".into(), vec![0, 1, 2, 3], String::new()),
            ("release".into(), vec![0, 1, 2, 3], String::new()),
        ]);
        assert_eq!(allocation_log(&engine, &second.id)[..2], [
            ("wait".into(), vec![], request(&first)), ("grant".into(), vec![0, 1], String::new()),
        ]);
        assert!(pool.place(&engine, &plan("other", 4), (9, None), None, &mut [None; 4]).is_some(), "exits end the lanes");
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    /// An older root still downloading its weights claims no GPU: a younger ready root runs
    /// meanwhile and hands its lane over at its next step boundary once the older one's first
    /// step is ready. Its step in flight, a served call's own child included, finishes first.
    #[test]
    fn a_younger_root_runs_while_an_older_one_downloads_and_yields_at_its_step_boundary() {
        let (root, pool, engine) = pool_of(2);
        let [big, small, _] = running_roots(&engine, ["big", "small", "unused"]);
        let holds = (pool.root(&engine, &big, 2).unwrap(), pool.root(&engine, &small, 2).unwrap());
        let (a, b) = (big.root_seq(), small.root_seq());
        let lane = pool.root_lane(&engine, b, &mut [None; 2]).unwrap();
        assert_eq!(lane, 0..2);
        let (_, image) = pool.place(&engine, &plan("image", 1), (b, None), Some(lane.clone()), &mut [None; 2]).unwrap();
        drop(image);
        let (_, served) = pool.place(&engine, &plan("served", 1), (b, Some("served")), Some(lane.clone()), &mut [None; 2]).unwrap();
        // The big root's first step is ready.
        let mut claims = [None; 2];
        assert!(pool.root_lane(&engine, a, &mut claims).is_none());
        assert!(pool.place(&engine, &plan("image", 1), (b, None), Some(lane.clone()), &mut claims).is_none(), "no new step");
        let (_, child) = pool.place(&engine, &plan("child", 1), (b, Some("served")), Some(lane.clone()), &mut claims).unwrap();
        drop((child, served));
        let mut claims = [None; 2];
        assert_eq!(pool.root_lane(&engine, a, &mut claims), Some(0..2));
        assert!(pool.root_lane(&engine, b, &mut claims).is_none());
        assert_eq!(allocation_log(&engine, &big.id), [
            ("wait".into(), vec![], request(&small)), ("grant".into(), vec![0, 1], String::new()),
        ]);
        assert_eq!(allocation_log(&engine, &small.id), [
            ("grant".into(), vec![0, 1], String::new()), ("release".into(), vec![0, 1], request(&big)),
            ("wait".into(), vec![], request(&big)),
        ]);
        drop(holds);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    /// A waiting root's wait names its current blocker: a new entry when it changes.
    #[test]
    fn a_waiting_roots_entry_follows_its_current_blocker() {
        let (root, pool, engine) = pool_of(2);
        let [x, y, a] = running_roots(&engine, ["x", "y", "a"]);
        let (x_hold, holds) = (pool.root(&engine, &x, 2).unwrap(), [pool.root(&engine, &y, 2).unwrap(), pool.root(&engine, &a, 2).unwrap()]);
        assert_eq!(pool.root_lane(&engine, x.root_seq(), &mut [None; 2]), Some(0..2));
        // y is still downloading; a waits behind x.
        assert!(pool.root_lane(&engine, a.root_seq(), &mut [None; 2]).is_none());
        assert!(pool.root_lane(&engine, a.root_seq(), &mut [None; 2]).is_none());
        drop(x_hold);
        let mut claims = [None; 2];
        assert_eq!(pool.root_lane(&engine, y.root_seq(), &mut claims), Some(0..2));
        assert!(pool.root_lane(&engine, a.root_seq(), &mut claims).is_none());
        assert_eq!(allocation_log(&engine, &a.id), [
            ("wait".into(), vec![], request(&x)), ("wait".into(), vec![], request(&y)),
        ]);
        drop(holds);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    /// Gate (f), runs 5673/5674 on one GPU: the older root's two ready first steps wait behind
    /// the younger holder's step in flight. Each pass visits both; the second's own claim must
    /// not hide the holder, so the root logs one wait naming it, not one per pass flipping to
    /// nobody (each wait entry also woke the next pass).
    #[test]
    fn a_roots_ready_steps_log_one_wait_for_one_blocker() {
        let (root, pool, engine) = pool_of(1);
        let [older, younger, _] = running_roots(&engine, ["older", "younger", "unused"]);
        let holds = (pool.root(&engine, &older, 1).unwrap(), pool.root(&engine, &younger, 1).unwrap());
        let (a, b) = (older.root_seq(), younger.root_seq());
        let lane = pool.root_lane(&engine, b, &mut [None]).unwrap();
        let (_, step) = pool.place(&engine, &plan("step", 1), (b, None), Some(lane.clone()), &mut [None]).unwrap();
        for _ in 0..3 {
            let mut claims = [None];
            assert!(pool.root_lane(&engine, a, &mut claims).is_none());
            assert!(pool.root_lane(&engine, a, &mut claims).is_none());
            assert_eq!(pool.root_lane(&engine, b, &mut claims), Some(lane.clone()));
            assert!(pool.place(&engine, &plan("step", 1), (b, None), Some(lane.clone()), &mut claims).is_none(), "no new step");
        }
        drop(step);
        assert_eq!(pool.root_lane(&engine, a, &mut [None]), Some(0..1));
        assert_eq!(allocation_log(&engine, &older.id), [
            ("wait".into(), vec![], request(&younger)), ("grant".into(), vec![0], String::new()),
        ]);
        drop(holds);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    #[test]
    fn an_explicit_root_keeps_its_children_in_its_two_gpu_lane() {
        let (root, pool, engine) = pool_of(4);
        let job = root_job(&engine, "cpu-job", 2);
        let allocation = pool.root(&engine, &job, 2).unwrap();
        let seq = job.root_seq();
        let lane = pool.root_lane(&engine, seq, &mut [None; 4]).unwrap();
        assert_eq!(lane, 0..2);
        let group = plan("child", 2);
        let (gpu, child) = pool.place(&engine, &group, (seq, None), Some(lane.clone()), &mut [None; 4]).unwrap();
        assert_eq!(gpu, 0);
        assert!(pool.place(&engine, &group, (seq, None), Some(lane), &mut [None; 4]).is_none(),
            "a simultaneous sibling cannot escape to the other two GPUs");
        let (gpu, other) = pool.place(&engine, &group, (8, None), None, &mut [None; 4]).unwrap();
        assert_eq!(gpu, 2, "the unrelated request uses the other half");
        drop(allocation); // canceled/ended root; its live child still runs on its GPUs.
        assert!(pool.place(&engine, &group, (9, None), None, &mut [None; 4]).is_none());
        drop(child);
        assert_eq!(pool.place(&engine, &group, (9, None), None, &mut [None; 4]).unwrap().0, 0);
        drop(other);
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }

    /// A fresh pod's burst of one unlearned plan: its first call starts on GPU 0, the second at
    /// once on empty GPU 1, the third waits for either.
    #[test]
    fn a_cold_burst_runs_on_every_empty_gpu_at_once() {
        let (root, pool, engine) = pool_of(2);
        let image = plan("image", 1);
        let mut claims = [None; 2];
        let (gpu, first) = pool.place(&engine, &image, (1, None), None, &mut claims).unwrap();
        assert_eq!(gpu, 0);
        pool.leaders.lock().unwrap().insert(("image".into(), 0));
        let (gpu, second) = pool.place(&engine, &image, (2, None), None, &mut claims).unwrap();
        assert_eq!(gpu, 1);
        pool.leaders.lock().unwrap().insert(("image".into(), 1));
        assert!(pool.place(&engine, &image, (3, None), None, &mut claims).is_none());
        assert_eq!(claims, [Some(3), Some(3)]);
        drop((first, second));
        pool.stop().unwrap();
        crate::scratch::remove(&root);
    }
}
