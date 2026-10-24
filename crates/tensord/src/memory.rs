//! TensorD hosts varena's lease manager (design varena-leases): each envelope GPU the inventory
//! names has one, which creates the GPU's chunks and leases them to its executors, admits and
//! places their requests, and keeps their measured peaks (capacity min(card, `gpu_memory_limit`)).
//! Admission is [`admit`] (`Manager::admit`: a group on all of its GPUs or none, waiting while
//! refused). A GPU whose chunks TensorD cannot create has a chunk-less manager: its executors run
//! standalone, admitted and counted all the same. Host RAM is the OS page cache; [`Host`] watches
//! what the running calls must keep free of it. One thread blocks on every GPU's events and the
//! host's short event; each GPU's [`Watch`] reads NVML while cozy holds bytes there.
use crate::{execution::Engine, machine::receipt};
use platform::contain::host_memory;
use platform::contain::host_pressure::{self, Growth, HostMode, Nudge};
use platform::memory::{wait_any, Signal, Waitable};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use platform::{
    handles::Peer,
    process::{Birth, Exact},
};
use varena::lease::{
    manager::{Opener, Record},
    Hosting, Identity, Manager, Mode, Peak,
};
use varena::{Backend, CudaBackend, GpuId};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// One envelope GPU: its lease manager (chunk-less where TensorD cannot create its chunks), the
/// GPU model its peaks are kept under, and the watch that reads its foreign and standalone bytes.
pub struct Gpu {
    pub manager: Arc<Manager>,
    pub model: String,
    watch: Arc<Watch>,
}

/// What hosting a GPU's leases takes: who the peers on its listener are, its chunks (none where
/// TensorD cannot create them: its executors run standalone), which processes are cozy's, and
/// the derived-weights disk cache every GPU's managers share.
pub struct Leases {
    pub peers: Arc<Peers>,
    pub open: Option<Opener>,
    pub tenants: Tenants,
    pub derived: Arc<crate::derived::Derived>,
}

/// One GPU's hosting: its peers, its free bytes as NVML reads them, which needs no CUDA
/// context (none where NVML does not open it: the manager reads its open backend's then), and
/// the derived-weights disk cache.
struct Hosted {
    peers: Arc<Peers>,
    device: Option<platform::gpu::Device>,
    derived: Arc<crate::derived::Derived>,
}

impl Hosting for Hosted {
    fn identify(&self, peer: &Peer) -> Option<Identity> {
        self.peers.identify(peer)
    }

    fn record(&self, record: Record) {
        self.peers.record(record);
    }

    fn free(&self) -> Option<u64> {
        self.device.as_ref()?.free().ok()
    }

    fn derive(&self, domain: &str, key: &str, bytes: u64) -> Option<varena::lease::DerivedEntry> {
        match self.derived.create(domain, key, bytes) {
            Ok(entry) => entry,
            Err(error) => {
                self.peers.notes.note(serde_json::json!({"event": "derived_unkept", "key": key, "bytes": bytes, "why": error.to_string()}));
                None
            }
        }
    }

    fn derived(&self, domain: &str, key: &str, entry: u64) {
        self.derived.written(domain, key, entry);
    }

    fn read_derived(&self, domain: &str, key: &str) -> Option<std::fs::File> {
        self.derived.open(domain, key)
    }

    fn forget_derived(&self, domain: &str, key: &str) {
        self.derived.touch(domain, key);
    }
}

impl Gpu {
    /// `entry`'s manager, sized by the inventory's memory (an integrated GPU's is the host's) and
    /// never above `limit`, in the machine's sharing `mode`, with the peaks this machine measured
    /// on its model. None (a memory-log note says why) when the inventory does not name it, or
    /// its listener or manager cannot be made.
    pub fn create(entry: &str, inventory: &[receipt::Gpu], limit: Option<u64>, mode: HostMode, leases: Leases, peaks: &Peaks, log: &MemoryLog) -> Option<Gpu> {
        let row = row(entry, inventory)?;
        let id = GpuId { uuid: row.device_uuid.clone(), model: row.device_name.clone(), memory: row.memory_bytes };
        let mode = match mode {
            HostMode::Shared => Mode::Shared,
            HostMode::Dedicated => Mode::Dedicated,
        };
        let chunks = leases.open.is_some();
        let manager = varena::lease::listen(&id)
            .map_err(|error| format!("its listener: {error}"))
            .and_then(|listener| listener.ok_or_else(|| "another process serves its leases".to_string()))
            .and_then(|listener| {
                let hosting = Arc::new(Hosted { peers: leases.peers, device: platform::gpu::Device::open(&id.uuid).ok(), derived: leases.derived });
                Manager::create(&id, limit, mode, leases.open, listener, hosting).map_err(|error| error.to_string())
            });
        let manager = match manager {
            Ok(manager) => manager,
            Err(why) => {
                log.note(serde_json::json!({"event": "no_manager", "gpu": entry, "why": why}));
                return None;
            }
        };
        peaks.restore(&id.model, &manager);
        log.note(serde_json::json!({"event": "manager", "gpu": entry, "card": id.memory, "chunks": chunks}));
        let watch = Watch::start(&id.uuid, manager.clone(), leases.tenants, log);
        Some(Gpu { manager, model: id.model, watch })
    }

    /// Executors to end: those answering no revoke.
    pub fn stalled(&self) -> Vec<u32> {
        self.manager.stalled()
    }

    /// The manager's books changed: the watch looks whether to read again.
    pub fn changed(&self) {
        self.watch.change();
    }
}

/// How often the watch reads NVML while cozy holds bytes on the GPU (design vram-yield).
const SAMPLE: std::time::Duration = std::time::Duration::from_millis(500);
/// A foreign process that appears and crosses this is new: idle pages make room for it.
const NEW_FOREIGN: u64 = 256 << 20;

/// One GPU's watch: every half second while cozy holds bytes there (chunks or executors), else
/// parked until the books change, it hands the manager a reading (`set_foreign`, its tick: the
/// manager re-reads the driver's free bytes and reclaims in the background below its low
/// watermark). NVML's bytes no cozy process holds go with it for display; a foreign pid absent
/// when sampling began that crosses 256 MiB marks its reading new. Where NVML names no pid of ours
/// (a container's), foreign is 0: a pod is cozy's alone.
struct Watch {
    epoch: Mutex<u64>,
    changed: std::sync::Condvar,
}

impl Watch {
    fn start(uuid: &str, manager: Arc<Manager>, tenants: Tenants, log: &MemoryLog) -> Arc<Watch> {
        let watch = Arc::new(Watch { epoch: Mutex::default(), changed: std::sync::Condvar::new() });
        let (uuid, watching, notes) = (uuid.to_string(), Arc::downgrade(&watch), log.clone());
        let spawned = std::thread::Builder::new().name("gpu-watch".into()).spawn(move || watching_gpu(&uuid, &manager, &tenants, watching, &notes));
        if let Err(error) = spawned {
            log.note(serde_json::json!({"event": "gpu_unwatched", "why": error.to_string()}));
        }
        watch
    }

    fn change(&self) {
        *self.epoch.lock().unwrap_or_else(|e| e.into_inner()) += 1;
        self.changed.notify_all();
    }
}

fn watching_gpu(uuid: &str, manager: &Manager, tenants: &Tenants, watch: std::sync::Weak<Watch>, log: &MemoryLog) {
    let device = platform::gpu::Device::open(uuid).map_err(|error| error.to_string());
    let (mut known, mut stale, mut drift) = (None::<std::collections::BTreeSet<u32>>, false, None);
    while let Some(watch) = watch.upgrade() {
        {
            let mut epoch = watch.epoch.lock().unwrap_or_else(|e| e.into_inner());
            let held = manager.holdings();
            if held.created == 0 && held.executors.is_empty() {
                known = None;
                let seen = *epoch;
                while *epoch == seen {
                    epoch = watch.changed.wait(epoch).unwrap_or_else(|e| e.into_inner());
                }
                continue;
            }
        }
        drop(watch);
        let reading = device.as_ref().map_err(Clone::clone).and_then(|device| device.read().map_err(|error| error.to_string()));
        let processes = reading.and_then(|reading| reading.processes.ok_or_else(|| "NVML names no process of this machine".to_string()));
        match (&processes, stale) {
            (Err(why), false) => log.note(serde_json::json!({"event": "gpu_unmeasured", "gpu": uuid, "why": why})),
            (Ok(_), true) => log.note(serde_json::json!({"event": "gpu_measured", "gpu": uuid})),
            _ => {}
        }
        stale = processes.is_err();
        let (foreign, new_pid) = match &processes {
            Ok(processes) => {
                drift = books(processes, &manager.holdings(), drift, uuid, log);
                let ours: std::collections::BTreeSet<u32> = tenants.book().keys().copied().chain([std::process::id()]).collect();
                let others: BTreeMap<u32, u64> = processes.iter().filter(|(pid, _)| !ours.contains(pid)).map(|(pid, bytes)| (*pid, *bytes)).collect();
                let seen = known.get_or_insert_with(|| others.keys().copied().collect());
                let new = others.iter().filter(|(pid, bytes)| !seen.contains(pid) && **bytes >= NEW_FOREIGN).map(|(pid, _)| *pid).collect::<Vec<_>>();
                seen.extend(&new);
                (others.values().sum(), !new.is_empty())
            }
            Err(_) => (0, false),
        };
        manager.set_foreign(foreign, new_pid);
        std::thread::sleep(SAMPLE);
    }
}

/// The inventory's GPU `entry` names: an index, or a `GPU-…` UUID.
pub fn row<'a>(entry: &str, inventory: &'a [receipt::Gpu]) -> Option<&'a receipt::Gpu> {
    inventory.iter().find(|g| g.device_index.to_string() == entry || g.device_uuid == entry)
}

/// One GPU executor process: its plan, package generation and lane, and a handle that ends
/// exactly it.
pub struct Tenant {
    pub plan: String,
    pub generation: String,
    pub start: usize,
    pub exact: Exact,
}

/// Every live GPU executor process (leaders and followers).
pub type Tenants = Arc<Registry>;

/// The tenants by pid, and the groups whose Start is forming them: a follower that dials before
/// its rank 0 reports it waits for its registration ([`Peers`]).
#[derive(Default)]
pub struct Registry {
    book: Mutex<Book>,
    changed: std::sync::Condvar,
}

/// What the registry holds: its tenants (the map it derefs to), and the leaders whose groups form
/// now.
#[derive(Default)]
pub struct Book {
    tenants: BTreeMap<u32, Tenant>,
    forming: BTreeMap<u32, Group>,
}

/// A forming group: its rank 0, the followers its Start expects, and its peers waiting now.
struct Group {
    leader: Birth,
    followers: usize,
    parked: usize,
}

impl std::ops::Deref for Book {
    type Target = BTreeMap<u32, Tenant>;
    fn deref(&self) -> &Self::Target {
        &self.tenants
    }
}

impl std::ops::DerefMut for Book {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tenants
    }
}

impl Registry {
    pub fn lock(&self) -> std::sync::LockResult<std::sync::MutexGuard<'_, Book>> {
        self.book.lock()
    }

    fn book(&self) -> std::sync::MutexGuard<'_, Book> {
        self.book.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Registers `tenants`; a peer waiting for one is admitted.
    pub fn register(&self, tenants: impl IntoIterator<Item = (u32, Tenant)>) {
        self.book().tenants.extend(tenants);
        self.changed.notify_all();
    }

    /// `leader`'s Start forms its group of `followers` (ranks - 1) until the returned guard drops:
    /// up to that many members of its process group that dial meanwhile wait to be registered
    /// (more are refused, so a package's children cannot hold threads by dialing), and are refused
    /// if the Start ends without registering them.
    pub fn forming(self: &Arc<Self>, leader: &Birth, followers: usize) -> Forming {
        self.book().forming.insert(leader.pid, Group { leader: leader.clone(), followers, parked: 0 });
        Forming { registry: self.clone(), leader: leader.pid }
    }

    /// Peers waiting for their registration now.
    pub fn parked(&self) -> usize {
        self.book().forming.values().map(|group| group.parked).sum()
    }
}

/// A group forming ([`Registry::forming`]).
pub struct Forming {
    registry: Arc<Registry>,
    leader: u32,
}

impl Drop for Forming {
    fn drop(&mut self) {
        self.registry.book().forming.remove(&self.leader);
        self.registry.changed.notify_all();
    }
}

/// Who a peer on a GPU's lease listener is: a live tenant (an executor, or a follower its rank 0
/// reported), running as one of `uids` (this machine's, its launch identity's). A member of a
/// group whose Start forms it now waits until it is registered, or refused when that Start ends
/// without it: TensorD knows the Start's state, so nothing redials or times out. Anything else a
/// package starts is refused at once, so every leased pid is one the health path can end. Its
/// executor is its group (plan and lane), its trust domain its package generation and user. Its
/// grants, revokes, reclaims and adoptions are memory-log lines.
pub struct Peers {
    tenants: Tenants,
    uids: Vec<u32>,
    notes: MemoryLog,
}

impl Peers {
    pub fn new(tenants: Tenants, uids: Vec<u32>, notes: MemoryLog) -> Peers {
        Peers { tenants, uids, notes }
    }
}

impl Hosting for Peers {
    fn identify(&self, peer: &Peer) -> Option<Identity> {
        if !self.uids.contains(&peer.uid) {
            return None;
        }
        let forming: Vec<Birth> = self.tenants.book().forming.values().map(|group| group.leader.clone()).collect();
        let leader = forming.iter().find(|leader| platform::process::group_members(leader).iter().any(|m| m.pid == peer.pid)).map(|leader| leader.pid);
        let mut book = self.tenants.book();
        let mut waited = false;
        let identity = loop {
            if let Some(tenant) = book.tenants.get(&peer.pid).filter(|tenant| !tenant.exact.ended()) {
                break Some(Identity { executor: format!("{}#{}", tenant.plan, tenant.start), domain: format!("{}/{}", tenant.generation, peer.uid) });
            }
            let Some(group) = leader.and_then(|leader| book.forming.get_mut(&leader)) else { break None };
            if !waited {
                if group.parked >= group.followers {
                    break None;
                }
                group.parked += 1;
                waited = true;
            }
            book = self.tenants.changed.wait(book).unwrap_or_else(|e| e.into_inner());
        };
        if let Some(group) = leader.filter(|_| waited).and_then(|leader| book.forming.get_mut(&leader)) {
            group.parked -= 1;
        }
        identity
    }

    fn record(&self, record: Record) {
        match serde_json::to_value(&record) {
            Ok(event) => self.notes.note(event),
            Err(error) => tracing::warn!("memory: a lease record unwritten: {error}"),
        }
    }
}

/// The books check (design varena-leases R2-6): NVML charges each chunk to its creator, so
/// TensorD's own bytes are what its manager created plus its context. A gap past one chunk is a
/// memory-log line when it starts, and again whenever it moves by another chunk; the gap is
/// returned for the next reading.
fn books(processes: &BTreeMap<u32, u64>, held: &varena::lease::Holdings, was: Option<i128>, uuid: &str, log: &MemoryLog) -> Option<i128> {
    let own = processes.get(&std::process::id()).copied().unwrap_or(0);
    let gap = i128::from(own) - i128::from(held.created + held.context);
    let chunk = i128::from(varena::lease::CHUNK);
    if gap.abs() <= chunk {
        return None;
    }
    if was.is_none_or(|was| (gap - was).abs() > chunk) {
        log.note(serde_json::json!({"event": "books_drift", "gpu": uuid, "own_bytes": own, "created_bytes": held.created, "context_bytes": held.context, "gap_bytes": gap}));
        return Some(gap);
    }
    was
}

/// GPU `uuid`'s chunks, opened in TensorD's own context at its manager's first ask and released
/// when the manager holds no chunk and serves no executor (a context costs VRAM, and keeps a
/// laptop's GPU awake). None (a memory-log note says why) where its compute mode admits one
/// process at a time, whose context TensorD's would lock out, or CUDA cannot share its pages.
/// Only NVML and CUDA's driver are read here: no context is made.
pub fn chunks(uuid: &str, log: &MemoryLog) -> Option<Opener> {
    let off = |why: String| log.note(serde_json::json!({"event": "leases_off", "gpu": uuid, "why": why}));
    match platform::gpu::Device::open(uuid).and_then(|device| device.exclusive()) {
        Ok(true) => {
            off("its compute mode admits one process at a time".into());
            return None;
        }
        Ok(false) => {}
        Err(error) => tracing::warn!("memory: GPU {uuid}'s compute mode is unread ({error}): CUDA decides"),
    }
    let ordinal = shareable(uuid).inspect_err(|why| off(why.clone())).ok()?;
    Some(Box::new(move || CudaBackend::open(ordinal).map(|backend| Arc::new(backend) as Arc<dyn Backend>)))
}

/// The CUDA ordinal of GPU `uuid` in this process (neither NVML's index nor an executor's,
/// `CUDA_VISIBLE_DEVICES`), whose pages CUDA can create and share.
fn shareable(uuid: &str) -> Result<i32, String> {
    use tensorfs::io::cuda;
    let count = cuda::device_count().map_err(|error| error.to_string())?;
    let device = |ordinal: i32| {
        let device = cuda::device_at(ordinal).ok()?;
        cuda::uuid(device).is_ok_and(|named| named == uuid).then_some((ordinal, device))
    };
    let (ordinal, device) = (0..count).find_map(device).ok_or_else(|| format!("CUDA has no device {uuid}"))?;
    #[cfg(unix)]
    let handles = cuda::ATTR_POSIX_FD_SUPPORTED;
    #[cfg(windows)]
    let handles = cuda::ATTR_WIN32_HANDLE_SUPPORTED;
    for (attribute, what) in [(cuda::ATTR_VMM_SUPPORTED, "virtual memory management"), (handles, "shareable handles")] {
        if cuda::attribute(device, attribute).map_err(|error| error.to_string())? == 0 {
            return Err(format!("CUDA device {uuid} has no {what}"));
        }
    }
    Ok(ordinal)
}

/// Admits request `id` at `<package>|<shape>` on every one of `managers` or none
/// (`Manager::admit`, which names the key its peak is recorded under). Refused, it waits for the
/// engine's activity, which every manager event and every cancel bumps, and asks again: running
/// requests end, and a GPU nothing runs on always admits. False: it was canceled meanwhile. A
/// manager that cannot answer lets it run unrecorded.
pub fn admit(engine: &Engine, managers: &[&Manager], id: &str, package: &str, shape: &str) -> io::Result<bool> {
    loop {
        let seen = engine.activity_epoch();
        match Manager::admit(managers, id, package, shape) {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(error) => {
                tracing::warn!("memory: request {id} runs unrecorded: {error}");
                return Ok(true);
            }
        }
        if engine.get(id)?.cancel_actor.is_some() {
            return Ok(false);
        }
        engine.wait_activity(seen, None);
    }
}

/// The GPU of the group (an index into `managers`, GPU order) that keeps each set of `sets`, by
/// `Manager::place` for `package`'s request at `shape`. A set's bytes are what its load recorded
/// on a GPU of the group: a set never loaded goes where the most room is free. A manager that
/// cannot answer keeps every set on GPU 0, where it pages.
pub fn homes(managers: &[&Manager], package: &str, shape: &str, sets: &[&str], log: &MemoryLog) -> Vec<u32> {
    let unknown: Vec<(&str, Option<u64>)> = sets.iter().map(|set| (*set, None)).collect();
    match Manager::place(managers, package, shape, &unknown) {
        Ok(homes) => homes,
        Err(error) => {
            log.note(serde_json::json!({"event": "homes_on_gpu_0", "package": package, "why": error.to_string()}));
            vec![0; sets.len()]
        }
    }
}

/// Measured peaks, persisted per GPU model as `<dir>/<model>.json`: varena's records, kept and
/// restored whole. A key names its package release (`<construction>|<shape>`, or a set's
/// `<package>@<release>/<component>|…`), so one file holds every release's. Each GPU of the
/// model contributes its records; a key two of them measured is merged by the manager's restore.
pub struct Peaks {
    dir: PathBuf,
    saved: Mutex<BTreeMap<String, Vec<Peak>>>,
}

impl Peaks {
    pub fn at(dir: PathBuf) -> Self {
        Self { dir, saved: Mutex::default() }
    }

    fn path(&self, model: &str) -> PathBuf {
        let name: String = model.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        self.dir.join(format!("{name}.json"))
    }

    fn restore(&self, model: &str, manager: &Manager) {
        let mut saved = self.saved.lock().unwrap();
        let held = saved.entry(model.into()).or_insert_with(|| {
            let read = fs::read(self.path(model)).ok().and_then(|bytes| serde_json::from_slice::<Vec<Peak>>(&bytes).ok());
            read.unwrap_or_default()
        });
        if let Err(error) = manager.restore(held) {
            tracing::warn!("memory: {model}'s peaks not restored: {error}");
        }
    }

    /// Saves what `managers` (every GPU of `model`, each holding what was restored into it) now
    /// hold, when it changed.
    pub fn save<'a>(&self, model: &str, managers: impl Iterator<Item = &'a Manager>) {
        let mut saved = self.saved.lock().unwrap();
        let mut held: Vec<Peak> = vec![];
        for peak in managers.flat_map(Manager::peaks) {
            if !held.contains(&peak) {
                held.push(peak);
            }
        }
        if saved.get(model).is_some_and(|before| before.len() == held.len() && held.iter().all(|peak| before.contains(peak))) {
            return;
        }
        let written = fs::create_dir_all(&self.dir)
            .map_err(platform::Error::from)
            .and_then(|()| platform::durable::replace(&self.path(model), &serde_json::to_vec(&held).unwrap_or_default(), 0o600));
        match written {
            Ok(()) => drop(saved.insert(model.into(), held)),
            Err(error) => tracing::warn!("memory: {model}'s peaks not saved: {error}"),
        }
    }
}

/// One call running on the machine's GPUs: its executor and its `<package>|<shape>`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Call {
    pub pid: u32,
    pub package: String,
    pub shape: String,
}

/// The calls running now, by request: TensorD's own record (admitted until they end), which the
/// host rule reads beside each GPU's measured host peaks.
pub type Running = Arc<Mutex<BTreeMap<String, Call>>>;

/// A process running calls, as the host rule reads it: its calls' measured anonymous peak (none
/// while one is unmeasured), its anonymous bytes now, and how many calls it runs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostCall {
    pub pid: u32,
    pub peak: Option<u64>,
    pub anon: u64,
    pub running: u64,
}

/// The running calls as the host rule reads them: each call's host peak is the largest any GPU
/// measured for its `<package>|<shape>`.
pub fn host_calls(running: &Running, gpus: &[Arc<Gpu>]) -> Vec<HostCall> {
    let calls: Vec<Call> = running.lock().unwrap_or_else(|e| e.into_inner()).values().cloned().collect();
    calls
        .into_iter()
        .map(|call| HostCall {
            pid: call.pid,
            peak: gpus.iter().filter_map(|gpu| gpu.manager.host_peak(&call.package, &call.shape)).max(),
            anon: host_memory::resident(call.pid).map_or(0, |now| now.anon),
            running: 1,
        })
        .collect()
}

/// What woke the events thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Woke {
    /// A GPU's manager: room freed, an executor exited, a peak was recorded.
    Gpu,
    /// The host is short of what its running calls must keep free ([`Host`]).
    HostShort,
}

/// A source the events thread waits on: a GPU manager's events, or the host's short event.
pub enum Source {
    Gpu(Arc<Gpu>),
    Host(Arc<Host>),
}

impl Source {
    fn waitable(&self) -> Waitable<'_> {
        match self {
            Source::Gpu(gpu) => gpu.manager.events(),
            Source::Host(host) => host.short.waitable(),
        }
    }

    /// Clears what it signalled (a ready event: no wait).
    fn clear(&self) -> Woke {
        match self {
            Source::Gpu(gpu) => {
                gpu.manager.clear_events();
                Woke::Gpu
            }
            Source::Host(host) => {
                host.short();
                Woke::HostShort
            }
        }
    }
}

/// Blocks on every source until the returned guard drops; `wake` runs once per wake with what
/// woke it. Every ready source is cleared on each wake, so none starves; one that broke (hung up,
/// failed) is dropped, never spun on. It never times out.
pub fn react(sources: Vec<Source>, wake: impl Fn(Woke) + Send + 'static) -> io::Result<Reacting> {
    let stop = Arc::new(Signal::new().map_err(io::Error::other)?);
    let stopped = stop.clone();
    std::thread::Builder::new().name("memory-events".into()).spawn(move || {
        let mut live: Vec<&Source> = sources.iter().collect();
        loop {
            let waitables: Vec<Waitable<'_>> = std::iter::once(stopped.waitable()).chain(live.iter().map(|source| source.waitable())).collect();
            let ready = match wait_any(&waitables) {
                Ok(ready) => ready,
                Err(error) => return tracing::warn!("memory: events: {error}"),
            };
            if ready.iter().any(|ready| ready.index == 0) {
                return;
            }
            let mut woke = vec![];
            for ready in ready.iter().filter(|ready| !ready.broken) {
                let what = live[ready.index - 1].clear();
                if !woke.contains(&what) {
                    woke.push(what);
                }
            }
            let broken: Vec<usize> = ready.iter().filter(|ready| ready.broken).map(|ready| ready.index - 1).collect();
            live = live.into_iter().enumerate().filter(|(i, _)| !broken.contains(i)).map(|(_, source)| source).collect();
            woke.into_iter().for_each(&wake);
        }
    })?;
    Ok(Reacting(stop))
}

/// The events thread's life: dropped, it ends.
pub struct Reacting(Arc<Signal>);

impl Drop for Reacting {
    fn drop(&mut self) {
        self.0.set();
    }
}

/// The host rule (#1851) on the platform's watcher: each pressure event, or a nudge (TensorD's
/// own events, where the kernel gives none), is a look. What the calls running on these GPUs
/// must keep free, each process's measured anonymous peak less its anonymous bytes now plus
/// the growth seen past it ([`Growth`]), against the live budget, page cache counted as room.
/// Short, the events thread hears it ([`Woke::HostShort`]): the pool ends one idle process, and
/// the next event looks again. Another is ended only once the last end measurably raised the
/// budget: when ending cannot help (a growing call alone exceeds the host), the call meets its
/// own limit and the idle processes stay. A look that is not short starts over. No timer.
pub struct Host {
    short: Arc<Signal>,
    nudge: Nudge,
    /// Looks begun and made so far, and a signal for each made.
    looks: Arc<(Mutex<(u64, u64)>, std::sync::Condvar)>,
}

impl Host {
    /// Watches while the process lives; None (a memory-log note says why) where the platform
    /// gives no pressure events.
    pub fn watch(mode: HostMode, gpus: Vec<Arc<Gpu>>, running: Running, log: &MemoryLog) -> Option<Arc<Host>> {
        let armed = host_pressure::Pressure::arm(mode).and_then(|pressure| Ok((pressure, Arc::new(Signal::new()?))));
        let (mut pressure, short) = armed
            .inspect_err(|error| log.note(serde_json::json!({"event": "host_unwatched", "mode": format!("{mode:?}"), "why": error.to_string()})))
            .ok()?;
        let (nudge, looks) = (pressure.nudge(), Arc::new((Mutex::new((0, 0)), std::sync::Condvar::new())));
        let (looked, signal) = (looks.clone(), short.clone());
        let watching = std::thread::Builder::new().name("host-pressure".into()).spawn(move || {
            let mut keep = Keep::default();
            // The budget when short was last signalled: the next signal needs more than it.
            let mut signalled: Option<u64> = None;
            while pressure.wait().is_ok() {
                looked.0.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
                let need = keep.look(host_calls(&running, &gpus));
                let budget = u64::try_from(host_memory::read().available).unwrap_or(u64::MAX);
                if budget >= need {
                    signalled = None;
                } else if signalled.is_none_or(|was| budget > was) {
                    signal.set();
                    signalled = Some(budget);
                }
                looked.0.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                looked.1.notify_all();
            }
        });
        if let Err(error) = watching {
            log.note(serde_json::json!({"event": "host_unwatched", "why": error.to_string()}));
            return None;
        }
        Some(Arc::new(Host { short, nudge, looks }))
    }

    /// Something happened (a call began, an executor's frame came): the watcher looks.
    pub fn nudge(&self) {
        self.nudge.send();
    }

    /// Nudges the watcher and blocks until a look begun since has been made.
    pub fn look(&self) {
        let begun = self.looks.0.lock().unwrap_or_else(|e| e.into_inner()).0;
        self.nudge();
        self.await_look(begun);
    }

    /// Looks made so far.
    pub fn looks(&self) -> u64 {
        self.looks.0.lock().unwrap_or_else(|e| e.into_inner()).1
    }

    /// Blocks until more than `made` looks have been made.
    pub fn await_look(&self, made: u64) {
        let (count, signal) = &*self.looks;
        let mut looks = count.lock().unwrap_or_else(|e| e.into_inner());
        while looks.1 <= made {
            looks = signal.wait(looks).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Whether a look found the host short since this was last asked (the events thread asks).
    pub fn short(&self) -> bool {
        self.short.clear()
    }
}

/// What the host keeps free for the calls running on it: each process's anonymous growth still
/// to come (its requests' measured anonymous peak less its anonymous bytes now) and [`Growth`],
/// the most a look saw them grow past what was measured to come (an unmeasured call's every
/// byte). Page cache is never need: the budget holds it.
#[derive(Default)]
struct Keep {
    growth: Growth,
    /// Each process's requests running at the last look: each that ended settles the growth.
    running: BTreeMap<u32, u64>,
}

impl Keep {
    /// A look at the processes running calls: what must stay free.
    fn look(&mut self, calls: Vec<HostCall>) -> u64 {
        // A process running several calls is one entry.
        let mut by: BTreeMap<u32, HostCall> = BTreeMap::new();
        for call in calls {
            by.entry(call.pid)
                .and_modify(|seen| {
                    seen.peak = seen.peak.zip(call.peak).map(|(a, b)| a.max(b));
                    seen.running += call.running;
                })
                .or_insert(call);
        }
        let (mut need, mut unforeseen) = (0u64, BTreeMap::new());
        for call in by.values() {
            let peak = call.peak.unwrap_or(0);
            need = need.saturating_add(peak.saturating_sub(call.anon));
            unforeseen.insert(call.pid.to_string(), call.anon.saturating_sub(peak));
        }
        for (pid, was) in &self.running {
            for _ in by.get(pid).map_or(0, |call| call.running)..*was {
                self.growth.settle();
            }
        }
        self.running = by.values().map(|call| (call.pid, call.running)).collect();
        need.saturating_add(self.growth.observe(unforeseen))
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// Where one machine's memory events go: `memory.log` in its store's `logs`, beside TensorFS's
/// transport log, read by `cozy machine logs --memory`. None logs to tracing.
#[derive(Clone, Debug, Default)]
pub struct MemoryLog(Option<Arc<PathBuf>>);

impl MemoryLog {
    pub fn at(dir: PathBuf) -> Self {
        Self(Some(Arc::new(dir)))
    }

    pub(crate) fn note(&self, event: serde_json::Value) {
        let Some(dir) = self.0.as_deref() else {
            return tracing::info!("memory: {event}");
        };
        let path = dir.join("memory.log");
        if fs::metadata(&path).is_ok_and(|meta| meta.len() > ROTATE_BYTES) {
            let _ = fs::rename(&path, dir.join("memory.log.1"));
        }
        let line = format!("{:.3} {event}\n", now_ms() as f64 / 1e3);
        let written = fs::create_dir_all(dir)
            .and_then(|_| OpenOptions::new().create(true).append(true).open(&path))
            .and_then(|mut file| file.write_all(line.as_bytes()));
        if let Err(error) = written {
            tracing::warn!("memory: {event} ({}: {error})", path.display());
        }
    }
}
