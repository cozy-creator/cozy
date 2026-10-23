//! TensorD hosts varena's lease manager (design varena-leases): each envelope GPU the inventory
//! names has one, which creates the GPU's chunks and leases them to its executors, and owns the
//! GPU's [`Ledger`] (capacity min(card, `gpu_memory_limit`)) until the ledger folds into it. An
//! executor is granted its row at spawn, its row is reaped once its exit is proven, and admission
//! is [`admit`] (`Ledger::admit`: a group on all of its GPUs or none, waiting while refused).
//! A GPU TensorD cannot create chunks on keeps the ledger alone. Host RAM is the OS page cache;
//! [`Host`] watches what the running calls must keep free of it. One thread blocks on every
//! GPU's events and the host's short event.
use crate::{execution::Engine, machine::receipt};
use platform::contain::host_memory;
use platform::contain::host_pressure::{self, Growth, HostMode, Nudge};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, RawFd},
        unix::net::UnixStream,
    },
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use platform::{handles::Peer, process::Exact};
use varena::lease::{Hosting, Identity, Manager, Record};
use varena::{
    ledger::{FixedRow, Mode},
    Backend, CudaBackend, GpuId, HostCall, Ledger, Peak,
};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// One envelope GPU: its lease manager, or its ledger alone; and the GPU model its peaks are
/// kept under.
pub struct Gpu {
    held: Held,
    pub model: String,
    uuid: String,
    fixed: Arc<Mutex<Fixed>>,
}

/// Standalone executors' fixed rows by pid (varena-leases R2-2), and whether their sampler runs.
#[derive(Default)]
struct Fixed {
    rows: BTreeMap<u32, FixedRow>,
    sampling: bool,
}

/// How often a fixed row's bytes are read again while one lives.
const FIXED_SAMPLE: std::time::Duration = std::time::Duration::from_millis(500);

enum Held {
    Leases(Arc<Manager>),
    Ledger(Ledger),
}

/// What hosting a GPU's leases takes: who the peers on its listener are, and its chunks.
pub struct Leases {
    pub hosting: Arc<dyn Hosting>,
    pub open: Opener,
}

impl Gpu {
    /// `entry`'s memory, sized by the inventory's memory (an integrated GPU's is the host's) and
    /// never above `limit`, in the machine's sharing `mode`, with the peaks this machine measured
    /// on its model: a lease manager serving `leases`, else (none given, or its listener or
    /// manager cannot be made: a memory-log note says why) the ledger alone. None when the
    /// inventory does not name it or its table cannot be made.
    pub fn create(entry: &str, inventory: &[receipt::Gpu], limit: Option<u64>, mode: HostMode, leases: Option<Leases>, peaks: &Peaks, log: &MemoryLog) -> Option<Gpu> {
        let row = row(entry, inventory)?;
        let id = GpuId { uuid: row.device_uuid.clone(), model: row.device_name.clone(), memory: row.memory_bytes };
        let mode = match mode {
            HostMode::Shared => Mode::Shared,
            HostMode::Dedicated => Mode::Dedicated,
        };
        let held = match leases.map(|leases| manage(&id, limit, mode, leases)) {
            Some(Ok(manager)) => Held::Leases(manager),
            refused => {
                if let Some(Err(why)) = refused {
                    log.note(serde_json::json!({"event": "leases_off", "gpu": entry, "why": why}));
                }
                let ledger = match Ledger::create(&id, limit) {
                    Ok(ledger) => ledger,
                    Err(error) => {
                        tracing::warn!("memory: GPU {entry} has no ledger: {error}");
                        return None;
                    }
                };
                if let Err(error) = ledger.set_mode(mode) {
                    tracing::warn!("memory: GPU {entry} stays shared: {error}");
                }
                Held::Ledger(ledger)
            }
        };
        let gpu = Gpu { held, model: id.model.clone(), uuid: id.uuid.clone(), fixed: Arc::default() };
        peaks.restore(&id.model, gpu.ledger());
        let leased = gpu.manager().is_some();
        log.note(serde_json::json!({"event": "ledger", "gpu": entry, "card": id.memory, "capacity": gpu.ledger().status().capacity, "leases": leased}));
        Some(gpu)
    }

    /// A GPU on `ledger` alone.
    pub fn ledgered(ledger: Ledger, model: &str) -> Gpu {
        Gpu { held: Held::Ledger(ledger), model: model.into(), uuid: String::new(), fixed: Arc::default() }
    }

    /// Standalone executor `pid` on it: a fixed row, never asked and never foreign, holding its
    /// bytes as NVML charges its pid, read every half second while any fixed row lives (where
    /// NVML names no pid of ours, a container's, they stay unmeasured).
    pub fn fix(&self, pid: u32) {
        let row = match self.ledger().fixed(pid) {
            Ok(row) => row,
            Err(error) => return tracing::warn!("memory: standalone executor {pid} has no fixed row: {error}"),
        };
        let mut fixed = self.fixed.lock().unwrap();
        fixed.rows.insert(pid, row);
        if !std::mem::replace(&mut fixed.sampling, true) {
            let (uuid, rows) = (self.uuid.clone(), Arc::downgrade(&self.fixed));
            let spawned = std::thread::Builder::new().name("fixed-rows".into()).spawn(move || sample_fixed(&uuid, rows));
            if let Err(error) = spawned {
                fixed.sampling = false;
                tracing::warn!("memory: fixed rows unsampled: {error}");
            }
        }
    }

    /// After `waitpid`: `pid`'s rows go.
    pub fn reap(&self, pid: u32) {
        self.fixed.lock().unwrap().rows.remove(&pid);
        self.ledger().reap(pid);
    }

    /// The GPU's admission and placement facts (the manager's ledger, until step 4).
    pub fn ledger(&self) -> &Ledger {
        match &self.held {
            Held::Leases(manager) => manager.ledger(),
            Held::Ledger(ledger) => ledger,
        }
    }

    /// Its lease manager, where TensorD creates its chunks.
    pub fn manager(&self) -> Option<&Arc<Manager>> {
        match &self.held {
            Held::Leases(manager) => Some(manager),
            Held::Ledger(_) => None,
        }
    }

    /// Executors to end: the ledger's stalled rows, and those answering no revoke.
    pub fn stalled(&self) -> Vec<u32> {
        match &self.held {
            Held::Leases(manager) => manager.stalled(),
            Held::Ledger(ledger) => ledger.stalled(),
        }
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

/// Every live GPU executor process (leaders and followers) by pid.
pub type Tenants = Arc<Mutex<BTreeMap<u32, Tenant>>>;

/// Who a peer on a GPU's lease listener is: a live tenant (an executor, or a follower its rank 0
/// reported), running as one of `uids` (this machine's, its launch identity's); anything else a
/// package starts is refused, so every leased pid is one the health path can end. Its executor is
/// its group (plan and lane), its trust domain its package generation and user. Its grants,
/// revokes, reclaims and adoptions are memory-log lines.
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
        let tenants = self.tenants.lock().unwrap();
        let tenant = tenants.get(&peer.pid).filter(|tenant| !tenant.exact.ended())?;
        Some(Identity { executor: format!("{}#{}", tenant.plan, tenant.start), domain: format!("{}/{}", tenant.generation, peer.uid) })
    }

    fn record(&self, record: Record) {
        match serde_json::to_value(&record) {
            Ok(event) => self.notes.note(event),
            Err(error) => tracing::warn!("memory: a lease record unwritten: {error}"),
        }
    }
}

/// Sets each fixed row's bytes from NVML's per-pid reading until none is left.
fn sample_fixed(uuid: &str, fixed: std::sync::Weak<Mutex<Fixed>>) {
    let device = platform::gpu::Device::open(uuid).inspect_err(|error| tracing::warn!("memory: fixed rows unmeasured: {error}")).ok();
    while let Some(fixed) = fixed.upgrade() {
        let processes = device.as_ref().and_then(|device| device.read().ok()).and_then(|reading| reading.processes);
        let mut held = fixed.lock().unwrap();
        if held.rows.is_empty() {
            held.sampling = false;
            return;
        }
        for (pid, row) in processes.iter().flat_map(|processes| held.rows.iter().map(move |(pid, row)| (processes.get(pid), row))) {
            row.set(pid.copied().unwrap_or(0));
        }
        drop((held, fixed));
        std::thread::sleep(FIXED_SAMPLE);
    }
}

/// `id`'s manager on its listener (`lease::listen`: named by its UUID in this user's namespace).
fn manage(id: &GpuId, limit: Option<u64>, mode: Mode, leases: Leases) -> Result<Arc<Manager>, String> {
    let listener = varena::lease::listen(id)
        .map_err(|error| format!("its listener: {error}"))?
        .ok_or_else(|| "another process serves its leases".to_string())?;
    let open = leases.open;
    Manager::create(id, limit, mode, Box::new(move || open()), listener, leases.hosting).map_err(|error| error.to_string())
}

/// Opens a GPU's chunks: TensorD's own context, made at its manager's first ask and released
/// when the manager holds no chunk and serves no executor (a context costs VRAM, and keeps a
/// laptop's GPU awake).
pub type Opener = Arc<dyn Fn() -> varena::Result<Arc<dyn Backend>> + Send + Sync>;

/// GPU `uuid`'s opener. None (a memory-log note says why) where its compute mode admits one
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
    Some(Arc::new(move || CudaBackend::open(ordinal).map(|backend| Arc::new(backend) as Arc<dyn Backend>)))
}

/// The CUDA ordinal of GPU `uuid` in this process (neither NVML's index nor an executor's,
/// `CUDA_VISIBLE_DEVICES`), whose pages CUDA can create and share.
fn shareable(uuid: &str) -> Result<i32, String> {
    use tensorfs::io::cuda;
    let driver = cuda::driver().map_err(|error| error.to_string())?;
    let mut count = 0;
    // SAFETY: cuDeviceGetCount's out-param, a live local.
    cuda::check("cuDeviceGetCount", unsafe { (driver.device_get_count)(&mut count) }).map_err(|error| error.to_string())?;
    let device = |ordinal: i32| {
        let mut device = 0;
        // SAFETY: cuDeviceGet's out-param, a live local, for an ordinal below the count.
        cuda::check("cuDeviceGet", unsafe { (driver.device_get)(&mut device, ordinal) }).ok()?;
        cuda::uuid(device).is_ok_and(|named| named == uuid).then_some((ordinal, device))
    };
    let (ordinal, device) = (0..count).find_map(device).ok_or_else(|| format!("CUDA has no device {uuid}"))?;
    for (attribute, what) in [(cuda::ATTR_VMM_SUPPORTED, "virtual memory management"), (cuda::ATTR_POSIX_FD_SUPPORTED, "shareable handles")] {
        if cuda::attribute(device, attribute).map_err(|error| error.to_string())? == 0 {
            return Err(format!("CUDA device {uuid} has no {what}"));
        }
    }
    Ok(ordinal)
}

/// Admits request `id` of `family` at `<package>|<shape>` on every one of `ledgers` or none
/// (`Ledger::admit`, which names the key its peak is recorded under; a yielded request of the
/// same family, waiting on this one, is left out). Refused, it waits for the engine's activity,
/// which every ledger event and every cancel bumps, and asks again: running requests end, and a
/// GPU nothing runs on always admits. False: it was canceled meanwhile. A ledger that cannot
/// answer lets it run unrecorded.
pub fn admit(engine: &Engine, ledgers: &[&Ledger], id: &str, package: &str, shape: &str, family: Option<&str>) -> io::Result<bool> {
    loop {
        let seen = engine.activity_epoch();
        match Ledger::admit(ledgers, id, package, shape, family) {
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

/// The GPU of the group (an index into `ledgers`, GPU order) that keeps each set of `sets`, by
/// `Ledger::place` for `package`'s request at `shape` in `family`. A set's bytes are what its
/// load recorded on a GPU of the group (its layout's, exact; its region split is the model's,
/// known only where it loads): a set never loaded goes where the most room is free. A ledger
/// that cannot answer keeps every set on GPU 0, where it pages.
pub fn homes(ledgers: &[&Ledger], package: &str, shape: &str, family: Option<&str>, sets: &[&str], log: &MemoryLog) -> Vec<u32> {
    let unknown: Vec<(&str, Option<u64>)> = sets.iter().map(|set| (*set, None)).collect();
    match Ledger::place(ledgers, package, shape, family, &unknown) {
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
/// model contributes its records; a key two of them measured is merged by the ledger's restore.
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

    fn restore(&self, model: &str, ledger: &Ledger) {
        let mut saved = self.saved.lock().unwrap();
        let held = saved.entry(model.into()).or_insert_with(|| {
            let read = fs::read(self.path(model)).ok().and_then(|bytes| serde_json::from_slice::<Vec<Peak>>(&bytes).ok());
            read.unwrap_or_default()
        });
        if let Err(error) = ledger.restore(held) {
            tracing::warn!("memory: {model}'s peaks not restored: {error}");
        }
    }

    /// Saves what `ledgers` (every GPU of `model`, each holding what was restored into it) now
    /// hold, when it changed.
    pub fn save<'a>(&self, model: &str, ledgers: impl Iterator<Item = &'a Ledger>) {
        let mut saved = self.saved.lock().unwrap();
        let mut held: Vec<Peak> = vec![];
        for peak in ledgers.flat_map(Ledger::peaks) {
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

/// What woke the events thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Woke {
    /// A ledger: room freed, a row died, a peak was recorded.
    Ledger,
    /// The host is short of what its running calls must keep free ([`Host`]).
    HostShort,
}

/// A source the events thread waits on: a ledger's events, or the host's short event.
pub enum Source {
    Ledger(Arc<Gpu>),
    Host(Arc<Host>),
}

impl Source {
    fn fd(&self) -> RawFd {
        match self {
            Source::Ledger(gpu) => gpu.ledger().events().as_raw_fd(),
            Source::Host(host) => host.short.as_raw_fd(),
        }
    }

    /// Clears what it signalled (a ready event: no wait).
    fn clear(&self) -> Woke {
        match self {
            Source::Ledger(gpu) => {
                gpu.ledger().clear_events();
                Woke::Ledger
            }
            Source::Host(host) => {
                host.short();
                Woke::HostShort
            }
        }
    }
}

/// Blocks on every source until the returned end drops; `wake` runs once per wake with what
/// woke it. It never times out.
pub fn react(sources: Vec<Source>, wake: impl Fn(Woke) + Send + 'static) -> io::Result<UnixStream> {
    let (held, end) = UnixStream::pair()?;
    std::thread::Builder::new().name("memory-events".into()).spawn(move || {
        let watched = std::iter::once(end.as_raw_fd()).chain(sources.iter().map(Source::fd));
        let mut fds: Vec<libc::pollfd> = watched.map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 }).collect();
        loop {
            // SAFETY: poll on descriptors this thread owns; no timeout.
            if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return tracing::warn!("memory: events: {}", io::Error::last_os_error());
            }
            let (pool, events) = fds.split_at_mut(1);
            if pool.iter().any(|fd| fd.revents != 0) {
                return;
            }
            let mut woke = vec![];
            for (fd, source) in events.iter_mut().zip(&sources).filter(|(fd, _)| fd.revents != 0) {
                if fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    fd.fd = -1; // closed or failed: poll skips it from now on
                    continue;
                }
                let what = source.clear();
                if !woke.contains(&what) {
                    woke.push(what);
                }
            }
            woke.into_iter().for_each(&wake);
        }
    })?;
    Ok(held)
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
    short: UnixStream,
    nudge: Nudge,
    /// Looks begun and made so far, and a signal for each made.
    looks: Arc<(Mutex<(u64, u64)>, std::sync::Condvar)>,
}

impl Host {
    /// Watches while the process lives; None (a memory-log note says why) where the platform
    /// gives no pressure events.
    pub fn watch(mode: HostMode, gpus: Vec<Arc<Gpu>>, log: &MemoryLog) -> Option<Arc<Host>> {
        let armed = host_pressure::Pressure::arm(mode).and_then(|pressure| {
            let (short, signal) = UnixStream::pair()?;
            short.set_nonblocking(true)?;
            Ok((pressure, (short, signal)))
        });
        let (mut pressure, (short, mut signal)) = armed
            .inspect_err(|error| log.note(serde_json::json!({"event": "host_unwatched", "mode": format!("{mode:?}"), "why": error.to_string()})))
            .ok()?;
        let (nudge, looks) = (pressure.nudge(), Arc::new((Mutex::new((0, 0)), std::sync::Condvar::new())));
        let looked = looks.clone();
        let watching = std::thread::Builder::new().name("host-pressure".into()).spawn(move || {
            let mut keep = Keep::default();
            // The budget when short was last signalled: the next signal needs more than it.
            let mut signalled: Option<u64> = None;
            while pressure.wait().is_ok() {
                looked.0.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
                let need = keep.look(gpus.iter().map(|gpu| gpu.ledger().host_calls()).collect::<Option<Vec<_>>>().map(|calls| calls.concat()));
                let budget = u64::try_from(host_memory::read().available).unwrap_or(u64::MAX);
                if budget >= need {
                    signalled = None;
                } else if signalled.is_none_or(|was| budget > was) {
                    if signal.write_all(&[1]).is_err() {
                        return;
                    }
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
        let (count, made) = &*self.looks;
        let mut looks = count.lock().unwrap_or_else(|e| e.into_inner());
        let begun = looks.0;
        self.nudge();
        while looks.1 <= begun {
            looks = made.wait(looks).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Whether a look found the host short since this was last asked (the events thread asks).
    pub fn short(&self) -> bool {
        let mut buf = [0; 64];
        (&self.short).read(&mut buf).is_ok_and(|n| n > 0)
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
    /// A look at the processes running requests (`None`: a GPU's table cannot be read): what
    /// must stay free.
    fn look(&mut self, calls: Option<Vec<HostCall>>) -> u64 {
        // A process with rows on several GPUs is one call.
        let mut by: BTreeMap<u32, HostCall> = BTreeMap::new();
        for call in calls.unwrap_or_default() {
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
