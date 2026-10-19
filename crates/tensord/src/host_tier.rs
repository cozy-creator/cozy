//! Degree 1: shared CPU weight buffers filled and retained by TensorD through TensorFS.
//! Each layout is a memfd sealed at creation (other processes cannot write, resize or punch
//! it). TensorFS fills regions from verified stored bytes; Runtime adopts the layout read-only
//! (`Plane.register_sealed`) and waits for Ready regions before using its own plane for pinned
//! staging and GPU copies. Model construction and CUDA operations stay in the executor.
//! TensorD holds one descriptor per layout across executor replacement and model switches.
//! `TierLimit` bounds retained CPU buffers using live host headroom; unheld complete layouts
//! are released least recently used first or after `ttl`. No lock is held across a fill.
//! A layout can instead expose unstaged regions through granted object descriptors, or use
//! a sealed streaming window that TensorD refills as Runtime claims regions. These are data
//! paths for a granted weight set, not permission for the executor to open the managed Store.
use crate::access::Fetch;
use platform::contain::host_memory::HostMemory;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io,
    os::{
        fd::AsRawFd,
        unix::fs::{FileExt, MetadataExt},
    },
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc, Arc, Condvar, Mutex, Weak,
    },
    time::{Duration, Instant},
};
use crate::storage::{self, Store};
use tensorfs_core::{
    format,
    ids,
    read::{self, Source as Item},
    store2::Checkpoint,
};
use varena::{
    host::{self, OpenFill},
    io::{IoTally, Source},
    layout::Layout,
    window::{self, OpenWindow},
};

#[derive(Clone, Debug)]
pub struct HostTierConfig {
    /// Threads of one fill (read + copy). Default: the CPUs this process may use, at most 8:
    /// more contended on RunPod (an A5000 pod's UNet fill: 8 threads 0.44 s, 16 threads 1.08 s).
    pub fill_threads: usize,
    /// An unheld layout unused this long is released.
    pub ttl: Duration,
}
impl Default for HostTierConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism().map_or(4, |n| n.get());
        Self {
            fill_threads: cpus.clamp(1, 8),
            ttl: Duration::from_secs(30 * 60),
        }
    }
}

/// The most the tier may charge, given live host memory and what it charges now.
pub trait TierLimit: Send + Sync {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64;
    /// The most a streamed layout's slots may take now (`charged` counts every layout): by
    /// default the room left under `limit`. A window gets one slot (its largest region)
    /// whatever this says.
    fn staging(&self, host: &HostMemory, charged: u64) -> u64 {
        self.limit(host, charged).saturating_sub(charged)
    }
}
/// The machine's one pinned host total: half the live headroom plus the tier's retained
/// bytes. Executors' pinned budgets split the same total (`Pool::pinned_budgets`).
pub struct HalfOfHeadroom;
impl TierLimit for HalfOfHeadroom {
    fn limit(&self, host: &HostMemory, charged: u64) -> u64 {
        if host.available < 0 {
            return 0; // unreadable: no retained tier capacity; use a supported streaming path
        }
        (host.available as u64 + charged) / 2
    }
}

/// CPU layouts one executor may adopt: these components of this exact model manifest.
#[derive(Clone)]
pub struct HostGrant {
    pub manifest: String,
    pub checkpoint: Checkpoint,
    pub components: BTreeSet<String>,
}

/// Runtime `SealedPlan`: the read plan and region grouping the model code chose.
#[derive(Clone, Debug, Deserialize)]
struct SealedPlan {
    manifest: String,
    name: String,
    window: u64,
    traversal: Vec<(String, String)>,
    components: Vec<String>,
    regions: Vec<Vec<String>>,
    #[serde(default)]
    parts: Vec<String>,
}

/// The `SealedTier` request's envelope; the plan itself arrives as a sealed memfd.
pub struct SealedRequest<'a> {
    pub sha256: &'a str,
    pub length: u64,
    /// The regions to stage (the blocks that stream every step); the rest are holes the
    /// executor reads from object files. None: every region.
    pub stage: Option<&'a [u32]>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Fill {
    pub name: String,
    pub bytes: u64,
    pub ms: f64,
    pub cached_bytes: u64,
    pub direct_bytes: u64,
    pub buffered_bytes: u64,
    /// Bytes this process read from storage meanwhile (`/proc/self/io`): the fill's disk
    /// reads, plus any other reader's in the machine at the same time.
    pub disk_read_bytes: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Ledger {
    /// The latest fills, newest last (at most 32).
    pub fills: Vec<Fill>,
    pub hits: u64,
    /// Layouts streamed through a window because they did not fit, and the bytes ended windows
    /// read from disk.
    pub windows_opened: u64,
    pub streamed_bytes: u64,
    /// Fills that failed: their adopters got an error, the layout was dropped.
    pub failed: u64,
    pub released: u64,
    /// What the released layouts charged, and what the kernel's shared memory fell by.
    pub released_bytes: u64,
    pub freed_bytes: u64,
    /// Released layouts whose memory did not come back (something still maps them): they
    /// stay charged.
    pub stranded_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct HostTierFacts {
    pub host: HostMemory,
    pub limit: u64,
    pub charged_bytes: u64,
    pub held_bytes: u64,
    pub filling_bytes: u64,
    /// Live windows (streamed layouts), what they charge, and what they have read so far.
    pub windows: usize,
    pub window_bytes: u64,
    pub window_read_bytes: u64,
    pub entries: usize,
    pub peers: usize,
    #[serde(flatten)]
    pub ledger: Ledger,
}

struct Entry {
    fd: File,
    /// A layout staged in part: the owner's mapping, kept to stage more.
    staging: Option<Staging>,
    charged: u64,
    used: Instant,
    holders: BTreeSet<u64>,
    /// Every region is in; until then nothing releases it.
    complete: bool,
    /// A streamed layout: its machine's staging loop. Released once its last holder exits.
    window: Option<Streamer>,
    granted: bool,
}
struct Streamer {
    stop: Arc<AtomicBool>,
    tally: Arc<IoTally>,
}
impl Streamer {
    fn read_bytes(&self) -> u64 {
        let t = &self.tally;
        [&t.cached_bytes, &t.direct_bytes, &t.buffered_bytes]
            .iter()
            .map(|n: &&AtomicU64| n.load(Ordering::Relaxed))
            .sum()
    }
}
/// What a layout staged in part needs to stage more of it.
struct Staging {
    open: Arc<OpenFill>,
    layout: Arc<Layout>,
    manifest: String,
    staged: BTreeSet<u32>,
}
enum Slot {
    /// Admitted, its memfd being created.
    Opening(u64),
    Open(Entry),
}
/// Work for the tier's filler, in order: a layout's first fill (of what it stages), or more
/// regions staged into a layout staged in part.
enum Job {
    Fill {
        key: String,
        plan: SealedPlan,
        layout: Arc<Layout>,
        open: Arc<OpenFill>,
        fetch: Option<Arc<Fetch>>,
    },
    Stage {
        key: String,
        regions: Vec<u32>,
        fetch: Option<Arc<Fetch>>,
    },
}
struct Peer {
    pidfd: File,
}
#[derive(Default)]
struct State {
    /// By TensorFS layout digest: the same bytes at the same offsets, whoever asks.
    slots: BTreeMap<String, Slot>,
    peers: BTreeMap<u64, Peer>,
    next_peer: u64,
    ledger: Ledger,
}
impl State {
    fn charged(&self) -> u64 {
        self.slots
            .values()
            .map(|s| match s {
                Slot::Opening(n) => *n,
                Slot::Open(e) => e.charged,
            })
            .sum::<u64>()
            + self.ledger.stranded_bytes
    }
}

pub struct HostTier {
    store: Arc<Store>,
    config: HostTierConfig,
    limit: Box<dyn TierLimit>,
    state: Mutex<State>,
    filled: Condvar,
    /// The filler's queue: one fill at a time, in the order layouts were opened.
    fills: Mutex<mpsc::Sender<Job>>,
}

fn failure(error: impl std::fmt::Display) -> io::Error {
    io::Error::other(error.to_string())
}
fn denied(what: &str) -> io::Error {
    io::Error::new(io::ErrorKind::PermissionDenied, what.to_string())
}
fn hex(value: &str) -> &str {
    value.strip_prefix("sha256:").unwrap_or(value)
}
fn disk_read_bytes() -> u64 {
    std::fs::read_to_string("/proc/self/io")
        .ok()
        .and_then(|s| {
            s.lines()
                .find_map(|l| l.strip_prefix("read_bytes: ")?.trim().parse().ok())
        })
        .unwrap_or(0)
}

impl HostTier {
    pub fn new(
        store: Arc<Store>,
        config: HostTierConfig,
        limit: Box<dyn TierLimit>,
    ) -> io::Result<Arc<Self>> {
        let (fills, jobs) = mpsc::channel::<Job>();
        let tier = Arc::new(Self {
            store,
            config,
            limit,
            state: Mutex::default(),
            filled: Condvar::new(),
            fills: Mutex::new(fills),
        });
        // Ends with the tier; a job it never runs poisons its layout (`OpenFill` on drop).
        let weak: Weak<Self> = Arc::downgrade(&tier);
        std::thread::spawn(move || {
            for job in jobs {
                match weak.upgrade() {
                    Some(tier) => tier.run(job),
                    None => return,
                }
            }
        });
        Ok(tier)
    }

    /// Start on what an executor of these grants will ask for, while it starts: read its
    /// objects ahead into the page cache.
    pub fn prepare(self: &Arc<Self>, grants: Vec<HostGrant>) {
        // What is not here yet comes as its executor directs; only what is here is started on.
        let grants: Vec<HostGrant> = (grants.into_iter())
            .filter(|g| storage::chunks(&g.checkpoint, &g.components.iter().cloned().collect::<Vec<_>>())
                .iter().all(|o| storage::present(&self.store, o)))
            .collect();
        // Its executor stages only what streams and reads the rest from the files: warm the
        // page cache (reclaimable), allocate nothing.
        let tier = self.clone();
        std::thread::spawn(move || {
            for grant in &grants {
                for component in &grant.components {
                    if let Ok(plan) = component_plan(&grant.checkpoint, component) {
                        tier.read_ahead(&plan);
                    }
                }
            }
        });
    }

    /// Ask the kernel to read `plan`'s objects into the page cache (asynchronous, reclaimable).
    fn read_ahead(&self, plan: &read::ReadPlan) {
        for item in &plan.items {
            if let Item::Object(range) = &item.source {
                if let Ok(file) = self.store.path(&v2(&range.obj)).map_err(storage::io_error).and_then(File::open) {
                    // SAFETY: advice on a descriptor we hold.
                    #[cfg(target_os = "linux")]
                    unsafe {
                        libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_WILLNEED)
                    };
                    #[cfg(not(target_os = "linux"))]
                    drop(file);
                }
            }
        }
    }

    /// An executor, by its pidfd: what it adopted stays held until that exact process exits.
    /// It reads a layout's holes from object files, so the tier stages only what it streams.
    pub fn register_peer(&self, pidfd: File) -> u64 {
        let mut state = self.state.lock().unwrap();
        state.next_peer += 1;
        let id = state.next_peer;
        state.peers.insert(id, Peer { pidfd });
        id
    }

    fn live_peer(&self, peer: u64) -> io::Result<()> {
        match self.state.lock().unwrap().peers.contains_key(&peer) {
            true => Ok(()),
            false => Err(denied("host tier peer is not a registered live executor")),
        }
    }

    /// One weight set's layout for `peer`, read-only: from the tier, or opened now (whole and
    /// filling, or streamed when it does not fit). The executor waits per region either way.
    pub fn seal(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
        fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Option<File>> {
        self.live_peer(peer)?;
        let stage = request.stage.map(<[u32]>::to_vec);
        let plan = Self::plan(request, plan)?;
        let mut key = self.ensure(&plan, grants, stage.clone(), fetch)?;
        match stage.as_deref() {
            // A layout another executor had staged in part: this one stages what it asked for.
            Some(regions) => {
                self.stage_regions(&key, regions, fetch);
            }
            // It names no regions (a CPU executor): the rest is staged when the tier has room
            // for all of it; else it reads through a window of its own.
            None if !self.stage_rest(&key, fetch) => {
                let (read_plan, layout) = Self::layout(&plan, grants)?;
                key = self.stream(format!("{key}.whole"), &plan, read_plan, layout, fetch)?;
            }
            None => {}
        }
        let mut state = self.state.lock().unwrap();
        match state.slots.get(&key) {
            Some(Slot::Open(_)) => self.grant(&mut state, &key, peer).map(Some),
            _ => Err(failure("the layout's fill failed before it was granted")),
        }
    }

    /// The read plan and layout `plan` names, inside the executor's selection.
    fn layout(plan: &SealedPlan, grants: &[HostGrant]) -> io::Result<(read::ReadPlan, Layout)> {
        let grant = grants
            .iter()
            .find(|g| hex(&g.manifest) == hex(&plan.manifest))
            .ok_or_else(|| denied("manifest outside this executor's selection"))?;
        if plan.components.is_empty()
            || plan
                .components
                .iter()
                .chain(plan.traversal.iter().map(|(c, _)| c))
                .any(|c| !grant.components.contains(c))
        {
            return Err(denied("plan leaves the selected components"));
        }
        let mut read_plan = plan_for_traversal(
            &grant.checkpoint,
            &plan.traversal,
            &plan.components,
            plan.window,
        )?;
        if !plan.parts.is_empty() {
            read_plan = read_plan.select(&plan.parts).map_err(failure)?;
        }
        let layout = Layout::build(&read_plan, &plan.regions).map_err(failure)?;
        Ok((read_plan, layout))
    }

    /// The layout `plan` names, held or opened now: whole, or with only `stage` staged (the
    /// rest holes, read from object files), its fill queued, when the tier can make room. A
    /// staged layout that does not fit opens with nothing staged; a whole one streams through
    /// a window. Its key.
    fn ensure(
        &self,
        plan: &SealedPlan,
        grants: &[HostGrant],
        mut stage: Option<Vec<u32>>,
        fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<String> {
        let (read_plan, layout) = Self::layout(plan, grants)?;
        let key = layout.digest.clone();
        let mut state = self.state.lock().unwrap();
        loop {
            self.reap(&mut state);
            match state.slots.get(&key) {
                Some(Slot::Open(_)) => {
                    state.ledger.hits += 1;
                    return Ok(key);
                }
                Some(Slot::Opening(_)) => state = self.filled.wait(state).unwrap(),
                None => break,
            }
        }
        // Admission over live headroom, read now: release unheld layouts, oldest first.
        let spans = |regions: &[u32]| -> u64 {
            regions
                .iter()
                .filter_map(|&r| layout.regions.get(r as usize))
                .map(|r| r.span)
                .sum()
        };
        let mut need = stage.as_deref().map_or(layout.nbytes, spans);
        loop {
            let host = platform::contain::host_memory::read();
            let charged = state.charged();
            if charged + need <= self.limit.limit(&host, charged) {
                break;
            }
            if !self.release_lru(&mut state) {
                if stage.is_some() {
                    // Nothing staged: every region reads the object files (the disk rung).
                    stage = Some(Vec::new());
                    need = 0;
                    break;
                }
                drop(state);
                return self.stream(key, plan, read_plan, layout, fetch);
            }
        }
        state.slots.insert(key.clone(), Slot::Opening(need));
        drop(state);
        // Sealed now, filled by the filler: adopters wait per region, never for the whole.
        let opened =
            host::open_sealed(&plan.name, &layout, None, stage.as_deref()).map_err(failure);
        let mut state = self.state.lock().unwrap();
        let result = match opened {
            Ok((fd, open)) => {
                let (open, layout) = (Arc::new(open), Arc::new(layout));
                let staging = stage.map(|regions| Staging {
                    open: open.clone(),
                    layout: layout.clone(),
                    manifest: hex(&plan.manifest).to_string(),
                    staged: regions.into_iter().collect(),
                });
                state.slots.insert(
                    key.clone(),
                    Slot::Open(Entry {
                        fd: File::from(fd),
                        staging,
                        charged: need,
                        used: Instant::now(),
                        holders: BTreeSet::new(),
                        complete: false,
                        window: None,
                        granted: false,
                    }),
                );
                let job = Job::Fill {
                    key: key.clone(),
                    plan: plan.clone(),
                    layout,
                    open,
                    fetch: fetch.cloned(),
                };
                // A send fails only once the filler is gone with the tier: the job's drop
                // poisons the layout, so no adopter waits for it.
                let _ = self.fills.lock().unwrap().send(job);
                Ok(key)
            }
            Err(error) => {
                state.slots.remove(&key);
                Err(error)
            }
        };
        self.filled.notify_all();
        result
    }

    /// The disk rung: `layout` streamed through a window of as many of its largest regions as
    /// the tier has room for (one at least: the indivisible working set; two let one refill
    /// while another is read), served from disk on its own thread until its last holder exits.
    /// The room is read and reserved under one lock, so concurrent openers never spend the
    /// same bytes.
    fn stream(
        &self,
        key: String,
        plan: &SealedPlan,
        read_plan: read::ReadPlan,
        layout: Layout,
        fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<String> {
        // A window reads any region at any time: everything it may read is here first.
        if let Some(fetch) = fetch {
            let objects: Vec<_> = read_plan.items.iter().filter_map(|item| match &item.source {
                Item::Object(range) => Some(v2(&range.obj)),
                Item::Inline(_) => None,
            }).collect();
            fetch.wait(&plan.manifest, &objects)?;
        }
        let span = layout
            .regions
            .iter()
            .map(|r| r.span)
            .max()
            .unwrap_or(1)
            .max(1);
        let mut state = self.state.lock().unwrap();
        loop {
            match state.slots.get(&key) {
                Some(Slot::Open(_)) => return Ok(key),
                Some(Slot::Opening(_)) => state = self.filled.wait(state).unwrap(),
                None => break,
            }
        }
        let room = self.limit.staging(&platform::contain::host_memory::read(), state.charged());
        let slots = ((room / span) as usize).clamp(1, layout.regions.len().max(1));
        state.slots.insert(key.clone(), Slot::Opening(slots as u64 * span));
        drop(state);
        let opened = window::open_window(&plan.name, &layout, slots).map_err(failure);
        let mut state = self.state.lock().unwrap();
        let (fd, open) = match opened {
            Ok(opened) => opened,
            Err(error) => {
                state.slots.remove(&key);
                drop(state);
                self.filled.notify_all();
                return Err(error);
            }
        };
        let streamer = Streamer {
            stop: Arc::new(AtomicBool::new(false)),
            tally: Arc::new(IoTally::default()),
        };
        let (stop, tally) = (streamer.stop.clone(), streamer.tally.clone());
        let charged = open.bytes();
        state.slots.insert(
            key.clone(),
            Slot::Open(Entry {
                fd: File::from(fd),
                staging: None,
                charged,
                used: Instant::now(),
                holders: BTreeSet::new(),
                complete: false,
                window: Some(streamer),
                granted: false,
            }),
        );
        state.ledger.windows_opened += 1;
        drop(state);
        self.filled.notify_all();
        let (store, threads) = (self.store.clone(), self.config.fill_threads);
        std::thread::spawn(move || {
            if let Err(error) = serve(
                &store, &read_plan, &layout, open, threads, &tally, &stop,
            ) {
                // Dropping the window failed every region: its readers get the error.
                tracing::warn!("host tier window stopped: {error}");
            }
        });
        Ok(key)
    }

    /// The filler: one layout's bytes, then it is complete (charged what it holds) or, failed,
    /// dropped (its adopters see the failure in its Ready words).
    fn run(&self, job: Job) {
        let (key, plan, layout, open, fetch) = match job {
            Job::Fill { key, plan, layout, open, fetch } => (key, plan, layout, open, fetch),
            Job::Stage { key, regions, fetch } => return self.run_stage(&key, &regions, fetch.as_deref()),
        };
        let filled = self.fill(&plan, &layout, &open, fetch.as_deref());
        drop(open); // a layout staged in part keeps the mapping in its entry; a whole one lets go
        let mut state = self.state.lock().unwrap();
        match filled {
            Ok(fill) => {
                if let Some(Slot::Open(entry)) = state.slots.get_mut(&key) {
                    entry.complete = true;
                    if let Ok(meta) = entry.fd.metadata() {
                        entry.charged = meta.blocks() * 512;
                    }
                }
                if state.ledger.fills.len() == 32 {
                    state.ledger.fills.remove(0);
                }
                state.ledger.fills.push(fill);
            }
            Err(error) => {
                tracing::warn!("host tier fill of {} failed: {error}", plan.name);
                state.slots.remove(&key);
                state.ledger.failed += 1;
            }
        }
        self.filled.notify_all();
    }

    /// Stage `regions` of a layout staged in part, in order, as far as the tier has room;
    /// returns the regions now staged or staging (all of them for a whole layout). The filler
    /// copies them; adopters wait per region, and a region left a hole reads the files.
    fn stage_regions(&self, key: &str, regions: &[u32], fetch: Option<&Arc<Fetch>>) -> Vec<u32> {
        let mut state = self.state.lock().unwrap();
        let limit = |state: &State| {
            let host = platform::contain::host_memory::read();
            let charged = state.charged();
            (charged, self.limit.limit(&host, charged))
        };
        let (mut charged, ceiling) = limit(&state);
        let Some(Slot::Open(entry)) = state.slots.get_mut(key) else {
            return Vec::new();
        };
        let Some(staging) = entry.staging.as_mut() else {
            return regions.to_vec();
        };
        let mut queued = Vec::new();
        for &r in regions {
            let Some(region) = staging.layout.regions.get(r as usize) else {
                continue;
            };
            if staging.staged.contains(&r) {
                continue;
            }
            if charged + region.span > ceiling {
                break; // the rest stream from the files
            }
            charged += region.span;
            entry.charged += region.span;
            staging.staged.insert(r);
            queued.push(r);
        }
        let staged = regions
            .iter()
            .copied()
            .filter(|r| staging.staged.contains(r))
            .collect();
        // Taken before the answer: the executor waits for these from now on, never reads them
        // as holes. The filler copies them.
        let taken = staging.open.take(&queued);
        if !taken.is_empty() {
            let job = Job::Stage {
                key: key.to_string(),
                regions: taken,
                fetch: fetch.cloned(),
            };
            let _ = self.fills.lock().unwrap().send(job);
        }
        staged
    }

    /// Every remaining region of a layout staged in part, for an adopter that reads no holes:
    /// all of them when the tier has room, else none (the sparse copy does not grow). False:
    /// the layout keeps holes.
    fn stage_rest(&self, key: &str, fetch: Option<&Arc<Fetch>>) -> bool {
        let rest: Vec<u32> = {
            let state = self.state.lock().unwrap();
            let Some(Slot::Open(Entry {
                staging: Some(staging),
                ..
            })) = state.slots.get(key)
            else {
                return true; // whole, or a window
            };
            let rest: Vec<u32> = (0..staging.layout.regions.len() as u32)
                .filter(|r| !staging.staged.contains(r))
                .collect();
            let need: u64 = rest
                .iter()
                .map(|&r| staging.layout.regions[r as usize].span)
                .sum();
            let charged = state.charged();
            if charged + need > self.limit.limit(&platform::contain::host_memory::read(), charged) {
                return false;
            }
            rest
        };
        self.stage_regions(key, &rest, fetch).len() == rest.len()
    }

    /// The filler's half of `stage_regions`: each region copied in order as its objects land.
    fn run_stage(&self, key: &str, regions: &[u32], fetch: Option<&Fetch>) {
        let taken = {
            let state = self.state.lock().unwrap();
            match state.slots.get(key) {
                Some(Slot::Open(Entry {
                    staging: Some(s), ..
                })) => Some((s.open.clone(), s.layout.clone(), s.manifest.clone())),
                _ => None,
            }
        };
        let Some((open, layout, manifest)) = taken else {
            return;
        };
        let tally = IoTally::default();
        let staged = self.landing(&manifest, &layout, regions, fetch, |batch, source| {
            open.stage(batch, &layout, source, self.config.fill_threads, &tally).map(drop)
        });
        let mut state = self.state.lock().unwrap();
        if let Some(Slot::Open(entry)) = state.slots.get_mut(key) {
            if let Ok(meta) = entry.fd.metadata() {
                entry.charged = entry.charged.max(meta.blocks() * 512);
            }
        }
        if let Err((error, rest)) = staged {
            // Its adopters see the regions failed; the next ask opens the layout afresh.
            tracing::warn!("host tier staging of {regions:?} failed: {error}");
            fail(&open, &layout, &rest);
            state.slots.remove(key);
            state.ledger.failed += 1;
        }
        drop(state);
        self.filled.notify_all();
    }

    /// `regions` of `layout` copied in their order: the next one once its objects are here
    /// (absent ones fetched now through `fetch`; none: they must be here), with those after it
    /// already here in the same lease. On failure, the regions not copied.
    fn landing(
        &self,
        manifest: &str,
        layout: &Layout,
        regions: &[u32],
        fetch: Option<&Fetch>,
        copy: impl Fn(&[u32], &Source) -> varena::Result<()>,
    ) -> Result<(), (io::Error, Vec<u32>)> {
        let objects = |batch: &[u32]| -> Vec<ids::ObjectRef> {
            let mut objects = BTreeMap::new();
            for &r in batch {
                for item in &layout.regions[r as usize].items {
                    if let varena::layout::ItemSource::Object(range) = &item.source {
                        objects.insert(range.obj.sha256.clone(), range.obj.clone());
                    }
                }
            }
            objects.into_values().collect()
        };
        let mut at = 0;
        while at < regions.len() {
            let rest = || regions[at..].to_vec();
            if let Some(fetch) = fetch {
                let wanted: Vec<_> = objects(&regions[at..=at]).iter().map(v2).collect();
                fetch.wait(manifest, &wanted).map_err(|error| (error, rest()))?;
            }
            let here = |r: &u32| objects(&[*r]).iter().all(|o| storage::present(&self.store, &v2(o)));
            let end = at + 1 + regions[at + 1..].iter().take_while(|r| here(r)).count();
            let batch = &regions[at..end];
            let copied = match objects(batch) {
                listed if listed.is_empty() => copy(batch, &Source::files(Vec::new(), false, 0)).map_err(failure),
                listed => files(&self.store, listed)
                    .and_then(|files| copy(batch, &Source::files(files, true, 0)).map_err(failure)),
            };
            copied.map_err(|error| (error, rest()))?;
            at = end;
        }
        Ok(())
    }

    /// More of a layout staged in part, for `peer`: `regions` it now streams every step
    /// (`SealedStage`). Returns the regions staged or staging; the rest stay holes.
    pub fn stage(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
        fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Vec<u32>> {
        self.live_peer(peer)?;
        let regions = request.stage.unwrap_or_default().to_vec();
        let plan = Self::plan(request, plan)?;
        let (_, layout) = Self::layout(&plan, grants)?;
        Ok(self.stage_regions(&layout.digest, &regions, fetch))
    }

    /// The verified object files behind `plan`, read-only, for `peer`'s holes: the store's
    /// own descriptors (the file verified is the file read), opened per ask and handed over.
    pub fn object_files(
        &self,
        peer: u64,
        grants: &[HostGrant],
        request: SealedRequest<'_>,
        plan: File,
        fetch: Option<&Arc<Fetch>>,
    ) -> io::Result<Vec<(tensorfs_core::ids::ObjectRef, File)>> {
        self.live_peer(peer)?;
        let plan = Self::plan(request, plan)?;
        let (read_plan, _) = Self::layout(&plan, grants)?;
        let mut objects = BTreeMap::new();
        for item in &read_plan.items {
            if let Item::Object(range) = &item.source {
                objects.insert(range.obj.sha256.clone(), range.obj.clone());
            }
        }
        if let Some(fetch) = fetch {
            let all: Vec<_> = objects.values().map(v2).collect();
            fetch.wait(&plan.manifest, &all)?;
        }
        files(&self.store, objects.into_values().collect())
    }

    /// Release unheld layouts, least recently used first, until `want` bytes were charged
    /// to them or none is left; returns the bytes released. For the memory policy module.
    pub fn release(&self, want: u64) -> u64 {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state);
        let before = state.ledger.released_bytes;
        while state.ledger.released_bytes - before < want && self.release_lru(&mut state) {}
        state.ledger.released_bytes - before
    }

    pub fn facts(&self) -> HostTierFacts {
        let mut state = self.state.lock().unwrap();
        self.reap(&mut state);
        let host = platform::contain::host_memory::read();
        let charged = state.charged();
        let (mut held, mut filling) = (0, 0);
        let (mut windows, mut window_bytes, mut window_read_bytes) = (0, 0, 0);
        for slot in state.slots.values() {
            match slot {
                Slot::Open(
                    e @ Entry {
                        window: Some(w), ..
                    },
                ) => {
                    windows += 1;
                    window_bytes += e.charged;
                    window_read_bytes += w.read_bytes();
                }
                Slot::Opening(n)
                | Slot::Open(Entry {
                    charged: n,
                    complete: false,
                    ..
                }) => filling += n,
                Slot::Open(e) if !e.holders.is_empty() => held += e.charged,
                Slot::Open(_) => (),
            }
        }
        HostTierFacts {
            host,
            limit: self.limit.limit(&host, charged),
            charged_bytes: charged,
            held_bytes: held,
            filling_bytes: filling,
            windows,
            window_bytes,
            window_read_bytes,
            entries: state.slots.len(),
            peers: state.peers.len(),
            ledger: state.ledger.clone(),
        }
    }

    /// The plan the executor sent: a sealed memfd of exactly the declared bytes and digest.
    fn plan(request: SealedRequest<'_>, plan: File) -> io::Result<SealedPlan> {
        let body = Self::body(request, plan)?;
        serde_json::from_slice(&body).map_err(failure)
    }

    /// A sealed memfd of exactly the declared bytes and digest.
    fn body(request: SealedRequest<'_>, plan: File) -> io::Result<Vec<u8>> {
        if platform::shm::seals(std::os::fd::AsFd::as_fd(&plan))? & platform::shm::NO_WRITE_SEALS != platform::shm::NO_WRITE_SEALS
            || plan.metadata()?.len() != request.length
            || request.length > crate::seam::MAX_FRAME as u64
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "a sealed plan must be sealed at its declared length",
            ));
        }
        let mut body = vec![0; request.length as usize];
        plan.read_exact_at(&mut body, 0)?;
        if format!("{:x}", Sha256::digest(&body)) != request.sha256 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sealed plan digest differs",
            ));
        }
        Ok(body)
    }

    /// What this layout stages (not its holes), region by region in layout order as its
    /// objects land: an adopter waits per region, never for the whole.
    fn fill(&self, plan: &SealedPlan, layout: &Layout, open: &OpenFill, fetch: Option<&Fetch>) -> io::Result<Fill> {
        let regions: Vec<u32> = (0..layout.regions.len() as u32).filter(|&r| !open.hole(r)).collect();
        let started = (Instant::now(), disk_read_bytes());
        let tally = IoTally::default();
        let filled = self.landing(&plan.manifest, layout, &regions, fetch, |batch, source| {
            open.stage(batch, layout, source, self.config.fill_threads, &tally).map(drop)
        });
        if let Err((error, rest)) = filled {
            fail(open, layout, &rest);
            return Err(error);
        }
        let read = |n: &std::sync::atomic::AtomicU64| n.load(std::sync::atomic::Ordering::Relaxed);
        Ok(Fill {
            name: plan.name.clone(),
            bytes: regions.iter().map(|&r| layout.regions[r as usize].span).sum(),
            ms: started.0.elapsed().as_secs_f64() * 1e3,
            cached_bytes: read(&tally.cached_bytes),
            direct_bytes: read(&tally.direct_bytes),
            buffered_bytes: read(&tally.buffered_bytes),
            disk_read_bytes: disk_read_bytes().saturating_sub(started.1),
        })
    }

    fn grant(&self, state: &mut State, key: &str, peer: u64) -> io::Result<File> {
        if !state.peers.contains_key(&peer) {
            return Err(denied("host tier peer is not a registered live executor"));
        }
        let Some(Slot::Open(entry)) = state.slots.get_mut(key) else {
            unreachable!()
        };
        entry.used = Instant::now();
        entry.holders.insert(peer);
        entry.granted = true;
        // A new read-only description: the recipient can neither write nor resize, and the
        // seals forbid it any other way.
        File::open(format!("/proc/self/fd/{}", entry.fd.as_raw_fd()))
    }

    /// Exited executors hold nothing; unheld layouts past their TTL go.
    fn reap(&self, state: &mut State) {
        let ended: Vec<u64> = state
            .peers
            .iter()
            .filter(|(_, peer)| platform::process::ended(&peer.pidfd))
            .map(|(id, _)| *id)
            .collect();
        for id in &ended {
            state.peers.remove(id);
        }
        let ttl = self.config.ttl;
        let mut expired = Vec::new();
        for (key, slot) in &mut state.slots {
            if let Slot::Open(entry) = slot {
                entry.holders.retain(|h| !ended.contains(h));
                let unused = entry.used.elapsed() >= ttl;
                let done = match entry.window {
                    // A window holds nothing worth keeping: it goes with its last holder.
                    Some(_) => entry.granted || unused,
                    None => entry.complete && unused,
                };
                if entry.holders.is_empty() && done {
                    expired.push(key.clone());
                }
            }
        }
        for key in expired {
            self.release_entry(state, &key);
        }
    }

    fn release_lru(&self, state: &mut State) -> bool {
        let oldest = state
            .slots
            .iter()
            .filter_map(|(key, slot)| match slot {
                Slot::Open(e) if e.holders.is_empty() && e.complete => Some((e.used, key.clone())),
                _ => None,
            })
            .min();
        match oldest {
            Some((_, key)) => {
                self.release_entry(state, &key);
                true
            }
            None => false,
        }
    }

    /// Close the tier's descriptor of an unheld layout. Its memory returns once nothing else
    /// holds it; a process that still does (a descriptor or a mapping of its file) leaves the
    /// bytes charged (stranded). Asked of that file, not of the host's shared-memory total,
    /// which every other process moves too.
    fn release_entry(&self, state: &mut State, key: &str) {
        let Some(Slot::Open(entry)) = state.slots.remove(key) else {
            return;
        };
        if let Some(window) = &entry.window {
            // Its staging loop unmaps the slots within a poll; nothing else maps them.
            window.stop.store(true, Ordering::Release);
            state.ledger.streamed_bytes += window.read_bytes();
            state.ledger.released += 1;
            state.ledger.released_bytes += entry.charged;
            return;
        }
        let file = fs::metadata(format!("/proc/self/fd/{}", entry.fd.as_raw_fd()))
            .map(|m| (m.dev(), m.ino()));
        drop(entry.fd);
        state.ledger.released += 1;
        state.ledger.released_bytes += entry.charged;
        match file {
            Ok((dev, ino)) if platform::shm::held(dev, ino).is_ok_and(|held| !held) => state.ledger.freed_bytes += entry.charged,
            _ => state.ledger.stranded_bytes += entry.charged,
        }
    }
}

/// A window's staging loop: a read lease over its objects for as long as it serves.
#[allow(clippy::too_many_arguments)]
fn serve(
    store: &Arc<Store>,
    read_plan: &read::ReadPlan,
    layout: &Layout,
    open: OpenWindow,
    threads: usize,
    tally: &IoTally,
    stop: &AtomicBool,
) -> io::Result<()> {
    let mut objects = BTreeMap::new();
    for item in &read_plan.items {
        if let Item::Object(range) = &item.source {
            objects.insert(range.obj.sha256.clone(), range.obj.clone());
        }
    }
    let source = Source::files(files(store, objects.into_values().collect())?, true, 0);
    open.serve(layout, &source, threads, tally, stop)
        .map_err(failure)
}

/// `regions` that will never be copied, marked failed so no adopter waits for them: staging
/// them from no files fails each region an object backs.
fn fail(open: &OpenFill, layout: &Layout, regions: &[u32]) {
    if !regions.is_empty() {
        let _ = open.stage(regions, layout, &Source::files(Vec::new(), false, 0), 1, &IoTally::default());
    }
}

/// The SHA-256 (hex) of `files`' objects in order, one `<sha256>:<length>\n` line each: what an
/// `object_files` answer names, checked against the list the executor derives from its plan.
pub fn objects_digest(files: &[(tensorfs_core::ids::ObjectRef, File)]) -> String {
    let lines: String = files
        .iter()
        .map(|(obj, _)| format!("{}:{}\n", obj.sha256, obj.length))
        .collect();
    format!("{:x}", Sha256::digest(lines.as_bytes()))
}

fn v1(object: &format::ObjectRef) -> ids::ObjectRef {
    ids::ObjectRef { sha256: object.sha256.clone(), length: object.length }
}

fn v2(object: &ids::ObjectRef) -> format::ObjectRef {
    format::ObjectRef { sha256: object.sha256.clone(), length: object.length }
}

/// `objects`' files, opened through the store, for the plane to read.
fn files(store: &Store, objects: Vec<ids::ObjectRef>) -> io::Result<Vec<(ids::ObjectRef, File)>> {
    objects
        .into_iter()
        .map(|object| Ok((store.file(&v2(&object)).map_err(storage::io_error)?, object)))
        .map(|opened: io::Result<_>| opened.map(|(file, object)| (object, file)))
        .collect()
}

/// One component's whole read plan.
fn component_plan(checkpoint: &Checkpoint, component: &str) -> io::Result<read::ReadPlan> {
    let traversal: Vec<(String, String)> = (checkpoint.header().component(component).into_iter())
        .flat_map(|c| c.tensors.iter().map(|t| (component.to_string(), t.key.clone())))
        .collect();
    plan_for_traversal(checkpoint, &traversal, &[component.to_string()], 4 << 20)
}

/// The plane's read plan over `checkpoint` in the executor's `traversal` order: every part of
/// every tensor of the `requested` components, as ranges of the chunks that hold it, split at
/// `window` (0: whole). A plain tensor's part is `value`. Completeness is per requested
/// component: a traversal that omits a tensor of one, or names one outside them, refuses.
pub fn plan_for_traversal(
    checkpoint: &Checkpoint,
    traversal: &[(String, String)],
    requested: &[String],
    window: u64,
) -> io::Result<read::ReadPlan> {
    let refuse = |why: String| Err(io::Error::new(io::ErrorKind::InvalidInput, why));
    let header = checkpoint.header();
    let scope: BTreeSet<&str> = requested.iter().map(String::as_str).collect();
    if scope.is_empty() || scope.len() != requested.len() {
        return refuse("a plan requests a set of components".into());
    }
    let mut carried: BTreeSet<(&str, &str)> = BTreeSet::new();
    for component in &scope {
        let Some(c) = header.component(component) else {
            return refuse(format!("the checkpoint has no component {component:?}"));
        };
        carried.extend(c.tensors.iter().map(|t| (*component, t.key.as_str())));
    }
    let (mut items, mut dest) = (Vec::new(), 0u64);
    for (component, key) in traversal {
        if !carried.remove(&(component.as_str(), key.as_str())) {
            return refuse(format!("the traversal names {component}/{key} twice or outside {scope:?}"));
        }
        let tensor = header.tensor(component, key).expect("a carried tensor");
        for (role, at) in tensor.locations() {
            let role = if role.is_empty() { "value" } else { role };
            let what = format!("{component}/{key}#{role}");
            for span in header.spans(&at) {
                let mut off = 0;
                while off < span.length {
                    let len = match window {
                        0 => span.length - off,
                        window => window.min(span.length - off),
                    };
                    let range = read::ObjectRange { obj: v1(&header.chunks[span.chunk]), off: span.start + off, len };
                    items.push(read::PlanItem { what: what.clone(), dest_off: dest, len, source: Item::Object(range) });
                    (dest, off) = (dest + len, off + len);
                }
            }
        }
    }
    if let Some((component, key)) = carried.into_iter().next() {
        return refuse(format!("the traversal omits {component}/{key} of {scope:?}"));
    }
    Ok(read::ReadPlan { items, bytes: dest, io_bytes: dest, inline_parts: 0, order: read::Order::Destination })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A sealed plan is as large as its model's keys (the old 64 MiB refused a valid one with
    /// long keys); its declared length must be its sealed bytes and within the shared guard.
    #[test]
    fn a_sealed_plan_over_64_mib_is_read_and_a_lying_length_is_refused() {
        let body = vec![b' '; (64 << 20) + 1];
        let mut file = platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        file.write_all(&body).unwrap();
        platform::shm::seal_with(std::os::fd::AsFd::as_fd(&file), platform::shm::NO_WRITE_SEALS).unwrap();
        let sha256 = format!("{:x}", Sha256::digest(&body));
        let request = |length| SealedRequest { sha256: &sha256, length, stage: None };
        let read = HostTier::body(request(body.len() as u64), file.try_clone().unwrap()).unwrap();
        assert!(read == body);
        let lying = HostTier::body(request(crate::seam::MAX_FRAME as u64 + 1), file);
        assert_eq!(lying.unwrap_err().kind(), io::ErrorKind::InvalidInput);
    }
}
