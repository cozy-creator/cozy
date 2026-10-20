//! Applies the single placement policy to CUDA mappings and the shared ledger.
//!
//! Model descriptions and ordered requests come from the caller. The coordinator never
//! parses a checkpoint, predicts a plan, copies bytes, or holds its table lock across I/O.

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::thread::{self, JoinHandle, ThreadId};

use parking_lot::{Condvar, Mutex};
use tensorfs::io::source::Source;
use tensorfs::io::ReadStats;

use crate::gpu::{BindState, FillAttempt, FillPhase, FillPin, Gpu, Pin};
use crate::layout::Layout;
use crate::ledger::{self, Channel, Ledger, PageKey};
use crate::plan::{
    self, Action, Class, Kind, ModelId, PageId, Position, RequestId, Step, Tier, Trigger,
};
use crate::{Error, Result};

pub type Shrink = dyn Fn(u64) -> Result<u64> + Send + Sync;
pub type Transform = dyn Fn(PageId, FillPin) -> Result<()> + Send + Sync;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relief {
    Complete,
    Cut(u64),
    Lower(u64),
    Exhausted(u64),
}

fn activation_oom_target(available: u64, need: u64) -> Result<u64> {
    available
        .checked_add(1)
        .map(|progress| need.max(progress))
        .ok_or_else(|| invalid("OOM reclaim target overflow"))
}

#[derive(Default)]
struct BindRetry {
    available: Option<u64>,
}

impl BindRetry {
    fn after_oom(&self, available: u64, need: u64) -> Result<u64> {
        if self.available.is_some_and(|prior| available <= prior) {
            // A driver OOM disproves the unchanged ledger estimate. Asking for
            // that same absolute room again is not recovery: reclaim additional
            // bytes through the ordinary policy, or reach Lower/Exhausted.
            available
                .checked_add(need)
                .ok_or_else(|| invalid("mapping recovery room overflow"))
        } else {
            // The first checked recovery, or genuinely increased room, gets one
            // retry before a further OOM requires additional reclamation.
            Ok(need)
        }
    }

    fn advance(&mut self, available: u64) -> bool {
        if self.available.is_some_and(|prior| available <= prior) {
            return false;
        }
        self.available = Some(available);
        true
    }
}

#[derive(Clone, Debug)]
pub struct Event {
    pub sequence: u64,
    pub request: RequestId,
    pub step: u32,
    pub pass_index: u64,
    pub model: Option<ModelId>,
    pub page: PageId,
    pub class: Class,
    pub kind: &'static str,
    pub reason: &'static str,
    pub bytes: u64,
    pub capacity: u64,
    pub working_set: Option<u64>,
    pub resident: u64,
    pub orphan_bytes: u64,
    pub speculative_budget: Option<u64>,
    pub fill: Option<FillResult>,
}

#[derive(Clone, Debug)]
pub struct FillResult {
    pub phase: &'static str,
    pub outcome: &'static str,
    pub error: Option<String>,
    pub io_runs: usize,
}

#[derive(Clone, Debug)]
pub struct Copy {
    pub request: RequestId,
    pub model: ModelId,
    pub page: PageId,
    pub tier: Tier,
    pub speculative: bool,
    pub streamed: bool,
    pub stats: ReadStats,
    /// Completed I/O, retained even when a later transform or publication fails.
    pub io: Vec<tensorfs::io::metrics::Copy>,
}

#[derive(Clone, Debug)]
pub struct ProofPage {
    pub page: PageId,
    pub content: String,
    pub payload: u64,
    pub mapped: u64,
}

impl ProofPage {
    fn catalog(pages: &[PageId], fingerprint: &[u8; 32], layout: &Layout) -> Vec<Self> {
        pages
            .iter()
            .zip(&layout.regions)
            .enumerate()
            .map(|(index, (&page, region))| Self {
                page,
                content: format!("{}/{}", tensorfs::sha256::hex(fingerprint), index),
                payload: region.items.iter().map(|item| item.len).sum(),
                mapped: region.span,
            })
            .collect()
    }
}

/// A conservative H2D bound may use this catalog only for an isolated request/GPU.
/// Page identities include the encoding/transform/rank fingerprint and region number.
#[derive(Clone, Debug, Default)]
pub struct TransferProof {
    pub version: u32,
    pub complete: bool,
    pub isolated: bool,
    pub capacity: u64,
    pub pages: Vec<ProofPage>,
    pub initial: Vec<PageId>,
    pub invalid: Vec<&'static str>,
}

impl TransferProof {
    fn knows(&self, page: PageId) -> bool {
        self.pages
            .binary_search_by_key(&page, |known| known.page)
            .is_ok()
    }

    fn invalidate(&mut self, reason: &'static str) {
        self.complete = false;
        if !self.invalid.contains(&reason) {
            self.invalid.push(reason);
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub events: Vec<Event>,
    pub copies: Vec<Copy>,
    pub peak: u64,
    pub proof: TransferProof,
}

struct Model {
    name: String,
    pages: Vec<PageId>,
    fingerprint: [u8; 32],
    ram_key: [u8; 32],
    transform: Option<Arc<Transform>>,
    proof_pages: Vec<ProofPage>,
    source: Arc<Source>,
    regions: Vec<ledger::RamRegion>,
    ram: Arc<RamWarm>,
    ram_generation: Option<u64>,
}

struct RamWarm {
    token: u64,
    pending: AtomicBool,
    cancelled: AtomicBool,
}
impl RamWarm {
    fn new() -> Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(1);
        let token = NEXT
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.checked_add(1)
            })
            .map_err(|_| invalid("RAM registration token exhausted"))?;
        Ok(Self {
            token,
            pending: AtomicBool::new(false),
            cancelled: AtomicBool::new(false),
        })
    }
}

struct Request {
    steps: Vec<Vec<ModelId>>,
    explicit: bool,
    package: String,
    shape: String,
    stats: Stats,
    budgets: BTreeMap<u32, u64>,
    pass_index: u64,
}

struct Cache {
    shrink: Arc<Shrink>,
    active: u32,
    blocked: bool,
    busy: bool,
    resized: bool,
}

/// Suppress a zero-progress allocator trim only until its facts or request
/// boundary change. An actual callback pin survives every such rearm.
#[derive(Default)]
struct SlackTrim {
    busy: bool,
    changed: bool,
    blocked: bool,
    published: bool,
}
impl SlackTrim {
    fn publish(&mut self, changed: bool) {
        self.published |= self.busy;
        if changed {
            self.rearm();
        }
    }
    fn rearm(&mut self) {
        self.blocked = false;
        self.changed |= self.busy;
    }
    fn begin(&mut self) {
        debug_assert!(!self.busy);
        self.busy = true;
        self.changed = false;
        self.published = false;
    }
    fn finish(&mut self, no_progress: bool) {
        self.busy = false;
        self.blocked = no_progress && !self.changed;
        self.changed = false;
        self.published = false;
    }
    fn pins(&self) -> u32 {
        u32::from(self.busy || self.blocked)
    }
}

struct Inner {
    table: plan::Table,
    models: BTreeMap<ModelId, Model>,
    requests: BTreeMap<RequestId, Request>,
    caches: BTreeMap<PageId, Cache>,
    orphaned: BTreeMap<PageId, PageKey>,
    retired: BTreeSet<PageId>,
    filling: BTreeMap<PageId, ThreadId>,
    next_page: PageId,
    progress: u64,
    trim: Option<Arc<Shrink>>,
    slack_trim: SlackTrim,
}
impl Inner {
    fn rearm_slack(&mut self) {
        self.slack_trim.rearm();
        self.sync_slack_pin();
    }
    fn sync_slack_pin(&mut self) {
        if let Some(page) = self
            .table
            .pages
            .iter_mut()
            .find(|page| page.kind == Kind::Slack)
        {
            page.pins = self.slack_trim.pins();
        }
    }
}

impl TransferProof {
    fn snapshot(inner: &Inner) -> Self {
        let mut pages: Vec<_> = inner
            .models
            .values()
            .flat_map(|model| model.proof_pages.iter().cloned())
            .collect();
        pages.sort_unstable_by_key(|page| page.page);
        let initial: Vec<_> = inner
            .table
            .pages
            .iter()
            .filter(|page| page.kind == Kind::Weight && page.mapped())
            .map(|page| page.id)
            .collect();
        let mut proof = Self {
            version: 1,
            complete: true,
            isolated: inner.requests.is_empty() && !foreign_holdings(inner),
            capacity: inner.table.capacity,
            pages,
            initial,
            invalid: vec![],
        };
        if proof.initial.iter().any(|id| !proof.knows(*id)) {
            proof.invalidate("unknown-initial-content");
        }
        proof
    }
}

fn foreign_holdings(inner: &Inner) -> bool {
    inner.table.remote.iter().any(|row| {
        row.context != 0 || row.classes.iter().any(|bytes| *bytes != 0) || !row.remaining.is_empty()
    })
}

pub struct Placement {
    gpu: Arc<Gpu>,
    ledger: Arc<Ledger>,
    row: Arc<ledger::Row>,
    channel: Arc<Channel>,
    inner: Mutex<Inner>,
    operations: Mutex<()>,
    changed: Condvar,
    stop: Arc<AtomicBool>,
    worker: Mutex<Option<JoinHandle<()>>>,
}

impl Placement {
    /// Claims the current process after fork. Descriptors are owned by this call.
    pub fn open(device: i32, grant: Option<(OwnedFd, OwnedFd, OwnedFd)>) -> Result<Arc<Self>> {
        let gpu = Gpu::open(device)?;
        let (ledger, channel) = match grant {
            Some((table, events, channel)) => {
                (Ledger::adopt(table.as_fd(), events.as_fd())?, channel)
            }
            None => {
                let ledger = Ledger::create(&gpu.uuid()?, gpu.capacity()?)?;
                let grant = ledger.grant(std::process::id())?;
                (ledger, grant.channel)
            }
        };
        let row = Arc::new(ledger.claim(std::process::id())?);
        // NVML inventory may include driver-reserved bytes absent from CUDA's
        // usable total. Correct the one shared ceiling after the fork boundary,
        // before this owner can register pages or publish an allocator budget.
        let before = ledger.capacity();
        let usable = gpu.capacity()?;
        let capacity = ledger.limit_capacity(usable)?;
        if capacity < before {
            tracing::info!(gpu = %gpu.uuid()?, previous_limit_bytes = before,
                cuda_usable_bytes = usable, effective_limit_bytes = capacity,
                "GPU memory capacity corrected");
        }
        let channel = Arc::new(Channel::new(channel));
        channel.bind(row.owner())?;
        let owner = row.owner().row;
        let out = Arc::new(Self {
            gpu,
            ledger: Arc::new(ledger),
            row: row.clone(),
            channel,
            inner: Mutex::new(Inner {
                table: plan::Table {
                    rows: vec![plan::Row {
                        id: owner,
                        alive: true,
                        reaping: false,
                    }],
                    ..Default::default()
                },
                models: BTreeMap::new(),
                requests: BTreeMap::new(),
                caches: BTreeMap::new(),
                orphaned: BTreeMap::new(),
                retired: BTreeSet::new(),
                filling: BTreeMap::new(),
                next_page: 1,
                progress: 0,
                trim: None,
                slack_trim: SlackTrim::default(),
            }),
            operations: Mutex::new(()),
            changed: Condvar::new(),
            stop: Arc::new(AtomicBool::new(false)),
            worker: Mutex::new(None),
        });
        out.inner.lock().table.capacity = out.ledger.capacity();
        let wake = row.clone();
        out.gpu.set_wakeup(Arc::new(move || {
            let _ = wake.wake();
        }));
        let weak = Arc::downgrade(&out);
        let stop = out.stop.clone();
        let worker = std::thread::Builder::new()
            .name("varena-reclaim".into())
            .spawn(move || {
                Self::serve(weak, row, stop);
            })
            .map_err(|error| Error::Io(error.to_string()))?;
        *out.worker.lock() = Some(worker);
        Ok(out)
    }

    fn serve(weak: Weak<Self>, row: Arc<ledger::Row>, stop: Arc<AtomicBool>) {
        let mut seen = 0;
        while !stop.load(Ordering::Acquire) {
            let ask = match row.wait_event(&mut seen, &stop) {
                Ok(ask) => ask,
                Err(error) => {
                    tracing::warn!(%error, "reclaim wait failed");
                    break;
                }
            };
            let Some(this) = weak.upgrade() else {
                break;
            };
            let release = match row.cache_release() {
                Ok(release) => release,
                Err(error) => {
                    tracing::warn!(%error, "cache-release owner disappeared");
                    break;
                }
            };
            let released = if release.is_some() {
                let through = this
                    .inner
                    .lock()
                    .table
                    .pages
                    .iter()
                    .filter(|page| page.row == row.owner().row && page.kind == Kind::Cache)
                    .map(|page| page.id)
                    .max()
                    .unwrap_or(0);
                this.drive(
                    0,
                    Trigger::ReleaseUnusedCaches {
                        row: row.owner().row,
                        after: 0,
                        through,
                    },
                )
                .map(|_| ())
            } else {
                Ok(())
            };
            let result = if let Some(ask) = ask {
                let result = this.drive(
                    0,
                    Trigger::Ask {
                        row: row.owner().row,
                        need: ask.bytes,
                        ceiling: ask.ceiling,
                    },
                );
                let _ = row.ack(ask);
                result.map(|_| ())
            } else {
                this.refresh().and_then(|()| {
                    let requests: Vec<_> = this.inner.lock().requests.keys().copied().collect();
                    for request in requests {
                        this.drive(request, Trigger::PinReleased(request))?;
                    }
                    Ok(())
                })
            };
            let settled = this.settle();
            // Do not erase pending evidence while an untrusted callback or its
            // actual CUDA consumer completion is still running.
            drop(release);
            if let Err(error) = released.and(result).and(settled) {
                tracing::warn!(%error, "reclaim failed");
            }
        }
    }

    /// A CUDA completion does not ring the ledger's eventfd. After a release wake,
    /// wait on actual submitted work before sleeping again, then publish and reconsider
    /// pages that were still protected when the wake was first processed.
    fn settle(&self) -> Result<()> {
        loop {
            let pending = self.gpu.prepare_pending()?;
            // Preparation may dequeue a last pin. Publish that release and its
            // still-pending consumer before waiting, so observers/foregrounds
            // never need the consumer to finish just to see the new facts.
            self.refresh()?;
            if self.stop.load(Ordering::Acquire) {
                return Ok(());
            }
            let requests: Vec<_> = self.inner.lock().requests.keys().copied().collect();
            for request in requests {
                self.drive(request, Trigger::PinReleased(request))?;
            }
            if !self.gpu.wait_prepared(pending)? {
                return Ok(());
            }
        }
    }

    pub fn device(&self) -> i32 {
        self.gpu.device()
    }

    pub fn gpu_uuid(&self) -> Result<String> {
        self.gpu.uuid()
    }

    pub fn reserve_host(&self, bytes: u64) -> Result<Option<Arc<ledger::HostReservation>>> {
        if self.stop.load(Ordering::Acquire) {
            return Err(Error::Closed("placement".into()));
        }
        let active = !self.inner.lock().requests.is_empty();
        if !active {
            return Err(invalid("a mapped output requires an active request"));
        }
        self.channel.reserve_host(self.row.owner(), bytes)
    }

    pub fn available(&self) -> Result<u64> {
        self.refresh()?;
        Ok(self.inner.lock().table.available())
    }

    pub fn close(&self) -> Result<()> {
        let models: Vec<_> = self.inner.lock().models.keys().copied().collect();
        let mut error = None;
        for model in models {
            if let Err(failure) = self.close_model(model) {
                if error.is_none() {
                    error = Some(failure);
                }
            }
        }
        self.stop.store(true, Ordering::Release);
        self.changed.notify_all();
        let wake = self.row.wake();
        match error {
            Some(error) => Err(error),
            None => wake,
        }
    }

    pub fn load(&self, model: ModelId, level: Tier) -> Result<Relief> {
        if level == Tier::Ram {
            return self.warm_ram(model);
        }
        self.drive(
            0,
            Trigger::Load {
                row: self.row.owner().row,
                model,
                level,
            },
        )
    }

    /// Warming is a cache hint. Downloading and registration never delay admission or
    /// hold the demand channel; a GPU miss continues through its verified disk source.
    fn warm_ram(&self, model: ModelId) -> Result<Relief> {
        let (fingerprint, source, regions, state) = {
            let mut inner = self.inner.lock();
            let data = inner
                .models
                .get(&model)
                .ok_or_else(|| invalid("unknown model"))?;
            let work = (
                data.ram_key,
                data.source.clone(),
                data.regions.clone(),
                data.ram.clone(),
            );
            for page in inner
                .table
                .pages
                .iter_mut()
                .filter(|page| page.model == Some(model))
            {
                page.loaded = true;
            }
            work
        };
        if state.cancelled.load(Ordering::Acquire) || state.pending.swap(true, Ordering::AcqRel) {
            return Ok(Relief::Complete);
        }
        let demand = self.channel.clone();
        let row = self.row.clone();
        let worker_state = state.clone();
        let started = std::thread::Builder::new()
            .name("varena-warm".into())
            .spawn(move || {
                let warm = || -> Result<()> {
                    if worker_state.cancelled.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    let items: Vec<_> = regions
                        .iter()
                        .flat_map(|region| region.items.iter().cloned())
                        .collect();
                    let files = source
                        .export_files(&items)
                        .map_err(|error| Error::Io(error.to_string()))?
                        .into_iter()
                        .map(|(object, file)| (object, file.into()))
                        .collect::<Vec<_>>();
                    if worker_state.cancelled.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    let channel = demand.fork(row.owner())?;
                    channel.cache_ram_tagged(
                        row.owner(),
                        fingerprint,
                        &files,
                        &regions,
                        worker_state.token,
                    )?;
                    if worker_state.cancelled.load(Ordering::Acquire) {
                        channel.forget_ram(row.owner(), fingerprint, worker_state.token)?;
                    }
                    Ok(())
                };
                if let Err(error) = warm() {
                    tracing::warn!(%error, "RAM warming skipped; disk source remains available");
                }
                worker_state.pending.store(false, Ordering::Release);
                let _ = row.wake();
            });
        if let Err(error) = started {
            state.pending.store(false, Ordering::Release);
            tracing::warn!(%error, "RAM warm worker unavailable; disk source remains available");
        }
        Ok(Relief::Complete)
    }

    pub fn register(
        &self,
        id: ModelId,
        name: String,
        fingerprint: [u8; 32],
        layout: &Layout,
        source: Arc<Source>,
        transform: Option<Arc<Transform>>,
    ) -> Result<Vec<PageId>> {
        let _operation = self.operations.lock();
        if self.inner.lock().models.contains_key(&id) {
            return Err(invalid("model already registered"));
        }
        let range = self.gpu.reserve(layout.nbytes)?;
        let mut rollback = Registration {
            owner: self,
            model: id,
            pages: Vec::new(),
            committed: false,
        };
        let mut pages = Vec::new();
        for (index, region) in layout.regions.iter().enumerate() {
            let page = self.new_page()?;
            self.gpu.register(
                page,
                range.clone(),
                region.offset,
                region.span,
                source.clone(),
                region.items.clone(),
                region.offset,
                tensorfs::sha256::hex(&fingerprint),
            )?;
            rollback.pages.push(page);
            self.inner.lock().table.pages.push(plan::Page {
                id: page,
                row: self.row.owner().row,
                model: Some(id),
                bytes: region.span,
                tier: Tier::Disk,
                kind: Kind::Weight,
                pins: 0,
                filling: false,
                loaded: false,
                last_used: 0,
                spread_rank: spread_rank(index, layout.regions.len()),
                uses: vec![],
                speculative_for: None,
                attempted: BTreeSet::new(),
                ram_ready: false,
                lowered: BTreeSet::new(),
            });
            let key = PageKey {
                fingerprint,
                page: index as u64,
            };
            let imported = self.channel.take(self.row.owner(), key)?;
            if !imported.is_empty() {
                let (sender, receiver) =
                    UnixStream::pair().map_err(|error| Error::Io(error.to_string()))?;
                let spans = imported.iter().map(|(bytes, _)| *bytes).collect::<Vec<_>>();
                for (_, fd) in &imported {
                    platform::handles::send_fd(&sender, fd)
                        .map_err(crate::os("repackage kept page"))?;
                }
                let _ = self.gpu.import(
                    page,
                    &tensorfs::sha256::hex(&fingerprint),
                    &spans,
                    receiver.into(),
                )?;
            }
            pages.push(page);
        }
        self.inner.lock().models.insert(
            id,
            Model {
                name,
                pages: pages.clone(),
                fingerprint,
                ram_key: tensorfs::sha256::digest(layout.digest.as_bytes()),
                transform,
                proof_pages: ProofPage::catalog(&pages, &fingerprint, layout),
                source,
                regions: layout
                    .regions
                    .iter()
                    .map(|region| ledger::RamRegion {
                        offset: region.offset,
                        len: region.span,
                        items: region.items.clone(),
                    })
                    .collect(),
                ram: Arc::new(RamWarm::new()?),
                ram_generation: None,
            },
        );
        for run in self.inner.lock().requests.values_mut() {
            run.stats.proof.invalidate("registration-changed");
        }
        self.refresh()?;
        rollback.committed = true;
        Ok(pages)
    }

    fn new_page(&self) -> Result<PageId> {
        let mut inner = self.inner.lock();
        let page = inner.next_page;
        inner.next_page = page
            .checked_add(1)
            .ok_or_else(|| invalid("page id exhausted"))?;
        Ok(page)
    }

    pub fn begin(
        &self,
        id: RequestId,
        steps: Vec<Vec<ModelId>>,
        package: String,
        shape: String,
    ) -> Result<()> {
        let _operation = self.operations.lock();
        self.refresh()?;
        if self.inner.lock().requests.contains_key(&id) {
            return Err(invalid("request already active"));
        }
        let mut inner = self.inner.lock();
        let proof = TransferProof::snapshot(&inner);
        for run in inner.requests.values_mut() {
            run.stats.proof.isolated = false;
        }
        inner.table.requests.push(plan::Request {
            id,
            row: self.row.owner().row,
            position: Position { step: 0, order: 0 },
            working_sets: vec![None; steps.len().max(1)],
            starved: false,
            sharing_limit: None,
            resident_steps: BTreeSet::new(),
            joint_steps: BTreeSet::new(),
            model_positions: BTreeMap::new(),
            can_cut: true,
            can_lower: true,
            reclaim_round: 0,
        });
        inner.requests.insert(
            id,
            Request {
                steps: vec![],
                explicit: false,
                package,
                shape,
                stats: Stats {
                    proof,
                    ..Stats::default()
                },
                budgets: BTreeMap::new(),
                pass_index: 0,
            },
        );
        if let Err(error) = Self::declare_locked(&mut inner, id, steps) {
            Self::remove_request(&mut inner, id);
            return Err(error);
        }
        // A new request gets one opportunity to trim blocks whose asynchronous
        // consumers finished without changing allocator byte counters.
        inner.rearm_slack();
        Self::event_locked(&mut inner, id, 0, "begin", "request")?;
        drop(inner);
        let mut rollback = Beginning {
            owner: self,
            request: id,
            committed: false,
        };
        self.hydrate(id)?;
        self.refresh()?;
        rollback.committed = true;
        Ok(())
    }

    pub fn declare(&self, id: RequestId, steps: Vec<Vec<ModelId>>) -> Result<()> {
        let _operation = self.operations.lock();
        Self::declare_locked(&mut self.inner.lock(), id, steps)?;
        self.hydrate(id)?;
        self.refresh()
    }

    fn declare_locked(inner: &mut Inner, id: RequestId, steps: Vec<Vec<ModelId>>) -> Result<()> {
        let mut groups: Vec<(Vec<ModelId>, u32)> = vec![];
        for mut models in steps {
            models.sort_unstable();
            models.dedup();
            if models.iter().any(|model| !inner.models.contains_key(model)) {
                return Err(invalid("plan names unknown model"));
            }
            if let Some((prior, passes)) = groups.last_mut().filter(|(prior, _)| *prior == models) {
                let _ = prior;
                *passes = passes
                    .checked_add(1)
                    .ok_or_else(|| invalid("too many passes"))?;
            } else {
                groups.push((models, 1));
            }
        }
        for page in &mut inner.table.pages {
            page.uses.retain(|usage| usage.request != id);
        }
        for (step, (models, passes)) in groups.iter().enumerate() {
            let mut orders = BTreeMap::<ModelId, u32>::new();
            for _ in 0..*passes {
                for model in models {
                    let order = orders.entry(*model).or_default();
                    let pages = inner
                        .models
                        .get(model)
                        .ok_or_else(|| invalid("unknown model"))?
                        .pages
                        .clone();
                    for page in pages {
                        let page = inner
                            .table
                            .pages
                            .iter_mut()
                            .find(|page_| page_.id == page)
                            .ok_or_else(|| invalid("model page absent"))?;
                        page.uses.push(plan::Use {
                            request: id,
                            position: Position {
                                step: u32::try_from(step).map_err(|_| invalid("too many steps"))?,
                                order: *order,
                            },
                        });
                        *order = order
                            .checked_add(1)
                            .ok_or_else(|| invalid("plan is too long"))?;
                    }
                }
            }
        }
        let facts = inner
            .table
            .requests
            .iter_mut()
            .find(|request| request.id == id)
            .ok_or_else(|| invalid("request facts absent"))?;
        facts.working_sets.resize(groups.len().max(1), None);
        facts.resident_steps = groups
            .iter()
            .enumerate()
            .filter(|(_, (_, passes))| *passes >= 2)
            .map(|(step, _)| step as u32)
            .collect();
        facts.joint_steps = groups
            .iter()
            .enumerate()
            .filter(|(_, (models, _))| models.len() > 1)
            .map(|(step, _)| step as u32)
            .collect();
        facts.position = Position { step: 0, order: 0 };
        facts.model_positions.clear();
        facts.starved = false;
        let request = inner
            .requests
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown request"))?;
        request.steps = groups.into_iter().map(|(models, _)| models).collect();
        request.explicit = !request.steps.is_empty();
        request.budgets.clear();
        Ok(())
    }

    /// No plan means demand-only operation. Record just the model actually entered;
    /// there is no future use to prefetch and earlier models become reclaimable.
    fn demand_step(inner: &mut Inner, request: RequestId, model: ModelId) -> Result<()> {
        let pages = inner
            .models
            .get(&model)
            .ok_or_else(|| invalid("unknown model"))?
            .pages
            .clone();
        let run = inner
            .requests
            .get_mut(&request)
            .ok_or_else(|| invalid("unknown request"))?;
        let step = u32::try_from(run.steps.len()).map_err(|_| invalid("too many steps"))?;
        run.steps.push(vec![model]);
        for page in &mut inner.table.pages {
            page.uses.retain(|usage| usage.request != request);
        }
        for (order, id) in pages.into_iter().enumerate() {
            let page = inner
                .table
                .pages
                .iter_mut()
                .find(|page| page.id == id)
                .ok_or_else(|| invalid("model page absent"))?;
            page.uses.push(plan::Use {
                request,
                position: Position {
                    step,
                    order: u32::try_from(order).map_err(|_| invalid("too many pages"))?,
                },
            });
        }
        let facts = inner
            .table
            .requests
            .iter_mut()
            .find(|row| row.id == request)
            .ok_or_else(|| invalid("request facts absent"))?;
        facts.position = Position { step, order: 0 };
        facts.model_positions.clear();
        facts.working_sets.resize(step as usize + 1, None);
        facts.starved = false;
        Ok(())
    }

    pub fn enter(&self, request: RequestId, model: ModelId) -> Result<Relief> {
        let demand;
        {
            let mut inner = self.inner.lock();
            demand = !inner
                .requests
                .get(&request)
                .ok_or_else(|| invalid("unknown request"))?
                .explicit;
            if demand {
                Self::demand_step(&mut inner, request, model)?;
            }
            let from = inner
                .table
                .request(request)
                .ok_or_else(|| invalid("unknown request"))?
                .position
                .step;
            let next = inner.requests.get(&request).and_then(|run| {
                run.steps
                    .iter()
                    .enumerate()
                    .skip(from as usize)
                    .find(|(_, models)| models.contains(&model))
                    .map(|(step, _)| step as u32)
            });
            if let Some(next) = next {
                if let Some(facts) = inner
                    .table
                    .requests
                    .iter_mut()
                    .find(|row| row.id == request)
                {
                    facts.entered(next, model);
                }
            } else {
                let name = inner
                    .models
                    .get(&model)
                    .map_or("<unknown>", |data| data.name.as_str());
                return Err(Error::Invalid(format!(
                    "model is absent from the remaining plan (request {request}, model {model} [{name}], current step {from})"
                )));
            }
        }
        if demand {
            self.hydrate(request)?;
        }
        let result = self.drive(request, Trigger::StepStart(request))?;
        self.capture_budget(request);
        Ok(result)
    }

    pub fn bind(&self, request: RequestId, page: PageId, stream: usize) -> Result<Pin> {
        let (need, demand) = {
            let inner = self.inner.lock();
            let page = inner
                .table
                .page(page)
                .ok_or_else(|| invalid("unknown page"))?;
            let model = page.model.ok_or_else(|| invalid("page has no model"))?;
            let run = inner
                .requests
                .get(&request)
                .ok_or_else(|| invalid("unknown request"))?;
            let step = inner
                .table
                .request(request)
                .ok_or_else(|| invalid("request facts absent"))?
                .position
                .step;
            let active = run
                .steps
                .get(step as usize)
                .is_some_and(|models| models.contains(&model));
            (page.bytes, (!run.explicit && !active).then_some(model))
        };
        if let Some(model) = demand {
            self.enter(request, model)?;
        }
        let mut retry = BindRetry::default();
        loop {
            self.wait_fill(page)?;
            if let Some(row) = self
                .inner
                .lock()
                .table
                .requests
                .iter_mut()
                .find(|row| row.id == request)
            {
                // A weight-page map has no arithmetic op to cut; activation OOMs still cut first.
                row.can_cut = false;
            }
            let outcome = match self.drive(request, Trigger::Bind { request, page }) {
                Ok(outcome) => outcome,
                Err(error) if error.is_oom() => {
                    self.gpu.recover()?;
                    self.refresh()?;
                    let available = self.inner.lock().table.available();
                    let need = retry.after_oom(available, need)?;
                    self.drive(request, Trigger::Oom { request, need })?
                }
                Err(error) => return Err(error),
            };
            match outcome {
                Relief::Complete => {
                    self.wait_fill(page)?;
                    let operation = self.operations.lock();
                    let mut inner = self.inner.lock();
                    let model = inner
                        .table
                        .page(page)
                        .and_then(|row| row.model)
                        .ok_or_else(|| invalid("page model absent"))?;
                    if !inner
                        .models
                        .get(&model)
                        .is_some_and(|model| model.pages.contains(&page))
                        || !inner.requests.contains_key(&request)
                    {
                        return Err(invalid("binding owner disappeared"));
                    }
                    match self.gpu.try_pin(page, stream)? {
                        BindState::Ready { pin, pins } => {
                            let row = inner
                                .table
                                .pages
                                .iter_mut()
                                .find(|row| row.id == page)
                                .ok_or_else(|| invalid("bound page absent"))?;
                            row.tier = Tier::Gpu;
                            row.pins = u32::try_from(pins).unwrap_or(u32::MAX);
                            Self::event_locked(&mut inner, request, page, "bind", "demand")?;
                            inner.table.bound(request, page);
                            inner.progress = inner.progress.saturating_add(1);
                            self.publish(&inner)?;
                            drop(inner);
                            drop(operation);
                            // The actual pin protects the mapping while its stream
                            // dependency is enqueued outside both publication gates.
                            // An error drops the pin through checked deferred cleanup.
                            if let Err(error) = pin.wait_on(stream) {
                                if let Some(run) = self.inner.lock().requests.get_mut(&request) {
                                    run.stats.proof.invalidate("bind-stream-wait-failed");
                                }
                                return Err(error);
                            }
                            self.refresh()?;
                            self.row.wake()?;
                            return Ok(pin);
                        }
                        BindState::Filling(pending) => {
                            drop(inner);
                            drop(operation);
                            self.refresh()?;
                            self.gpu.wait_prepared(pending)?;
                            // Retry only after the captured generation progresses.
                            continue;
                        }
                        BindState::Empty => {
                            // A reclaimer won the handoff after drive completed.
                            // The existing progress check bounds the new decision.
                        }
                    }
                }
                Relief::Lower(short) => {
                    self.refresh()?;
                    let reclaim_from = self
                        .inner
                        .lock()
                        .table
                        .occupied()
                        .ok_or_else(|| invalid("occupied bytes overflow"))?;
                    if !matches!(
                        self.drive(
                            request,
                            Trigger::Lower {
                                request,
                                // The prospective ring already covers this weight page.
                                allocation: 0,
                                reclaim_from,
                                reclaim: short,
                            }
                        )?,
                        Relief::Complete
                    ) {
                        return Err(Error::DeviceFull {
                            need,
                            deficit: short,
                        });
                    }
                }
                Relief::Cut(short) | Relief::Exhausted(short) => {
                    return Err(Error::DeviceFull {
                        need,
                        deficit: short,
                    });
                }
            }
            self.refresh()?;
            let available = self.inner.lock().table.available();
            // Failed fills emit diagnostics too. Only increasing actual room,
            // never an event sequence or an oscillating snapshot, permits retry.
            if !retry.advance(available) {
                return Err(Error::DeviceFull {
                    need,
                    deficit: need,
                });
            }
        }
    }

    fn wait_fill(&self, page: PageId) -> Result<()> {
        let mut inner = self.inner.lock();
        while let Some(owner) = inner.filling.get(&page) {
            if self.stop.load(Ordering::Acquire) {
                break;
            }
            if *owner == thread::current().id() {
                return Err(invalid("page fill depends on its own unfinished page"));
            }
            self.changed.wait(&mut inner);
        }
        if self.stop.load(Ordering::Acquire) {
            return Err(Error::Closed("placement".into()));
        }
        Ok(())
    }

    fn step_key(inner: &Inner, request: RequestId, step: usize) -> Result<String> {
        let run = inner
            .requests
            .get(&request)
            .ok_or_else(|| invalid("unknown request"))?;
        let models = run
            .steps
            .get(step)
            .ok_or_else(|| invalid("unknown request step"))?;
        let mut fingerprint = run.package.as_bytes().to_vec();
        for model in models {
            fingerprint.extend_from_slice(
                &inner
                    .models
                    .get(model)
                    .ok_or_else(|| invalid("unknown model"))?
                    .fingerprint,
            );
        }
        Ok(format!(
            "step:{}",
            tensorfs::sha256::hex_digest(&fingerprint)
        ))
    }

    fn hydrate(&self, request: RequestId) -> Result<()> {
        let mut inner = self.inner.lock();
        let run = inner
            .requests
            .get(&request)
            .ok_or_else(|| invalid("unknown request"))?;
        let shape = run.shape.clone();
        let peaks = (0..run.steps.len())
            .map(|step| {
                Self::step_key(&inner, request, step).map(|key| self.ledger.measured(&key, &shape))
            })
            .collect::<Result<Vec<_>>>()?;
        if let Some(facts) = inner
            .table
            .requests
            .iter_mut()
            .find(|row| row.id == request)
        {
            facts.working_sets = if peaks.is_empty() { vec![None] } else { peaks };
        }
        Ok(())
    }

    pub fn leave(&self, request: RequestId, peak: u64) -> Result<()> {
        let (key, shape) = {
            let mut inner = self.inner.lock();
            let step = inner
                .table
                .request(request)
                .ok_or_else(|| invalid("unknown request"))?
                .position
                .step as usize;
            let key = Self::step_key(&inner, request, step)?;
            let models = inner
                .requests
                .get(&request)
                .ok_or_else(|| invalid("unknown request"))?
                .steps
                .get(step)
                .cloned()
                .ok_or_else(|| invalid("unknown request step"))?;
            let resident = inner
                .table
                .pages
                .iter()
                .filter(|page| page.mapped() && page.model.is_some_and(|id| models.contains(&id)))
                .map(|page| page.bytes)
                .fold(0_u64, u64::saturating_add);
            let facts = inner
                .table
                .requests
                .iter_mut()
                .find(|row| row.id == request)
                .ok_or_else(|| invalid("unknown request"))?;
            if let Some(measured) = facts.working_sets.get_mut(step) {
                *measured = Some(measured.unwrap_or(0).max(peak));
            }
            let run = inner
                .requests
                .get_mut(&request)
                .ok_or_else(|| invalid("unknown request"))?;
            run.stats.peak = run.stats.peak.max(resident.saturating_add(peak));
            run.pass_index = run.pass_index.saturating_add(1);
            (key, run.shape.clone())
        };
        self.ledger.record_peak(&key, &shape, peak)?;
        self.refresh()?;
        self.capture_budget(request);
        self.row.wake()
    }

    pub fn end(&self, request: RequestId) -> Result<Stats> {
        let recovered = self.gpu.recover();
        self.event(request, 0, "end", "request")?;
        let mut inner = self.inner.lock();
        let measured = inner
            .table
            .request(request)
            .is_some_and(|row| row.working_sets.iter().all(Option::is_some));
        let peak = inner
            .requests
            .get(&request)
            .filter(|_| measured)
            .map(|run| (run.package.clone(), run.shape.clone(), run.stats.peak));
        let stats = Self::remove_request(&mut inner, request)
            .ok_or_else(|| invalid("unknown request"))?
            .stats;
        if inner.requests.is_empty() {
            // Admission runs before the next Begin. Do not leave an idle owner
            // with a zero-trim suppression pin that no request can refresh.
            inner.rearm_slack();
        }
        drop(inner);
        self.refresh()?;
        recovered?;
        if let Some((package, shape, peak)) = peak {
            self.ledger.record_peak(&package, &shape, peak)?;
        }
        Ok(stats)
    }

    fn remove_request(inner: &mut Inner, request: RequestId) -> Option<Request> {
        inner.table.requests.retain(|row| row.id != request);
        inner
            .table
            .reclaims
            .retain(|row| row.step.request != request);
        for page in &mut inner.table.pages {
            page.uses.retain(|usage| usage.request != request);
            page.attempted.retain(|step| step.request != request);
            page.lowered.remove(&request);
            if page
                .speculative_for
                .is_some_and(|step| step.request == request)
            {
                page.speculative_for = None;
            }
        }
        inner.requests.remove(&request)
    }

    pub fn stats(&self, request: RequestId) -> Result<Stats> {
        self.inner
            .lock()
            .requests
            .get(&request)
            .map(|run| run.stats.clone())
            .ok_or_else(|| invalid("unknown request"))
    }

    pub fn relief(&self, request: RequestId, need: u64, lower: bool) -> Result<Relief> {
        self.gpu.recover()?;
        self.refresh()?;
        let (reclaim_from, target) = {
            let mut inner = self.inner.lock();
            if let Some(facts) = inner
                .table
                .requests
                .iter_mut()
                .find(|row| row.id == request)
            {
                facts.can_cut = !lower;
                if !lower {
                    // A new actual OOM may ask the same owner again. An ACK
                    // within this operation still bounds every re-decision.
                    facts.reclaim_round = facts
                        .reclaim_round
                        .checked_add(1)
                        .ok_or_else(|| invalid("OOM reclaim round overflow"))?;
                }
            }
            let occupied = inner
                .table
                .occupied()
                .ok_or_else(|| invalid("occupied bytes overflow"))?;
            // CUDA already refused the allocation. A logical free estimate is
            // not proof of physical room (for example, a new CUDA context may
            // not yet have a process measurement). Keep one fixed target across
            // remote acknowledgements and re-decisions: Complete must reclaim
            // actual bytes, without enlarging the allocator's real budget.
            let target = if lower {
                need
            } else {
                activation_oom_target(inner.table.available(), need)?
            };
            (occupied, target)
        };
        self.drive(
            request,
            if lower {
                Trigger::Lower {
                    request,
                    allocation: need,
                    reclaim_from,
                    // At least one real mapped page must leave after a failed
                    // allocation, even if the logical estimate says it fits.
                    reclaim: 1,
                }
            } else {
                Trigger::Oom {
                    request,
                    need: target,
                }
            },
        )
    }

    pub fn resident(&self) -> Result<Vec<(String, u64)>> {
        self.refresh()?;
        let inner = self.inner.lock();
        Ok(inner
            .models
            .iter()
            .map(|(id, model)| {
                (
                    model.name.clone(),
                    inner
                        .table
                        .pages
                        .iter()
                        .filter(|page| page.model == Some(*id) && page.mapped())
                        .map(|page| page.bytes)
                        .sum(),
                )
            })
            .collect())
    }

    pub fn cache(&self, bytes: u64, shrink: Arc<Shrink>) -> Result<PageId> {
        let id = self.new_page()?;
        let mut inner = self.inner.lock();
        let last_used = inner.progress;
        inner.caches.insert(
            id,
            Cache {
                shrink,
                active: 0,
                blocked: false,
                busy: false,
                resized: false,
            },
        );
        inner.table.pages.push(plan::Page {
            id,
            row: self.row.owner().row,
            model: None,
            bytes,
            tier: Tier::Gpu,
            kind: Kind::Cache,
            pins: 0,
            filling: false,
            loaded: false,
            last_used,
            spread_rank: 0,
            uses: vec![],
            speculative_for: None,
            attempted: BTreeSet::new(),
            ram_ready: false,
            lowered: BTreeSet::new(),
        });
        Ok(id)
    }

    pub fn cache_bytes(&self, id: PageId, bytes: u64) -> Result<()> {
        let mut inner = self.inner.lock();
        let cache = inner
            .caches
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown cache"))?;
        cache.blocked = false;
        cache.resized |= cache.busy;
        let pins = cache.active.saturating_add(u32::from(cache.busy));
        let page = inner
            .table
            .pages
            .iter_mut()
            .find(|page| page.id == id && page.kind == Kind::Cache)
            .ok_or_else(|| invalid("unknown cache"))?;
        page.bytes = bytes;
        page.pins = pins;
        Ok(())
    }

    pub fn close_cache(&self, id: PageId) {
        let mut inner = self.inner.lock();
        inner.caches.remove(&id);
        inner.table.pages.retain(|page| page.id != id);
        self.changed.notify_all();
    }

    pub fn cache_active(&self, id: PageId, active: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        while active && inner.caches.get(&id).is_some_and(|cache| cache.busy) {
            self.changed.wait(&mut inner);
        }
        let cache = inner
            .caches
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown cache"))?;
        cache.active = if active {
            cache.active.saturating_add(1)
        } else {
            cache.active.saturating_sub(1)
        };
        let pins = cache
            .active
            .saturating_add(u32::from(cache.blocked))
            .saturating_add(u32::from(cache.busy));
        let page = inner
            .table
            .pages
            .iter_mut()
            .find(|page| page.id == id && page.kind == Kind::Cache)
            .ok_or_else(|| invalid("unknown cache"))?;
        page.pins = pins;
        Ok(())
    }

    pub fn retained_bind(&self, request: RequestId, model: ModelId) -> Result<()> {
        self.enter(request, model)?;
        let pages = self
            .inner
            .lock()
            .models
            .get(&model)
            .ok_or_else(|| invalid("unknown model"))?
            .pages
            .clone();
        for page in pages {
            self.event(request, page, "bind", "retained")?;
        }
        Ok(())
    }

    pub fn close_model(&self, id: ModelId) -> Result<()> {
        let (pages, fingerprint, token) = {
            let mut inner = self.inner.lock();
            let Some(model) = inner.models.remove(&id) else {
                return Ok(());
            };
            for run in inner.requests.values_mut() {
                run.stats.proof.invalidate("registration-changed");
            }
            model.ram.cancelled.store(true, Ordering::Release);
            inner.retired.extend(model.pages.iter().copied());
            for page in inner
                .table
                .pages
                .iter_mut()
                .filter(|page| page.model == Some(id))
            {
                page.loaded = false;
                page.uses.clear();
            }
            (model.pages, model.ram_key, model.ram.token)
        };
        let forgotten = self
            .channel
            .forget_ram(self.row.owner(), fingerprint, token);
        for page in pages {
            self.gpu.unregister(page)?;
        }
        self.refresh().and(forgotten)
    }

    pub fn trim_with(&self, trim: Arc<Shrink>) {
        self.inner.lock().trim = Some(trim);
    }

    pub fn allocator(&self, context: u64, allocated: u64, reserved: u64) -> Result<()> {
        let mut inner = self.inner.lock();
        let mut changed = false;
        let cached: u64 = inner
            .table
            .pages
            .iter()
            .filter(|page| page.kind == Kind::Cache)
            .map(|page| page.bytes)
            .sum();
        for (kind, bytes) in [
            (Kind::Context, context),
            (Kind::Scratch(0), allocated.saturating_sub(cached)),
            (Kind::Slack, reserved.saturating_sub(allocated)),
        ] {
            if let Some(page) = inner.table.pages.iter_mut().find(|page| page.kind == kind) {
                changed |= page.bytes != bytes;
                page.bytes = bytes;
            } else {
                changed = true;
                let id = inner.next_page;
                inner.next_page = id
                    .checked_add(1)
                    .ok_or_else(|| invalid("page id exhausted"))?;
                inner.table.pages.push(plan::Page {
                    id,
                    row: self.row.owner().row,
                    model: None,
                    bytes,
                    tier: Tier::Gpu,
                    kind,
                    pins: 0,
                    filling: false,
                    loaded: false,
                    last_used: 0,
                    spread_rank: 0,
                    uses: vec![],
                    speculative_for: None,
                    attempted: BTreeSet::new(),
                    ram_ready: false,
                    lowered: BTreeSet::new(),
                });
            }
        }
        inner.slack_trim.publish(changed);
        if changed {
            inner.sync_slack_pin();
        }
        drop(inner);
        if changed {
            self.row.wake()?;
        }
        Ok(())
    }

    /// Absolute Torch allocator ceiling, derived from actual non-Torch allocations.
    pub fn allocator_budget(&self) -> Result<u64> {
        self.refresh()?;
        let inner = self.inner.lock();
        let own = inner
            .table
            .pages
            .iter()
            .filter(|page| page.mapped() && matches!(page.kind, Kind::Weight | Kind::Context))
            .map(|page| page.bytes)
            .fold(0_u64, u64::saturating_add);
        let others = inner
            .table
            .remote
            .iter()
            .map(|row| {
                row.classes
                    .iter()
                    .copied()
                    .fold(row.context, u64::saturating_add)
            })
            .fold(0_u64, u64::saturating_add);
        Ok(inner
            .table
            .capacity
            .saturating_sub(own.saturating_add(others)))
    }

    fn refresh(&self) -> Result<()> {
        let status = self.ledger.status()?;
        let orphans = self.channel.orphans(self.row.owner())?;
        let models: Vec<_> = self
            .inner
            .lock()
            .models
            .iter()
            .map(|(id, model)| (*id, model.ram_key, model.ram.clone()))
            .collect();
        let ram = models
            .into_iter()
            .map(|(id, fingerprint, state)| {
                self.channel
                    .ram_status(self.row.owner(), fingerprint)
                    .map(|status| (id, state, status))
            })
            .collect::<Result<Vec<_>>>()?;
        self.gpu.poll()?;
        let mut inner = self.inner.lock();
        // Capture and publish GPU facts under the same gate used by try_pin.
        // No earlier capture survives the IPC above and overwrites a newer pin.
        // Poll (including checked CUDA cleanup) stays outside this gate.
        let facts = self.gpu.facts_current()?;
        for (id, state, status) in ram {
            let pages = match inner
                .models
                .get_mut(&id)
                .filter(|model| Arc::ptr_eq(&model.ram, &state))
            {
                Some(model) => {
                    if status
                        .as_ref()
                        .zip(model.ram_generation)
                        .is_some_and(|(status, prior)| status.generation < prior)
                    {
                        continue;
                    }
                    model.ram_generation = status.as_ref().map(|status| status.generation);
                    model.pages.clone()
                }
                None => continue,
            };
            for (index, id) in pages.into_iter().enumerate() {
                if let Some(page) = inner.table.pages.iter_mut().find(|page| page.id == id) {
                    page.ram_ready = status
                        .as_ref()
                        .and_then(|status| status.ready.get(index))
                        .copied()
                        .unwrap_or(false);
                }
            }
        }
        inner.table.capacity = status.capacity;
        let present: BTreeSet<_> = facts.iter().map(|(id, _, _, _)| *id).collect();
        let finished: BTreeSet<_> = inner.retired.difference(&present).copied().collect();
        inner
            .table
            .pages
            .retain(|page| !finished.contains(&page.id));
        inner.retired.retain(|id| !finished.contains(id));
        for (id, bytes, pins, pending) in facts {
            let reserved = inner.filling.contains_key(&id);
            if let Some(page) = inner.table.pages.iter_mut().find(|page| page.id == id) {
                page.tier = if bytes == 0 && !reserved {
                    Tier::Disk
                } else {
                    Tier::Gpu
                };
                page.pins = u32::try_from(pins).unwrap_or(u32::MAX);
                page.filling = pending || reserved;
            }
        }
        inner.table.remote = status
            .rows
            .iter()
            .filter(|row| row.owner != self.row.owner())
            .map(|row| plan::Remote {
                row: row.owner.row,
                classes: row.snapshot.classes,
                context: row.snapshot.context,
                remaining: row
                    .snapshot
                    .requests
                    .iter()
                    .map(|request| request.remaining)
                    .collect(),
                reaping: row.state == ledger::State::Reaping,
            })
            .collect();
        let shared = inner.requests.len() > 1 || foreign_holdings(&inner);
        let capacity = inner.table.capacity;
        for run in inner.requests.values_mut() {
            run.stats.proof.isolated &= !shared;
            run.stats.proof.capacity = run.stats.proof.capacity.max(capacity);
        }
        inner.table.rows = status
            .rows
            .iter()
            .map(|row| plan::Row {
                id: row.owner.row,
                alive: row.state == ledger::State::Live,
                reaping: row.state == ledger::State::Reaping,
            })
            .collect();
        if !inner
            .table
            .rows
            .iter()
            .any(|row| row.id == self.row.owner().row)
        {
            return Err(Error::Closed("own ledger row disappeared".into()));
        }
        const ORPHANS: u32 = u32::MAX;
        inner.table.rows.push(plan::Row {
            id: ORPHANS,
            alive: false,
            reaping: false,
        });
        let present: BTreeSet<_> = orphans.iter().map(|(key, _)| *key).collect();
        let gone: BTreeSet<_> = inner
            .orphaned
            .iter()
            .filter(|(_, key)| !present.contains(key))
            .map(|(id, _)| *id)
            .collect();
        inner.orphaned.retain(|id, _| !gone.contains(id));
        inner.table.pages.retain(|page| !gone.contains(&page.id));
        for (key, bytes) in orphans {
            if inner.orphaned.values().any(|held| *held == key) {
                continue;
            }
            let id = inner.next_page;
            inner.next_page = id
                .checked_add(1)
                .ok_or_else(|| invalid("page id exhausted"))?;
            inner.orphaned.insert(id, key);
            inner.table.pages.push(plan::Page {
                id,
                row: ORPHANS,
                model: None,
                bytes,
                tier: Tier::Kept,
                kind: Kind::Weight,
                pins: 0,
                filling: false,
                loaded: false,
                last_used: 0,
                spread_rank: 0,
                uses: vec![],
                speculative_for: None,
                attempted: BTreeSet::new(),
                ram_ready: false,
                lowered: BTreeSet::new(),
            });
        }
        self.publish(&inner)
    }

    /// Atomic row publication only; no CUDA calls, fill or custody IPC.
    fn publish(&self, inner: &Inner) -> Result<()> {
        let snapshot = ledger::Snapshot {
            classes: inner
                .table
                .published_classes(self.row.owner().row)
                .map_err(invalid)?,
            context: inner
                .table
                .pages
                .iter()
                .filter(|page| page.kind == Kind::Context)
                .map(|page| page.bytes)
                .sum(),
            requests: inner
                .table
                .requests
                .iter()
                .map(|request| ledger::Running {
                    id: request.id,
                    remaining: inner.table.remaining(request.id),
                })
                .collect(),
        };
        self.row.publish(&snapshot, inner.progress)
    }

    fn drive(&self, request: RequestId, mut trigger: Trigger) -> Result<Relief> {
        let reason = match trigger {
            Trigger::FillLanded(_) | Trigger::PinReleased(_) => "speculation",
            Trigger::Lower { .. } => "lower",
            _ => "demand",
        };
        'decide: loop {
            if self.stop.load(Ordering::Acquire) {
                return Err(Error::Closed("placement".into()));
            }
            let operation = self.operations.lock();
            self.refresh()?;
            let actions = plan::decide(&self.inner.lock().table, trigger);
            let mut wait = None;
            for action in actions {
                if let (Trigger::ReleaseUnusedCaches { after, .. }, Action::Shrink { page, .. }) =
                    (&mut trigger, &action)
                {
                    // One callback per cache in the initial fixed set, including
                    // callbacks that report progress without resizing their page.
                    *after = *page;
                }
                match action {
                    Action::DropHome { request, page } => {
                        self.event(request, page, "lower", "lower")?;
                        let mut inner = self.inner.lock();
                        if let Some(active) = inner
                            .table
                            .requests
                            .iter_mut()
                            .find(|row| row.id == request)
                        {
                            active.starved = true;
                        }
                        if let Some(page) = inner.table.pages.iter_mut().find(|row| row.id == page)
                        {
                            page.lowered.insert(request);
                            page.speculative_for = None;
                        }
                    }
                    Action::Map(_) => {} // Gpu.fill atomically allocates and fills the page.
                    Action::Fill {
                        page,
                        from,
                        to,
                        speculative,
                    } => {
                        if to == Tier::Ram {
                            let model = self
                                .inner
                                .lock()
                                .table
                                .page(page)
                                .and_then(|page| page.model)
                                .ok_or_else(|| invalid("RAM page model absent"))?;
                            drop(operation);
                            return self.warm_ram(model);
                        }
                        self.inner
                            .lock()
                            .filling
                            .insert(page, thread::current().id());
                        let filling = Filling { owner: self, page };
                        drop(operation);
                        let result = self.fill(request, page, from, speculative);
                        drop(filling);
                        result?;
                        continue 'decide;
                    }
                    Action::Unmap(page) => {
                        if !self.gpu.unmap(page)? {
                            drop(operation);
                            if self.gpu.wait_pending()? {
                                continue 'decide;
                            }
                            return Err(invalid("unmap stayed pinned without a progress event"));
                        }
                        self.event(request, page, "evict", reason)?;
                    }
                    Action::ReleaseKept(page) => {
                        let victim = {
                            let inner = self.inner.lock();
                            inner.orphaned.get(&page).copied().zip(
                                inner.table.page(page).map(|facts| {
                                    (inner.table.class(facts), facts.bytes, facts.model)
                                }),
                            )
                        };
                        if let Some((key, facts)) = victim {
                            let released = self.channel.release_orphan(self.row.owner(), key)?;
                            // Another owner can take the kept handle before our
                            // release. Re-read custody in either case, and report
                            // an eviction only for our acknowledged release.
                            self.refresh()?;
                            if released {
                                Self::event_facts_locked(
                                    &mut self.inner.lock(),
                                    request,
                                    page,
                                    "evict",
                                    "orphan",
                                    facts,
                                )?;
                            }
                            // Later actions must use the actual released bytes,
                            // not the plan's old orphan snapshot.
                            continue 'decide;
                        }
                    }
                    Action::Trim(page) => {
                        let (trim, bytes) = {
                            let mut inner = self.inner.lock();
                            let trim = inner.trim.clone().ok_or_else(|| {
                                invalid("allocator trim callback was not registered")
                            })?;
                            let bytes = inner.table.page(page).map_or(0, |row| row.bytes);
                            inner.slack_trim.begin();
                            let pins = inner.slack_trim.pins();
                            if let Some(row) =
                                inner.table.pages.iter_mut().find(|row| row.id == page)
                            {
                                row.pins = pins;
                            }
                            (trim, bytes)
                        };
                        let mut callback = Callback {
                            owner: self,
                            page,
                            keep_pin: false,
                            trim: true,
                        };
                        drop(operation);
                        let freed = trim(bytes)?;
                        {
                            let mut inner = self.inner.lock();
                            // An allocator publication is authoritative, including
                            // a partial trim already reflected in reserved bytes.
                            // Only callbacks without a publication use delta accounting.
                            if !inner.slack_trim.published {
                                if let Some(row) =
                                    inner.table.pages.iter_mut().find(|row| row.id == page)
                                {
                                    row.bytes = row.bytes.saturating_sub(freed);
                                }
                            }
                        }
                        callback.keep_pin = freed == 0;
                        drop(callback);
                        continue 'decide;
                    }
                    Action::Shrink { page, bytes } => {
                        let shrink = {
                            let mut inner = self.inner.lock();
                            let cache = inner
                                .caches
                                .get_mut(&page)
                                .ok_or_else(|| invalid("cache callback absent"))?;
                            cache.busy = true;
                            cache.resized = false;
                            let shrink = cache.shrink.clone();
                            let pins = cache.active.saturating_add(1);
                            if let Some(row) =
                                inner.table.pages.iter_mut().find(|row| row.id == page)
                            {
                                row.pins = pins;
                            }
                            shrink
                        };
                        let callback = Callback {
                            owner: self,
                            page,
                            keep_pin: false,
                            trim: false,
                        };
                        drop(operation);
                        let freed = shrink(bytes)?;
                        {
                            let mut inner = self.inner.lock();
                            let state = inner.caches.get_mut(&page).map(|cache| {
                                cache.blocked = freed == 0;
                                (
                                    cache
                                        .active
                                        .saturating_add(u32::from(cache.blocked))
                                        .saturating_add(u32::from(cache.busy)),
                                    cache.resized,
                                )
                            });
                            if let Some(page) =
                                inner.table.pages.iter_mut().find(|row| row.id == page)
                            {
                                let (pins, resized) = state.unwrap_or((0, false));
                                if !resized {
                                    page.bytes = page.bytes.saturating_sub(freed);
                                }
                                page.pins = pins;
                            }
                        }
                        drop(callback);
                        continue 'decide;
                    }
                    Action::Starve(step) => {
                        if let Some(row) = self
                            .inner
                            .lock()
                            .table
                            .requests
                            .iter_mut()
                            .find(|row| row.id == step.request)
                        {
                            row.starved = true;
                        }
                    }
                    Action::Ask { row, need, ceiling } => {
                        let owner = self
                            .ledger
                            .status()?
                            .rows
                            .into_iter()
                            .find(|item| item.owner.row == row)
                            .ok_or_else(|| invalid("asked row disappeared"))?
                            .owner;
                        let ask = self.ledger.ask(owner, need, ceiling)?;
                        wait = Some((owner, ask));
                        break;
                    }
                    Action::Cut(id)
                    | Action::Lower { request: id, .. }
                    | Action::Exhausted { request: id, .. } => {
                        drop(operation);
                        if self.gpu.wait_pending()? {
                            continue 'decide;
                        }
                        return Ok(match action {
                            Action::Cut(_) => Relief::Cut(self.need(id)),
                            Action::Lower { need, .. } => Relief::Lower(need),
                            Action::Exhausted { need, .. } => Relief::Exhausted(need),
                            _ => Relief::Exhausted(0),
                        });
                    }
                    Action::Acknowledge { .. } => {}
                    Action::Forget(_) => {}
                    Action::Invalid(message) => return Err(invalid(message)),
                }
            }
            drop(operation);
            if let Some((owner, ask)) = wait {
                self.ledger.wait_answer(owner, ask, &self.stop)?;
                let mut inner = self.inner.lock();
                if let Some(row) = inner.table.request(request) {
                    let step = row.step();
                    let round = row.reclaim_round;
                    inner.table.reclaims.push(plan::Reclaim {
                        step,
                        round,
                        row: owner.row,
                        ceiling: ask.ceiling,
                        complete: true,
                    });
                }
                continue;
            }
            self.refresh()?;
            return Ok(Relief::Complete);
        }
    }

    fn need(&self, request: RequestId) -> u64 {
        self.inner
            .lock()
            .table
            .request(request)
            .and_then(plan::Request::measured)
            .unwrap_or(0)
    }

    fn event(
        &self,
        request: RequestId,
        page: PageId,
        kind: &'static str,
        reason: &'static str,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        Self::event_locked(&mut inner, request, page, kind, reason)
    }

    fn event_locked(
        inner: &mut Inner,
        request: RequestId,
        page: PageId,
        kind: &'static str,
        reason: &'static str,
    ) -> Result<()> {
        let facts = inner
            .table
            .page(page)
            .map_or((Class::Free, 0, None), |facts| {
                (inner.table.class(facts), facts.bytes, facts.model)
            });
        Self::event_facts_locked(inner, request, page, kind, reason, facts)
    }

    /// An acknowledged removal retains its victim identity, while every total
    /// describes the refreshed table in which that victim is already absent.
    fn event_facts_locked(
        inner: &mut Inner,
        request: RequestId,
        page: PageId,
        kind: &'static str,
        reason: &'static str,
        (class, bytes, model): (Class, u64, Option<ModelId>),
    ) -> Result<()> {
        let step = inner
            .table
            .request(request)
            .map_or(0, |row| row.position.step);
        let event = Event {
            sequence: inner.progress,
            request,
            step,
            pass_index: inner.requests.get(&request).map_or(0, |run| run.pass_index),
            page,
            model,
            class,
            kind,
            reason,
            bytes,
            capacity: inner.table.capacity,
            working_set: inner.table.working_set(request),
            resident: inner
                .table
                .pages
                .iter()
                .filter(|page| page.mapped())
                .map(|page| page.bytes)
                .sum(),
            orphan_bytes: inner
                .table
                .pages
                .iter()
                .filter(|page| page.mapped() && inner.table.class(page) == Class::Orphan)
                .map(|page| page.bytes)
                .sum(),
            speculative_budget: inner
                .requests
                .get(&request)
                .and_then(|run| run.budgets.get(&step))
                .copied(),
            fill: None,
        };
        if let Some(run) = inner.requests.get_mut(&request) {
            if kind == "bind" && !run.stats.proof.knows(page) {
                run.stats.proof.invalidate("unknown-demand-content");
            }
            run.stats.events.push(event);
        }
        inner.progress = inner.progress.saturating_add(1);
        Ok(())
    }

    fn fill_result_locked(
        inner: &mut Inner,
        request: RequestId,
        sequence: u64,
        phase: FillPhase,
        error: Option<&Error>,
        io: &[tensorfs::io::metrics::Copy],
    ) {
        let Some(run) = inner.requests.get_mut(&request) else {
            return;
        };
        // An empty metric alone cannot disprove partial I/O. Only the mechanism's
        // allocation phase proves this attempt never entered the source engine.
        let refused =
            phase == FillPhase::Allocate && error.is_some_and(Error::is_oom) && io.is_empty();
        if error.is_some() && !refused {
            run.stats.proof.invalidate("fill-failed");
        }
        let Some(event) = run
            .stats
            .events
            .iter_mut()
            .find(|event| event.sequence == sequence)
        else {
            run.stats.proof.invalidate("missing-fill-attempt");
            return;
        };
        if refused {
            event.kind = "fill-refused";
        }
        event.fill = Some(FillResult {
            phase: phase.name(),
            outcome: if refused {
                "refused"
            } else if error.is_some() {
                "failed"
            } else {
                "complete"
            },
            error: error.map(|error| match error {
                Error::Cuda { code, .. } => format!("CUDA:{code}"),
                _ => error.code().to_owned(),
            }),
            io_runs: io.len(),
        });
    }

    fn fill(
        &self,
        request: RequestId,
        page: PageId,
        _from: Tier,
        speculative: Option<Step>,
    ) -> Result<()> {
        let sequence = {
            let mut inner = self.inner.lock();
            let sequence = inner.progress;
            Self::event_locked(
                &mut inner,
                request,
                page,
                "fill",
                if speculative.is_some() {
                    "speculation"
                } else {
                    "demand"
                },
            )?;
            sequence
        };
        if let Some(step) = speculative {
            if let Some(row) = self
                .inner
                .lock()
                .table
                .pages
                .iter_mut()
                .find(|row| row.id == page)
            {
                row.attempted.insert(step);
            }
        }
        let (model, transform, key, ram_key, streamed) = {
            let inner = self.inner.lock();
            let model = inner
                .table
                .page(page)
                .and_then(|row| row.model)
                .ok_or_else(|| invalid("page model absent"))?;
            let data = inner
                .models
                .get(&model)
                .ok_or_else(|| invalid("model absent"))?;
            let index = data
                .pages
                .iter()
                .position(|id| *id == page)
                .ok_or_else(|| invalid("page not in model"))?;
            (
                model,
                data.transform.clone(),
                PageKey {
                    fingerprint: data.fingerprint,
                    page: index as u64,
                },
                data.ram_key,
                inner
                    .table
                    .page(page)
                    .is_some_and(|row| row.lowered.contains(&request)),
            )
        };
        let apply = |pin| transform.as_ref().map_or(Ok(()), |fill| fill(page, pin));
        let ram = self.channel.ram(self.row.owner(), ram_key, key.page)?
            .and_then(|view| match Ram::map(view) {
                Ok(mapped) => Some(mapped),
                Err(error) => {
                    tracing::warn!(%error, "RAM view unavailable; reading the verified disk source");
                    None
                }
            });
        let actual = if ram.is_some() { Tier::Ram } else { Tier::Disk };
        let (FillAttempt { result, phase }, io) = tensorfs::io::metrics::capture_direct(|| {
            self.gpu.fill_attempt_with(page, ram.map(Arc::new), apply)
        });
        // Record before propagating the error: a transform may fail after real copies
        // completed, and a retry must not erase their traffic.
        let mut stats = result.as_ref().copied().unwrap_or_default();
        stats.bytes = io
            .iter()
            .filter(|copy| copy.kind == "h2d")
            .map(|copy| copy.bytes)
            .sum();
        let mut inner = self.inner.lock();
        Self::fill_result_locked(
            &mut inner,
            request,
            sequence,
            phase,
            result.as_ref().err(),
            &io,
        );
        if let Some(run) = inner.requests.get_mut(&request) {
            if result
                .as_ref()
                .is_ok_and(|reported| stats.bytes < reported.bytes)
            {
                run.stats.proof.invalidate("missing-fill-copy-records");
            }
            run.stats.copies.push(Copy {
                request,
                model,
                page,
                tier: actual,
                speculative: speculative.is_some(),
                streamed,
                stats,
                io,
            });
        }
        drop(inner);
        result?;
        let (spans, container) = self.gpu.export(page)?;
        self.channel
            .keep(self.row.owner(), key, &spans, container.as_fd())?;
        let mut inner = self.inner.lock();
        if let Some(row) = inner.table.pages.iter_mut().find(|row| row.id == page) {
            row.speculative_for = speculative;
            if let Some(step) = speculative {
                row.attempted.insert(step);
            }
        }
        inner.progress = inner.progress.saturating_add(1);
        Ok(())
    }

    fn capture_budget(&self, request: RequestId) {
        let mut inner = self.inner.lock();
        let current = inner.table.request(request).map(|row| row.position.step);
        let budget = inner.table.speculative_budget(request);
        if let (Some(step), Some(budget)) = (current, budget) {
            if let Some(run) = inner.requests.get_mut(&request) {
                run.budgets.entry(step).or_insert(budget);
            }
        }
    }
}

struct Beginning<'a> {
    owner: &'a Placement,
    request: RequestId,
    committed: bool,
}
impl Drop for Beginning<'_> {
    fn drop(&mut self) {
        if !self.committed {
            Placement::remove_request(&mut self.owner.inner.lock(), self.request);
            let _ = self.owner.row.wake();
        }
    }
}

struct Registration<'a> {
    owner: &'a Placement,
    model: ModelId,
    pages: Vec<PageId>,
    committed: bool,
}

struct Filling<'a> {
    owner: &'a Placement,
    page: PageId,
}

struct Callback<'a> {
    owner: &'a Placement,
    page: PageId,
    keep_pin: bool,
    trim: bool,
}
impl Drop for Callback<'_> {
    fn drop(&mut self) {
        let mut inner = self.owner.inner.lock();
        let pins = if self.trim {
            // Measurements may have changed while the callback ran outside the
            // gate. Its old zero result must not suppress those newer facts.
            inner.slack_trim.finish(self.keep_pin);
            inner.slack_trim.pins()
        } else if let Some(cache) = inner.caches.get_mut(&self.page) {
            cache.busy = false;
            cache.active.saturating_add(u32::from(cache.blocked))
        } else {
            u32::from(self.keep_pin)
        };
        if let Some(page) = inner
            .table
            .pages
            .iter_mut()
            .find(|page| page.id == self.page)
        {
            page.pins = pins;
        }
        drop(inner);
        self.owner.changed.notify_all();
        let _ = self.owner.row.wake();
    }
}
impl Drop for Filling<'_> {
    fn drop(&mut self) {
        self.owner.inner.lock().filling.remove(&self.page);
        self.owner.changed.notify_all();
        let _ = self.owner.row.wake();
    }
}

impl Drop for Registration<'_> {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let mut gone = BTreeSet::new();
        let mut retained = BTreeSet::new();
        for page in &self.pages {
            match self.owner.gpu.unregister(*page) {
                Ok(true) => {
                    gone.insert(*page);
                }
                Ok(false) | Err(_) => {
                    retained.insert(*page);
                }
            }
        }
        let mut inner = self.owner.inner.lock();
        if let Some(model) = inner.models.remove(&self.model) {
            model.ram.cancelled.store(true, Ordering::Release);
        }
        inner.table.pages.retain(|page| !gone.contains(&page.id));
        inner.retired.extend(retained);
        let _ = self.owner.row.wake();
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
        for model in self.inner.get_mut().models.values() {
            model.ram.cancelled.store(true, Ordering::Release);
            let _ = self
                .channel
                .forget_ram(self.row.owner(), model.ram_key, model.ram.token);
        }
        self.stop.store(true, Ordering::Release);
        let _ = self.row.wake();
        if let Some(worker) = self.worker.get_mut().take() {
            if worker.thread().id() != std::thread::current().id() {
                let _ = worker.join();
            }
        }
    }
}

struct Ram {
    address: usize,
    length: usize,
    _view: ledger::RamView,
}
impl Ram {
    fn map(view: ledger::RamView) -> Result<Self> {
        let length = usize::try_from(view.len).map_err(|_| invalid("RAM view length overflows"))?;
        let offset =
            libc::off_t::try_from(view.offset).map_err(|_| invalid("RAM view offset overflows"))?;
        let address = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                length,
                libc::PROT_READ,
                libc::MAP_SHARED,
                view.file.as_raw_fd(),
                offset,
            )
        };
        if address == libc::MAP_FAILED {
            return Err(Error::Io(std::io::Error::last_os_error().to_string()));
        }
        Ok(Self {
            address: address as usize,
            length,
            _view: view,
        })
    }
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.address as *const u8, self.length) }
    }
}
impl Drop for Ram {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.address as *mut _, self.length);
        }
    }
}
impl AsRef<[u8]> for Ram {
    fn as_ref(&self) -> &[u8] {
        self.bytes()
    }
}

fn invalid(message: &str) -> Error {
    Error::Invalid(message.to_owned())
}

fn spread_rank(index: usize, count: usize) -> u32 {
    let bits = (usize::BITS - count.saturating_sub(1).leading_zeros()).clamp(1, 32);
    (index as u32).reverse_bits() >> (32 - bits)
}

#[cfg(test)]
mod gpu_tests {
    use super::*;
    use std::ffi::c_void;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};
    use tensorfs::io::cuda;
    use tensorfs::io::source::{Item, ItemSource};

    #[test]
    fn zero_trim_suppression_ends_only_at_a_new_opportunity() {
        let mut trim = SlackTrim::default();
        trim.begin();
        trim.finish(true);
        assert_eq!(trim.pins(), 1, "unchanged zero-progress facts stay blocked");
        trim.rearm();
        assert_eq!(trim.pins(), 0);
        trim.begin();
        trim.finish(true);
        assert_eq!(trim.pins(), 1, "one rearm must not enable a retry loop");
        trim.rearm();
        trim.begin();
        trim.finish(false);
        assert_eq!(trim.pins(), 0);
    }

    fn event_test_inner() -> Inner {
        Inner {
            table: plan::Table {
                capacity: 1024,
                rows: vec![plan::Row {
                    id: 1,
                    alive: true,
                    reaping: false,
                }],
                pages: vec![plan::Page {
                    id: 8,
                    row: 1,
                    model: None,
                    bytes: 128,
                    tier: Tier::Gpu,
                    kind: Kind::Weight,
                    pins: 0,
                    filling: false,
                    loaded: false,
                    last_used: 0,
                    spread_rank: 0,
                    uses: vec![],
                    speculative_for: None,
                    attempted: BTreeSet::new(),
                    ram_ready: false,
                    lowered: BTreeSet::new(),
                }],
                ..Default::default()
            },
            models: BTreeMap::new(),
            requests: BTreeMap::from([(
                9,
                Request {
                    steps: vec![],
                    explicit: true,
                    package: "orphan-events".into(),
                    shape: "shape".into(),
                    stats: Stats::default(),
                    budgets: BTreeMap::new(),
                    pass_index: 0,
                },
            )]),
            caches: BTreeMap::new(),
            orphaned: BTreeMap::new(),
            retired: BTreeSet::new(),
            filling: BTreeMap::new(),
            next_page: 1,
            progress: 40,
            trim: None,
            slack_trim: SlackTrim::default(),
        }
    }

    #[test]
    fn released_orphan_events_keep_victim_bytes_and_use_current_totals() {
        let mut inner = event_test_inner();
        // Custody's acknowledged release and refresh removed the last orphan.
        // The event still names its original bytes, not an absent-page zero.
        Placement::event_facts_locked(
            &mut inner,
            9,
            7,
            "evict",
            "orphan",
            (Class::Orphan, 512, None),
        )
        .unwrap();
        Placement::event_locked(&mut inner, 9, 8, "evict", "speculation").unwrap();
        let events = &inner.requests[&9].stats.events;
        assert_eq!(
            (events[0].page, events[0].bytes, events[0].class),
            (7, 512, Class::Orphan)
        );
        assert_eq!(
            events[0].resident, 128,
            "live mapped weights must remain counted"
        );
        assert_eq!(events[0].orphan_bytes, 0);
        assert_eq!(
            events[1].orphan_bytes, 0,
            "later actions resurrected a released orphan"
        );
        assert_eq!((events[0].sequence, events[1].sequence), (40, 41));
    }

    #[test]
    fn only_proven_pre_io_oom_is_neutral_and_outcomes_follow_exact_attempts() {
        let oom = Error::Cuda {
            op: "allocation",
            code: 2,
            name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
        };
        let mut inner = event_test_inner();
        inner.requests.get_mut(&9).unwrap().stats.proof.complete = true;
        Placement::event_locked(&mut inner, 9, 8, "fill", "demand").unwrap();
        Placement::event_locked(&mut inner, 9, 8, "fill", "demand").unwrap();
        Placement::fill_result_locked(&mut inner, 9, 40, FillPhase::Allocate, Some(&oom), &[]);
        let stats = &inner.requests[&9].stats;
        assert!(stats.proof.complete);
        assert_eq!(stats.events[0].kind, "fill-refused");
        assert_eq!(
            stats.events[0].fill.as_ref().unwrap().error.as_deref(),
            Some("CUDA:2")
        );
        assert!(
            stats.events[1].fill.is_none(),
            "another same-page attempt was overwritten"
        );
        Placement::fill_result_locked(&mut inner, 9, 41, FillPhase::Transform, Some(&oom), &[]);
        assert_eq!(inner.requests[&9].stats.proof.invalid, ["fill-failed"]);
        assert_eq!(inner.requests[&9].stats.events[1].kind, "fill");
        for (phase, error, io) in [
            (FillPhase::Read, oom.clone(), vec![]),
            (
                FillPhase::Allocate,
                oom.clone(),
                vec![tensorfs::io::metrics::Copy::default()],
            ),
            (FillPhase::Allocate, invalid("bad input"), vec![]),
        ] {
            let mut inner = event_test_inner();
            inner.requests.get_mut(&9).unwrap().stats.proof.complete = true;
            Placement::event_locked(&mut inner, 9, 8, "fill", "demand").unwrap();
            Placement::fill_result_locked(&mut inner, 9, 40, phase, Some(&error), &io);
            assert!(!inner.requests[&9].stats.proof.complete);
            assert_eq!(inner.requests[&9].stats.events[0].kind, "fill");
        }
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn allocation_oom_records_zero_io_before_the_source_is_entered() {
        use std::sync::atomic::AtomicUsize;
        let gpu = Gpu::open(0).unwrap();
        let uuid = assigned_gpu(&gpu);
        let span = crate::layout::align_up(
            gpu.capacity().unwrap() + crate::layout::REGION_ALIGN,
            crate::layout::REGION_ALIGN,
        );
        let source = Arc::new(Source::files(vec![], false, 0));
        // This poison source fails if entered. First make a real VMM allocation
        // larger than CUDA's physical capacity: no tensor payload is allocated.
        let items = vec![Item {
            offset: 0,
            len: 1,
            part_end: true,
            source: ItemSource::Object(tensorfs::io::ObjectRange {
                obj: tensorfs::format::ObjectRef {
                    sha256: "a".repeat(64),
                    length: 1,
                },
                off: 0,
                len: 1,
            }),
        }];
        gpu.register(
            1,
            gpu.reserve(span).unwrap(),
            0,
            span,
            source.clone(),
            items.clone(),
            0,
            "oom-before-source".into(),
        )
        .unwrap();
        let called = AtomicUsize::new(0);
        let (attempt, io) = tensorfs::io::metrics::capture_direct(|| {
            gpu.fill_attempt_with::<Vec<u8>>(1, None, |_| {
                called.fetch_add(1, Ordering::Relaxed);
                Ok(())
            })
        });
        assert!(attempt.result.unwrap_err().is_oom());
        assert_eq!(attempt.phase, FillPhase::Allocate);
        assert!(io.is_empty());
        assert_eq!(called.load(Ordering::Relaxed), 0);
        assert_eq!(gpu.facts().unwrap()[0].1, 0);
        assert!(gpu.unregister(1).unwrap());

        let small = crate::layout::REGION_ALIGN;
        gpu.register(
            2,
            gpu.reserve(small).unwrap(),
            0,
            small,
            source.clone(),
            items,
            0,
            "source-guard".into(),
        )
        .unwrap();
        let attempt = gpu.fill_attempt_with::<Vec<u8>>(2, None, |_| {
            called.fetch_add(1, Ordering::Relaxed);
            Ok(())
        });
        assert_eq!(attempt.phase, FillPhase::Read);
        assert!(
            matches!(attempt.result, Err(Error::Source(_))),
            "the same poison source must refuse once entered"
        );
        assert_eq!(called.load(Ordering::Relaxed), 0);
        gpu.recover().unwrap();
        assert!(gpu.unregister(2).unwrap());

        gpu.register(
            3,
            gpu.reserve(small).unwrap(),
            0,
            small,
            source,
            vec![Item {
                offset: 0,
                len: 1,
                part_end: true,
                source: ItemSource::Inline(vec![7].into()),
            }],
            0,
            "after-oom".into(),
        )
        .unwrap();
        let attempt = gpu.fill_attempt_with::<Vec<u8>>(3, None, |_| Ok(()));
        assert_eq!(attempt.phase, FillPhase::Finish);
        attempt.result.unwrap();
        let pin = gpu.bind(3, 0).unwrap();
        let mut bytes = [0_u8; 1];
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .unwrap();
        assert_eq!(bytes, [7]);
        drop(pin);
        gpu.wait_pending().unwrap();
        assert!(gpu.unregister(3).unwrap());
        println!("PASS allocation OOM records zero I/O before source entry {uuid}");
    }

    #[test]
    fn rearmed_facts_keep_the_callback_pin_and_outlive_its_zero_result() {
        let mut trim = SlackTrim::default();
        trim.begin();
        trim.publish(false);
        assert!(trim.published);
        trim.finish(true);
        assert_eq!(trim.pins(), 1, "unchanged publication is not progress");
        trim.rearm();
        trim.begin();
        trim.publish(true);
        trim.rearm();
        assert_eq!(trim.pins(), 1, "a running callback still owns its pin");
        trim.finish(true);
        assert_eq!(
            trim.pins(),
            0,
            "the old zero result cannot block newer facts"
        );
        trim.begin();
        trim.finish(true);
        assert_eq!(trim.pins(), 1, "unchanged facts still stop another retry");
    }

    #[test]
    fn activation_oom_reclaims_foreign_bytes_even_when_the_logical_estimate_fits() {
        let mut table = plan::Table {
            capacity: 25_153_044_480,
            rows: vec![
                plan::Row {
                    id: 1,
                    alive: true,
                    reaping: false,
                },
                plan::Row {
                    id: 2,
                    alive: true,
                    reaping: false,
                },
            ],
            requests: vec![plan::Request {
                id: 1,
                row: 1,
                position: plan::Position { step: 0, order: 0 },
                working_sets: vec![None],
                starved: false,
                sharing_limit: None,
                resident_steps: BTreeSet::new(),
                joint_steps: BTreeSet::new(),
                model_positions: BTreeMap::new(),
                can_cut: true,
                can_lower: true,
                reclaim_round: 0,
            }],
            remote: vec![plan::Remote {
                row: 2,
                classes: [4_313_555_456, 0, 0, 20_021_510_144, 0, 0, 100_949_504],
                context: 331_350_016,
                remaining: vec![],
                reaping: false,
            }],
            ..Default::default()
        };
        let need = 96 << 20; // Actual 32MiB failure plus the caller's existing 64MiB slack.
        let available = table.available();
        assert_eq!(available, 385_679_360);
        assert!(
            plan::decide(&table, Trigger::Oom { request: 1, need }).is_empty(),
            "the old target did not ask this idle owner to release anything"
        );
        let target = activation_oom_target(available, need).unwrap();
        let trigger = Trigger::Oom {
            request: 1,
            need: target,
        };
        assert_eq!(
            plan::decide(&table, trigger),
            vec![Action::Ask {
                row: 2,
                need: 1,
                ceiling: Class::Orphan,
            }]
        );
        table.reclaims.push(plan::Reclaim {
            step: plan::Step {
                request: 1,
                index: 0,
            },
            round: 0,
            row: 2,
            ceiling: Class::Orphan,
            complete: true,
        });
        assert_eq!(
            plan::decide(&table, trigger),
            vec![Action::Ask {
                row: 2,
                need: 1,
                ceiling: Class::Unused,
            }],
            "a no-op acknowledgement is not real freed room"
        );
        table.reclaims[0].ceiling = Class::Declared;
        assert!(
            matches!(plan::decide(&table, trigger).as_slice(), [Action::Cut(1)]),
            "unreclaimable remote bytes must not produce Complete"
        );
        table.remote[0].classes[Class::Free.as_usize()] -= 1;
        assert!(
            plan::decide(&table, trigger).is_empty(),
            "credit only the new published byte"
        );
        table.requests[0].reclaim_round += 1;
        let next = Trigger::Oom {
            request: 1,
            need: activation_oom_target(table.available(), need).unwrap(),
        };
        assert_eq!(
            plan::decide(&table, next),
            vec![Action::Ask {
                row: 2,
                need: 1,
                ceiling: Class::Orphan,
            }],
            "a fresh OOM must not reuse the previous operation's acknowledgement"
        );
        assert_eq!(table.capacity, 25_153_044_480);
        assert_eq!(table.remote[0].context, 331_350_016);
        assert_eq!(
            table.remote[0].classes[Class::Pinned.as_usize()],
            100_949_504
        );
    }

    #[test]
    fn activation_oom_target_preserves_real_shortfalls_and_checks_overflow() {
        assert_eq!(activation_oom_target(8, 9).unwrap(), 9);
        assert_eq!(activation_oom_target(8, 4).unwrap(), 9);
        assert!(matches!(
            activation_oom_target(u64::MAX, 1),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn bind_retry_requires_increasing_room_after_checked_recovery() {
        let mut retry = BindRetry::default();
        assert_eq!(retry.after_oom(100, 20).unwrap(), 20);
        assert!(retry.advance(100));
        assert_eq!(retry.after_oom(100, 20).unwrap(), 120);
        assert!(!retry.advance(100));
        assert!(!retry.advance(90));
        assert_eq!(retry.after_oom(90, 20).unwrap(), 110);
        assert!(!retry.advance(100), "oscillation is not new room");
        assert_eq!(retry.after_oom(120, 20).unwrap(), 20);
        assert!(retry.advance(120));
        assert_eq!(retry.after_oom(120, 20).unwrap(), 140);
    }

    #[test]
    fn bind_retry_does_not_wrap_additional_room() {
        let mut retry = BindRetry::default();
        assert!(retry.advance(u64::MAX));
        assert!(matches!(
            retry.after_oom(u64::MAX, 1),
            Err(Error::Invalid(_))
        ));
    }

    #[test]
    fn content_catalog_counts_payload_separately_from_padding_and_mapping() {
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: 4099,
                span: crate::layout::REGION_ALIGN,
                items: vec![
                    Item {
                        offset: 0,
                        len: 3,
                        source: ItemSource::Inline(vec![1, 2, 3].into()),
                        part_end: true,
                    },
                    Item {
                        offset: 4096,
                        len: 3,
                        source: ItemSource::Inline(vec![4, 5, 6].into()),
                        part_end: true,
                    },
                ],
            }],
            nbytes: crate::layout::REGION_ALIGN,
            digest: "test-layout".into(),
            names: vec![],
        };
        let first = ProofPage::catalog(&[7], &[1; 32], &layout);
        let other_transform = ProofPage::catalog(&[7], &[2; 32], &layout);
        assert_eq!(first[0].payload, 6);
        assert_eq!(first[0].mapped, crate::layout::REGION_ALIGN);
        assert_ne!(first[0].content, other_transform[0].content);
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn failed_begin_rolls_back_and_reuses_the_request_identity() {
        let placement = Placement::open(0, None).expect("real CUDA placement");
        let uuid = assigned_gpu(&placement.gpu);
        assert!(placement
            .begin(7, vec![vec![999]], "test".into(), "shape".into())
            .is_err());
        assert!(
            placement.stats(7).is_err(),
            "failed begin must not retain request state"
        );
        placement
            .begin(7, vec![], "test".into(), "shape".into())
            .expect("same id starts again");
        let stats = placement.end(7).expect("request ends");
        assert_eq!(stats.events.first().map(|event| event.kind), Some("begin"));
        assert_eq!(stats.events.last().map(|event| event.kind), Some("end"));
        assert_eq!(stats.proof.version, 1);
        assert!(stats.proof.complete);
        placement.close().expect("close placement");
        println!("PASS failed begin rolls back and reuses its request {uuid}");
    }

    fn assigned_gpu(gpu: &Gpu) -> String {
        let driver = cuda::driver().expect("driver");
        let mut count = 0;
        cuda::check("cuDeviceGetCount", unsafe {
            (driver.device_get_count)(&mut count)
        })
        .expect("visible devices");
        assert_eq!(count, 1, "this proof requires exactly one assigned GPU");
        gpu.uuid().expect("GPU UUID")
    }

    // An old recursive pending wait must fail one bounded test, not strand the
    // native proof runner. All CUDA owners live on the worker if it wedges.
    fn fill_worker(run: impl FnOnce() -> String + Send + 'static) -> String {
        let (sent, done) = mpsc::channel();
        let worker = thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
            let _ = sent.send(result);
        });
        let result = done
            .recv_timeout(Duration::from_secs(10))
            .expect("fill callback waited for its own unfinished fill");
        worker.join().unwrap();
        result.unwrap_or_else(|error| std::panic::resume_unwind(error))
    }

    fn fill_layout(regions: u64, value: u8) -> Layout {
        let span = 4 << 20;
        let bytes: Arc<[u8]> = vec![value; span as usize].into();
        Layout {
            parts: vec![],
            regions: (0..regions)
                .map(|index| crate::layout::Region {
                    offset: index * span,
                    nbytes: span,
                    span,
                    items: vec![Item {
                        offset: index * span,
                        len: span,
                        source: ItemSource::Inline(bytes.clone()),
                        part_end: true,
                    }],
                })
                .collect(),
            nbytes: regions * span,
            digest: format!("fill-dependency-{regions}-{value}"),
            names: vec![],
        }
    }

    struct FillLedger {
        ledger: Ledger,
        name: String,
    }
    impl FillLedger {
        fn new() -> Self {
            let name = format!(
                "fill-dependency-{}-{}",
                std::process::id(),
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            );
            Self {
                ledger: Ledger::create(&name, 24 << 20).unwrap(),
                name,
            }
        }
        fn placement(&self) -> Arc<Placement> {
            let grant = self.ledger.grant_unbound().unwrap();
            Placement::open(0, Some((grant.table, grant.events, grant.channel))).unwrap()
        }
    }
    impl Drop for FillLedger {
        fn drop(&mut self) {
            let uid = unsafe { libc::geteuid() };
            let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.name));
        }
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn a_fill_can_bind_another_model_under_pressure_without_waiting_for_itself() {
        let uuid = fill_worker(|| {
            let fixture = FillLedger::new();
            let placement = fixture.placement();
            let uuid = assigned_gpu(&placement.gpu);
            let source = Arc::new(Source::files(vec![], false, 0));
            let factors = placement
                .register(
                    2,
                    "factors".into(),
                    [91; 32],
                    &fill_layout(1, 11),
                    source.clone(),
                    None,
                )
                .unwrap();
            let weak = Arc::downgrade(&placement);
            let active = Arc::new(AtomicBool::new(false));
            let enabled = active.clone();
            let first = Arc::new(AtomicU64::new(0));
            let target = first.clone();
            let transform = Arc::new(move |page, write: FillPin| {
                if !enabled.load(Ordering::Acquire) || page != target.load(Ordering::Acquire) {
                    return Ok(());
                }
                let placement = weak.upgrade().unwrap();
                // The outer page is actually Filling, and every other base page
                // is Working. The nested dependency must reach Lower, not wait
                // for this same callback to finish before returning that rung.
                let factor = placement.bind(9, factors[0], write.stream)?;
                unsafe {
                    tensorfs::io::read(
                        &[tensorfs::io::Copy {
                            src: tensorfs::io::Src::Device {
                                device: 0,
                                ptr: factor.ptr,
                                stream: write.stream,
                            },
                            dst: tensorfs::io::Dst::Device {
                                device: 0,
                                ptr: write.ptr,
                                len: write.bytes as usize,
                                stream: write.stream,
                            },
                        }],
                        &tensorfs::io::ReadOpts::default(),
                    )?;
                }
                Ok(())
            });
            let base = placement
                .register(
                    1,
                    "base".into(),
                    [92; 32],
                    &fill_layout(6, 7),
                    source,
                    Some(transform),
                )
                .unwrap();
            first.store(base[0], Ordering::Release);
            assert_eq!(placement.load(1, Tier::Gpu).unwrap(), Relief::Complete);
            assert_eq!(
                placement
                    .resident()
                    .unwrap()
                    .iter()
                    .map(|(_, n)| n)
                    .sum::<u64>(),
                24 << 20
            );
            assert!(placement.gpu.unmap(base[0]).unwrap());
            active.store(true, Ordering::Release);
            placement
                .begin(
                    9,
                    vec![vec![1, 2]],
                    "fill-dependency".into(),
                    "shape".into(),
                )
                .unwrap();
            placement.enter(9, 1).unwrap();
            let pin = placement.bind(9, base[0], 0).unwrap();
            let mut bytes = [0_u8; 32];
            unsafe {
                tensorfs::io::read(
                    &[tensorfs::io::Copy {
                        src: tensorfs::io::Src::Device {
                            device: 0,
                            ptr: pin.ptr,
                            stream: 0,
                        },
                        dst: tensorfs::io::Dst::host(&mut bytes),
                    }],
                    &tensorfs::io::ReadOpts::default(),
                )
            }
            .unwrap();
            assert_eq!(
                bytes, [11; 32],
                "the callback must read the actual other model"
            );
            assert!(placement
                .stats(9)
                .unwrap()
                .events
                .iter()
                .any(|event| event.kind == "lower"));
            active.store(false, Ordering::Release);
            drop(pin);
            placement.gpu.wait_pending().unwrap();
            placement.end(9).unwrap();
            placement.close().unwrap();
            uuid
        });
        println!(
            "PASS a fill binds another model under pressure without waiting for itself {uuid}"
        );
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn self_region_fill_refuses_and_the_same_request_recovers() {
        let uuid = fill_worker(|| {
            // A returned FillPin can already belong to another release poller.
            // Force that interleaving before exercising failed-fill reuse; it
            // must wait for real completion, never report an escaped live view.
            crate::gpu::returned_write_release_interleaving().unwrap();
            let fixture = FillLedger::new();
            let placement = fixture.placement();
            let uuid = assigned_gpu(&placement.gpu);
            let weak = Arc::downgrade(&placement);
            let recurse = Arc::new(AtomicBool::new(true));
            let enabled = recurse.clone();
            let transform = Arc::new(move |page, write: FillPin| {
                if enabled.load(Ordering::Acquire) {
                    let placement = weak.upgrade().unwrap();
                    let direct = placement.gpu.bind(page, write.stream).err().unwrap();
                    assert!(
                        matches!(direct, Error::Invalid(message) if message.contains("own unfinished page"))
                    );
                    let error = placement.bind(9, page, write.stream).err().unwrap();
                    assert!(
                        matches!(&error, Error::Invalid(message) if message.contains("own unfinished page"))
                    );
                    return Err(error);
                }
                Ok(())
            });
            let pages = placement
                .register(
                    1,
                    "self".into(),
                    [93; 32],
                    &fill_layout(1, 7),
                    Arc::new(Source::files(vec![], false, 0)),
                    Some(transform),
                )
                .unwrap();
            placement
                .begin(9, vec![vec![1]], "self-fill".into(), "shape".into())
                .unwrap();
            let error = placement.bind(9, pages[0], 0).err().unwrap();
            assert!(
                matches!(error, Error::Invalid(message) if message.contains("own unfinished page"))
            );
            assert!(placement.inner.lock().filling.is_empty());
            placement.gpu.recover().unwrap();
            recurse.store(false, Ordering::Release);
            let pin = placement
                .bind(9, pages[0], 0)
                .expect("the same request recovers");
            let mut bytes = [0_u8; 32];
            unsafe {
                tensorfs::io::read(
                    &[tensorfs::io::Copy {
                        src: tensorfs::io::Src::Device {
                            device: 0,
                            ptr: pin.ptr,
                            stream: 0,
                        },
                        dst: tensorfs::io::Dst::host(&mut bytes),
                    }],
                    &tensorfs::io::ReadOpts::default(),
                )
            }
            .unwrap();
            assert_eq!(bytes, [7; 32]);
            drop(pin);
            placement.gpu.wait_pending().unwrap();
            placement.end(9).unwrap();
            placement.close().unwrap();
            uuid
        });
        println!("PASS self-region fill refuses and the same request recovers {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn idle_cached_weight_pins_cannot_block_their_own_warm_admission() {
        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        let capacity = 32 << 20;
        let weight = 20 << 20; // actual mapped CUDA bytes, more than half the scoped budget
        assert!(probe.capacity().unwrap() >= capacity);
        drop(probe);
        // A private test ledger avoids changing the machine service's own cap.
        let name = format!(
            "admission-gpu-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        struct Named(String);
        impl Drop for Named {
            fn drop(&mut self) {
                let uid = unsafe { libc::geteuid() };
                let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.0));
            }
        }
        let _named = Named(name.clone());
        let ledger = Ledger::create(&name, capacity).expect("scoped admission ledger");
        let grant = ledger.grant_unbound().expect("rank grant");
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("actual CUDA placement");
        assert_eq!(placement.gpu_uuid().unwrap(), uuid);
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: weight,
                span: weight,
                items: vec![Item {
                    offset: 0,
                    len: weight,
                    part_end: true,
                    source: ItemSource::Inline(vec![7; weight as usize].into()),
                }],
            }],
            nbytes: weight,
            digest: "warm-admission".into(),
            names: vec![],
        };
        let pages = placement
            .register(
                1,
                "warm-admission".into(),
                [43; 32],
                &layout,
                Arc::new(Source::files(vec![], false, 0)),
                None,
            )
            .expect("real model region");
        placement
            .begin(1, vec![vec![1]], "warm-admission".into(), "shape".into())
            .unwrap();
        placement.enter(1, 1).unwrap();
        let pin = placement
            .bind(1, pages[0], 0)
            .expect("actual filled CUDA page");
        let held = Arc::new(Mutex::new(Some(pin)));
        let dropped = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (retained, calls) = (held.clone(), dropped.clone());
        let created = Arc::new(Mutex::new(None));
        let created_calls = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let (created_id, new_calls) = (created.clone(), created_calls.clone());
        let weak = Arc::downgrade(&placement);
        let cache = placement
            .cache(
                0,
                Arc::new(move |_| {
                    calls.fetch_add(1, Ordering::AcqRel);
                    let pin = retained.lock().take();
                    drop(pin);
                    let new_calls = new_calls.clone();
                    let id = weak.upgrade().expect("live placement").cache(
                        0,
                        Arc::new(move |_| {
                            new_calls.fetch_add(1, Ordering::AcqRel);
                            Ok(0)
                        }),
                    )?;
                    *created_id.lock() = Some(id);
                    Ok(0) // zero graph-pool bytes still retain real weight pins
                }),
            )
            .unwrap();
        placement.leave(1, 0).unwrap();
        placement.end(1).unwrap();
        assert_eq!(
            ledger
                .peaks()
                .iter()
                .find(|(key, _)| key == "warm-admission|shape")
                .map(|(_, bytes)| *bytes),
            Some(weight)
        );
        assert_eq!(
            ledger.status().unwrap().rows[0].snapshot.classes[Class::Pinned.as_usize()],
            weight
        );
        assert!(
            !ledger.fits("warm-admission", "shape").unwrap(),
            "the first snapshot is conservatively pinned"
        );
        observe(
            &ledger,
            "idle cache release never admitted the warm request",
            || ledger.fits("warm-admission", "shape").unwrap(),
        );
        assert_eq!(dropped.load(Ordering::Acquire), 1);
        assert_eq!(
            created_calls.load(Ordering::Acquire),
            0,
            "a cache created by a release callback must wait for a later signal"
        );
        assert!(held.lock().is_none());
        assert_eq!(
            placement
                .resident()
                .unwrap()
                .iter()
                .map(|(_, bytes)| *bytes)
                .sum::<u64>(),
            weight,
            "admission must not unmap warm weights"
        );
        placement
            .begin(2, vec![vec![1]], "warm-admission".into(), "shape".into())
            .unwrap();
        let pin = placement.bind(2, pages[0], 0).expect("warm CUDA binding");
        let mut bytes = [0_u8; 32];
        // The live pin owns this CUDA region through the synchronous copy;
        // bytes is a live, disjoint host destination for the full read.
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .expect("real GPU contents");
        assert_eq!(bytes, [7; 32]);
        drop(pin);
        placement.leave(2, 0).unwrap();
        let stats = placement.end(2).unwrap();
        assert!(
            stats.copies.is_empty(),
            "warm admission refilled weight bytes"
        );
        placement.close_cache(cache);
        placement.close_cache(created.lock().take().expect("callback-created cache"));

        // Exercise allocator slack with actual CUDA allocations, not synthetic
        // resident bytes. A live 4MiB owner later joins the idle 12MiB cache.
        let live = tensorfs::io::DeviceBuffer::new(0, 4 << 20).unwrap();
        let idle = Arc::new(Mutex::new(vec![tensorfs::io::DeviceBuffer::new(
            0,
            12 << 20,
        )
        .unwrap()]));
        let attempts = Arc::new(AtomicU64::new(0));
        let (entered, waiting) = mpsc::channel();
        let (release, finish) = mpsc::channel();
        let finish = Arc::new(Mutex::new(finish));
        let release = ReleaseGate(Some(release));
        let (buffers, calls) = (idle.clone(), attempts.clone());
        let weak = Arc::downgrade(&placement);
        placement.trim_with(Arc::new(move |_| {
            match calls.fetch_add(1, Ordering::AcqRel) {
                0 => {
                    entered.send(()).unwrap();
                    finish.lock().recv().unwrap();
                    Ok(0) // This trim observed the old, still-busy allocation state.
                }
                1 => {
                    let (retired, remaining) = {
                        let mut buffers = buffers.lock();
                        let retired = buffers.remove(0);
                        let remaining = buffers.iter().map(|buffer| buffer.len() as u64).sum();
                        (retired, remaining)
                    };
                    let freed = retired.len() as u64;
                    drop(retired);
                    // A real partial free can be measured before the callback
                    // returns. The published remainder must not be subtracted twice.
                    weak.upgrade().unwrap().allocator(0, 0, remaining)?;
                    Ok(freed)
                }
                2 => {
                    let buffers = std::mem::take(&mut *buffers.lock());
                    let freed = buffers.iter().map(|buffer| buffer.len() as u64).sum();
                    drop(buffers); // Actual CUDA frees precede the freed-byte report.
                    Ok(freed)
                }
                _ => Err(invalid(
                    "zero-progress slack trim retried without new facts",
                )),
            }
        }));
        placement.allocator(0, 4 << 20, 16 << 20).unwrap();
        placement
            .begin(3, vec![vec![1]], "warm-admission".into(), "shape".into())
            .unwrap();
        let pin = placement.bind(3, pages[0], 0).unwrap();
        let trimming = placement.clone();
        let worker = std::thread::spawn(move || trimming.relief(3, 1, false));
        waiting.recv_timeout(Duration::from_secs(10)).unwrap();
        placement.allocator(0, 4 << 20, 16 << 20).unwrap();
        // This is the actual owner transition: the former live allocation now
        // belongs to the idle cache. Publish it while the old trim is in flight.
        idle.lock().push(live);
        placement.allocator(0, 0, 16 << 20).unwrap();
        {
            let inner = placement.inner.lock();
            let slack = inner
                .table
                .pages
                .iter()
                .find(|page| page.kind == Kind::Slack)
                .unwrap();
            assert_eq!(slack.bytes, 16 << 20);
            assert_eq!(
                slack.pins, 1,
                "new facts must not release an active trim pin"
            );
            assert!(
                inner.table.page(pages[0]).unwrap().pins > 0,
                "a real CUDA view must remain pinned"
            );
        }
        drop(release);
        assert_eq!(
            worker.join().unwrap().unwrap(),
            Relief::Complete,
            "a stale zero trim suppressed the now-idle CUDA allocations"
        );
        assert_eq!(attempts.load(Ordering::Acquire), 2);
        assert_eq!(idle.lock().len(), 1);
        assert_eq!(
            placement.available().unwrap(),
            capacity - weight - (4 << 20),
            "partial trim bytes were subtracted again after their publication"
        );
        assert_eq!(
            placement.relief(3, 9 << 20, false).unwrap(),
            Relief::Complete
        );
        assert_eq!(attempts.load(Ordering::Acquire), 3);
        assert!(idle.lock().is_empty());
        assert_eq!(
            placement.available().unwrap(),
            capacity - weight,
            "unchanged-facts callback must retain ordinary freed-byte accounting"
        );
        assert_eq!(
            placement.resident().unwrap()[0].1,
            weight,
            "slack reclaim must keep the real warm weight mapping"
        );
        drop(pin);
        placement.gpu.wait_pending().unwrap();
        placement.leave(3, 0).unwrap();
        placement.end(3).unwrap();
        assert!(
            ledger.fits("warm-admission", "shape").unwrap(),
            "allocator slack must not block its owner's next admission"
        );
        placement
            .begin(4, vec![vec![1]], "warm-admission".into(), "shape".into())
            .unwrap();
        let pin = placement.bind(4, pages[0], 0).unwrap();
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .unwrap();
        assert_eq!(bytes, [7; 32]);
        drop(pin);
        placement.leave(4, 0).unwrap();
        assert!(placement.end(4).unwrap().copies.is_empty());
        placement.close().unwrap();
        println!(
            "PASS idle cache pins release before warm admission without weight eviction {uuid}"
        );
    }

    struct ConsumerGate {
        started: mpsc::Sender<()>,
        release: mpsc::Receiver<()>,
    }

    unsafe extern "C" fn hold_consumer(pointer: *mut c_void) {
        // CUDA owns this box until it invokes the callback exactly once. No CUDA
        // calls are allowed here; the channel only keeps its stream unfinished.
        let gate = unsafe { Box::from_raw(pointer.cast::<ConsumerGate>()) };
        let _ = gate.started.send(());
        let _ = gate.release.recv();
    }

    struct ReleaseGate(Option<mpsc::Sender<()>>);
    impl Drop for ReleaseGate {
        fn drop(&mut self) {
            if let Some(release) = self.0.take() {
                let _ = release.send(());
            }
        }
    }

    struct Primary(cuda::CUdevice);
    impl Drop for Primary {
        fn drop(&mut self) {
            if let Ok(driver) = cuda::driver() {
                unsafe { (driver.primary_ctx_release)(self.0) };
            }
        }
    }

    fn observe(ledger: &Ledger, description: &str, mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if ready() {
                return;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(!remaining.is_zero(), "{description}");
            let mut poll = libc::pollfd {
                fd: ledger.events().as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut poll, 1, remaining.as_millis() as i32) };
            if result == -1 && std::io::Error::last_os_error().raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            assert!(result > 0, "{description}");
            let mut count = 0_u64;
            let read = unsafe {
                libc::read(
                    ledger.events().as_raw_fd(),
                    (&mut count as *mut u64).cast(),
                    std::mem::size_of::<u64>(),
                )
            };
            assert_eq!(read, 8, "read ledger event");
        }
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn failed_direct_ram_fill_retains_its_source_until_checked_recovery() {
        let gpu = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&gpu);
        let bytes = crate::layout::REGION_ALIGN;
        let range = gpu.reserve(bytes).expect("page reservation");
        gpu.register(
            1,
            range,
            0,
            bytes,
            Arc::new(Source::files(vec![], false, 0)),
            vec![],
            0,
            "owned-ram-error".into(),
        )
        .expect("page");
        let source = Arc::new(vec![7_u8; bytes as usize]);
        let lifetime = Arc::downgrade(&source);
        let error = gpu.fill_from_ram_owned_with(1, source.clone(), |_| {
            Err(Error::Invalid("injected transform failure".into()))
        });
        assert!(error.is_err());
        drop(source);
        assert!(
            lifetime.upgrade().is_some(),
            "failed direct I/O released its host owner"
        );
        gpu.recover().expect("checked CUDA recovery");
        assert!(
            lifetime.upgrade().is_none(),
            "checked recovery retained a completed source"
        );
        println!("PASS failed RAM fill retains its source through recovery {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn ram_warm_returns_before_export_and_close_prevents_stale_registration() {
        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        let ledger =
            Ledger::create(&uuid, probe.capacity().expect("capacity")).expect("observer ledger");
        let grant = ledger.grant_unbound().expect("rank grant");
        drop(probe);
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("placement");
        let blocker =
            platform::shm::frozen("blocked-ram-export", b"data").expect("verified source");
        blocker.lock().expect("hold the export's shared-lock gate");
        let file = std::fs::File::from(
            platform::shm::reopen(blocker.as_fd(), false).expect("source description"),
        );
        let object = tensorfs::format::ObjectRef::of(b"data");
        let length = crate::layout::REGION_ALIGN;
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: 4,
                span: length,
                items: vec![Item {
                    offset: 0,
                    len: 4,
                    part_end: true,
                    source: ItemSource::Object(tensorfs::io::ObjectRange {
                        obj: object.clone(),
                        off: 0,
                        len: 4,
                    }),
                }],
            }],
            nbytes: length,
            digest: "blocked-ram-export".into(),
            names: vec![],
        };
        let pages = placement
            .register(
                1,
                "blocked-export".into(),
                [28; 32],
                &layout,
                Arc::new(Source::files(vec![(object, file)], false, 0)),
                None,
            )
            .expect("model");
        let state = placement
            .inner
            .lock()
            .models
            .get(&1)
            .expect("registered model")
            .ram
            .clone();
        let (finished, result) = mpsc::channel();
        let warming = placement.clone();
        let caller = std::thread::spawn(move || {
            finished
                .send(warming.load(1, Tier::Ram))
                .expect("report load")
        });
        assert_eq!(
            result
                .recv_timeout(Duration::from_secs(5))
                .expect("RAM load waited for export")
                .expect("load"),
            Relief::Complete
        );
        caller.join().expect("load caller");
        assert!(state.pending.load(Ordering::Acquire));
        placement
            .begin(1, vec![vec![1]], "ram-export-test".into(), "tiny".into())
            .expect("request starts while warming is blocked");
        let pin = placement
            .bind(1, *pages.first().expect("page"), 0)
            .expect("disk demand does not wait for RAM");
        assert_eq!(
            placement
                .stats(1)
                .expect("stats")
                .copies
                .first()
                .expect("actual demand copy")
                .tier,
            Tier::Disk
        );
        drop(pin);
        placement.end(1).expect("request ends");
        placement
            .close_model(1)
            .expect("close while export is blocked");
        blocker.unlock().expect("release exporter");
        observe(&ledger, "cancelled export did not finish", || {
            !state.pending.load(Ordering::Acquire)
        });
        assert!(
            ledger
                .ram_status(tensorfs::sha256::digest(layout.digest.as_bytes()))
                .is_none(),
            "a closed model published a stale cache"
        );
        placement.close().expect("close placement");
        println!(
            "PASS RAM warming stays asynchronous and closes without stale registration {uuid}"
        );
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn actual_readiness_survives_concurrent_fill_and_unmap() {
        let gpu = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&gpu);
        let length = crate::layout::REGION_ALIGN;
        let range = gpu.reserve(length).expect("real reservation");
        gpu.register(
            1,
            range,
            0,
            length,
            Arc::new(Source::files(vec![], false, 0)),
            vec![Item {
                offset: 0,
                len: length,
                source: ItemSource::Inline(vec![7; length as usize].into()),
                part_end: true,
            }],
            0,
            "concurrent-readiness".into(),
        )
        .unwrap();
        assert!(matches!(gpu.try_pin(1, 0).unwrap(), BindState::Empty));
        let (started, waiting) = mpsc::channel();
        let (release, finish) = mpsc::channel();
        let gate = ReleaseGate(Some(release));
        let filling = gpu.clone();
        let worker = std::thread::spawn(move || {
            filling.fill_with(1, move |_| {
                started.send(()).unwrap();
                finish.recv().unwrap();
                Ok(())
            })
        });
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("actual fill entered");
        let pending = match gpu.try_pin(1, 0).unwrap() {
            BindState::Filling(pending) => pending,
            _ => panic!("an actual in-flight fill must remain a typed pending state"),
        };
        assert!(
            !gpu.unmap(1).unwrap(),
            "an in-flight fill must not be unmapped"
        );
        drop(gate);
        worker.join().unwrap().expect("actual GPU fill completes");
        // Completion deliberately precedes the prepared wait: no notification
        // may be lost between observing Filling and beginning the wait.
        assert!(gpu.wait_prepared(pending).unwrap());
        let pin = match gpu.try_pin(1, 0).unwrap() {
            BindState::Ready { pin, pins: 1 } => pin,
            _ => panic!("the completed mapping must be pinned atomically"),
        };
        pin.wait_on(0).unwrap();
        let reclaiming = gpu.clone();
        assert!(
            !std::thread::spawn(move || reclaiming.unmap(1))
                .join()
                .unwrap()
                .unwrap(),
            "a concurrent reclaimer must preserve the actual pin"
        );
        let mut bytes = [0_u8; 32];
        // The live pin owns this source through the synchronous device read.
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .expect("actual pinned GPU contents");
        assert_eq!(bytes, [7; 32]);
        drop(pin);
        gpu.wait_pending().unwrap();
        assert!(
            gpu.unmap(1).unwrap(),
            "the completed last pin must release the mapping"
        );
        assert!(matches!(gpu.try_pin(1, 0).unwrap(), BindState::Empty));
        gpu.unregister(1).unwrap();
        println!("PASS actual readiness survives concurrent fill and unmap {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn repeated_component_forward_keeps_its_declared_step() {
        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        let capacity = 32 << 20;
        assert!(probe.capacity().unwrap() >= capacity);
        drop(probe);
        let name = format!(
            "declared-step-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        struct Named(String);
        impl Drop for Named {
            fn drop(&mut self) {
                let uid = unsafe { libc::geteuid() };
                let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.0));
            }
        }
        let _named = Named(name.clone());
        let ledger = Ledger::create(&name, capacity).unwrap();
        let grant = ledger.grant_unbound().unwrap();
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("actual CUDA placement");
        let mut pages = BTreeMap::new();
        for (model, label) in [(1, "A"), (2, "VAE"), (3, "audio"), (4, "DiT"), (5, "LoRA")] {
            let count = if model == 2 { 2 } else { 1 };
            let layout = Layout {
                parts: vec![],
                regions: (0..count)
                    .map(|index| crate::layout::Region {
                        offset: index * crate::layout::REGION_ALIGN,
                        nbytes: 32,
                        span: crate::layout::REGION_ALIGN,
                        items: vec![Item {
                            offset: index * crate::layout::REGION_ALIGN,
                            len: 32,
                            source: ItemSource::Inline(vec![7; 32].into()),
                            part_end: true,
                        }],
                    })
                    .collect(),
                nbytes: count * crate::layout::REGION_ALIGN,
                digest: format!("declared-step-{model}"),
                names: vec![],
            };
            pages.insert(
                model,
                placement
                    .register(
                        model,
                        label.into(),
                        [model as u8; 32],
                        &layout,
                        Arc::new(Source::files(vec![], false, 0)),
                        None,
                    )
                    .unwrap(),
            );
        }
        placement
            .begin(
                1,
                vec![vec![1], vec![2, 3], vec![4, 5], vec![3], vec![2]],
                "declared-step".into(),
                "".into(),
            )
            .unwrap();
        let forward = |model| {
            placement.enter(1, model).expect("declared model entry");
            for page in &pages[&model] {
                let pin = placement.bind(1, *page, 0).unwrap();
                // A nested/repeated pin does not start another model forward.
                let nested = placement.bind(1, *page, 0).unwrap();
                drop(nested);
                drop(pin);
            }
            placement.gpu.wait_pending().unwrap();
            placement.leave(1, 0).expect("completed model forward");
        };
        forward(1);
        forward(3);
        for _ in 0..50 {
            forward(2);
            let inner = placement.inner.lock();
            let facts = inner.table.request(1).unwrap();
            assert_eq!(
                facts.position.step, 1,
                "VAE encode selected a future decode step"
            );
            assert_eq!(
                facts.model_positions.get(&3),
                Some(&1),
                "joint peer rewound"
            );
        }
        forward(4);
        forward(5);
        assert_eq!(
            placement
                .inner
                .lock()
                .table
                .request(1)
                .unwrap()
                .position
                .step,
            2
        );
        forward(3);
        assert_eq!(
            placement
                .inner
                .lock()
                .table
                .request(1)
                .unwrap()
                .position
                .step,
            3
        );
        forward(2);
        assert_eq!(
            placement
                .inner
                .lock()
                .table
                .request(1)
                .unwrap()
                .position
                .step,
            4
        );
        let error = placement
            .enter(1, 1)
            .expect_err("past steps remain invalid");
        let detail = error.to_string();
        assert!(detail.contains("model 1 [A]") && detail.contains("current step 4"));
        placement.end(1).unwrap();
        placement.close().unwrap();
        println!("PASS repeated component forward keeps its declared step {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn failed_fills_require_real_room_and_recover_in_the_same_placement() {
        use std::sync::atomic::AtomicUsize;

        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        let capacity = 32 << 20;
        assert!(probe.capacity().unwrap() >= capacity);
        drop(probe);
        let name = format!(
            "bind-oom-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        struct Named(String);
        impl Drop for Named {
            fn drop(&mut self) {
                let uid = unsafe { libc::geteuid() };
                let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.0));
            }
        }
        let _named = Named(name.clone());
        let ledger = Ledger::create(&name, capacity).expect("scoped ledger");
        let grant = ledger.grant_unbound().expect("rank grant");
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("actual CUDA placement");
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: 32,
                span: crate::layout::REGION_ALIGN,
                items: vec![Item {
                    offset: 0,
                    len: 32,
                    source: ItemSource::Inline(vec![7; 32].into()),
                    part_end: true,
                }],
            }],
            nbytes: crate::layout::REGION_ALIGN,
            digest: "bind-oom".into(),
            names: vec![],
        };
        let source = Arc::new(Source::files(vec![], false, 0));
        let victim = placement
            .register(2, "idle".into(), [45; 32], &layout, source.clone(), None)
            .unwrap()[0];
        assert_eq!(placement.load(2, Tier::Gpu).unwrap(), Relief::Complete);
        let failing = Arc::new(AtomicBool::new(true));
        let calls = Arc::new(AtomicUsize::new(0));
        let weak = Arc::downgrade(&placement);
        let fail = failing.clone();
        let attempted = calls.clone();
        let target = placement
            .register(
                1,
                "target".into(),
                [46; 32],
                &layout,
                source,
                Some(Arc::new(move |_, _| {
                    if !fail.load(Ordering::Acquire) {
                        return Ok(());
                    }
                    let attempt = attempted.fetch_add(1, Ordering::AcqRel) + 1;
                    let placement = weak.upgrade().expect("live placement");
                    let idle_mapped = placement
                        .gpu
                        .facts()?
                        .iter()
                        .any(|(id, bytes, _, _)| *id == victim && *bytes != 0);
                    // The unchanged implementation reaches this sentinel rather
                    // than hanging the GPU harness: its own failed-fill events
                    // masquerade as progress, without reclaiming the idle page.
                    if attempt >= 3 && idle_mapped {
                        return Err(invalid("failed fill retried without real reclaim"));
                    }
                    Err(Error::Cuda {
                        op: "injected fill allocation",
                        code: 2,
                        name: "CUDA_ERROR_OUT_OF_MEMORY".into(),
                    })
                })),
            )
            .unwrap()[0];
        placement
            .begin(1, vec![vec![1]], "bind-oom".into(), "".into())
            .unwrap();
        let error = match placement.bind(1, target, 0) {
            Err(error) => error,
            Ok(_) => panic!("injected OOM unexpectedly bound"),
        };
        assert!(matches!(error, Error::DeviceFull { .. }), "{error}");
        assert_eq!(calls.load(Ordering::Acquire), 3);
        assert!(placement
            .gpu
            .facts()
            .unwrap()
            .iter()
            .any(|(id, bytes, _, _)| *id == victim && *bytes == 0));
        let stats = placement.end(1).expect("failed request ends normally");
        assert_eq!(stats.copies.len(), 3);
        assert!(placement.inner.lock().requests.is_empty());

        failing.store(false, Ordering::Release);
        placement
            .begin(1, vec![vec![1]], "bind-oom".into(), "".into())
            .expect("same request identity is reusable");
        let pin = placement.bind(1, target, 0).expect("same-process recovery");
        let mut bytes = [0_u8; 32];
        // The actual pin protects the source through this synchronous read.
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .expect("recovered content");
        assert_eq!(bytes, [7; 32]);
        drop(pin);
        placement.gpu.wait_pending().unwrap();
        placement.end(1).unwrap();
        placement.close().unwrap();
        println!("PASS failed fills require real room and recover in the same placement {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn idle_worker_retires_after_pending_consumer_finishes() {
        let probe = Gpu::open(0).expect("real CUDA device");
        let driver = cuda::driver().expect("driver");
        let mut count = 0;
        cuda::check("cuDeviceGetCount", unsafe {
            (driver.device_get_count)(&mut count)
        })
        .expect("visible devices");
        assert_eq!(count, 1, "this proof requires exactly one assigned GPU");
        let uuid = probe.uuid().expect("GPU UUID");
        let ledger =
            Ledger::create(&uuid, probe.capacity().expect("capacity")).expect("observer ledger");
        let grant = ledger.grant_unbound().expect("unbound rank grant");
        drop(probe);
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("real placement");
        assert_eq!(placement.gpu_uuid().expect("placement GPU UUID"), uuid);
        let consumer = cuda::Stream::new(0).expect("consumer stream");
        let length = crate::layout::REGION_ALIGN;
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: length,
                span: length,
                items: vec![Item {
                    offset: 0,
                    len: length,
                    source: ItemSource::Inline(vec![7; length as usize].into()),
                    part_end: true,
                }],
            }],
            nbytes: length,
            digest: "idle-retirement-test".into(),
            names: vec![],
        };
        let pages = placement
            .register(
                1,
                "idle-retirement".into(),
                [7; 32],
                &layout,
                Arc::new(Source::files(vec![], false, 0)),
                None,
            )
            .expect("register");
        let page = *pages.first().expect("one real page");
        placement
            .begin(1, vec![vec![1]], "idle-retirement".into(), "".into())
            .expect("request");
        let pin = placement
            .bind(1, page, consumer.raw())
            .expect("fill and bind");
        let (started, waiting) = mpsc::channel();
        let (release, finish) = mpsc::channel();
        let gate = ReleaseGate(Some(release));
        let callback = Box::into_raw(Box::new(ConsumerGate {
            started,
            release: finish,
        }));
        let (device, context) = cuda::primary_context(0).expect("context");
        let retained = Primary(device);
        let result = {
            let _context = cuda::CtxGuard::enter(context).expect("enter context");
            unsafe {
                (driver.launch_host_func)(
                    consumer.raw() as cuda::CUstream,
                    hold_consumer,
                    callback.cast(),
                )
            }
        };
        drop(retained);
        if result != cuda::SUCCESS {
            drop(unsafe { Box::from_raw(callback) });
        }
        cuda::check("cuLaunchHostFunc", result).expect("gate consumer");
        waiting
            .recv_timeout(Duration::from_secs(10))
            .expect("consumer is pending");
        placement.close_model(1).expect("unregister pinned model");
        drop(pin);
        // Observe only the coordinator's already-published snapshot. Calling
        // Gpu::poll/facts or Placement::resident here would hide the regression.
        observe(&ledger, "worker did not dequeue the last pin", || {
            placement
                .inner
                .lock()
                .table
                .page(page)
                .is_some_and(|row| row.pins == 0 && row.filling && row.mapped())
        });
        drop(gate);
        observe(&ledger, "idle worker did not retire completed page", || {
            let status = ledger.status().expect("published rows");
            placement.inner.lock().table.page(page).is_none()
                && status
                    .rows
                    .iter()
                    .find(|row| row.owner == placement.row.owner())
                    .is_some_and(|row| row.snapshot.classes.iter().all(|bytes| *bytes == 0))
                && status
                    .kept
                    .iter()
                    .all(|(key, _)| key.fingerprint != [7; 32])
        });
        placement.end(1).expect("end request");
        placement.close().expect("close placement");
        println!("PASS idle worker retires after pending consumer {uuid}");
    }

    fn controlled_lower_layout() -> Layout {
        const MIB: u64 = 1 << 20;
        Layout {
            parts: vec![],
            regions: (0..6)
                .map(|index| crate::layout::Region {
                    offset: index * 4 * MIB,
                    nbytes: 4 * MIB,
                    span: 4 * MIB,
                    items: vec![Item {
                        offset: index * 4 * MIB,
                        len: 4 * MIB,
                        part_end: true,
                        source: ItemSource::Inline(vec![7; (4 * MIB) as usize].into()),
                    }],
                })
                .collect(),
            nbytes: 24 * MIB,
            digest: "controlled-lower".into(),
            names: vec![],
        }
    }

    #[test]
    fn controlled_lower_source_plan_reads_all_six_host_regions() {
        use tensorfs::io::source::{IoTally, Mode};
        let layout = controlled_lower_layout();
        let source = Source::files(vec![], false, 0);
        let tally = IoTally::default();
        for region in &layout.regions {
            let mut bytes = vec![0_u8; region.nbytes as usize];
            unsafe {
                source.read_items(
                    &region.items,
                    region.offset,
                    tensorfs::io::Dst::host(&mut bytes),
                    1,
                    Mode::Buffered,
                    &tally,
                )
            }
            .expect("controlled lower region has a valid source/destination plan");
            assert!(bytes.iter().all(|byte| *byte == 7));
        }
        assert_eq!(tally.inline_bytes.load(Ordering::Relaxed), layout.nbytes);
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn lower_relief_keeps_fitting_home_and_requires_real_reclaim() {
        const MIB: u64 = 1 << 20;
        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        assert!(probe.capacity().unwrap() >= 56 * MIB);
        drop(probe);
        // The real CUDA allocations use an isolated 56 MiB ledger; the
        // machine service's cap and every other executor remain untouched.
        let name = format!(
            "lower-target-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        struct Named(String);
        impl Drop for Named {
            fn drop(&mut self) {
                let uid = unsafe { libc::geteuid() };
                let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.0));
            }
        }
        let _named = Named(name.clone());
        let ledger = Ledger::create(&name, 56 * MIB).expect("scoped lower ledger");
        let grant = ledger.grant_unbound().expect("rank grant");
        let placement = Placement::open(0, Some((grant.table, grant.events, grant.channel)))
            .expect("actual CUDA placement");
        let layout = controlled_lower_layout();
        let pages = placement
            .register(
                1,
                "lower".into(),
                [62; 32],
                &layout,
                Arc::new(Source::files(vec![], false, 0)),
                None,
            )
            .unwrap();
        assert_eq!(pages.len(), 6);
        assert_eq!(placement.load(1, Tier::Gpu).unwrap(), Relief::Complete);
        let context = tensorfs::io::DeviceBuffer::new(0, (4 * MIB) as usize).unwrap();
        let scratch = tensorfs::io::DeviceBuffer::new(0, (8 * MIB) as usize).unwrap();
        placement.allocator(4 * MIB, 8 * MIB, 8 * MIB).unwrap();
        placement
            .begin(1, vec![vec![1]], "controlled-lower".into(), "shape".into())
            .unwrap();
        placement.enter(1, 1).unwrap();
        placement.leave(1, 8 * MIB).unwrap();
        placement.enter(1, 1).unwrap();
        let pin = placement
            .bind(1, pages[0], 0)
            .expect("protected current source");

        // Scaled six-page witness: 4 context + 8 live activation + 8
        // retained weights + 12 ring + 24 upcoming allocation = 56 MiB.
        assert_eq!(
            placement.relief(1, 24 * MIB, true).unwrap(),
            Relief::Complete
        );
        let stats = placement.stats(1).unwrap();
        let lowered: Vec<_> = stats
            .events
            .iter()
            .filter(|event| event.kind == "lower")
            .map(|event| event.page)
            .collect();
        assert_eq!(lowered.len(), 4, "only four home pages need to leave");
        assert!(
            !lowered.contains(&pages[0]),
            "a current source pin was lowered"
        );
        // The worker may already have prefetched up to three streaming slots.
        // Those mapped bytes do not restore the lowered home watermark.
        let resident = placement.resident().unwrap()[0].1;
        assert!((8 * MIB..=20 * MIB).contains(&resident));
        let allocated = tensorfs::io::DeviceBuffer::new(0, (24 * MIB) as usize)
            .expect("the requested allocation fits after controlled relief");
        placement.allocator(4 * MIB, 32 * MIB, 32 * MIB).unwrap();
        drop(allocated);
        placement.allocator(4 * MIB, 8 * MIB, 8 * MIB).unwrap();

        // A reported allocator failure still requires real reclaim when the
        // logical free room already covers both the allocation and its ring.
        assert!(placement.available().unwrap() >= 16 * MIB);
        assert_eq!(
            placement.relief(1, 4 * MIB, true).unwrap(),
            Relief::Complete
        );
        assert_eq!(
            placement
                .stats(1)
                .unwrap()
                .events
                .iter()
                .filter(|event| event.kind == "lower")
                .count(),
            5
        );
        assert!(
            matches!(placement.relief(1, 4 * MIB, true).unwrap(), Relief::Exhausted(n) if n > 0),
            "a repeated failure with no reclaimable home must not succeed as a no-op"
        );
        let mut bytes = [0_u8; 32];
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: pin.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .expect("pinned source stays readable through every relief");
        assert_eq!(bytes, [7; 32]);
        drop(pin);
        placement.gpu.wait_pending().unwrap();
        placement.leave(1, 32 * MIB).unwrap();
        placement.end(1).unwrap();
        drop(scratch);
        drop(context);
        placement.allocator(0, 0, 0).unwrap();
        placement.close().unwrap();
        println!("PASS controlled lower preserves fitting home, pins and bounded physical reclaim {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn idle_observed_peak_allows_adaptive_warm_use_without_dropping_live_pins() {
        const MIB: u64 = 1 << 20;
        let probe = Gpu::open(0).expect("real CUDA device");
        let uuid = assigned_gpu(&probe);
        assert!(probe.capacity().unwrap() >= 32 * MIB);
        drop(probe);
        let name = format!(
            "idle-observed-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        struct Named(String);
        impl Drop for Named {
            fn drop(&mut self) {
                let uid = unsafe { libc::geteuid() };
                let _ = std::fs::remove_file(format!("/dev/shm/varena-{}-{uid}", self.0));
            }
        }
        let _named = Named(name.clone());
        let ledger = Ledger::create(&name, 32 * MIB).unwrap();
        let grant = ledger.grant_unbound().unwrap();
        let placement =
            Placement::open(0, Some((grant.table, grant.events, grant.channel))).unwrap();
        let weight = 20 * MIB;
        let layout = Layout {
            parts: vec![],
            regions: vec![crate::layout::Region {
                offset: 0,
                nbytes: weight,
                span: weight,
                items: vec![Item {
                    offset: 0,
                    len: weight,
                    part_end: true,
                    source: ItemSource::Inline(vec![7; weight as usize].into()),
                }],
            }],
            nbytes: weight,
            digest: "idle-observed-peak".into(),
            names: vec![],
        };
        let pages = placement
            .register(
                1,
                "idle-observed".into(),
                [61; 32],
                &layout,
                Arc::new(Source::files(vec![], false, 0)),
                None,
            )
            .unwrap();
        let context = tensorfs::io::DeviceBuffer::new(0, (4 * MIB) as usize).unwrap();
        placement.allocator(4 * MIB, 0, 0).unwrap();
        placement
            .begin(1, vec![vec![1]], "idle-observed".into(), "shape".into())
            .unwrap();
        let retained = placement.bind(1, pages[0], 0).unwrap();
        let scratch = tensorfs::io::DeviceBuffer::new(0, (8 * MIB) as usize).unwrap();
        placement.allocator(4 * MIB, 8 * MIB, 8 * MIB).unwrap();
        placement.leave(1, 8 * MIB).unwrap();
        drop(scratch);
        placement.allocator(4 * MIB, 0, 0).unwrap();
        placement.end(1).unwrap();
        assert_eq!(ledger.measured("idle-observed", "shape"), Some(28 * MIB));
        assert_eq!(placement.allocator_budget().unwrap(), 8 * MIB);
        assert!(
            !ledger.fits("idle-observed", "shape").unwrap(),
            "idle cache acknowledgement must come first"
        );
        observe(
            &ledger,
            "no-op idle acknowledgement never admitted adaptive warm use",
            || ledger.fits("idle-observed", "shape").unwrap(),
        );
        let idle = ledger.status().unwrap();
        assert_eq!(idle.capacity, 32 * MIB);
        assert_eq!(
            idle.rows[0].snapshot.classes[Class::Pinned.as_usize()],
            weight,
            "admission silently discarded a live pin"
        );
        assert_eq!(
            placement.allocator_budget().unwrap(),
            8 * MIB,
            "admission inflated actual available bytes"
        );
        placement
            .begin(2, vec![vec![1]], "idle-observed".into(), "shape".into())
            .unwrap();
        let warm = placement.bind(2, pages[0], 0).unwrap();
        let scratch = tensorfs::io::DeviceBuffer::new(0, (8 * MIB) as usize).unwrap();
        placement.allocator(4 * MIB, 8 * MIB, 8 * MIB).unwrap();
        let mut bytes = [0_u8; 32];
        // Both live pins own the mapping throughout this checked real D2H read.
        unsafe {
            tensorfs::io::read(
                &[tensorfs::io::Copy {
                    src: tensorfs::io::Src::Device {
                        device: 0,
                        ptr: warm.ptr,
                        stream: 0,
                    },
                    dst: tensorfs::io::Dst::host(&mut bytes),
                }],
                &tensorfs::io::ReadOpts::default(),
            )
        }
        .expect("unchanged pinned CUDA content");
        assert_eq!(bytes, [7; 32]);
        drop(warm);
        placement.leave(2, 8 * MIB).unwrap();
        let stats = placement.end(2).unwrap();
        assert!(
            stats.copies.is_empty(),
            "warm use unnecessarily refilled held weight bytes"
        );
        drop((scratch, retained));
        placement.gpu.wait_pending().unwrap();
        drop(context);
        placement.allocator(0, 0, 0).unwrap();
        placement.close().unwrap();
        println!("PASS idle observed peak admits adaptive warm use without dropping pins {uuid}");
    }

    #[test]
    #[ignore = "requires an owned rental; never run on the local GPU"]
    fn activation_oom_waits_for_foreign_idle_reclaim_and_keeps_live_pins() {
        let uuid = fill_worker(|| {
            const MIB: u64 = 1 << 20;
            let fixture = FillLedger::new();
            let owner = fixture.placement();
            let requester = fixture.placement();
            let uuid = assigned_gpu(&owner.gpu);
            assert_eq!(requester.gpu_uuid().unwrap(), uuid);
            let pages = owner
                .register(
                    1,
                    "foreign-idle".into(),
                    [62; 32],
                    &fill_layout(4, 7),
                    Arc::new(Source::files(vec![], false, 0)),
                    None,
                )
                .unwrap();
            owner.load(1, Tier::Gpu).unwrap();
            let retained = owner.gpu.bind(pages[0], 0).unwrap();
            owner.refresh().unwrap();
            let slack = Arc::new(Mutex::new(Some(
                tensorfs::io::DeviceBuffer::new(0, (4 * MIB) as usize).unwrap(),
            )));
            let (entered, waiting) = mpsc::channel();
            let (release, finish) = mpsc::channel();
            let gate = ReleaseGate(Some(release));
            let finish = Mutex::new(finish);
            let weak = Arc::downgrade(&owner);
            let held = slack.clone();
            owner.trim_with(Arc::new(move |_| {
                entered.send(()).unwrap();
                finish.lock().recv().unwrap();
                // This real allocator cache remains busy. Its acknowledgement
                // must not be mistaken for physical progress by the requester.
                assert!(held.lock().is_some());
                weak.upgrade().unwrap().allocator(0, 0, 4 * MIB)?;
                Ok(0)
            }));
            owner.allocator(0, 0, 4 * MIB).unwrap();
            requester
                .begin(1, vec![], "foreign-requester".into(), "shape".into())
                .unwrap();
            let available = requester.available().unwrap();
            assert_eq!(available, 4 * MIB);
            let budget = requester.allocator_budget().unwrap();
            assert_eq!(budget, available);
            let (sent, done) = mpsc::channel();
            let asking = requester.clone();
            let worker = thread::spawn(move || {
                sent.send(asking.relief(1, MIB, false)).unwrap();
            });
            waiting
                .recv_timeout(Duration::from_secs(5))
                .expect("no foreign trim request");
            assert!(
                matches!(done.try_recv(), Err(mpsc::TryRecvError::Empty)),
                "relief returned before the actual foreign callback acknowledged"
            );
            drop(gate);
            assert_eq!(
                done.recv_timeout(Duration::from_secs(5)).unwrap().unwrap(),
                Relief::Complete
            );
            worker.join().unwrap();
            assert!(
                slack.lock().is_some(),
                "the callback did not actually free its cache"
            );
            assert!(
                requester.available().unwrap() >= available + 4 * MIB,
                "no-op acknowledgement reset the fixed reclaim target"
            );
            assert!(
                requester.allocator_budget().unwrap() > budget,
                "actual remote reclamation did not reach the allocator ceiling"
            );
            for remaining in [8 * MIB, 4 * MIB] {
                assert_eq!(
                    requester.relief(1, MIB, false).unwrap(),
                    Relief::Complete,
                    "a new OOM in the same unchanged request reused an old remote ACK"
                );
                assert_eq!(
                    owner.resident().unwrap()[0].1,
                    remaining,
                    "each actual OOM must reclaim one more unpinned CUDA region"
                );
            }
            assert!(
                matches!(
                    requester.relief(1, MIB, false).unwrap(),
                    Relief::Cut(_) | Relief::Lower(_) | Relief::Exhausted(_)
                ),
                "a no-progress owner produced Complete or discarded its protected bytes"
            );
            assert_eq!(owner.ledger.capacity(), 24 * MIB);
            let mut bytes = [0_u8; 32];
            // The independent row's real pin survives another row's pressure.
            unsafe {
                tensorfs::io::read(
                    &[tensorfs::io::Copy {
                        src: tensorfs::io::Src::Device {
                            device: 0,
                            ptr: retained.ptr,
                            stream: 0,
                        },
                        dst: tensorfs::io::Dst::host(&mut bytes),
                    }],
                    &tensorfs::io::ReadOpts::default(),
                )
            }
            .expect("foreign pinned CUDA content");
            assert_eq!(bytes, [7; 32]);
            let allocated = tensorfs::io::DeviceBuffer::new(0, (MIB) as usize).unwrap();
            requester.allocator(0, MIB, MIB).unwrap();
            drop(allocated);
            requester.allocator(0, 0, 0).unwrap();
            requester.end(1).unwrap();
            drop(retained);
            owner.gpu.wait_pending().unwrap();
            drop(slack.lock().take());
            owner.allocator(0, 0, 0).unwrap();
            requester.close().unwrap();
            owner.close().unwrap();
            uuid
        });
        println!(
            "PASS activation OOM acknowledges foreign idle reclaim and preserves live pins {uuid}"
        );
    }


}
