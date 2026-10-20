//! Applies the single placement policy to CUDA mappings and the shared ledger.
//!
//! Model descriptions and ordered requests come from the caller. The coordinator never
//! parses a checkpoint, predicts a plan, copies bytes, or holds its table lock across I/O.

use std::collections::{BTreeMap, BTreeSet};
use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::thread::JoinHandle;

use parking_lot::{Condvar, Mutex};
use tensorfs::io::source::Source;
use tensorfs::io::ReadStats;

use crate::gpu::{FillPin, Gpu, Pin};
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

#[derive(Clone, Debug)]
pub struct Event {
    pub sequence: u64,
    pub request: RequestId,
    pub step: u32,
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
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub events: Vec<Event>,
    pub copies: Vec<Copy>,
    pub peak: u64,
}

struct Model {
    name: String,
    pages: Vec<PageId>,
    fingerprint: [u8; 32],
    transform: Option<Arc<Transform>>,
}

struct Request {
    steps: Vec<Vec<ModelId>>,
    explicit: bool,
    package: String,
    shape: String,
    stats: Stats,
    budgets: BTreeMap<u32, u64>,
}

struct Cache {
    shrink: Arc<Shrink>,
    active: u32,
    blocked: bool,
}

struct Inner {
    table: plan::Table,
    models: BTreeMap<ModelId, Model>,
    requests: BTreeMap<RequestId, Request>,
    caches: BTreeMap<PageId, Cache>,
    orphaned: BTreeMap<PageId, PageKey>,
    retired: BTreeSet<PageId>,
    filling: BTreeSet<PageId>,
    next_page: PageId,
    progress: u64,
    trim: Option<Arc<Shrink>>,
}

pub struct Placement {
    gpu: Arc<Gpu>,
    ledger: Arc<Ledger>,
    row: Arc<ledger::Row>,
    channel: Channel,
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
        let channel = Channel::new(channel);
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
                filling: BTreeSet::new(),
                next_page: 1,
                progress: 0,
                trim: None,
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
            if let Err(error) = result.and(settled) {
                tracing::warn!(%error, "reclaim failed");
            }
        }
    }

    /// A CUDA completion does not ring the ledger's eventfd. After a release wake,
    /// wait on actual submitted work before sleeping again, then publish and reconsider
    /// pages that were still protected when the wake was first processed.
    fn settle(&self) -> Result<()> {
        loop {
            let pending = self.gpu.wait_pending()?;
            self.refresh()?;
            if self.stop.load(Ordering::Acquire) {
                return Ok(());
            }
            let requests: Vec<_> = self.inner.lock().requests.keys().copied().collect();
            for request in requests {
                self.drive(request, Trigger::PinReleased(request))?;
            }
            if !pending {
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
        self.drive(
            0,
            Trigger::Load {
                row: self.row.owner().row,
                model,
                level,
            },
        )
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
                transform,
            },
        );
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
        if self.inner.lock().requests.contains_key(&id) {
            return Err(invalid("request already active"));
        }
        let mut inner = self.inner.lock();
        inner.table.requests.push(plan::Request {
            id,
            row: self.row.owner().row,
            position: Position { step: 0, order: 0 },
            working_sets: vec![None; steps.len().max(1)],
            starved: false,
            sharing_limit: None,
            resident_steps: BTreeSet::new(),
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
                stats: Stats::default(),
                budgets: BTreeMap::new(),
            },
        );
        if let Err(error) = Self::declare_locked(&mut inner, id, steps) {
            inner.table.requests.retain(|request| request.id != id);
            inner.requests.remove(&id);
            return Err(error);
        }
        drop(inner);
        self.hydrate(id)?;
        self.refresh()?;
        self.event(id, 0, "begin", "request")
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
            let mut order = 0_u32;
            for _ in 0..*passes {
                for model in models {
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
                                order,
                            },
                        });
                        order = order
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
        facts.position = Position { step: 0, order: 0 };
        facts.starved = false;
        let request = inner
            .requests
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown request"))?;
        request.steps = groups.into_iter().map(|(models, _)| models).collect();
        request.explicit = !request.steps.is_empty();
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
                    if facts.position.step != next {
                        facts.position = Position {
                            step: next,
                            order: 0,
                        };
                        facts.starved = false;
                    }
                }
            } else {
                return Err(invalid("model is absent from the remaining plan"));
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
        let mut previous = None;
        let pin = loop {
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
                    self.drive(request, Trigger::Oom { request, need })?
                }
                Err(error) => return Err(error),
            };
            match outcome {
                Relief::Complete => {
                    self.wait_fill(page)?;
                    match self.gpu.bind(page, stream) {
                        Ok(pin) => break pin,
                        Err(error) if error.is_oom() => {
                            self.gpu.recover()?;
                        }
                        Err(error) => return Err(error),
                    }
                }
                Relief::Lower(short) => {
                    if !matches!(
                        self.drive(
                            request,
                            Trigger::Lower {
                                request,
                                need: short
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
            let facts = {
                let inner = self.inner.lock();
                (inner.progress, inner.table.available())
            };
            if previous == Some(facts) {
                return Err(Error::DeviceFull {
                    need,
                    deficit: need,
                });
            }
            previous = Some(facts);
        };
        self.event(request, page, "bind", "demand")?;
        {
            let mut inner = self.inner.lock();
            let current = inner
                .table
                .request(request)
                .ok_or_else(|| invalid("unknown request"))?
                .position;
            let next = inner.table.page(page).and_then(|page| {
                page.uses
                    .iter()
                    .filter(|usage| usage.request == request && usage.position >= current)
                    .map(|usage| usage.position)
                    .min()
            });
            if let Some(next) = next {
                if let Some(facts) = inner
                    .table
                    .requests
                    .iter_mut()
                    .find(|row| row.id == request)
                {
                    facts.position = Position {
                        step: next.step,
                        order: next.order.saturating_add(1),
                    };
                }
            }
            inner.progress = inner.progress.saturating_add(1);
        }
        self.refresh()?;
        self.row.wake()?;
        Ok(pin)
    }

    fn wait_fill(&self, page: PageId) -> Result<()> {
        let mut inner = self.inner.lock();
        while inner.filling.contains(&page) && !self.stop.load(Ordering::Acquire) {
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
        let stats = inner
            .requests
            .remove(&request)
            .ok_or_else(|| invalid("unknown request"))?
            .stats;
        drop(inner);
        self.refresh()?;
        recovered?;
        if let Some((package, shape, peak)) = peak {
            self.ledger.record_peak(&package, &shape, peak)?;
        }
        Ok(stats)
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
        if let Some(facts) = self
            .inner
            .lock()
            .table
            .requests
            .iter_mut()
            .find(|row| row.id == request)
        {
            facts.can_cut = !lower;
        }
        self.drive(
            request,
            if lower {
                Trigger::Lower { request, need }
            } else {
                Trigger::Oom { request, need }
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
        let pins = cache.active;
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
    }

    pub fn cache_active(&self, id: PageId, active: bool) -> Result<()> {
        let mut inner = self.inner.lock();
        let cache = inner
            .caches
            .get_mut(&id)
            .ok_or_else(|| invalid("unknown cache"))?;
        cache.active = if active {
            cache.active.saturating_add(1)
        } else {
            cache.active.saturating_sub(1)
        };
        let pins = cache.active.saturating_add(u32::from(cache.blocked));
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
        let pages = {
            let mut inner = self.inner.lock();
            let Some(model) = inner.models.remove(&id) else {
                return Ok(());
            };
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
            model.pages
        };
        for page in pages {
            self.gpu.unregister(page)?;
        }
        self.refresh()
    }

    pub fn trim_with(&self, trim: Arc<Shrink>) {
        self.inner.lock().trim = Some(trim);
    }

    pub fn allocator(&self, context: u64, allocated: u64, reserved: u64) -> Result<()> {
        let mut inner = self.inner.lock();
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
                page.bytes = bytes;
            } else {
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
        let facts = self.gpu.facts()?;
        let status = self.ledger.status()?;
        let orphans = self.channel.orphans(self.row.owner())?;
        let mut inner = self.inner.lock();
        inner.table.capacity = status.capacity;
        let present: BTreeSet<_> = facts.iter().map(|(id, _, _, _)| *id).collect();
        let finished: BTreeSet<_> = inner.retired.difference(&present).copied().collect();
        inner
            .table
            .pages
            .retain(|page| !finished.contains(&page.id));
        inner.retired.retain(|id| !finished.contains(id));
        for (id, bytes, pins, pending) in facts {
            let reserved = inner.filling.contains(&id);
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

    fn drive(&self, request: RequestId, trigger: Trigger) -> Result<Relief> {
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
                            return Err(Error::Unsupported(
                                "shared RAM source registration is unavailable".into(),
                            ));
                        }
                        self.inner.lock().filling.insert(page);
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
                        let key = self.inner.lock().orphaned.get(&page).copied();
                        if let Some(key) = key {
                            self.channel.release_orphan(self.row.owner(), key)?;
                            self.event(request, page, "evict", "orphan")?;
                        }
                    }
                    Action::Trim(page) => {
                        let trim =
                            self.inner.lock().trim.clone().ok_or_else(|| {
                                invalid("allocator trim callback was not registered")
                            })?;
                        let bytes = self
                            .inner
                            .lock()
                            .table
                            .page(page)
                            .map_or(0, |row| row.bytes);
                        let freed = trim(bytes)?;
                        if let Some(row) = self
                            .inner
                            .lock()
                            .table
                            .pages
                            .iter_mut()
                            .find(|row| row.id == page)
                        {
                            row.bytes = row.bytes.saturating_sub(freed);
                            if freed == 0 {
                                row.pins = 1;
                            }
                        }
                        if freed < bytes {
                            drop(operation);
                            continue 'decide;
                        }
                    }
                    Action::Shrink { page, bytes } => {
                        let shrink = self
                            .inner
                            .lock()
                            .caches
                            .get(&page)
                            .map(|cache| cache.shrink.clone())
                            .ok_or_else(|| invalid("cache callback absent"))?;
                        let freed = shrink(bytes)?;
                        let pins = {
                            let mut inner = self.inner.lock();
                            inner.caches.get_mut(&page).map(|cache| {
                                cache.blocked = freed == 0;
                                cache.active.saturating_add(u32::from(cache.blocked))
                            })
                        };
                        if let Some(page) = self
                            .inner
                            .lock()
                            .table
                            .pages
                            .iter_mut()
                            .find(|row| row.id == page)
                        {
                            page.bytes = page.bytes.saturating_sub(freed);
                            page.pins = pins.unwrap_or(0);
                        }
                        drop(operation);
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
        let (class, bytes, model) = inner
            .table
            .page(page)
            .map_or((Class::Free, 0, None), |facts| {
                (inner.table.class(facts), facts.bytes, facts.model)
            });
        let step = inner
            .table
            .request(request)
            .map_or(0, |row| row.position.step);
        let event = Event {
            sequence: inner.progress,
            request,
            step,
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
        };
        if let Some(run) = inner.requests.get_mut(&request) {
            run.stats.events.push(event);
        }
        inner.progress = inner.progress.saturating_add(1);
        Ok(())
    }

    fn fill(
        &self,
        request: RequestId,
        page: PageId,
        _from: Tier,
        speculative: Option<Step>,
    ) -> Result<()> {
        self.event(
            request,
            page,
            "fill",
            if speculative.is_some() {
                "speculation"
            } else {
                "demand"
            },
        )?;
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
        let (model, transform, key, streamed) = {
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
                inner
                    .table
                    .page(page)
                    .is_some_and(|row| row.lowered.contains(&request)),
            )
        };
        let apply = |pin| transform.as_ref().map_or(Ok(()), |fill| fill(page, pin));
        let ram = self
            .channel
            .ram(self.row.owner(), key.fingerprint, key.page)?;
        let actual = if ram.is_some() { Tier::Ram } else { Tier::Disk };
        let stats = if let Some(ram) = ram {
            let mapped = Ram::map(ram)?;
            self.gpu.fill_from_ram_with(page, mapped.bytes(), apply)?
        } else {
            self.gpu.fill_with(page, apply)?
        };
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
        if let Some(run) = inner.requests.get_mut(&request) {
            run.stats.copies.push(Copy {
                request,
                model,
                page,
                tier: actual,
                speculative: speculative.is_some(),
                streamed,
                stats,
            });
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
        inner.models.remove(&self.model);
        inner.table.pages.retain(|page| !gone.contains(&page.id));
        inner.retired.extend(retained);
        let _ = self.owner.row.wake();
    }
}

impl Drop for Placement {
    fn drop(&mut self) {
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

fn invalid(message: &str) -> Error {
    Error::Invalid(message.to_owned())
}

fn spread_rank(index: usize, count: usize) -> u32 {
    let bits = (usize::BITS - count.saturating_sub(1).leading_zeros()).clamp(1, 32);
    (index as u32).reverse_bits() >> (32 - bits)
}
