//! The machine's sole dispatch policy; the engine only journals/supervises. Every run takes
//! one path: its plan (journaled when it binds models, else weightless) on the pool.
use platform::process::Birth;
use crate::{
    catalog::Catalog,
    execution::Engine,
    gpu_service::{KeptMember, Level, Member, Plan, Pool},
    journal::{Execution, SubmissionContext},
};
use serde_json::Value;
use std::{
    collections::{BTreeSet, HashMap, HashSet},
    fs::File,
    io,
    path::Path,
    sync::{Arc, Mutex},
};

/// What one public submission calls on its generation.
#[derive(Default)]
pub struct Call {
    pub entrypoint: String,
    pub input: Value,
    pub attention_kernel: String,
    pub gpus: u32,
    pub bits: u32,
    pub inputs: Vec<crate::journal::InputFile>,
    /// GPUs its plan runs on: 0 deviceless.
    pub degree: u32,
    /// A child run's parent execution id.
    pub parent: String,
    /// The callee App a child run calls in its parent's environment; empty for the root's.
    pub application: String,
}

pub struct Service {
    pub engine: Arc<Engine>,
    pub catalog: Catalog,
    parallelism: usize,
    stopped: Mutex<bool>,
    retained: Mutex<HashMap<String, Arc<File>>>,
    pool: Mutex<Option<Arc<Pool>>>,
    /// The child-call broker and weights grants.
    calls: Mutex<Option<Arc<crate::calls::Calls>>>,
    /// The TensorFS store's caches (`Publisher::reclaim`), swept with the machine's.
    publisher: Mutex<Option<Arc<crate::published::Publisher>>>,
    startup_gpu_births: Mutex<Vec<Birth>>,
    /// Package/model preparations in flight: work a rental's idle release must wait for.
    preparing: std::sync::atomic::AtomicUsize,
    /// This machine's birth: a fenced leader's group members born before it are its
    /// previous incarnation's followers.
    started_ticks: u64,
    queue: Mutex<crate::queue::Queue>,
    /// While it holds (an update activating), no new root run starts: queued runs are journaled
    /// and start after the restart.
    held: std::sync::OnceLock<Box<dyn Fn() -> bool + Send + Sync>>,
    /// A rental's report to its Hub that a preparation made no progress (`on_stall`).
    stall: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
}
/// Held while one preparation runs; the machine is not idle meanwhile.
pub struct Preparing(Arc<Service>);
impl Drop for Preparing {
    fn drop(&mut self) {
        self.0
            .preparing
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        self.0.engine.notify_activity();
    }
}

const SWEEP_EVERY: std::time::Duration = crate::reclaim::IDLE;
impl Service {
    pub fn open(root: &Path, generations: &Path, parallelism: usize) -> io::Result<Arc<Self>> {
        if parallelism == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "CPU parallelism must be positive",
            ));
        }
        let engine = Engine::open(&root.join("execution"))?;
        // A completed request does not prove its retained CUDA context ended. A prior
        // machine's executor cannot be adopted (nothing can reach it), so it is ended now
        // and GPU admission stays fenced until its exit is observed.
        let mut cursor = 0;
        let mut seen = HashSet::new();
        let mut startup_gpu_births = vec![];
        let started_ticks = Birth::own()?.start_ticks;
        loop {
            let page = engine.gpu_births_after(cursor, 256)?;
            if page.is_empty() {
                break;
            }
            for (id, birth) in page {
                cursor = id;
                let alive = !birth.ended().unwrap_or(false)
                    || crate::process::group_outlives(&birth, started_ticks);
                if alive && seen.insert((birth.pid, birth.boot_id.clone(), birth.start_ticks)) {
                    // A follower left behind by a leader that already ended is ended too.
                    for member in platform::process::group_members(&birth) {
                        if member.start_ticks < started_ticks {
                            if let Ok(Some(exact)) = platform::process::Exact::open(&member) {
                                let _ = exact.kill();
                            }
                        }
                    }
                    if let Err(error) = engine.end_orphan(birth.clone()) {
                        tracing::warn!("GPU startup birth remains fenced: {error}");
                    }
                    startup_gpu_births.push(birth);
                }
            }
        }
        // Descendants of those executors (a setsid daemon, a compile worker) live on in the
        // executors' own scopes; every such scope of this machine is ended now.
        match platform::contain::Scope::sweep(&platform::contain::namespace(root)) {
            Ok(0) => (),
            Ok(held) => tracing::info!("ended {held} process(es) left in earlier executor scopes"),
            Err(error) => tracing::warn!("earlier executor scopes remain: {error}"),
        }
        let service = Arc::new(Self {
            engine,
            catalog: Catalog::new(generations)?,
            parallelism,
            stopped: Mutex::new(false),
            retained: Mutex::new(HashMap::new()),
            pool: Mutex::new(None),
            calls: Mutex::new(None),
            publisher: Mutex::new(None),
            startup_gpu_births: Mutex::new(startup_gpu_births),
            preparing: std::sync::atomic::AtomicUsize::new(0),
            started_ticks,
            queue: Default::default(),
            held: Default::default(),
            stall: Default::default(),
        });
        service.engine.reconcile()?;
        service.reclaim();
        // Keep queued runs' generations alive, including a prior boot's.
        for record in service.engine.nonterminal(usize::MAX)? {
            if let Ok(held) = service.catalog.resolve(&record.invocation.generation) {
                service
                    .retained
                    .lock()
                    .unwrap()
                    .insert(record.id.clone(), held.retention());
            }
            if let Some(pending) = record.pending_completion {
                for birth in pending.processes {
                    service.engine.end_orphan(birth)?;
                }
            } else if let Some(birth) = record.process {
                service.engine.end_orphan(birth)?;
            }
        }
        let owner = service.clone();
        std::thread::Builder::new()
            .name("machine-dispatch".into())
            .spawn(move || owner.dispatch_loop())?;
        let weak = Arc::downgrade(&service);
        std::thread::Builder::new()
            .name("machine-reclaim".into())
            .spawn(move || Self::reclaim_loop(weak))?;
        Ok(service)
    }
    /// The executor pool every run runs on, once configured.
    pub fn pool(&self) -> Option<Arc<Pool>> {
        self.pool.lock().unwrap().clone()
    }
    /// The pool when this machine has GPUs.
    pub fn gpu(&self) -> Option<Arc<Pool>> {
        self.pool().filter(|pool| pool.width() > 0)
    }
    pub fn calls(&self) -> Option<Arc<crate::calls::Calls>> {
        self.calls.lock().unwrap().clone()
    }
    pub fn configure_publisher(&self, publisher: Arc<crate::published::Publisher>) {
        *self.publisher.lock().unwrap() = Some(publisher);
    }
    pub fn configure_calls(&self, calls: Arc<crate::calls::Calls>) {
        *self.calls.lock().unwrap() = Some(calls);
        self.engine.notify_activity();
    }
    /// Observation only. Unknown births stay reserved; CPU dispatch stays available.
    pub fn gpu_startup_fences(&self) -> usize {
        let mut births = self.startup_gpu_births.lock().unwrap();
        // A group's followers die with their leader (one group kill); each exit is observed.
        births.retain(|birth| {
            !birth.ended().unwrap_or(false)
                || crate::process::group_outlives(birth, self.started_ticks)
        });
        births.len()
    }
    pub fn configure_pool(&self, pool: Arc<Pool>) -> io::Result<()> {
        let runtime = tokio::runtime::Handle::try_current().ok();
        self.configure_pool_with(pool, runtime.as_ref())
    }
    /// Production setup runs outside Tokio's entered context, just like dispatch.
    /// Use its existing runtime explicitly; synchronous bookkeeping fixtures need none.
    pub fn configure_pool_on(&self, pool: Arc<Pool>, runtime: &tokio::runtime::Handle) -> io::Result<()> {
        self.configure_pool_with(pool, Some(runtime))
    }
    fn configure_pool_with(&self, pool: Arc<Pool>, runtime: Option<&tokio::runtime::Handle>) -> io::Result<()> {
        let mut current = self.pool.lock().unwrap();
        if current.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                "the executor pool is already configured",
            ));
        }
        if let Some(runtime) = runtime {
            pool.watch_memory(&self.engine, runtime)?;
        }
        *current = Some(pool.clone());
        drop(current);
        if pool.width() > 0 {
            // Kernels compile at boot, before any install. An executor starts for a request or
            // an install: nothing here prepares a package nobody asked for.
            pool.image_kernel_boot();
            for held in self.catalog.installed() {
                pool.kernel_boot(held);
            }
            self.restore_warm_sets(&pool)?;
        }
        self.changed_environment()
    }
    /// Each caller's warm set as the journal keeps it, brought up again in the background: a
    /// restarted machine re-warms without its controller. A member whose installation is gone
    /// leaves the set.
    fn restore_warm_sets(&self, gpu: &Arc<Pool>) -> io::Result<()> {
        let kept = self.engine.with_journal(|journal| journal.warm_sets())?;
        let actors: BTreeSet<&String> = kept.iter().map(|(actor, _)| actor).collect();
        for actor in actors {
            let all = kept.iter().filter(|(of, _)| of == actor);
            let (records, members): (Vec<String>, Vec<Arc<Member>>) = all
                .clone()
                .filter_map(|(_, record)| Some((record.clone(), self.restore_member(gpu, actor, record)?)))
                .unzip();
            if records.len() < all.count() {
                self.engine.with_journal(|journal| journal.replace_warm_set(actor, &records))?;
            }
            gpu.set_members(actor, members);
        }
        // A previous run's executors still exiting fence it, as they fence requests.
        let (fence, started) = (self.startup_gpu_births.lock().unwrap().clone(), self.started_ticks);
        let (gpu, engine) = (gpu.clone(), self.engine.clone());
        std::thread::Builder::new().name("warm-set-restore".into()).spawn(move || {
            let mut seen = engine.activity_epoch();
            let alive = |birth: &Birth| {
                !birth.ended().unwrap_or(false) || crate::process::group_outlives(birth, started)
            };
            while fence.iter().any(alive) {
                seen = engine.wait_activity(seen, Some(std::time::Duration::from_millis(100)));
            }
            gpu.keep_all(&engine);
        })?;
        Ok(())
    }

    /// One kept member, with its plan as it was bound while the store still holds its weights.
    fn restore_member(&self, gpu: &Pool, actor: &str, record: &str) -> Option<Arc<Member>> {
        let kept: KeptMember = serde_json::from_str(record).ok()?;
        let installed = self.engine.installation(actor, &kept.installation).ok()??;
        let held = self.catalog.resolve(&installed.generation).ok()?;
        let bound = match kept.preparation.as_str() {
            "" => None,
            id => self.engine.preparation(actor, id).ok()?,
        };
        let plan = bound.and_then(|bound| gpu.plan(&bound).ok()).filter(|plan| gpu.holds(plan));
        Some(Member::new(held, plan, Level::named(&kept.level)?, None))
    }

    pub fn submit(
        &self,
        key: &str,
        generation: &str,
        entrypoint: &str,
        input: Value,
    ) -> io::Result<Execution> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Err(io::Error::other("machine is stopping"));
        }
        let held = self.catalog.resolve(generation)?;
        let mut invocation = held.invocation(entrypoint, input)?;
        invocation.degree = self.weightless_degree(&held, &invocation)?;
        let record = self.engine.submit(key, invocation)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    pub fn submit_public(
        &self,
        context: SubmissionContext,
        generation: &str,
        call: Call,
        boot: &str,
    ) -> io::Result<Execution> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Err(io::Error::other("machine is stopping"));
        }
        let held = self.catalog.resolve(generation)?;
        let mut invocation = held.invocation(&call.entrypoint, call.input)?;
        invocation.attention_kernel = call.attention_kernel;
        invocation.gpus = call.gpus;
        invocation.bits = call.bits;
        invocation.inputs = call.inputs;
        invocation.degree = self.weightless_degree(&held, &invocation)?;
        let record = self
            .engine
            .submit_public_on_boot(context, invocation, boot)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    /// A run prepared inside itself (`runs`) names its generation, call and model plan, and how
    /// the machine memo met it; one the memo answered (`answer`) completes with that result.
    pub fn bind_prepared(
        &self,
        id: &str,
        generation: &str,
        call: Call,
        preparation: &str,
        memo: crate::journal::CallMemo,
        answer: Option<crate::journal::ResultRecord>,
    ) -> io::Result<Execution> {
        let held = self.catalog.resolve(generation)?;
        let mut invocation = held.invocation_of(&call.application, &call.entrypoint, call.input)?;
        invocation.attention_kernel = call.attention_kernel;
        invocation.gpus = call.gpus;
        invocation.bits = call.bits;
        invocation.inputs = call.inputs;
        invocation.degree = call.degree;
        invocation.parent = call.parent;
        let record = self.engine.bind_prepared(id, invocation, preparation, memo, answer)?;
        self.retain(&record, held.retention());
        Ok(record)
    }
    /// A call submitted without a preparation constructs nothing: its declaration's degree.
    fn weightless_degree(&self, held: &crate::catalog::HeldGeneration, invocation: &crate::journal::Invocation) -> io::Result<u32> {
        let (_, _, interface) = held.record.app(&invocation.module).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, "this environment holds no such application")
        })?;
        let entry = interface["entrypoints"].as_array()
            .and_then(|rows| rows.iter().find(|row| row["name"] == invocation.entrypoint.as_str()))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "this application declares no such entrypoint"))?;
        self.pool().map_or(Ok(0), |pool| pool.degree(entry, invocation.gpus))
    }
    fn retain(&self, record: &Execution, hold: Arc<File>) {
        if !record.state.terminal() {
            self.retained
                .lock()
                .unwrap()
                .insert(record.id.clone(), hold);
        }
    }
    /// An install or GPU configuration changed what can dispatch.
    pub fn changed_environment(&self) -> io::Result<()> {
        self.engine.notify_activity();
        Ok(())
    }
    pub fn idle(&self) -> io::Result<bool> {
        Ok(
            self.preparing.load(std::sync::atomic::Ordering::Acquire) == 0
                && self.engine.nonterminal(1)?.is_empty(),
        )
    }
    /// Nothing a restart would lose: no run started, no preparation in flight, and nothing waiting
    /// that could not prepare again. An update waits for this; queued runs run after it.
    pub fn settled(&self) -> io::Result<bool> {
        Ok(self.preparing.load(std::sync::atomic::Ordering::Acquire) == 0 && !self.engine.with_journal(|journal| journal.unsettled())?)
    }
    pub fn hold_while(&self, held: impl Fn() -> bool + Send + Sync + 'static) {
        let _ = self.held.set(Box::new(held));
    }
    /// What a rental does when a preparation fails `no_progress`: tells its Hub, which marks the
    /// offer it was bought on suspect.
    pub fn on_stall(&self, stall: impl Fn() + Send + Sync + 'static) {
        let _ = self.stall.set(Box::new(stall));
    }
    pub(crate) fn stalled(&self) {
        if let Some(stall) = self.stall.get() {
            stall();
        }
    }
    pub fn preparing(self: &Arc<Self>) -> Preparing {
        self.preparing
            .fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Preparing(self.clone())
    }
    pub fn stop(&self) -> io::Result<bool> {
        let mut stopped = self.stopped.lock().unwrap();
        if !self.idle()? {
            return Ok(false);
        }
        *stopped = true;
        if let Some(pool) = self.pool() {
            pool.stop()?;
        }
        self.engine.notify_activity();
        Ok(true)
    }
    fn dispatch_loop(self: Arc<Self>) {
        loop {
            let epoch = self.engine.activity_epoch();
            if *self.stopped.lock().unwrap() {
                return;
            }
            if let Err(error) = self.engine.reconcile() {
                tracing::warn!("executor birth observation: {error}");
            }
            self.retained.lock().unwrap().retain(|id, _| {
                self.engine
                    .get(id)
                    .map(|r| !r.state.terminal())
                    .unwrap_or(true)
            });
            if let Err(error) = self.dispatch_ready() {
                tracing::warn!("dispatch observation: {error}");
            }
            // Dispatch wakes on events only: an arrival, an exit, a cancel, freed room.
            self.engine.wait_activity(epoch, None);
        }
    }
    /// Caches expire by their TTL on a thread of their own, apart from dispatch.
    fn reclaim_loop(weak: std::sync::Weak<Self>) {
        loop {
            std::thread::sleep(SWEEP_EVERY);
            match weak.upgrade() {
                Some(service) if !*service.stopped.lock().unwrap() => drop(service.reclaim()),
                _ => return,
            }
        }
    }
    /// Caches manage themselves (`reclaim`): a TTL and a low disk, never a purge verb.
    pub fn reclaim(&self) -> crate::reclaim::Swept {
        // Run objects no unfinished run needs go first.
        if let Some(calls) = self.calls() {
            if let Err(error) = calls.sweep_objects() {
                tracing::warn!("reclaim object roots: {error}");
            }
        }
        let sweep = || -> io::Result<crate::reclaim::Swept> {
            let pool = self.pool();
            let bound = self.engine.bound_generations()?;
            let kernels = self.gpu().map(|gpu| gpu.kernel_caches());
            let publisher = self.publisher.lock().unwrap().clone();
            let uv_cache = publisher
                .as_ref()
                .and_then(|publisher| publisher.uv_cache());
            crate::reclaim::sweep(&crate::reclaim::Caches {
                engine: &self.engine,
                catalog: &self.catalog,
                bound: &bound,
                kernels: kernels.as_ref(),
                memo: pool.as_ref().map(|pool| pool.memo()),
                uv_cache: uv_cache.as_deref(),
                disk: &|| crate::reclaim::measure(&self.engine.root),
            })
        };
        match sweep() {
            Ok(swept) => {
                if swept != crate::reclaim::Swept::default() {
                    tracing::info!("reclaimed {swept:?}");
                }
                swept
            }
            Err(error) => {
                tracing::warn!("reclaim: {error}");
                crate::reclaim::Swept::default()
            }
        }
    }
    /// One pass over the ready runs in the queue's order (`queue`). An older request with a
    /// ready GPU step keeps the GPUs it waits for (`claims`) from later offers (Slurm). A deviceless
    /// call takes CPU room; one waiting on its own call takes none. Each root left queued shows
    /// its place in line.
    fn dispatch_ready(&self) -> io::Result<()> {
        let stopped = self.stopped.lock().unwrap();
        if *stopped {
            return Ok(());
        }
        let Some(pool) = self.pool() else {
            return Ok(());
        };
        // Orphan exact process births remain reservations; a new owner does not
        // pretend they are free merely because it has no local supervisor.
        let active = self.engine.active(usize::MAX)?;
        let gpu = |record: &Execution| record.invocation.degree > 0;
        // Every GPU call slot is a permit of this machine run; an earlier run's process still
        // alive holds devices nobody here can name, so it fences them all.
        let gpu_fenced = self.gpu_startup_fences() != 0
            || active.iter().any(|r| gpu(r) && !self.engine.owns(&r.id));
        let mut claims = vec![None; pool.width()];
        let calls = self.calls();
        let waits = |r: &Execution| calls.as_ref().is_some_and(|calls| calls.waiting(&self.engine, &r.id));
        let mut room = self.parallelism.saturating_sub(
            active.iter().filter(|r| !gpu(r) && !waits(r)).count(),
        );
        // A run whose plan cannot be read is its own model; its offer reports why.
        let model = |r: &Execution| self.plan(&pool, r).map_or_else(|_| r.id.clone(), |plan| plan.construction);
        let ready = self.queue.lock().unwrap().order(self.engine.ready()?, model);
        let (mut place, held) = (0, self.held.get().is_some_and(|held| held()));
        for record in ready {
            let invocation = &record.invocation;
            let full = room == 0 && (invocation.degree == 0 || claims.iter().all(Option::is_some));
            let root = invocation.parent.is_empty();
            if !(full || held && root || invocation.degree > 0 && gpu_fenced) && self.offer(&pool, &record, &mut claims, calls.clone())? {
                room -= usize::from(invocation.degree == 0);
                if root {
                    self.queue.lock().unwrap().started(&record);
                }
            } else if root {
                place += 1;
                let after = if held { ", after the machine's update" } else { "" };
                let shown = self.engine.waiting(&record.id, Some(format!("position {place} in the queue{after}")));
                shown.unwrap_or_else(|error| tracing::warn!("run {}: its place was not shown: {error}", record.id));
            }
        }
        Ok(())
    }
    /// The plan `record` runs: its journaled preparation's, else a weightless call's.
    fn plan(&self, pool: &Pool, record: &Execution) -> io::Result<Plan> {
        let invocation = &record.invocation;
        match record.submission.as_ref().map(|s| s.preparation_id.as_str()) {
            None | Some("") => Ok(Plan::weightless(&invocation.generation, &invocation.module, &invocation.entrypoint, invocation.degree)),
            Some(id) => {
                let actor = record.submission.as_ref().map(|s| s.actor.as_str()).unwrap_or_default();
                let preparation = self.engine.preparation(actor, id)?.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "accepted preparation is absent")
                })?;
                pool.plan(&preparation)
            }
        }
    }
    /// Start `record` if it can start now: false leaves it queued (or failed, when its
    /// environment is gone).
    fn offer(&self, pool: &Arc<Pool>, record: &Execution, claims: &mut [Option<u64>], calls: Option<Arc<crate::calls::Calls>>) -> io::Result<bool> {
        let held = match self.catalog.resolve(&record.invocation.generation) {
            Ok(held) => held,
            Err(error) => {
                self.engine.fail_unstarted(&record.id, "environment_unavailable",
                    format!("held generation unavailable: {error}"))?;
                return Ok(false);
            }
        };
        if !held.record.owns(&record.invocation.module, &record.invocation.package) {
            self.engine.fail_unstarted(&record.id, "environment_unavailable",
                "held package identity differs from accepted invocation")?;
            return Ok(false);
        }
        let plan = self.plan(pool, record)?;
        pool.dispatch(&self.engine, record, held.application(&record.invocation.module)?, plan, claims, calls)
    }
}

#[cfg(test)]
mod preparing_tests {
    use super::*;

    #[test]
    fn a_preparation_in_flight_keeps_the_machine_busy() {
        let root = std::env::temp_dir().join(format!("service-preparing-{}", uuid::Uuid::new_v4()));
        let service = Service::open(&root, &root.join("generations"), 1).unwrap();
        assert!(service.idle().unwrap());
        let preparing = service.preparing();
        assert!(
            !service.idle().unwrap(),
            "a rental must not release itself mid-install"
        );
        drop(preparing);
        assert!(service.idle().unwrap());
        assert!(service.stop().unwrap());
        let _ = std::fs::remove_dir_all(root);
    }
}
