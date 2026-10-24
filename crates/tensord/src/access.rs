//! What a call's weights are prefetched under (one-request-kind §2.7). TensorD chooses each
//! slot's checkpoint, lands its structure and wants its chunks ahead of the call; the Runtime
//! reads them through TensorFS, at the run's Hub under its owner's newest bearer when its request
//! carried one.
//! Its checkpoints stay open, so the store keeps their bytes, until
//! the call ends (or its run is canceled): then what it wanted drops. While what it wants is not
//! all here, its run's progress says so (`meter`).
use crate::{
    execution::Engine,
    held_models::Pulled,
    hub,
    journal::Weights,
    origin::{self, priority, Machine, Remote},
    stop::Stop,
    storage::{self, Store},
};
use std::{
    collections::BTreeSet,
    io,
    sync::{Arc, Mutex},
};
use tensorfs::{
    err::{Code, Refusal},
    format::ObjectRef,
    store::Checkpoint,
};

/// Where `header`, a checkpoint this machine pulled from a Hub (`held_models::Pulled`), comes
/// from: `source`'s Hub. None for one never pulled (local).
pub fn remote(machine: &Machine, source: &hub::Source, header: &ObjectRef) -> io::Result<Option<Remote>> {
    if Pulled::of(machine.store(), header)?.is_none() {
        return Ok(None);
    }
    origin::remote(machine, source).map(Some).map_err(io::Error::other)
}

/// The checkpoint `id` (`sha256:<hex>`), its structure here.
pub fn checkpoint(store: &Store, id: &str) -> io::Result<Checkpoint> {
    storage::checkpoint(store, id)
}

/// Whether every chunk of `components` of `header` is here (`downloaded`). Once true it is
/// remembered (Status asks every second); a later eviction is fetched again by its reader.
pub fn whole(store: &Store, header: &str, components: &[String]) -> bool {
    static WHOLE: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    let key = format!("{}\0{header}\0{components:?}", store.root().display());
    if WHOLE.lock().unwrap().contains(&key) {
        return true;
    }
    let whole = checkpoint(store, header)
        .is_ok_and(|ck| storage::chunks(&ck, components).iter().all(|o| storage::present(store, o)));
    if whole {
        WHOLE.lock().unwrap().insert(key);
    }
    whole
}

pub fn failed(refusal: Refusal) -> io::Error {
    let failure = match refusal.code {
        Code::CAPACITY_EXHAUSTED => format!("machine_disk_full: {refusal}"),
        Code::CREDENTIAL_REQUIRED => format!("hub_credential_refused: {}", refusal.detail),
        _ => format!("model_fetch_failed: {refusal}"),
    };
    io::Error::other(failure)
}

/// One call's fetches: its checkpoints, where each comes from, and what it wants held.
pub struct Fetch {
    machine: Arc<Machine>,
    /// Where its checkpoints come from.
    hub: Option<hub::Source>,
    /// Its schedules' order: the root run's acceptance, or after every run (no run).
    order: u64,
    stop: Stop,
    kept: Mutex<Vec<tokio::task::JoinHandle<tensorfs::err::Result<tensorfs::fetch::Landed>>>>,
    /// Its checkpoints, open until it ends: the store keeps their bytes.
    held: Mutex<Vec<Checkpoint>>,
    /// What fetched for this call before it started: ended with it.
    replaced: Mutex<Option<Arc<Fetch>>>,
    /// Each wanted checkpoint not all here when it was wanted: its repository, its bytes, and
    /// its objects still absent.
    arriving: Mutex<Vec<(String, u64, Vec<ObjectRef>)>>,
}

impl Fetch {
    /// Checkpoints fetched from `hub` for a run ordered `order`.
    pub fn new(machine: Arc<Machine>, hub: Option<hub::Source>, order: u64) -> Arc<Self> {
        Arc::new(Self {
            machine,
            hub,
            order,
            stop: Stop::default(),
            kept: Mutex::default(),
            held: Mutex::default(),
            replaced: Mutex::default(),
            arriving: Mutex::default(),
        })
    }

    fn store(&self) -> &Arc<Store> {
        self.machine.store()
    }

    /// Where the absent objects of `header` come from.
    fn source(&self, header: &ObjectRef) -> io::Result<Option<Remote>> {
        let Some(hub) = &self.hub else {
            return Ok(None);
        };
        remote(&self.machine, hub, header)
    }

    /// `previous` (a run's preparation's fetches) ends with the call.
    pub fn replaces(&self, previous: Option<Arc<Fetch>>) {
        *self.replaced.lock().unwrap() = previous;
    }

    /// Want each of `selected`'s (header, components) components next, before the call's
    /// executor reads them, in Runtime's Start-pass order: the smallest component first; then
    /// `later`'s (its on-demand models, built when a stage first uses one) the same way, right
    /// behind them. Held until the call ends. Its run's progress waits on `selected` only.
    pub fn ahead(&self, selected: &[(String, Vec<String>)], later: &[(String, Vec<String>)]) -> io::Result<()> {
        let mut wanted = self.components(selected, true)?;
        // A model the call may never read must not cost it the ones it reads first.
        let mut behind = self.components(later, false).unwrap_or_else(|error| {
            tracing::warn!("its on-demand models' early fetch: {error}");
            vec![]
        });
        wanted.sort_by_key(|(bytes, ..)| *bytes);
        behind.sort_by_key(|(bytes, ..)| *bytes);
        let mut kept = self.kept.lock().unwrap();
        for (rank, (_, remote, objects)) in wanted.into_iter().chain(behind).enumerate() {
            kept.push(self.machine.want(&remote, objects, priority::next(self.order, rank as u32)).map_err(failed)?);
        }
        Ok(())
    }

    /// Each component of `selected` (header, components) from a Hub: its bytes, origin and
    /// chunks, its checkpoint held until the call ends. With `metered`, what is absent now is
    /// what its run waits on (`meter`), under the repository it came from.
    fn components(&self, selected: &[(String, Vec<String>)], metered: bool) -> io::Result<Vec<(u64, Remote, Vec<ObjectRef>)>> {
        let mut wanted = vec![];
        for (header, components) in selected {
            let checkpoint = checkpoint(self.store(), header)?;
            let Some(remote) = self.source(checkpoint.id())? else {
                continue;
            };
            // A checkpoint known whole is looked at no more.
            let objects = match !metered || components.is_empty() || whole(self.store(), header, components) {
                true => vec![],
                false => storage::chunks(&checkpoint, components),
            };
            let absent: Vec<_> = objects.iter().filter(|o| !storage::present(self.store(), o)).cloned().collect();
            if let Some(pulled) = (!absent.is_empty()).then(|| Pulled::of(self.store(), checkpoint.id())).transpose()?.flatten() {
                let bytes = objects.iter().map(|o| o.length).sum();
                self.arriving.lock().unwrap().push((pulled.repository, bytes, absent));
            }
            for component in components {
                let objects = storage::chunks(&checkpoint, std::slice::from_ref(component));
                let bytes: u64 = objects.iter().map(|o| o.length).sum();
                wanted.push((bytes, remote.clone(), objects));
            }
            self.held.lock().unwrap().push(checkpoint);
        }
        Ok(wanted)
    }

    /// Want `components` of each of `selected` (header, components) behind everything, held
    /// until the call ends: a warm set member's `downloaded`.
    pub fn keep(&self, selected: &[(String, Vec<String>)]) -> io::Result<()> {
        let mut kept = self.kept.lock().unwrap();
        if !kept.is_empty() {
            return Ok(());
        }
        for (header, components) in selected {
            let checkpoint = checkpoint(self.store(), header)?;
            if let Some(remote) = self.source(checkpoint.id())? {
                let objects = storage::chunks(&checkpoint, components);
                kept.push(self.machine.want(&remote, objects, priority::BEHIND).map_err(failed)?);
            }
            self.held.lock().unwrap().push(checkpoint);
        }
        Ok(())
    }

    /// Every chunk of `selected` (header, components) here, reporting its bytes (a warm run
    /// of a function): those that came from the Hub and from the datacenter volume, and those held.
    pub fn download(&self, selected: &[(String, Vec<String>)], stop: &Stop, progress: &(dyn Fn(u64, u64) + Sync)) -> io::Result<tensorfs::fetch::Landed> {
        let counted = Arc::new(tensorfs::fetch::Progress::default());
        origin::watched(&counted, progress, || {
            let mut landed = tensorfs::fetch::Landed::default();
            for (header, components) in selected {
                let held = storage::held(self.store(), header)?;
                let Some(remote) = self.source(&held)? else {
                    continue;
                };
                let components = (!components.is_empty()).then_some(components.as_slice());
                let pulled = self.machine.pull(&remote, &held, components, priority::BEHIND, Some(counted.clone()), stop).map_err(failed)?;
                (landed.origin, landed.volume, landed.local) =
                    (landed.origin + pulled.landed.origin, landed.volume + pulled.landed.volume, landed.local + pulled.landed.local);
            }
            Ok(landed)
        })
    }

    /// The weights it wants that are not all here yet, while any are not: their repositories,
    /// and the bytes here of all it found absent. An object here stays: its checkpoint is held.
    pub fn arriving(&self) -> Option<Weights> {
        let mut arriving = self.arriving.lock().unwrap();
        let (mut models, mut done, mut total) = (vec![], 0, 0);
        for (repository, bytes, absent) in arriving.iter_mut() {
            absent.retain(|object| !storage::present(self.store(), object));
            let missing: u64 = absent.iter().map(|o| o.length).sum();
            if missing > 0 && !models.contains(repository) {
                models.push(repository.clone());
            }
            (done, total) = (done + *bytes - missing, total + *bytes);
        }
        (!models.is_empty()).then(|| Weights { models: models.join(" and "), done, total })
    }

    /// Whether it ended (`end`).
    pub fn ended(&self) -> bool {
        self.stop.is_cancelled()
    }

    /// Everything it wanted drops and its checkpoints close.
    pub fn end(&self) {
        self.stop.cancel();
        for want in self.kept.lock().unwrap().drain(..) {
            want.abort();
        }
        self.held.lock().unwrap().clear();
        if let Some(replaced) = self.replaced.lock().unwrap().take() {
            replaced.end();
        }
    }
}

/// While what `fetch` wants is not all here, run `id`'s progress says which weights it waits on
/// and how many of their bytes are here, from a quarter-second look at the store; once they are,
/// or the fetch ends, its own stage shows again. A run whose weights are here hears nothing.
pub fn meter(engine: Arc<Engine>, id: String, fetch: &Arc<Fetch>) {
    if fetch.arriving.lock().unwrap().is_empty() {
        return;
    }
    let (store, fetch, began) = (fetch.store().clone(), Arc::downgrade(fetch), std::time::Instant::now());
    let started = std::thread::Builder::new().name("weights-meter".into()).spawn(move || {
        let mut last = None;
        loop {
            let live = fetch.upgrade().filter(|f| !f.ended());
            let weights = live.as_ref().and_then(|f| f.arriving());
            if let Err(error) = engine.observe_weights(&id, weights.clone()) {
                tracing::warn!("run {id}: its weights' progress: {error}");
                return;
            }
            // What the run waited for lands in the transport log, as a pull's does.
            let Some(arriving) = weights else {
                if let (Some(Weights { models, total, .. }), true) = (last, live.is_some()) {
                    let waited = format!("bytes={total} seconds={:.1} run={id}", began.elapsed().as_secs_f64());
                    storage::transport(&store, "arrived", &models, &waited);
                }
                return;
            };
            if engine.get(&id).is_ok_and(|record| record.state.terminal()) {
                return;
            }
            drop(live);
            last = Some(arriving);
            std::thread::sleep(std::time::Duration::from_millis(250));
        }
    });
    if let Err(error) = started {
        tracing::warn!("its weights' progress: {error}");
    }
}

impl Drop for Fetch {
    fn drop(&mut self) {
        self.end();
    }
}
