//! What a call's weights are prefetched under (one-request-kind §2.7). TensorD chooses each
//! slot's checkpoint, lands its structure and wants its chunks ahead of the call; the Runtime
//! reads them through TensorFS. A release is read anonymously, the owner's checkpoint by hash
//! under the run's capability. Its checkpoints stay open, so the store keeps their bytes, until
//! the call ends (or its run is canceled): then what it wanted drops.
use crate::{
    held_models::Pulled,
    hub,
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

/// Where `header`, a checkpoint this machine pulled from `source`'s Hub, comes from: the Hub
/// names its tree as the store's ref records (`held_models::Pulled`). None for one never
/// pulled (local).
pub fn remote(machine: &Machine, source: &hub::Source, header: &ObjectRef) -> io::Result<Option<Remote>> {
    let Some(pulled) = Pulled::of(machine.store(), header)? else {
        return Ok(None);
    };
    let found = origin::remote(machine, source, &pulled.repository, &pulled.release, &pulled.lane, &pulled.tree);
    found.map(|(remote, _)| Some(remote)).map_err(|(_, why)| io::Error::other(why))
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
    let code = match refusal.code {
        Code::CAPACITY_EXHAUSTED => "machine_disk_full",
        _ => "model_fetch_failed",
    };
    io::Error::other(format!("{code}: {refusal}"))
}

/// One call's fetches: its checkpoints, where each comes from, and what it wants held.
pub struct Fetch {
    machine: Arc<Machine>,
    hub: Option<hub::Source>,
    /// Its schedules' order: the root run's acceptance, or after every run (no run).
    order: u64,
    stop: Stop,
    kept: Mutex<Vec<tokio::task::JoinHandle<tensorfs::err::Result<tensorfs::fetch::Landed>>>>,
    /// Its checkpoints, open until it ends: the store keeps their bytes.
    held: Mutex<Vec<Checkpoint>>,
    /// What fetched for this call before it started: ended with it.
    replaced: Mutex<Option<Arc<Fetch>>>,
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
        })
    }

    fn store(&self) -> &Arc<Store> {
        self.machine.store()
    }

    /// Where the absent objects of `header` come from.
    fn source(&self, header: &ObjectRef) -> io::Result<Option<Remote>> {
        let Some(hub) = self.hub.as_ref() else {
            return Ok(None);
        };
        remote(&self.machine, hub, header)
    }

    /// `previous` (a run's preparation's fetches) ends with the call.
    pub fn replaces(&self, previous: Option<Arc<Fetch>>) {
        *self.replaced.lock().unwrap() = previous;
    }

    /// Want each of `selected`'s (header, components) components next, before the call's
    /// executor reads them, in Runtime's Start-pass order: the smallest component first. Held
    /// until the call ends.
    pub fn ahead(&self, selected: &[(String, Vec<String>)]) -> io::Result<()> {
        let mut wanted = vec![];
        for (header, components) in selected {
            let checkpoint = checkpoint(self.store(), header)?;
            let Some(remote) = self.source(checkpoint.id())? else {
                continue;
            };
            for component in components {
                let objects = storage::chunks(&checkpoint, std::slice::from_ref(component));
                let bytes: u64 = objects.iter().map(|o| o.length).sum();
                wanted.push((bytes, remote.clone(), objects));
            }
            self.held.lock().unwrap().push(checkpoint);
        }
        wanted.sort_by_key(|(bytes, ..)| *bytes);
        let mut kept = self.kept.lock().unwrap();
        for (rank, (_, remote, objects)) in wanted.into_iter().enumerate() {
            kept.push(self.machine.want(&remote, objects, priority::next(self.order, rank as u32)).map_err(failed)?);
        }
        Ok(())
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
    /// of a function): the bytes that came from the Hub and from the datacenter volume.
    pub fn download(&self, selected: &[(String, Vec<String>)], stop: &Stop, progress: &(dyn Fn(u64, u64) + Sync)) -> io::Result<(u64, u64)> {
        let counted = Arc::new(tensorfs::fetch::Progress::default());
        origin::watched(&counted, progress, || {
            let mut landed = (0, 0);
            for (header, components) in selected {
                let held = storage::held(self.store(), header)?;
                let Some(remote) = self.source(&held)? else {
                    continue;
                };
                let components = (!components.is_empty()).then_some(components.as_slice());
                let pulled = self.machine.pull(&remote, &held, components, priority::BEHIND, Some(counted.clone()), stop).map_err(failed)?;
                landed = (landed.0 + pulled.landed.origin, landed.1 + pulled.landed.volume);
            }
            Ok(landed)
        })
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

impl Drop for Fetch {
    fn drop(&mut self) {
        self.end();
    }
}
