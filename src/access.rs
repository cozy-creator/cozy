//! What a call's weights are fetched under (one-request-kind §2.7). TensorD chooses each
//! slot's checkpoint and lands its structure; Runtime says which objects come and when, through
//! a channel scoped to the call's checkpoints, and the machine's own readers wait for what they
//! read. A release is read anonymously, the owner's checkpoint by hash under the run's
//! capability. Credentials stay in this process, with the call that holds them: when it ends
//! (or its run is canceled) its waits fail, its channels close and their tickets drop.
use crate::hub;
use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    io,
    net::Shutdown,
    os::{fd::OwnedFd, unix::net::UnixStream},
    sync::{Arc, Mutex},
};
use tensorfs_core::{
    err::{Code, Refusal},
    fetch::{self, Access, Fetcher, Origin, Priority, Ticket},
    header::{Body, Header},
    ids::ObjectRef,
    store::Store,
    transport::{CredentialProvider, PullCancellation, SourcePolicy},
};

impl Access for hub::Catalog {
    fn credential(&self) -> &dyn CredentialProvider {
        hub::Catalog::credential(self)
    }
    fn policy(&self) -> &SourcePolicy {
        hub::Catalog::policy(self)
    }
}

/// `repository` at `source`'s Hub: a release's lane, or (no release) the owner's checkpoint by
/// hash.
pub fn origin(source: &hub::Source, repository: &str, release: &str, lane: &str, manifest: &ObjectRef) -> Origin {
    let refspec = match release.is_empty() {
        true => format!("{repository}@{}", manifest.id()),
        false => format!("{repository}@{release}"),
    };
    Origin { hub: source.origin.clone(), refspec, lane: lane.into(), manifest: manifest.clone() }
}

/// What fetches of `origin` present at `source`: the run's capability for the owner's
/// checkpoint by hash when it names that read, else nothing. Nothing is traded until a fetch
/// needs it.
pub fn of(source: &hub::Source, origin: &Origin) -> Result<Arc<hub::Catalog>, (&'static str, String)> {
    let catalog = match origin.refspec.split_once("@sha256:") {
        Some((model, _)) => hub::Catalog::scoped(source, hub::Op::Read { model, manifest: &origin.manifest.id() })?,
        None => hub::Catalog::new(source).map_err(|e| ("catalog_read_failed", e.0))?,
    };
    Ok(Arc::new(catalog))
}

/// The manifest `id` (`sha256:<hex>`) as the store holds it.
pub fn held(store: &Store, id: &str) -> io::Result<ObjectRef> {
    let sha256 = id.trim_start_matches("sha256:").to_string();
    let length = std::fs::metadata(store.manifest_path(&sha256))?.len();
    Ok(ObjectRef { sha256, length })
}

/// The header of a checkpoint whose structure is here.
pub fn header(store: &Store, manifest: &ObjectRef) -> io::Result<Header> {
    let held = store.read_manifest(manifest).map_err(io::Error::other)?;
    let header = held.header().ok_or_else(|| io::Error::other(format!("{} has no model header", manifest.id())))?;
    tensorfs_core::checkpoint::load_header(store, header).map_err(io::Error::other)
}

/// Every object `components` of `header` read (all of them when empty), each once.
pub fn objects(header: &Header, components: &[String]) -> Vec<ObjectRef> {
    let mut objects: Vec<ObjectRef> = (header.components.iter())
        .filter(|(name, _)| components.is_empty() || components.contains(name))
        .flat_map(|(_, tensors)| tensors.iter().flat_map(|(_, tensor)| tensor.parts.iter()))
        .filter_map(|(_, part)| match &part.body {
            Body::Segments(segments) => Some(segments.iter().cloned()),
            Body::Inline(_) => None,
        })
        .flatten()
        .collect();
    objects.sort_by(|a, b| a.sha256.cmp(&b.sha256));
    objects.dedup();
    objects
}

/// Whether every object of `components` of `manifest` is here (`downloaded`). Once true it is
/// remembered (Status asks every second); a later eviction is fetched again by its reader.
pub fn whole(store: &Store, manifest: &str, components: &[String]) -> bool {
    static WHOLE: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());
    let key = format!("{}\0{manifest}\0{components:?}", store.root().display());
    if WHOLE.lock().unwrap().contains(&key) {
        return true;
    }
    let read = || -> io::Result<bool> {
        let header = header(store, &held(store, manifest)?)?;
        Ok(objects(&header, components).iter().all(|o| store.contains(&o.sha256)))
    };
    let whole = read().unwrap_or(false);
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

/// What the derived channel fetches a source with: the fetcher, the access, the call's stop.
pub type Fetching = (Fetcher, Arc<dyn Access>, PullCancellation);

/// One call's fetches: its checkpoints, the access each is fetched under, and what it holds.
pub struct Fetch {
    store: Arc<Store>,
    hub: Option<hub::Source>,
    /// Its schedules' order: the root run's acceptance, or after every run (no run).
    order: u64,
    manifests: Vec<String>,
    accesses: Mutex<HashMap<Origin, Arc<dyn Access>>>,
    stop: PullCancellation,
    channels: Mutex<Vec<UnixStream>>,
    kept: Mutex<Vec<Ticket>>,
}

impl Fetch {
    /// `manifests` (`sha256:<hex>`) fetched from `hub` for a run ordered `order`.
    pub fn new(store: Arc<Store>, hub: Option<hub::Source>, manifests: Vec<String>, order: u64) -> Arc<Self> {
        Arc::new(Self {
            store,
            hub,
            order,
            manifests,
            accesses: Mutex::default(),
            stop: PullCancellation::default(),
            channels: Mutex::default(),
            kept: Mutex::default(),
        })
    }

    fn fetcher(&self) -> io::Result<Fetcher> {
        Fetcher::of(&self.store).map_err(io::Error::other)
    }

    /// Where the absent objects of `manifest` come from, and under what.
    fn source(&self, manifest: &ObjectRef) -> io::Result<Option<(Origin, Arc<dyn Access>)>> {
        let Some(origin) = tensorfs_core::origin::of(self.store.root(), manifest).map_err(io::Error::other)? else {
            return Ok(None);
        };
        let hub = self.hub.as_ref().ok_or_else(|| {
            io::Error::other(format!("model_fetch_failed: {} is here in part and this call names no Hub", manifest.id()))
        })?;
        let mut accesses = self.accesses.lock().unwrap();
        let access = match accesses.get(&origin) {
            Some(access) => access.clone(),
            None => {
                let access: Arc<dyn Access> = of(hub, &origin).map_err(|(_, why)| io::Error::other(why))?;
                accesses.entry(origin.clone()).or_insert(access).clone()
            }
        };
        Ok(Some((origin, access)))
    }

    /// Wait until `objects` of `manifest` are here: a reader is blocked on them (Now).
    pub fn wait(&self, manifest: &ObjectRef, objects: &[ObjectRef]) -> io::Result<()> {
        let absent: Vec<ObjectRef> = objects.iter().filter(|o| !self.store.contains(&o.sha256)).cloned().collect();
        if absent.is_empty() {
            return Ok(());
        }
        let (origin, access) = self.source(manifest)?.ok_or_else(|| {
            io::Error::other(format!("model_fetch_failed: {} lacks {} and no Hub holds it", manifest.id(), absent[0].id()))
        })?;
        let ticket = self.fetcher()?.want(&origin, access, &[], Priority::Now);
        ticket.wait(&absent, &self.stop).map_err(failed)
    }

    /// The derived channel's fetch of `manifest` (a `Weights[M]` source held in part).
    pub fn fetching(&self, manifest: &ObjectRef) -> io::Result<Option<Fetching>> {
        let Some((_, access)) = self.source(manifest)? else {
            return Ok(None);
        };
        Ok(Some((self.fetcher()?, access, self.stop.clone())))
    }

    /// A fresh executor channel over the call's checkpoints held in part (`fetch::serve`): the
    /// end to hand over. A checkpoint held whole is not on it: nothing of it is fetched.
    pub fn channel(&self) -> io::Result<File> {
        let mut origins = vec![];
        for manifest in &self.manifests {
            if let Some((origin, access)) = held(&self.store, manifest).and_then(|m| self.source(&m))? {
                if !origins.iter().any(|(o, _): &(Origin, Arc<dyn Access>)| *o == origin) {
                    origins.push((origin, access));
                }
            }
        }
        let (client, server) = UnixStream::pair()?;
        let mut channels = self.channels.lock().unwrap();
        if self.stop.is_cancelled() {
            return Err(io::Error::other("the call ended"));
        }
        channels.push(server.try_clone()?);
        let (fetcher, grant) = (self.fetcher()?, fetch::Grant { run: self.order, origins, stop: PullCancellation::default() });
        std::thread::Builder::new().name("fetch-channel".into()).spawn(move || {
            if let Err(refusal) = fetch::serve(&fetcher, grant, server) {
                eprintln!("fetch channel: {refusal}");
            }
        })?;
        Ok(File::from(OwnedFd::from(client)))
    }

    /// Want `components` of each of `selected` (manifest, components) at Behind, held until the
    /// call ends: a warm set member's `downloaded`.
    pub fn keep(&self, selected: &[(String, Vec<String>)]) -> io::Result<()> {
        let mut kept = self.kept.lock().unwrap();
        if !kept.is_empty() {
            return Ok(());
        }
        for (manifest, components) in selected {
            let manifest = held(&self.store, manifest)?;
            if let Some((origin, access)) = self.source(&manifest)? {
                let objects = objects(&header(&self.store, &manifest)?, components);
                kept.push(self.fetcher()?.want(&origin, access, &objects, Priority::Behind));
            }
        }
        Ok(())
    }

    /// Its waits fail, its channels close and every ticket it holds drops.
    pub fn end(&self) {
        let mut channels = self.channels.lock().unwrap();
        self.stop.cancel();
        for channel in channels.drain(..) {
            let _ = channel.shutdown(Shutdown::Both);
        }
        self.kept.lock().unwrap().clear();
    }
}

impl Drop for Fetch {
    fn drop(&mut self) {
        self.end();
    }
}
