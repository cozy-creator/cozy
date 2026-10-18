//! Where the machine's checkpoints come from, and the one downloader that brings them (TensorFS
//! v2's `Fetcher`, design §5): tiers local store → datacenter volume → origin, whole-object
//! GETs, every object verified as the store admits it. TensorD's threads are synchronous; each
//! call here blocks on the fetch runtime and ends early when its [`Stop`] is cancelled.
//!
//! - [`hub`]: TensorFS's `TensorhubOrigin` under the run's Cozy credentials.
//! - [`publish`]: a checkpoint put in a Hub repository's custody.
//! - [`serve`]: the executor's fetch channel.
mod hub;
mod publish;
mod serve;

use std::{
    future::Future,
    io,
    path::Path,
    sync::{Arc, Mutex},
    time::Duration,
};

use tensorfs_core::{
    err::{refuse, Code, Refusal, Result},
    fetch::{Client, Config, Fetcher, Landed, Origin, Progress, Pulled, SourcePolicy, VolumeOrigin, Want},
    format::ObjectRef,
    store2::Store,
};

pub use hub::{remote, resolve_hash, resolve_named, Resolved};
pub use publish::{Publication, Published};
pub use serve::{serve, Grant};

use crate::{stop::Stop, storage};

/// How often a blocked call looks at its stop.
const TICK: Duration = Duration::from_millis(100);

/// Queue order: smaller first.
pub mod priority {
    /// A reader is blocked on it.
    pub const NOW: u64 = 0;
    /// Next for `run` (by acceptance), `rank` within it.
    pub fn next(run: u64, rank: u32) -> u64 {
        1 + (run.min((1 << 31) - 1) << 32 | u64::from(rank))
    }
    /// Kept for later (a warm set's `downloaded`).
    pub const BEHIND: u64 = u64::MAX;
}

/// A checkpoint's origin, with the CAs its hosts may present beyond the public roots (a PEM
/// bundle) and whether its deployment declared itself local (loopback plaintext allowed).
#[derive(Clone)]
pub struct Remote {
    pub origin: Arc<dyn Origin>,
    pub trust: Option<Arc<Vec<u8>>>,
    pub local: bool,
}

type Key = (Option<Arc<Vec<u8>>>, bool);

/// The machine's downloader: one `Fetcher` per trust (in practice one).
pub struct Machine {
    store: Arc<Store>,
    runtime: Arc<tokio::runtime::Runtime>,
    volume: Option<VolumeOrigin>,
    fetchers: Mutex<Vec<(Key, Arc<Fetcher>)>>,
}

impl Machine {
    /// Fetches into `store`, reading the datacenter volume mounted at `volume` first.
    pub fn new(store: Arc<Store>, volume: Option<&Path>) -> io::Result<Arc<Machine>> {
        let runtime = tensorfs_core::fetch::runtime().map_err(io::Error::other)?;
        let volume = volume.and_then(|path| VolumeOrigin::mounted(path, storage::TTL));
        Ok(Arc::new(Machine { store, runtime, volume, fetchers: Mutex::default() }))
    }

    pub fn store(&self) -> &Arc<Store> {
        &self.store
    }

    /// The client objects of a remote with this trust are fetched over (for an `HttpOrigin`).
    pub fn client(&self, trust: Option<&Arc<Vec<u8>>>, local: bool) -> Result<Client> {
        Ok(self.fetcher(&(trust.cloned(), local))?.client().clone())
    }

    fn fetcher(&self, key: &Key) -> Result<Arc<Fetcher>> {
        let mut fetchers = self.fetchers.lock().unwrap();
        if let Some((_, fetcher)) = fetchers.iter().find(|(k, _)| k == key) {
            return Ok(fetcher.clone());
        }
        // Every object is held to its digest: no host vouches for anything, addresses are
        // still checked.
        let policy = SourcePolicy { any_host: true, allow_local: key.1, max_redirects: 8, ..Default::default() };
        let client = match &key.0 {
            Some(pem) => Client::with_roots(policy, pem)?,
            None => Client::new(policy),
        };
        let _entered = self.runtime.enter();
        let store: Arc<dyn tensorfs_core::fetch::ObjectStore> = self.store.clone();
        let fetcher = Arc::new(Fetcher::new(store, client, self.volume.clone(), Config::default()));
        fetchers.push((key.clone(), fetcher.clone()));
        Ok(fetcher)
    }

    fn of(&self, remote: &Remote) -> Result<Arc<Fetcher>> {
        self.fetcher(&(remote.trust.clone(), remote.local))
    }

    /// Runs `work` on the fetch runtime and waits for it, or for `stop`.
    pub(crate) fn block<T: Send + 'static>(&self, stop: &Stop, work: impl Future<Output = Result<T>> + Send + 'static) -> Result<T> {
        let (sent, answer) = std::sync::mpsc::channel();
        let stop = stop.clone();
        self.runtime.spawn(async move {
            let stopped = async {
                let mut tick = tokio::time::interval(TICK);
                while !stop.is_cancelled() {
                    tick.tick().await;
                }
            };
            let answered = tokio::select! {
                answered = work => answered,
                () = stopped => refuse(Code::TRANSFER_FAILED, "stopped"),
            };
            let _ = sent.send(answered);
        });
        answer.recv().unwrap_or_else(|_| refuse(Code::TRANSFER_FAILED, "the fetch runtime ended"))
    }

    /// `manifest`'s checkpoint, the `components` given (all when `None`): manifest, header,
    /// files, then, once the disk has room for them, the chunks the components read.
    pub fn pull(
        &self,
        remote: &Remote,
        manifest: &ObjectRef,
        components: Option<&[String]>,
        priority: u64,
        progress: Option<Arc<Progress>>,
        stop: &Stop,
    ) -> Result<Pulled> {
        let (fetcher, store, origin) = (self.of(remote)?, self.store.clone(), remote.origin.clone());
        let (manifest, components) = (manifest.clone(), components.map(<[String]>::to_vec));
        self.block(stop, async move {
            let want = |components| Want { components, priority, progress: progress.as_deref() };
            let structure = fetcher.pull_manifest(&origin, &manifest, &want(Some(&[]))).await?;
            if components.as_deref() == Some(&[]) {
                return Ok(structure);
            }
            let (held, id) = (store.clone(), manifest.clone());
            let checkpoint = tokio::task::spawn_blocking(move || held.checkpoint(&id).map_err(|e| e.refusal()))
                .await
                .map_err(|e| Refusal { code: Code::IO_FAILED, detail: e.to_string() })??;
            admit(&store, &tensorfs_core::fetch::chunks(checkpoint.header(), components.as_deref())?)?;
            let mut pulled = fetcher.pull_manifest(&origin, &manifest, &want(components.as_deref())).await?;
            (pulled.landed.origin, pulled.landed.volume) =
                (pulled.landed.origin + structure.landed.origin, pulled.landed.volume + structure.landed.volume);
            Ok(pulled)
        })
    }

    /// Each of `objects` here, fetched from `remote` when absent.
    pub fn objects(&self, remote: &Remote, objects: &[ObjectRef], priority: u64, stop: &Stop) -> Result<Landed> {
        let (fetcher, store, origin) = (self.of(remote)?, self.store.clone(), remote.origin.clone());
        let objects = objects.to_vec();
        self.block(stop, async move {
            admit(&store, &objects)?;
            fetcher.objects(&origin, &objects, &Want { priority, ..Want::default() }).await
        })
    }

    /// Wants `objects` at `priority` until the returned handle is aborted or they land.
    pub(crate) fn want(&self, remote: &Remote, objects: Vec<ObjectRef>, priority: u64) -> Result<tokio::task::JoinHandle<Result<Landed>>> {
        let (fetcher, origin) = (self.of(remote)?, remote.origin.clone());
        Ok(self.runtime.spawn(async move {
            fetcher.objects(&origin, &objects, &Want { priority, ..Want::default() }).await
        }))
    }
}

/// Runs `work`, telling `report` the bytes `progress` holds of its total every quarter second
/// meanwhile, and once more at the end.
pub fn watched<T>(progress: &Progress, report: &(dyn Fn(u64, u64) + Sync), work: impl FnOnce() -> T) -> T {
    use std::sync::atomic::Ordering::Relaxed;
    let done = Stop::default();
    std::thread::scope(|scope| {
        scope.spawn(|| loop {
            report(progress.held.load(Relaxed), progress.total.load(Relaxed));
            if done.is_cancelled() {
                return;
            }
            std::thread::sleep(Duration::from_millis(250));
        });
        let answer = work();
        done.cancel();
        answer
    })
}

/// Room on the store's disk for the absent `objects`, past the reserve.
fn admit(store: &Store, objects: &[ObjectRef]) -> Result<()> {
    let absent: u64 = objects.iter().filter(|o| !store.reuse(o).unwrap_or(false)).map(|o| o.length).sum();
    match absent {
        0 => Ok(()),
        bytes => storage::admits(store, bytes).map_err(|e| Refusal { code: Code::CAPACITY_EXHAUSTED, detail: e.to_string() }),
    }
}

#[cfg(test)]
pub(crate) mod tests;
