//! Fetching into a store (design §5). An [`Origin`] names a model's tree and locates objects
//! ([`Endpoint`], the distribution protocol's client; [`TensorhubOrigin`] until TensorD moves to
//! it; [`VolumeOrigin`] for a datacenter volume);
//! the [`Fetcher`] walks the closure (tree → its repo files and the chosen variant's header →
//! the wanted components' chunks and assets), reads each object through the tiers local store →
//! volume → origin,
//! and admits it through the store, which verifies it. One HTTP client ([`Client`]) and one
//! downloader serve every origin.

pub mod conform;
mod download;
mod dpop;
mod endpoint;
mod net;
mod origin;
pub mod policy;
mod tensorhub;
pub mod token;
mod volume;

use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

pub use dpop::{DPoP, JwtBearer};
pub use endpoint::{upload_tree, upload_tree_with, with_tree, Endpoint, Fresh, Payload, Stream, UploadControl, Uploaded};
pub use net::Client;
pub use origin::{choose, release_of, variant_of_tree, Chosen, Location, Origin};
pub use policy::SourcePolicy;
pub use tensorhub::{Resolved, TensorhubOrigin, TENSORHUB};
pub use token::{bearer, SigV4, TokenProvider};
pub use volume::VolumeOrigin;

use crate::err::{refuse, Code, Refusal, Result};
use crate::format::{self, Header, ObjectRef, Tree};
use download::Shared;

/// The store objects are admitted into (design §4).
pub trait ObjectStore: Send + Sync + 'static {
    /// Whether the store holds `object` (verified when it was admitted).
    fn has(&self, object: &ObjectRef) -> bool;
    /// A held object's bytes.
    fn open(&self, object: &ObjectRef) -> std::io::Result<File>;
    /// The one door for `object`: its bytes in order, then `commit`, which hashes them and
    /// refuses anything but `object` (`OBJECT_ID_MISMATCH`, naming the digest it got).
    fn admit(&self, object: &ObjectRef) -> Result<Box<dyn Admission>>;
}

/// One object on its way into a store.
pub trait Admission: Send {
    fn write(&mut self, bytes: &[u8]) -> Result<()>;
    fn commit(self: Box<Self>) -> Result<()>;
}

/// Where an object came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tier {
    Local,
    Volume,
    Origin,
}

/// Bytes by the tier they came from.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Landed {
    pub local: u64,
    pub volume: u64,
    pub origin: u64,
}

impl Landed {
    fn add(&mut self, tier: Tier, bytes: u64) {
        match tier {
            Tier::Local => self.local += bytes,
            Tier::Volume => self.volume += bytes,
            Tier::Origin => self.origin += bytes,
        }
    }

    fn join(&mut self, other: Landed) {
        (self.local, self.volume, self.origin) = (
            self.local + other.local,
            self.volume + other.volume,
            self.origin + other.origin,
        );
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    /// The watch's sampling resolution: every verdict is a multiple of the fetch's own
    /// measurements, never shorter than a few of these.
    pub tick: Duration,
    /// Workers to start with, and the most the pace may reach.
    pub streams: usize,
    pub max_streams: usize,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            tick: Duration::from_millis(250),
            streams: 16,
            max_streams: 128,
        }
    }
}

/// What a pull wants: components (all when `None`), its place in the queue (smaller first),
/// and where it reports progress.
#[derive(Default)]
pub struct Want<'a> {
    pub components: Option<&'a [String]>,
    pub priority: u64,
    pub progress: Option<&'a Progress>,
}

/// A pull's bytes: all it needs, and those held so far.
#[derive(Debug, Default)]
pub struct Progress {
    pub total: AtomicU64,
    pub held: AtomicU64,
}

/// What a pull took: the tree (when it walked one), the variant and its header, and its objects.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub tree: Option<ObjectRef>,
    pub variant: String,
    pub header: ObjectRef,
    pub objects: usize,
    pub landed: Landed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stats {
    pub requests: u64,
    pub restarts: u64,
    pub streams: usize,
    pub volume_written: u64,
    pub volume_skipped: u64,
}

/// A store's one downloader. Construct it inside a tokio runtime ([`runtime`] serves callers
/// without one); dropping it stops its workers.
pub struct Fetcher {
    shared: Arc<Shared>,
}

impl Drop for Fetcher {
    fn drop(&mut self) {
        self.shared.close();
    }
}

impl Fetcher {
    /// Fetches into `store` over `client`, reading `volume` first when one is given.
    pub fn new(
        store: Arc<dyn ObjectStore>,
        client: Client,
        volume: Option<VolumeOrigin>,
        config: Config,
    ) -> Fetcher {
        let shared = Arc::new(Shared::new(
            store,
            client,
            volume,
            config.tick,
            config.streams,
            config.max_streams,
        ));
        tokio::spawn(Arc::clone(&shared).dispatch());
        tokio::spawn(Arc::clone(&shared).watch());
        Fetcher { shared }
    }

    pub fn client(&self) -> &Client {
        &self.shared.client
    }

    pub fn stats(&self) -> Stats {
        let counters = &self.shared.stats;
        Stats {
            requests: counters.requests.load(Ordering::Relaxed),
            restarts: counters.restarts.load(Ordering::Relaxed),
            streams: self.shared.streams(),
            volume_written: counters.volume_written.load(Ordering::Relaxed),
            volume_skipped: counters.volume_skipped.load(Ordering::Relaxed),
        }
    }

    /// Resolves once nothing is queued, downloading or being written back to the volume.
    pub async fn settled(&self) {
        self.shared.settled().await
    }

    /// `name` at `release` and `variant` from `origin`, the components `want` selects.
    pub async fn pull(
        &self,
        origin: &Arc<dyn Origin>,
        name: &str,
        release: Option<&str>,
        variant: Option<&str>,
        want: &Want<'_>,
    ) -> Result<Pulled> {
        let tree = origin.resolve(name, release, variant).await?;
        self.pull_tree(origin, &tree, variant, want).await
    }

    /// The tree `tree`: its repo files, and its `variant` (its only one when `None`) with the
    /// components `want` selects.
    pub async fn pull_tree(
        &self,
        origin: &Arc<dyn Origin>,
        tree: &ObjectRef,
        variant: Option<&str>,
        want: &Want<'_>,
    ) -> Result<Pulled> {
        let mut landed = self.objects(origin, std::slice::from_ref(tree), want).await?;
        let parsed = Tree::from_json(&self.read(tree).await?).map_err(|e| invalid(tree, e))?;
        let (variant, header) = variant_of_tree(&parsed, variant)?;
        let files: Vec<ObjectRef> = parsed.repo_files().map(|(_, r)| r.clone()).collect();
        landed.join(self.objects(origin, &files, want).await?);
        let mut pulled = self.pull_header(origin, &header, want).await?;
        pulled.landed.join(landed);
        pulled.objects += 1 + files.len();
        (pulled.tree, pulled.variant) = (Some(tree.clone()), variant);
        Ok(pulled)
    }

    /// The cozytensors whose header is `header`, the components `want` selects.
    pub async fn pull_header(
        &self,
        origin: &Arc<dyn Origin>,
        header: &ObjectRef,
        want: &Want<'_>,
    ) -> Result<Pulled> {
        let mut landed = self.objects(origin, std::slice::from_ref(header), want).await?;
        let parsed = Header::from_cbor(&self.read(header).await?).map_err(|e| invalid(header, e))?;
        let wanted = wanted(&parsed, want.components)?;
        landed.join(self.objects(origin, &wanted, want).await?);
        Ok(Pulled {
            tree: None,
            variant: String::new(),
            header: header.clone(),
            objects: wanted.len() + 1,
            landed,
        })
    }

    /// Each of `objects` in the store, fetched from `origin` when absent.
    pub async fn objects(
        &self,
        origin: &Arc<dyn Origin>,
        objects: &[ObjectRef],
        want: &Want<'_>,
    ) -> Result<Landed> {
        let mut landed = Landed::default();
        let mut waiting = Vec::new();
        for object in objects {
            if let Some(progress) = want.progress {
                progress.total.fetch_add(object.length, Ordering::Relaxed);
            }
            match self.shared.store.has(object) {
                true => {
                    landed.add(Tier::Local, object.length);
                    if let Some(progress) = want.progress {
                        progress.held.fetch_add(object.length, Ordering::Relaxed);
                    }
                }
                false => waiting.push((
                    object.length,
                    self.shared.want(object, origin, want.priority),
                )),
            }
        }
        let landed = Mutex::new(landed);
        futures::future::try_join_all(waiting.into_iter().map(|(length, waiter)| {
            let landed = &landed;
            async move {
                let tier = waiter.wait().await?;
                landed.lock().unwrap().add(tier, length);
                if let Some(progress) = want.progress {
                    progress.held.fetch_add(length, Ordering::Relaxed);
                }
                Ok::<(), Refusal>(())
            }
        }))
        .await?;
        Ok(landed.into_inner().unwrap())
    }

    async fn read(&self, object: &ObjectRef) -> Result<Vec<u8>> {
        let (store, object) = (Arc::clone(&self.shared.store), object.clone());
        let read = tokio::task::spawn_blocking(move || {
            let mut bytes = Vec::with_capacity(object.length as usize);
            store.open(&object)?.read_to_end(&mut bytes).map(|_| bytes)
        });
        read.await
            .map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: error.to_string(),
            })?
            .map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: format!("reading an admitted document: {error}"),
            })
    }
}

fn invalid(object: &ObjectRef, error: format::Error) -> Refusal {
    Refusal {
        code: Code::UNKNOWN_FORMAT,
        detail: format!("{}: {error}", object.sha256),
    }
}

/// The chunks the tensors of `components` (every component when `None`) live in, then their
/// assets; each once.
pub fn wanted(header: &Header, components: Option<&[String]>) -> Result<Vec<ObjectRef>> {
    if let Some(name) = components.into_iter().flatten().find(|n| header.component(n).is_none()) {
        let has: Vec<&str> = header.components().iter().map(|c| c.name.as_str()).collect();
        return refuse(Code::REF_NOT_FOUND, format!("no component {name:?} (has {has:?})"));
    }
    crate::store2::wanted(header, components).map_err(crate::store2::Error::refusal)
}

/// A runtime for callers without one (a CLI, Python): two threads for sockets and timers.
pub fn runtime() -> Result<Arc<tokio::runtime::Runtime>> {
    let mut held = runtime_slot()?
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(runtime) = held.as_ref() {
        return Ok(Arc::clone(runtime));
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("tensorfs-fetch")
        .enable_all()
        .build()
        .map_err(|error| Refusal {
            code: Code::TRANSFER_FAILED,
            detail: format!("the fetch runtime could not start: {error}"),
        })?;
    Ok(Arc::clone(held.insert(Arc::new(runtime))))
}

type RuntimeSlot = Mutex<Option<Arc<tokio::runtime::Runtime>>>;

#[cfg(not(unix))]
fn runtime_slot() -> Result<&'static RuntimeSlot> {
    static RUNTIME: RuntimeSlot = Mutex::new(None);
    Ok(&RUNTIME)
}

#[cfg(unix)]
fn runtime_slot() -> Result<&'static RuntimeSlot> {
    process_runtime::slot()
}

/// The POSIX process boundary for the lazy network runtime. An inherited Tokio runtime
/// has no workers in a fork child, and its mutex may be owned by a vanished parent thread.
#[cfg(unix)]
#[allow(unsafe_code)]
mod process_runtime {
    use super::{Code, Refusal, Result, RuntimeSlot};
    use std::ptr;
    use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

    static SLOT: AtomicPtr<RuntimeSlot> = AtomicPtr::new(ptr::null_mut());
    static REGISTERED: AtomicBool = AtomicBool::new(false);

    extern "C" fn after_fork() {
        // Only an atomic store: no allocation, lock, refcount change or destructor is
        // permitted here. Deliberately abandon the inherited holder in this child.
        SLOT.store(ptr::null_mut(), Ordering::Release);
    }

    pub(super) fn slot() -> Result<&'static RuntimeSlot> {
        if !REGISTERED.load(Ordering::Acquire) {
            // SAFETY: after_fork has a static C ABI and touches only the lock-free pointer.
            // Register before publishing a holder. There is deliberately no "initializing"
            // state or Once lock for a concurrent fork to inherit. Racing first callers may
            // register the same idempotent callback; every registration completes before use.
            let error = unsafe { libc::pthread_atfork(None, None, Some(after_fork)) };
            if error != 0 {
                return Err(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the fetch runtime fork hook could not register: {}", std::io::Error::from_raw_os_error(error)),
                });
            }
            REGISTERED.store(true, Ordering::Release);
        }
        // The registered handler itself is inherited, so REGISTERED stays true in children.
        // Resetting the pointer on every fork also covers descendant forks and PID reuse;
        // no process identity comparison can mistake an ancestor's holder for a live one.
        let mut slot = SLOT.load(Ordering::Acquire);
        if slot.is_null() {
            let candidate: *mut RuntimeSlot = Box::into_raw(Box::new(RuntimeSlot::new(None)));
            slot = match SLOT.compare_exchange(ptr::null_mut(), candidate, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => candidate,
                Err(published) => {
                    // SAFETY: a losing candidate was never published and contains no runtime.
                    // Its allocation belongs only to this thread in the current process.
                    unsafe { drop(Box::from_raw(candidate)); }
                    published
                }
            };
        }
        // SAFETY: published holders are never freed. Threads in this process share its
        // mutex; a fork child clears SLOT before returning to callers and never locks or
        // drops the inherited holder. Keeping its Arc also prevents an inherited temporary
        // Arc from being the last owner and dropping the parent-created Tokio runtime.
        Ok(unsafe { &*slot })
    }
}

#[cfg(all(test, unix))]
#[allow(unsafe_code)]
mod runtime_fork_tests;
