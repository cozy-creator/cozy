//! The machine's one TensorFS store and TensorD's policy over it: what is pinned and under
//! which name, how much disk is kept back, and when pinned caches expire or give way.
//!
//! - Pins are the one retention record. Their ids are `<kind>-<name>` ([`pin`]): run objects,
//!   package sources and result files ([`pin::object`]), the Hub checkpoints this machine pulled
//!   ([`pin::model`]), derived outputs not yet adopted ([`pin::derived`]), adapter views
//!   ([`pin::view`]) and retained file trees ([`pin::tree`]). Local models are refs
//!   (`local/<name>`), never dropped.
//! - Caches manage themselves, with no purge verb: a cache pin expires a TTL after its last use,
//!   and on a low disk the least recently used give way, counted by the bytes only they hold
//!   (`Store::usage`). Whatever a live executor or an unfinished run reads (`keep`) stays.
//! - A reserve of 1/50 of the filesystem (1–10 GiB) and 32 inodes is never admitted to a
//!   download or written into by an optional cache.
use std::{
    io,
    path::Path,
    sync::Arc,
    time::{Duration, SystemTime, UNIX_EPOCH},
};

pub use tensorfs_core::store2::{Collected, Pin, Store, Usage};
use tensorfs_core::format::ObjectRef;

/// A cache pin unused this long expires.
pub const TTL: Duration = Duration::from_secs(7 * 24 * 60 * 60);

/// 1/50 of the filesystem, at least 1 GiB and at most 10 GiB.
const RESERVE_BYTES: (u64, u64) = (1 << 30, 10 << 30);
const RESERVE_DIVISOR: u64 = 50;
const RESERVE_INODES: u64 = 32;

/// Pin ids, by what they keep.
pub mod pin {
    /// A run's input, a package source or a result file, by its digest.
    pub fn object(sha256: &str) -> String {
        format!("obj-{sha256}")
    }
    /// A Hub checkpoint this machine pulled, by its manifest's digest.
    pub fn model(sha256: &str) -> String {
        format!("model-{sha256}")
    }
    /// A derived output's checkpoint, by its transaction's digest, until it is adopted.
    pub fn derived(transaction: &str) -> String {
        format!("derived-{}", transaction.trim_start_matches("sha256:"))
    }
    /// An adapter view over a base checkpoint, by its composition's digest.
    pub fn view(composition: &str) -> String {
        format!("view-{composition}")
    }
    /// A retained file tree (a run's native inputs, a workspace upload), by its owner's digest.
    pub fn tree(owner: &str) -> String {
        format!("tree-{}", tensorfs_core::sha256::hex_digest(owner.as_bytes()))
    }

    /// Caches: dropped on a TTL or a low disk. Objects and trees are released by their owners.
    pub(crate) fn cache(id: &str) -> bool {
        ["model-", "derived-", "view-"].iter().any(|kind| id.starts_with(kind))
    }
}

/// Opens (or creates) the machine store at `root`.
pub fn open(root: &Path) -> io::Result<Arc<Store>> {
    Store::open(root).map(Arc::new).map_err(|error| match root.join(PRE_CUT_CATALOG).exists() {
        true => io::Error::new(io::ErrorKind::InvalidData, format!(
            "{}: this store predates TensorFS 0.12's format cut and isn't read by this release; published models re-pull into a fresh store; local models need the one-time `migrate`",
            root.display()
        )),
        false => io_error(error),
    })
}

/// A rental's store from before TensorFS 0.12's format cut (an older image's machine made it at
/// boot): its models re-pull, so its contents go and a fresh store takes their place. A
/// computer's machine keeps refusing one (`open`): its local models need the one-time migrate.
pub fn drop_pre_cut(root: &Path) -> io::Result<()> {
    if !root.join(PRE_CUT_CATALOG).exists() {
        return Ok(());
    }
    tracing::warn!("{}: dropping a store from before the format cut; its models re-pull", root.display());
    for entry in std::fs::read_dir(root)? {
        let path = entry?.path();
        match path.is_dir() && !path.is_symlink() {
            true => std::fs::remove_dir_all(&path)?,
            false => std::fs::remove_file(&path)?,
        }
    }
    Ok(())
}

/// The catalog at the root of a store from before TensorFS 0.12's format cut.
const PRE_CUT_CATALOG: &str = "tensorfs.sqlite";

pub fn io_error(error: tensorfs_core::store2::Error) -> io::Error {
    let kind = match &error {
        tensorfs_core::store2::Error::Full { .. } => io::ErrorKind::StorageFull,
        tensorfs_core::store2::Error::Absent(_) => io::ErrorKind::NotFound,
        tensorfs_core::store2::Error::Io { source, .. } => source.kind(),
        _ => io::ErrorKind::InvalidData,
    };
    io::Error::new(kind, error.to_string())
}

/// Pins `roots` under `id` (replacing what it pinned), last used now.
pub fn hold(store: &Store, id: &str, roots: Vec<ObjectRef>, owner: Option<String>) -> io::Result<()> {
    let mut pin = Pin::new(roots);
    pin.owner = owner;
    store.pin(id, &pin).map_err(io_error)
}

/// One filesystem: its size and what is free on it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disk {
    pub capacity: u64,
    pub available: u64,
    /// Zero when the filesystem has no inode limit.
    pub inodes: u64,
    pub available_inodes: u64,
}

impl Disk {
    pub fn measure(root: &Path) -> io::Result<Disk> {
        let space = platform::durable::space(root).map_err(|error| {
            io::Error::other(format!("cannot measure the filesystem under {}: {error}", root.display()))
        })?;
        Ok(Disk {
            capacity: space.total,
            available: space.free,
            inodes: space.inodes,
            available_inodes: space.free_inodes,
        })
    }

    /// Never admitted to a download, and never written into by an optional cache.
    pub fn reserve(&self) -> u64 {
        (self.capacity / RESERVE_DIVISOR).clamp(RESERVE_BYTES.0, RESERVE_BYTES.1)
    }

    /// A low disk has no more than its reserve free: the bytes to free before it has more.
    pub fn short(&self) -> Option<u64> {
        let low = self.available <= self.reserve()
            || (self.inodes > 0 && self.available_inodes <= RESERVE_INODES);
        low.then(|| (self.reserve() + 1).saturating_sub(self.available).max(1))
    }

    /// What a download may still take: what is free past the reserve.
    pub fn free(&self) -> u64 {
        self.available.saturating_sub(self.reserve())
    }
}

/// Room for `bytes` more on the store's disk, else why not.
pub fn admits(store: &Store, bytes: u64) -> io::Result<()> {
    let disk = Disk::measure(store.root())?;
    match bytes <= disk.free() {
        true => Ok(()),
        false => Err(io::Error::new(
            io::ErrorKind::StorageFull,
            format!("{bytes} bytes do not fit: {} are free past the {}-byte reserve", disk.free(), disk.reserve()),
        )),
    }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs())
}

/// Cache pins reaching any of `keep` (`sha256:<hex>`), which no sweep may drop.
fn kept(pin: &Pin, keep: &[String]) -> bool {
    pin.roots.iter().any(|root| keep.iter().any(|k| k.trim_start_matches("sha256:") == root.sha256))
}

/// Expires cache pins unused for `ttl` and writers abandoned as long, then sweeps.
pub fn expire(store: &Store, keep: &[String], ttl: Duration) -> io::Result<Collected> {
    let unused_since = now_secs().saturating_sub(ttl.as_secs());
    let mut dropped = vec![];
    for (id, pin) in store.pins().map_err(io_error)? {
        if pin::cache(&id) && pin.last_use <= unused_since && !kept(&pin, keep)
            && store.unpin_unused(&id, pin.last_use).map_err(io_error)?
        {
            dropped.push(id);
        }
    }
    store.sweep_writers(ttl).map_err(io_error)?;
    let mut collected = store.gc().map_err(io_error)?;
    collected.pins = dropped;
    Ok(collected)
}

/// A cache pin a low disk may drop: its id, last use and the bytes only it holds.
pub struct Droppable {
    pub id: String,
    pub last_use: u64,
    pub marginal: u64,
}

/// The cache pins a low disk may drop, least recently used first; and the unreferenced bytes a
/// sweep frees as it is.
pub fn reclaimable(store: &Store, keep: &[String]) -> io::Result<(Vec<Droppable>, u64)> {
    let usage = store.usage().map_err(io_error)?;
    let pins = store.pins().map_err(io_error)?;
    let mut droppable: Vec<Droppable> = pins
        .iter()
        .filter(|(id, pin)| pin::cache(id) && !kept(pin, keep))
        .filter_map(|(id, pin)| usage.pins.get(id).map(|u| Droppable { id: id.clone(), last_use: pin.last_use, marginal: u.marginal }))
        .collect();
    droppable.sort_by(|a, b| (a.last_use, &a.id).cmp(&(b.last_use, &b.id)));
    Ok((droppable, usage.unreferenced))
}

/// Frees `need` bytes on a low disk: the store's garbage first, then the fewest least recently
/// used cache pins whose bytes cover what is still missing. Nothing `keep` reaches goes.
pub fn relieve(store: &Store, keep: &[String], need: u64) -> io::Result<Collected> {
    let mut collected = store.gc().map_err(io_error)?;
    if collected.bytes >= need {
        return Ok(collected);
    }
    let (droppable, _) = reclaimable(store, keep)?;
    let mut covered = collected.bytes;
    for pin in droppable {
        if covered >= need {
            break;
        }
        if store.unpin_unused(&pin.id, pin.last_use).map_err(io_error)? {
            covered += pin.marginal;
            collected.pins.push(pin.id);
        }
    }
    if !collected.pins.is_empty() {
        let swept = store.gc().map_err(io_error)?;
        (collected.objects, collected.bytes) = (collected.objects + swept.objects, collected.bytes + swept.bytes);
    }
    Ok(collected)
}

/// The local model `local/<name>` (as `tfs import` refs one): its manifest, when it is here.
pub fn local(store: &Store, name: &str) -> io::Result<Option<ObjectRef>> {
    match store.resolve(&format!("local/{name}"), None) {
        Ok(named) => Ok(Some(named.manifest).filter(|manifest| store.reuse(manifest).unwrap_or(false))),
        Err(error) if error.kind == tensorfs_core::failure::Kind::NotFound => Ok(None),
        Err(error) => Err(io::Error::other(error.to_string())),
    }
}

/// Refs `manifest` as the local model `local/<name>`, replacing what the name held.
pub fn set_local(store: &Store, name: &str, manifest: &ObjectRef) -> io::Result<()> {
    store
        .update_ref(&format!("local/{name}"), |_| None)
        .map_err(io_error)?;
    store
        .set_lane(&format!("local/{name}"), tensorfs_core::client::IMPORT_RELEASE, None, manifest)
        .map(drop)
        .map_err(|error| io::Error::other(error.to_string()))
}

/// The object `id` (`sha256:<hex>` or bare hex) as the store holds it, its length read off disk.
pub fn held(store: &Store, id: &str) -> io::Result<ObjectRef> {
    let sha256 = id.trim_start_matches("sha256:").to_string();
    let path = store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).map_err(io_error)?;
    let length = std::fs::metadata(path)?.len();
    Ok(ObjectRef { sha256, length })
}

/// The checkpoint `id` names, held here (its structure; chunks may be absent).
pub fn checkpoint(store: &Store, id: &str) -> io::Result<tensorfs_core::store2::Checkpoint> {
    store.checkpoint(&held(store, id)?).map_err(io_error)
}

/// Whether the store holds `object` (no touch: a look, not a use).
pub fn present(store: &Store, object: &ObjectRef) -> bool {
    store.path(object).is_ok_and(|path| std::fs::metadata(path).is_ok_and(|m| m.len() == object.length))
}

/// The chunks the tensors of `components` live in (all components when empty), each once;
/// none when it names a component the checkpoint lacks.
pub fn chunks(checkpoint: &tensorfs_core::store2::Checkpoint, components: &[String]) -> Vec<ObjectRef> {
    tensorfs_core::fetch::chunks(checkpoint.header(), (!components.is_empty()).then_some(components)).unwrap_or_default()
}

/// `text` is 64 lowercase hex digits (a bare sha256), or `sha256:` and them when `prefixed`.
pub fn check_digest(what: &str, text: &str, prefixed: bool) -> io::Result<()> {
    let hex = match prefixed {
        true => text.strip_prefix("sha256:"),
        false => Some(text),
    };
    match hex.is_some_and(|h| h.len() == 64 && h.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))) {
        true => Ok(()),
        false => Err(io::Error::new(io::ErrorKind::InvalidInput, format!("{what}: {text:?} is not a sha256 digest"))),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn a_store_from_before_the_format_cut_is_refused_with_what_to_do() {
        let root = std::env::temp_dir().join(format!("tensord-pre-cut-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("blobs")).unwrap();
        std::fs::write(root.join(super::PRE_CUT_CATALOG), b"").unwrap();
        let Err(refused) = super::open(&root) else { panic!("a store from before the cut is not opened") };
        assert!(refused.to_string().contains("this store predates TensorFS 0.12's format cut and isn't read by this release"), "{refused}");
        let _ = std::fs::remove_dir_all(root);
    }
}
