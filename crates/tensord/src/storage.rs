//! The machine's one TensorFS store, opened with TensorFS's own cache policy: it evicts on its
//! own and never takes an open checkpoint's bytes. A model TensorD keeps, it holds open.
//!
//! - A checkpoint's id is its cozytensors header's object (`sha256:<hex>`).
//! - Local models are `local/<name>` repos, never evicted: one release, named by its tree.
//! - A Hub model this machine pulled is a cached ref, its release's tree (`held_models::Pulled`).
use std::{
    collections::BTreeMap,
    io::{self, Write as _},
    path::Path,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

pub use tensorfs::store::Store;
use tensorfs::format::{ObjectRef, Tree};

/// Opens (or creates) the machine store at `root`; one in an older format is refused (its
/// models pull again into a new one).
pub fn open(root: &Path) -> io::Result<Arc<Store>> {
    Store::open(root).map(|store| Arc::new(store.with_cache())).map_err(io_error)
}

/// The machine's claim on its store: `owner.lock` at the store's real path, held exclusively
/// (without waiting) for as long as the returned file lives; the OS releases it when the machine
/// exits. A second machine on the same store, by any path, is refused. Readers (`tfs`, the
/// library) never take it.
pub fn claim(root: &Path) -> io::Result<std::fs::File> {
    let real = std::fs::canonicalize(root)?;
    let file = std::fs::OpenOptions::new().read(true).write(true).create(true).truncate(false).open(real.join(OWNER_LOCK))?;
    match platform::lock::try_claim(&file)? {
        true => Ok(file),
        false => Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            format!("STORE_BUSY: this TensorFS store already has an owner ({})", real.display()),
        )),
    }
}

/// The machine's claim on its store (`claim`).
const OWNER_LOCK: &str = "owner.lock";

/// Past this the transport log rotates to `transport.log.1`, so it holds at most twice this.
const TRANSPORT_ROTATE: u64 = 4 << 20;

/// One line of the store's transport log (`cozy machine logs <m> tensorfs`): what a download or
/// import did, or why it failed: `<unix seconds> <kind> <subject> <detail>`, one per completed
/// fetch, never per object. Never fatal.
pub fn transport(store: &Store, kind: &str, subject: &str, detail: &str) {
    let dir = store.root().join("logs");
    let path = dir.join("transport.log");
    let _ = std::fs::create_dir_all(&dir);
    if std::fs::metadata(&path).is_ok_and(|meta| meta.len() > TRANSPORT_ROTATE) {
        let _ = std::fs::rename(&path, dir.join("transport.log.1"));
    }
    let now = SystemTime::now().duration_since(UNIX_EPOCH);
    let at = now.map_or(0.0, |since| since.as_secs_f64());
    let line = format!("{at:.3} {kind} {subject} {detail}\n");
    if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        let _ = file.write_all(line.as_bytes());
    }
}

pub fn io_error(error: tensorfs::store::Error) -> io::Error {
    let kind = match &error {
        tensorfs::store::Error::Full { .. } => io::ErrorKind::StorageFull,
        tensorfs::store::Error::Absent(_) => io::ErrorKind::NotFound,
        tensorfs::store::Error::Io { source, .. } => source.kind(),
        _ => io::ErrorKind::InvalidData,
    };
    io::Error::new(kind, error.to_string())
}

/// A local model: its release, tree, variant and that variant's header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Local {
    pub release: String,
    pub tree: ObjectRef,
    pub variant: String,
    pub header: ObjectRef,
}

/// The local model `repo` (`local/<name>`) at `release` (its newest when `None`) in `variant`
/// (its only one when `None`); None when the name holds nothing.
pub fn local(store: &Store, repo: &str, release: Option<&str>, variant: Option<&str>) -> io::Result<Option<Local>> {
    let Some(named) = store.get_ref(repo).map_err(io_error)? else {
        return Ok(None);
    };
    let (release, tree) = tensorfs::fetch::release_of(&named, release)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
    let parsed = store.tree(&tree).map_err(io_error)?;
    let (variant, header) = parsed.variant(variant).map_err(|e| io::Error::new(io::ErrorKind::NotFound, e.to_string()))?;
    Ok(Some(Local { release, variant: variant.into(), header: header.clone(), tree }))
}

/// Names `header` (with its source's repo `files`) the local model `local/<name>`: one release,
/// named by its tree, replacing what the name held.
pub fn set_local(store: &Store, name: &str, header: &ObjectRef, files: &BTreeMap<String, ObjectRef>) -> io::Result<ObjectRef> {
    let mut tree = Tree { files: files.clone() };
    tree.set_variant("", header.clone());
    let tree = store.put_tree(&tree).map_err(io_error)?;
    let repo = format!("local/{name}");
    store
        .update_ref(&repo, |_| Some(tensorfs::fetch::with_tree(None, &repo, &tree, Some(&tree.id()))))
        .map_err(io_error)?;
    Ok(tree)
}

/// A tree naming `header` alone, as its unnamed variant (`model.cozytensors`).
pub fn tree_of(store: &Store, header: &ObjectRef) -> io::Result<ObjectRef> {
    let mut tree = tensorfs::format::Tree::default();
    tree.set_variant("", header.clone());
    store.put_tree(&tree).map_err(io_error)
}

/// The object `id` (`sha256:<hex>` or bare hex) as the store holds it, its length read off disk.
pub fn held(store: &Store, id: &str) -> io::Result<ObjectRef> {
    let sha256 = id.trim_start_matches("sha256:").to_string();
    let path = store.path(&ObjectRef { sha256: sha256.clone(), length: 0 }).map_err(io_error)?;
    let length = std::fs::metadata(path)?.len();
    Ok(ObjectRef { sha256, length })
}

/// The checkpoint `id` names, held here (its structure; chunks may be absent).
pub fn checkpoint(store: &Store, id: &str) -> io::Result<tensorfs::store::Checkpoint> {
    store.checkpoint(&held(store, id)?).map_err(io_error)
}

/// Whether the store holds `object` (no touch: a look, not a use).
pub fn present(store: &Store, object: &ObjectRef) -> bool {
    store.path(object).is_ok_and(|path| std::fs::metadata(path).is_ok_and(|m| m.len() == object.length))
}

/// The chunks and assets `components` read (all components when empty), each once; none when
/// it names a component the checkpoint lacks.
pub fn chunks(checkpoint: &tensorfs::store::Checkpoint, components: &[String]) -> Vec<ObjectRef> {
    tensorfs::fetch::wanted(checkpoint.header(), (!components.is_empty()).then_some(components)).unwrap_or_default()
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
    fn one_machine_claims_a_store_by_any_path_until_it_lets_go() {
        let root = std::env::temp_dir().join(format!("tensord-claim-{}", uuid::Uuid::new_v4()));
        let store = root.join("real/store");
        super::open(&store).unwrap();
        let first = super::claim(&store).unwrap();
        platform::durable::symlink(&root.join("real"), &root.join("alias")).unwrap();
        let busy = super::claim(&root.join("alias/store")).expect_err("a second claim");
        assert!(busy.to_string().contains("STORE_BUSY: this TensorFS store already has an owner"), "{busy}");
        drop(first);
        super::claim(&root.join("alias/store")).unwrap();
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn a_store_in_an_older_format_is_refused_with_what_to_do() {
        let root = std::env::temp_dir().join(format!("tensord-old-store-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("format"), br#"{"format":"tensorfs.store/2"}"#).unwrap();
        let Err(refused) = super::open(&root) else { panic!("an older store is not opened") };
        assert!(refused.to_string().contains("pull again"), "{refused}");
        let _ = std::fs::remove_dir_all(root);
    }
}
