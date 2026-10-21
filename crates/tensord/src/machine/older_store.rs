//! A machine store in an older TensorFS format (`tensorfs.store/2`) is the machine's cache, which
//! this TensorFS cannot read. TensorD sets it aside inside its own root (which may be a mount
//! point) and opens a new store there; its models pull again on demand. It waits out the update
//! that met it, so a rollback restores it for the software that reads it, and goes once that
//! update commits, or at once when no update is in flight. Never two stores past an update.
use std::{
    fs, io,
    path::{Path, PathBuf},
    sync::Arc,
};

use super::update::Paths;
use crate::storage::Store;

/// Under `tmp/`, whose directories the store leaves to the layer that made them.
const ASIDE: &str = "tmp/older-store";
const PART: &str = "tmp/older-store.part";

/// The machine store at `root`, opened; one in an older format set aside first (`warn`ed and,
/// with no update in flight, deleted at once).
pub fn open(root: &Path, paths: &Paths) -> io::Result<Arc<Store>> {
    if let Some(format) = older(root)? {
        set_aside(root)?;
        super::update::write_marker(paths, MARKER, &root.to_string_lossy())?;
        tracing::warn!(
            "{}: a {format} store from an older TensorFS ({}) is set aside until this update commits; \
             its models pull again into a new one",
            root.display(),
            gigabytes(size(&root.join(ASIDE)))
        );
    }
    let store = crate::storage::open(root)?;
    if !super::update::in_flight(paths)? {
        if let Some(note) = reclaim(paths)? {
            tracing::warn!("{note}");
        }
    }
    Ok(store)
}

/// After the update that met it commits: the older store is deleted. Returns what that freed.
pub fn reclaim(paths: &Paths) -> io::Result<Option<String>> {
    let Some(root) = marked(paths)? else { return Ok(None) };
    let aside = root.join(ASIDE);
    let freed = size(&aside);
    remove(&aside)?;
    remove(&root.join(PART))?;
    super::update::remove_marker(paths, MARKER)?;
    Ok(Some(format!("deleted the older-format TensorFS store set aside at {} ({} reclaimed)", root.display(), gigabytes(freed))))
}

/// On rollback: the older store comes back in place of the new one, for the software that
/// reads it. Returns whether there was one.
pub fn restore(paths: &Paths) -> io::Result<bool> {
    let Some(root) = marked(paths)? else { return Ok(false) };
    let aside = root.join(ASIDE);
    if aside.is_dir() {
        // The new store goes, its record first: the root never reads as a half of either.
        remove(&root.join("format"))?;
        for name in names(&root)?.into_iter().filter(|n| n != "tmp") {
            remove(&root.join(name))?;
        }
        for name in names(&root.join("tmp"))?.into_iter().filter(|n| n != "older-store") {
            remove(&root.join("tmp").join(name))?;
        }
        // The older one returns, its record last.
        for name in names(&aside)?.into_iter().filter(|n| n != "tmp" && n != "format") {
            fs::rename(aside.join(&name), root.join(&name))?;
        }
        for name in names(&aside.join("tmp"))? {
            fs::rename(aside.join("tmp").join(&name), root.join("tmp").join(&name))?;
        }
        fs::rename(aside.join("format"), root.join("format"))?;
        remove(&aside)?;
        platform::durable::sync_dir(&root).map_err(io::Error::from)?;
        tracing::warn!("{}: the older-format TensorFS store is restored for the software rolled back to", root.display());
    }
    super::update::remove_marker(paths, MARKER)?;
    Ok(true)
}

/// The update record naming the store root whose older store waits.
const MARKER: &str = "older-store";

fn marked(paths: &Paths) -> io::Result<Option<PathBuf>> {
    Ok(super::update::read_marker(paths, MARKER)?.map(PathBuf::from))
}

/// The older format `root` holds, if it holds one this TensorFS refuses.
fn older(root: &Path) -> io::Result<Option<String>> {
    #[derive(serde::Deserialize)]
    struct Record {
        format: String,
    }
    let bytes = match fs::read(root.join("format")) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    };
    let major = |format: &str| format.strip_prefix("tensorfs.store/").and_then(|v| v.split('.').next()?.parse::<u32>().ok());
    let found = serde_json::from_slice::<Record>(&bytes).ok().map(|r| r.format);
    Ok(found.filter(|f| matches!((major(f), major(tensorfs::store::FORMAT)), (Some(old), Some(now)) if old < now)))
}

/// Moves everything of the store at `root` under `tmp/older-store`. Resumable: the record moves
/// last, so until it does the root still reads as the older store and this runs again.
fn set_aside(root: &Path) -> io::Result<()> {
    let part = root.join(PART);
    fs::create_dir_all(part.join("tmp"))?;
    for name in names(root)?.into_iter().filter(|n| n != "tmp" && n != "format") {
        fs::rename(root.join(&name), part.join(&name))?;
    }
    for name in names(&root.join("tmp"))?.into_iter().filter(|n| n != "older-store.part" && n != "older-store") {
        fs::rename(root.join("tmp").join(&name), part.join("tmp").join(&name))?;
    }
    fs::rename(root.join("format"), part.join("format"))?;
    fs::rename(&part, root.join(ASIDE))?;
    platform::durable::sync_dir(&root.join("tmp")).map_err(io::Error::from)
}

fn names(dir: &Path) -> io::Result<Vec<String>> {
    match fs::read_dir(dir) {
        Ok(entries) => entries.map(|e| e.map(|e| e.file_name().to_string_lossy().into_owned())).collect(),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(vec![]),
        Err(error) => Err(error),
    }
}

fn remove(path: &Path) -> io::Result<()> {
    let removed = match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::remove_dir_all(path),
        Ok(_) => fs::remove_file(path),
        Err(error) => Err(error),
    };
    match removed {
        Err(error) if error.kind() != io::ErrorKind::NotFound => Err(error),
        _ => Ok(()),
    }
}

/// Bytes under `path` (its files' lengths).
fn size(path: &Path) -> u64 {
    match fs::symlink_metadata(path) {
        Ok(meta) if meta.is_dir() => fs::read_dir(path).into_iter().flatten().flatten().map(|e| size(&e.path())).sum(),
        Ok(meta) => meta.len(),
        Err(_) => 0,
    }
}

fn gigabytes(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `tensorfs.store/2` root: its record, a blob, a ref, a temp, and the machine's own files.
    fn older_store(root: &Path) -> Vec<(String, Vec<u8>)> {
        let files = [
            ("format", br#"{"format":"tensorfs.store/2"}"#.to_vec()),
            ("blobs/ab/cd/abcd", vec![7; 4096]),
            ("refs/local/m.json", b"{}".to_vec()),
            ("tmp/put-1", b"temp".to_vec()),
            ("owner.lock", vec![]),
            ("logs/transport.log", b"1 pull x\n".to_vec()),
        ];
        for (path, bytes) in &files {
            fs::create_dir_all(root.join(path).parent().unwrap()).unwrap();
            fs::write(root.join(path), bytes).unwrap();
        }
        files.into_iter().map(|(p, b)| (p.to_owned(), b)).collect()
    }

    fn held(root: &Path, files: &[(String, Vec<u8>)]) -> bool {
        files.iter().all(|(path, bytes)| fs::read(root.join(path)).is_ok_and(|b| &b == bytes))
    }

    fn update_in_flight(engine: &Path) {
        fs::create_dir_all(engine.join("update")).unwrap();
        fs::write(engine.join("update/pending.json"), b"{}").unwrap();
    }

    /// Met during an update, the older store waits beside the new one; a rollback restores it
    /// byte for byte; met again and committed, it is deleted and the new store stands alone.
    #[test]
    fn an_older_store_waits_out_its_update_and_is_deleted_or_restored() {
        let dir = tempfile::tempdir().unwrap();
        let (root, engine) = (dir.path().join("tensorfs"), dir.path().join("engine"));
        let paths = Paths::new(&engine, dir.path());
        let files = older_store(&root);
        update_in_flight(&engine);

        let store = open(&root, &paths).unwrap();
        assert!(store.put_bytes(b"new").is_ok(), "the new store serves");
        assert!(held(&root.join(ASIDE), &files), "the older store waits whole");
        assert!(fs::read_to_string(root.join("format")).unwrap().contains(tensorfs::store::FORMAT));
        drop(store);

        assert!(restore(&paths).unwrap());
        assert!(held(&root, &files), "rolled back, the older store is back as it was");
        assert!(!root.join(ASIDE).exists() && !root.join("blobs/sha256").exists());
        assert_eq!(super::super::update::read_marker(&paths, MARKER).unwrap(), None);

        // Met again; this time the update commits.
        let store = open(&root, &paths).unwrap();
        let note = reclaim(&paths).unwrap().unwrap();
        assert!(note.contains("reclaimed") && note.contains("GB"), "{note}");
        assert!(!root.join(ASIDE).exists() && !root.join("blobs/ab").exists());
        assert!(store.put_bytes(b"new").is_ok());
        assert_eq!(reclaim(&paths).unwrap(), None);
    }

    /// With no update in flight (a restart onto new software) nothing can roll back to it: it
    /// goes at once.
    #[test]
    fn with_no_update_in_flight_an_older_store_goes_at_once() {
        let dir = tempfile::tempdir().unwrap();
        let (root, engine) = (dir.path().join("tensorfs"), dir.path().join("engine"));
        let paths = Paths::new(&engine, dir.path());
        older_store(&root);
        let store = open(&root, &paths).unwrap();
        assert!(!root.join(ASIDE).exists() && !root.join("blobs/ab").exists());
        assert!(store.put_bytes(b"new").is_ok());
        // A current store is opened as it is.
        drop(store);
        assert!(open(&root, &paths).is_ok());
    }
}
