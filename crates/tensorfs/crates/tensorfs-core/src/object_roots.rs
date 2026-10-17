//! Durable custody of individual verified blobs. The caller owns release policy; roots
//! survive process death and retain bytes through every store GC tier.
use std::fs::{self, OpenOptions};
use std::io::{Read, Write};
use std::path::PathBuf;

use crate::catalog::WriterGuard;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{hex64, ObjectRef};
use crate::storage::HeldKey;
use crate::store::Store;

fn io(error: std::io::Error) -> Refusal {
    crate::store::classify_io("object root", error)
}
fn directory(store: &Store) -> PathBuf {
    store.root().join("roots/objects")
}
fn path(store: &Store, sha256: &str) -> Result<PathBuf> {
    hex64("object root", sha256)?;
    Ok(directory(store).join(format!("{sha256}.json")))
}

/// Retain an already admitted blob. A caller importing bytes must keep its WriterGuard
/// from admission through retention, so GC cannot pass between the two operations.
pub fn retain(store: &Store, object: &ObjectRef) -> Result<()> {
    let path = path(store, &object.sha256)?;
    let _writer = WriterGuard::acquire(store.root())?;
    let verified = store.open_verified(&object.sha256)?;
    if verified.len() != object.length {
        return refuse(
            Code::LENGTH_MISMATCH,
            "object root differs from admitted length",
        );
    }
    if let Some(held) = get(store, &object.sha256)? {
        return if held == *object {
            Ok(())
        } else {
            refuse(
                Code::LENGTH_MISMATCH,
                "object root already holds another length",
            )
        };
    }
    let dir = directory(store);
    fs::create_dir_all(&dir).map_err(io)?;
    let temporary = path.with_extension(format!("tmp-{}", crate::meta::now_nanos_unique()));
    let result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(io)?;
        file.write_all(&crate::canon::write(&object.to_value()))
            .and_then(|()| file.sync_all())
            .map_err(io)?;
        fs::rename(&temporary, &path).map_err(io)?;
        crate::store::fsync_entry_dir(store.root(), &dir)
    })();
    let _ = fs::remove_file(temporary);
    result
}

pub fn get(store: &Store, sha256: &str) -> Result<Option<ObjectRef>> {
    let file = match cozy_platform::durable::open_nofollow(&path(store, sha256)?, OpenOptions::new().read(true))
        .map_err(std::io::Error::from)
    {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(io(error)),
    };
    let mut bytes = Vec::new();
    file.take(1025).read_to_end(&mut bytes).map_err(io)?;
    let object =
        ObjectRef::from_value("object root", &crate::canon::parse_canonical(&bytes, 1024)?)?;
    if object.sha256 != sha256 {
        return refuse(
            Code::CROSS_SUBJECT_REPLAY,
            "object root path names different bytes",
        );
    }
    Ok(Some(object))
}

pub fn list(store: &Store) -> Result<Vec<ObjectRef>> {
    let entries = match fs::read_dir(directory(store)) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(vec![]),
        Err(error) => return Err(io(error)),
    };
    let mut roots = vec![];
    for entry in entries {
        let name = entry.map_err(io)?.file_name();
        let Some(sha256) = name.to_str().and_then(|n| n.strip_suffix(".json")) else {
            continue;
        };
        if let Some(object) = get(store, sha256)? {
            roots.push(object);
        }
    }
    Ok(roots)
}

/// Release custody. Removal is serialized with GC; callers serialize it with new
/// journal references. Bytes then become garbage unless another native root holds them.
pub fn remove(store: &Store, sha256: &str) -> Result<bool> {
    let path = path(store, sha256)?;
    let _writer = WriterGuard::acquire(store.root())?;
    match fs::remove_file(path) {
        Ok(()) => crate::store::fsync_entry_dir(store.root(), &directory(store)).map(|()| true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(io(error)),
    }
}

pub(crate) fn held_objects(store: &Store) -> Result<Vec<HeldKey>> {
    list(store)?
        .into_iter()
        .map(|object| {
            Ok(HeldKey {
                key: crate::storage::blob_key(&object.sha256)?,
                kind: "blob".into(),
                length: object.length,
                sha256: object.sha256,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Fault;

    #[test]
    fn bare_blob_custody_survives_store_reopen_and_gc_tiers_until_release() {
        let root = std::env::temp_dir().join(format!(
            "tfs-object-roots-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let object = store
            .put_stream(&mut &b"accepted input"[..], None, &Fault::default())
            .unwrap()
            .obj;
        retain(&store, &object).unwrap();
        assert_eq!(list(&store).unwrap(), vec![object.clone()]);
        drop(store);
        crate::gc::collect(&root, false).unwrap();
        crate::gc::collect_cached_for(&root, &[], u64::MAX).unwrap();
        crate::gc::collect_idle(&root, &[], std::time::Duration::ZERO).unwrap();
        let store = Store::open(&root).unwrap();
        assert!(store.object_path(&object.sha256).is_file());
        assert!(remove(&store, &object.sha256).unwrap());
        crate::gc::collect(&root, false).unwrap();
        assert!(!store.object_path(&object.sha256).exists());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn admission_through_retention_excludes_gc() {
        let root = std::env::temp_dir().join(format!(
            "tfs-object-admission-{}",
            crate::meta::now_nanos_unique()
        ));
        let store = Store::init(&root).unwrap();
        let guard = WriterGuard::acquire(&root).unwrap();
        let object = store
            .put_stream(
                &mut &b"write awaiting acknowledgment"[..],
                None,
                &Fault::default(),
            )
            .unwrap()
            .obj;
        assert_eq!(
            crate::gc::collect(&root, false).unwrap_err().code,
            Code::STORE_BUSY
        );
        retain(&store, &object).unwrap();
        drop(guard);
        crate::gc::collect(&root, false).unwrap();
        assert!(store.object_path(&object.sha256).is_file());
        let wrong = ObjectRef {
            length: object.length + 1,
            ..object.clone()
        };
        assert_eq!(
            retain(&store, &wrong).unwrap_err().code,
            Code::LENGTH_MISMATCH
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn object_root_written_before_journal_ack_survives_process_exit() {
        const AREA: &str = "TENSORFS_OBJECT_ROOT_TEST_AREA";
        if let Some(root) = std::env::var_os(AREA) {
            let root = PathBuf::from(root);
            let store = Store::init(&root).unwrap();
            let _guard = WriterGuard::acquire(&root).unwrap();
            let object = store
                .put_stream(
                    &mut &b"rooted before journal acknowledgment"[..],
                    None,
                    &Fault::default(),
                )
                .unwrap()
                .obj;
            retain(&store, &object).unwrap();
            // No destructors, simulating the store/journal crash boundary.
            std::process::exit(0);
        }
        let root = std::env::temp_dir().join(format!(
            "tfs-object-crash-{}",
            crate::meta::now_nanos_unique()
        ));
        let fence = crate::fork_fence::spawning();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "object_roots::tests::object_root_written_before_journal_ack_survives_process_exit",
            ])
            .env(AREA, &root)
            .spawn()
            .unwrap();
        drop(fence);
        assert!(child.wait().unwrap().success());
        crate::gc::collect(&root, false).unwrap();
        let store = Store::open(&root).unwrap();
        let object = ObjectRef::of(b"rooted before journal acknowledgment");
        assert!(store.object_path(&object.sha256).is_file());
        assert_eq!(get(&store, &object.sha256).unwrap(), Some(object));
        let _ = fs::remove_dir_all(root);
    }
}
