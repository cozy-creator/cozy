//! Disposable local checkouts and explicit byte materialization.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use crate::canon;
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::manifest::{Entry, Manifest};
use crate::store::Store;

fn io(what: impl AsRef<str>, error: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {error}", what.as_ref()),
    }
}

#[derive(Debug, Default, Clone)]
pub struct Checkout {
    pub dirs: usize,
    pub links: usize,
    pub copies: usize,
    pub bytes: u64,
}

fn relative(from_dir: &Path, to: &Path) -> PathBuf {
    let left: Vec<Component> = from_dir.components().collect();
    let right: Vec<Component> = to.components().collect();
    let common = left
        .iter()
        .zip(right.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let mut path = PathBuf::new();
    for _ in common..left.len() {
        path.push("..");
    }
    for component in &right[common..] {
        path.push(component.as_os_str());
    }
    path
}

/// Create a normal-looking tree. Ordinary files are relative symlinks where possible;
/// CozyTensors is omitted because its header is not a conventional weight file.
pub fn checkout(
    store: &Store,
    manifest: &Manifest,
    destination: &Path,
    force_copy: bool,
) -> Result<Checkout> {
    fs::create_dir_all(destination).map_err(|error| io("mkdir checkout", error))?;
    let destination = destination
        .canonicalize()
        .map_err(|error| io("canonicalize checkout", error))?;
    let mut result = Checkout::default();
    let mut directories = HashSet::new();
    for (virtual_path, entry) in manifest.entries() {
        let blob = match entry {
            Entry::CozyTensors(_) => continue,
            Entry::File(blob) => blob,
        };
        let source = store.blob_path(&blob.sha256);
        let metadata = fs::metadata(&source).map_err(|error| {
            if error.kind() == std::io::ErrorKind::NotFound {
                Refusal {
                    code: Code::OBJECT_ABSENT,
                    detail: format!("{virtual_path:?}: {} is absent", blob.id()),
                }
            } else {
                io(format!("stat {}", source.display()), error)
            }
        })?;
        if metadata.len() != blob.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{virtual_path:?}: {} is {} bytes, manifest promises {}",
                    blob.id(),
                    metadata.len(),
                    blob.length
                ),
            );
        }
        let target = destination.join(virtual_path);
        let parent = target.parent().unwrap();
        fs::create_dir_all(parent).map_err(|error| io("mkdir checkout parent", error))?;
        let mut cursor = parent;
        while cursor.starts_with(&destination) && cursor != destination {
            if directories.insert(cursor.to_path_buf()) {
                result.dirs += 1;
            }
            cursor = cursor.parent().unwrap_or(&destination);
        }
        let _ = fs::remove_file(&target);
        let linked =
            !force_copy && std::os::unix::fs::symlink(relative(parent, &source), &target).is_ok();
        if linked {
            result.links += 1;
        } else {
            fs::copy(&source, &target).map_err(|error| io("copy checkout file", error))?;
            let mut permissions = fs::metadata(&target)
                .map_err(|error| io("stat checkout copy", error))?
                .permissions();
            std::os::unix::fs::PermissionsExt::set_mode(&mut permissions, 0o444);
            fs::set_permissions(&target, permissions)
                .map_err(|error| io("chmod checkout copy", error))?;
            result.copies += 1;
        }
        result.bytes = result
            .bytes
            .checked_add(blob.length)
            .ok_or_else(|| Refusal {
                code: Code::ARITH_OVERFLOW,
                detail: "checkout byte count overflow".into(),
            })?;
    }
    Ok(result)
}

/// Write an independent physical copy of one selected ordinary manifest entry.
pub fn materialize(store: &Store, blob: &ObjectRef, destination: &Path) -> Result<u64> {
    let directory = destination.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(directory).map_err(|error| io("mkdir materialize", error))?;
    let temporary = directory.join(format!(
        ".materialize-{}-{}-{}",
        std::process::id(),
        &blob.sha256[..12],
        crate::meta::now_nanos_unique(),
    ));
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| io("create materialize", error))?;
    let written = match store.read_into(&blob.sha256, &mut output) {
        Ok(written) => written,
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            return Err(error);
        }
    };
    if written != blob.length {
        let _ = fs::remove_file(&temporary);
        return refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{} delivered {written} bytes, expected {}",
                blob.id(),
                blob.length
            ),
        );
    }
    output
        .flush()
        .map_err(|error| io("flush materialized file", error))?;
    output
        .sync_all()
        .map_err(|error| io("fsync materialized file", error))?;
    drop(output);
    fs::rename(&temporary, destination).map_err(|error| io("rename materialized file", error))?;
    Ok(written)
}

pub fn render(bytes: &[u8], max: usize) -> Result<String> {
    if let Ok(header) = crate::header::Header::parse(bytes) {
        return Ok(canon::pretty(&header.diagnostic_value()?, 0));
    }
    let value = canon::parse_canonical(bytes, max)?;
    Ok(canon::pretty(&value, 0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Fault, Store};
    use std::sync::{Arc, Barrier};

    #[test]
    fn concurrent_materializations_do_not_share_a_temp_name() {
        let root = std::env::temp_dir().join(format!(
            "tensorfs-materialize-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let store = Arc::new(Store::init(&root).unwrap());
        let bytes = vec![7u8; 1 << 20];
        let object = store
            .put_stream(
                &mut bytes.as_slice(),
                Some(&ObjectRef::of(&bytes)),
                &Fault::default(),
            )
            .unwrap()
            .obj;
        let destination = root.join("out.bin");
        let barrier = Arc::new(Barrier::new(3));
        let mut threads = Vec::new();
        for _ in 0..2 {
            let store = Arc::clone(&store);
            let object = object.clone();
            let destination = destination.clone();
            let barrier = Arc::clone(&barrier);
            threads.push(std::thread::spawn(move || {
                barrier.wait();
                materialize(&store, &object, &destination)
            }));
        }
        barrier.wait();
        for thread in threads {
            assert_eq!(thread.join().unwrap().unwrap(), bytes.len() as u64);
        }
        assert_eq!(fs::read(destination).unwrap(), bytes);
        assert_eq!(
            fs::read_dir(&root)
                .unwrap()
                .flatten()
                .filter(|entry| entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".materialize-"))
                .count(),
            0
        );
        let _ = fs::remove_dir_all(root);
    }
}
