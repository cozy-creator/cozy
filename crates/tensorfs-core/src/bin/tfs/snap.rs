//! Manifest inspection over the real local store.

use std::path::Path;
use std::process::ExitCode;

use tensorfs_core::checkpoint;
use tensorfs_core::ids::{hex64, Doc, ObjectRef};
use tensorfs_core::store::Store;

use crate::{bail, flag, Flags};

fn open(root: &Path, hex: &str) -> tensorfs_core::err::Result<(Store, ObjectRef)> {
    let store = Store::open(root)?;
    let sha256 = hex64("manifest", hex.trim_start_matches("sha256:"))?;
    let length = std::fs::metadata(store.manifest_path(&sha256))
        .map_err(|error| tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::OBJECT_ABSENT,
            detail: format!("manifest sha256:{sha256} is absent: {error}"),
        })?
        .len();
    Ok((store, ObjectRef { sha256, length }))
}

pub fn cmd_get(root: &Path, hex: &str, flags: &Flags) -> ExitCode {
    let Some(output) = flag(flags, "out") else {
        eprintln!("--out <path> is required");
        return ExitCode::from(2);
    };
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    if let Err(error) = std::fs::write(output, manifest.canonical_bytes()) {
        return bail(tensorfs_core::err::Refusal {
            code: tensorfs_core::err::Code::IO_FAILED,
            detail: format!("write {output}: {error}"),
        });
    }
    println!("manifest {} ({} B)", reference.id(), reference.length);
    ExitCode::SUCCESS
}

pub fn cmd_walk(root: &Path, hex: &str, flags: &Flags) -> ExitCode {
    let (store, reference) = ok!(open(root, hex));
    let manifest = ok!(store.read_manifest(&reference));
    let walk = ok!(checkpoint::walk(&store, &manifest));
    if let Some(path) = flag(flags, "refs") {
        let mut references = std::collections::BTreeMap::new();
        for reached in &walk.objects {
            match references.insert(reached.obj.sha256.clone(), reached.obj.length) {
                Some(length) if length != reached.obj.length => {
                    return bail(tensorfs_core::err::Refusal {
                        code: tensorfs_core::err::Code::LENGTH_MISMATCH,
                        detail: format!(
                            "sha256:{} is reached at conflicting lengths",
                            reached.obj.sha256
                        ),
                    })
                }
                _ => {}
            }
        }
        let lines: Vec<Vec<u8>> = references
            .into_iter()
            .map(|(sha256, length)| {
                tensorfs_core::canon::write(&tensorfs_core::canon::Value::obj(vec![
                    ("length", tensorfs_core::canon::Value::uint(length)),
                    ("sha256", tensorfs_core::canon::Value::str(sha256)),
                ]))
            })
            .collect();
        if let Err(error) = tensorfs_core::storage::write_json_lines(Path::new(path), &lines) {
            return bail(error);
        }
    }
    println!(
        "manifest {} reaches {} references / {} distinct blobs",
        reference.id(),
        walk.objects.len(),
        walk.distinct().len()
    );
    ExitCode::SUCCESS
}
