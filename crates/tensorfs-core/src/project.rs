//! The projection plane (tfs-005) — tiers 2 and 3 of the permanent projection ladder, plus
//! the rule that keeps tier 1 the only way to reach a weight.
//!
//! ```text
//! 1  native reads          all our code, every tensor            (read.rs)
//! 2  project_snapshot      configs and documents as plain files through relative symlinks
//! 3  materialize_blob      a private streamed copy of an ALREADY-RETAINED opaque carrier
//! ```
//!
//! **Weight files have no path form at any tier.** A projected tree carries configs and
//! documents only, and this module refuses to project an entry whose object the checkpoint
//! reached as a storage part. The v1 pointer stub died with the tensor paths it stood in
//! for: with no tensor paths in trees, a stub has nothing to stand in for.
//!
//! **Pretty rendering is a PROJECTION, never storable bytes.** `render` exists so a human
//! can read a canonical document; its output has a different digest and is refused by every
//! decoder, so a SnapshotManifest can never reference one (tensorfs.md §7).

use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};

use crate::canon;
use crate::checkpoint;
use crate::err::{refuse, Code, Refusal, Result};
use crate::manifest::{BlobDescriptor, Entry, SnapshotManifest};
use crate::receipt::{Receipt, Receipts};
use crate::store::Store;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

#[derive(Debug, Default, Clone)]
pub struct Projection {
    pub dirs: usize,
    pub links: usize,
    pub copies: usize,
    pub bytes: u64,
}

/// The relative path from `from_dir` to `to`, so a projected tree survives the store and the
/// tree moving together. Both must be absolute.
fn relative(from_dir: &Path, to: &Path) -> PathBuf {
    let a: Vec<Component> = from_dir.components().collect();
    let b: Vec<Component> = to.components().collect();
    let common = a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count();
    let mut p = PathBuf::new();
    for _ in common..a.len() {
        p.push("..");
    }
    for c in &b[common..] {
        p.push(c.as_os_str());
    }
    p
}

/// Tier 2. Config assets and canonical documents become plain files a third party can open;
/// weights cannot appear, and the refusal says why rather than skipping quietly.
pub fn project_snapshot(
    store: &Store,
    snap: &SnapshotManifest,
    dest: &Path,
    force_copy: bool,
    rx: &mut Receipts,
) -> Result<Projection> {
    // The walk is the classifier: it says what each object was reached AS, and "part" is the
    // one answer a projected path may never carry.
    let walk = checkpoint::walk(store, snap)?;
    let weights: Vec<&str> = walk
        .objects
        .iter()
        .filter(|r| r.kind == "part")
        .map(|r| r.obj.sha256.as_str())
        .collect();

    let snap_hex = snap.snapshot_id();
    let snap_hex = snap_hex.strip_prefix("sha256:").unwrap_or(&snap_hex);
    fs::create_dir_all(dest).map_err(|e| io("mkdir projection", e))?;
    let dest = dest
        .canonicalize()
        .map_err(|e| io("canonicalize projection root", e))?;
    let objects = store
        .root()
        .canonicalize()
        .map_err(|e| io("canonicalize store root", e))?;

    let mut p = Projection::default();
    for (path, e) in snap.entries() {
        let target = dest.join(path);
        match e {
            Entry::Dir => {
                fs::create_dir_all(&target).map_err(|e| io("mkdir", e))?;
                p.dirs += 1;
            }
            Entry::File(o) => {
                if weights.contains(&o.sha256.as_str()) {
                    return refuse(
                        Code::PROJECTION_WEIGHT_PATH,
                        format!(
                            "{path:?} names {}, which this checkpoint reaches as a storage \
                             part — a projected tree cannot synthesize a weight-container \
                             path at any tier",
                            o.id()
                        ),
                    );
                }
                let src = objects
                    .join("objects/sha256")
                    .join(&o.sha256[0..2])
                    .join(&o.sha256[2..4])
                    .join(&o.sha256);
                if !src.is_file() {
                    return refuse(
                        Code::OBJECT_ABSENT,
                        format!("{path:?}: {} is not in this store", o.id()),
                    );
                }
                fs::create_dir_all(target.parent().unwrap()).map_err(|e| io("mkdir parent", e))?;
                let _ = fs::remove_file(&target);
                let linked = if force_copy {
                    false
                } else {
                    std::os::unix::fs::symlink(relative(target.parent().unwrap(), &src), &target)
                        .is_ok()
                };
                if linked {
                    p.links += 1;
                } else {
                    fs::copy(&src, &target).map_err(|e| io("copy asset", e))?;
                    // Read-only, like the object it stands for.
                    let mut perms = fs::metadata(&target)
                        .map_err(|e| io("stat copy", e))?
                        .permissions();
                    std::os::unix::fs::PermissionsExt::set_mode(&mut perms, 0o444);
                    let _ = fs::set_permissions(&target, perms);
                    p.copies += 1;
                    rx.emit(Receipt::new(
                        "project_snapshot",
                        "symlink-fell-back-to-copy",
                        snap_hex,
                        store.generation(),
                        format!("{path:?}: {} bytes copied — dedup lost, correctness kept", o.length),
                    ));
                }
                p.bytes += o.length;
            }
            Entry::Blob(_) => {
                // A retained opaque carrier is tier 3's subject, taken whole and on demand.
                // Projecting it as a path here would be exactly the weight-container path
                // the ladder exists to forbid.
                return refuse(
                    Code::PROJECTION_WEIGHT_PATH,
                    format!(
                        "{path:?} is a retained blob — tier 3 `materialize_blob` streams it \
                         on demand; a projected tree never carries one"
                    ),
                );
            }
        }
    }
    Ok(p)
}

/// Tier 3. A private streamed copy of an ALREADY-RETAINED opaque carrier or document:
/// O(one segment) resident, atomic by rename, and never a synthesis — a carrier that was
/// never retained is `export_checkpoint` in a named exporter package (tfs-016), not this.
pub fn materialize_blob(
    store: &Store,
    what: &str,
    b: &BlobDescriptor,
    dest: &Path,
) -> Result<u64> {
    for (i, s) in b.segments.iter().enumerate() {
        if !store.contains(&s.sha256) {
            return refuse(
                Code::BLOB_NOT_RETAINED,
                format!(
                    "{what}: segment {i} {} is not retained here — materialization streams a \
                     carrier the store HOLDS; it never synthesizes one",
                    s.id()
                ),
            );
        }
    }
    let dir = dest.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(dir).map_err(|e| io("mkdir materialize", e))?;
    let tmp = dir.join(format!(
        ".materialize-{}-{}",
        std::process::id(),
        b.whole_sha256[0..12].to_string()
    ));
    let mut f = File::create(&tmp).map_err(|e| io("create materialize temp", e))?;
    // read_blob checks every segment length AND the whole digest as the bytes pass.
    let r = checkpoint::read_blob(store, what, b, &mut f);
    if let Err(e) = r {
        let _ = fs::remove_file(&tmp);
        return Err(e);
    }
    f.flush().map_err(|e| io("flush", e))?;
    f.sync_all().map_err(|e| io("fsync materialized", e))?;
    drop(f);
    fs::rename(&tmp, dest).map_err(|e| io("rename materialized", e))?;
    Ok(b.logical_size)
}

/// Pretty rendering — an ephemeral CLI/projection output. It parses the CANONICAL bytes
/// first, so it can only render a document the store would admit, and what it returns is
/// deliberately not admissible anywhere.
pub fn render(bytes: &[u8], max: usize) -> Result<String> {
    let v = canon::parse_canonical(bytes, max)?;
    Ok(canon::pretty(&v, 0))
}
