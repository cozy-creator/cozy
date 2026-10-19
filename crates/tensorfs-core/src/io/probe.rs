//! The hardware ceiling copies are measured against (design §13). Here, the disk: a volume's
//! sequential read rate, measured once per volume in a process.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

use platform::durable;

use super::host::{read_at, Bounce};

/// Bytes a probe reads at most.
const PROBE: u64 = 1 << 30;
const BLOCK: usize = 8 << 20;
const THREADS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ceiling {
    /// Sequential read GB/s of the volume.
    pub disk_gbps: f64,
    /// Measured past the page cache (O_DIRECT); else the filesystem refused it.
    pub disk_direct: bool,
}

/// What reads from `path`'s volume (and, with `device`, copies to that GPU) can reach.
pub fn ceiling(device: Option<i32>, path: &Path) -> io::Result<Ceiling> {
    if device.is_some() {
        return Err(io::Error::new(io::ErrorKind::Unsupported, "the H2D ceiling needs the CUDA engine"));
    }
    let (disk_gbps, disk_direct) = disk(path)?;
    Ok(Ceiling { disk_gbps, disk_direct })
}

/// GB/s reading the volume holding `path` (a file, or a folder of them): its largest files in
/// whole blocks, up to 1 GiB, on 8 threads, O_DIRECT where the filesystem allows (else through
/// the page cache, and `false`).
pub fn disk(path: &Path) -> io::Result<(f64, bool)> {
    static SEEN: Mutex<BTreeMap<u64, (f64, bool)>> = Mutex::new(BTreeMap::new());
    let volume = durable::device(path).map_err(io::Error::other)?;
    if let Some(&seen) = SEEN.lock().unwrap().get(&volume) {
        return Ok(seen);
    }
    let mut found = Vec::new();
    files(path, &mut found)?;
    found.sort_by(|a, b| b.0.cmp(&a.0));
    let (mut opened, mut blocks, mut total) = (Vec::new(), Vec::new(), 0);
    for (len, p) in found.into_iter().take_while(|(len, _)| *len > 0) {
        if total >= PROBE {
            break;
        }
        let f = File::open(&p)?;
        opened.push((durable::open_direct(&f).ok().flatten(), f));
        for at in (0..len).step_by(BLOCK) {
            if total >= PROBE {
                break;
            }
            blocks.push((opened.len() - 1, at));
            total += len.min(at + BLOCK as u64) - at;
        }
    }
    if blocks.is_empty() {
        return Err(io::Error::new(io::ErrorKind::NotFound, format!("{}: no bytes to read", path.display())));
    }
    let direct = opened.iter().all(|(d, _)| d.is_some());
    let (next, bytes, error) = (AtomicUsize::new(0), AtomicU64::new(0), Mutex::new(None));
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..THREADS.min(blocks.len()) {
            s.spawn(|| {
                let mut bounce = Bounce::default();
                while let Some(&(i, at)) = blocks.get(next.fetch_add(1, Relaxed)) {
                    let (d, f) = &opened[i];
                    match read_at(if direct { d.as_ref().unwrap() } else { f }, bounce.get(BLOCK), at) {
                        Ok(n) => bytes.fetch_add(n as u64, Relaxed),
                        Err(e) => return *error.lock().unwrap() = Some(e),
                    };
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let seen = (bytes.into_inner() as f64 / start.elapsed().as_nanos().max(1) as f64, direct);
    SEEN.lock().unwrap().insert(volume, seen);
    Ok(seen)
}

fn files(path: &Path, out: &mut Vec<(u64, PathBuf)>) -> io::Result<()> {
    let meta = fs::metadata(path)?;
    if meta.is_file() {
        out.push((meta.len(), path.to_path_buf()));
    }
    if !meta.is_dir() {
        return Ok(());
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_symlink() {
            files(&entry.path(), out)?;
        }
    }
    Ok(())
}
