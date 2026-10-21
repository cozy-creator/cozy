//! The hardware ceilings copies are measured against (design §13): a volume's O_DIRECT
//! sequential read rate here, a GPU's pinned H2D rate in `metrics`. Each is kept per boot in
//! the TensorFS store ([`keep_ceilings_in`]'s, else [`crate::home::resolve`]'s) at `tmp/ceilings/table`,
//! tab-separated `key value…` lines; a key's highest first value wins, so a probe taken at a
//! contended moment, or a copy that ran faster than it, never leaves a ceiling stuck low.

use std::collections::BTreeMap;
use std::cmp::Reverse;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use platform::durable;

use super::host::{read_at, Bounce};

/// Bytes a probe reads at most.
const PROBE: u64 = 1 << 30;
/// Bytes a probe must read for its rate to stand for the volume (and be kept).
pub(super) const KEEP: u64 = 256 << 20;
/// Reads as a cold device read does: 1 MiB requests (`ReadOpts::default().request`) on 8 threads.
/// 8 MiB blocks measured 3-7% under that on the same volumes.
const BLOCK: usize = 1 << 20;
const THREADS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ceiling {
    /// Sequential read GB/s of the volume.
    pub disk_gbps: f64,
    /// Measured past the page cache (O_DIRECT); else the filesystem refused it.
    pub disk_direct: bool,
    /// Pinned host→device GB/s, when a CUDA device was requested.
    pub h2d_gbps: Option<f64>,
    /// Streams used by the fastest measured H2D probe.
    pub streams: Option<u32>,
}

/// What reads from `path`'s volume (and, with `device`, copies to that GPU) can reach.
pub fn ceiling(device: Option<i32>, path: &Path) -> io::Result<Ceiling> {
    #[cfg(unix)]
    let gpu = device.map(|device| super::metrics::h2d(device).ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, format!("CUDA device {device} has no measurable H2D ceiling")))).transpose()?;
    #[cfg(not(unix))]
    let gpu: Option<(f64, u32)> = if device.is_some() { return Err(io::Error::new(io::ErrorKind::Unsupported, "H2D ceiling is unavailable on this platform")); } else { None };
    let (disk_gbps, disk_direct) = disk(path)?;
    Ok(Ceiling {
        disk_gbps,
        disk_direct,
        h2d_gbps: gpu.map(|g| g.0),
        streams: gpu.map(|g| g.1),
    })
}

/// GB/s reading the volume holding `path` (a file, or a folder of them): its largest files in
/// whole blocks, up to 1 GiB, on 8 threads, O_DIRECT where the filesystem allows (else through
/// the page cache, and `false`). A probe of under 256 MiB is returned, not kept.
pub fn disk(path: &Path) -> io::Result<(f64, bool)> {
    let volume = durable::device(path).map_err(io::Error::other)?;
    if let Some(seen) = disk_known(volume) {
        return Ok(seen);
    }
    let mut found = Vec::new();
    files(path, volume, &mut found)?;
    found.sort_by_key(|f| Reverse(f.0));
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
            let n = (len - at).min(BLOCK as u64).min(PROBE - total);
            blocks.push((opened.len() - 1, at, n as usize));
            total += n;
        }
    }
    if blocks.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("{}: no bytes to read", path.display()),
        ));
    }
    let direct = opened.iter().all(|(d, _)| d.is_some());
    let (next, bytes, error) = (AtomicUsize::new(0), AtomicU64::new(0), Mutex::new(None));
    let start = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..THREADS.min(blocks.len()) {
            s.spawn(|| {
                let mut bounce = Bounce::default();
                while let Some(&(i, at, need)) = blocks.get(next.fetch_add(1, Relaxed)) {
                    let (d, f) = &opened[i];
                    match read_at(
                        if direct { d.as_ref().unwrap() } else { f },
                        bounce.get(BLOCK),
                        at,
                    ) {
                        Ok(n) if n >= need => bytes.fetch_add(need as u64, Relaxed),
                        Ok(_) => {
                            return *error.lock().unwrap() = Some(io::Error::new(
                                io::ErrorKind::UnexpectedEof,
                                "disk probe file changed while reading",
                            ))
                        }
                        Err(e) => return *error.lock().unwrap() = Some(e),
                    };
                }
            });
        }
    });
    if let Some(e) = error.into_inner().unwrap() {
        return Err(e);
    }
    let read = bytes.into_inner();
    let seen = (
        read as f64 / start.elapsed().as_nanos().max(1) as f64,
        direct,
    );
    if read >= KEEP {
        disk_keep(volume, seen);
    }
    Ok(seen)
}

/// The kept ceiling of volume `volume` this boot: GB/s and whether it was O_DIRECT.
pub(crate) fn disk_known(volume: u64) -> Option<(f64, bool)> {
    match known(&format!("disk {volume}"))?.as_slice() {
        &[rate, direct] if direct == 0.0 || direct == 1.0 => Some((rate, direct == 1.0)),
        _ => None,
    }
}

/// Keeps `seen` for volume `volume`, unless a higher rate is kept already.
pub(crate) fn disk_keep(volume: u64, (rate, direct): (f64, bool)) {
    keep(&format!("disk {volume}"), &[rate, f64::from(u8::from(direct))]);
}

/// The kept values of `key` this boot: the line with the highest first value.
pub(crate) fn known(key: &str) -> Option<Vec<f64>> {
    let key = format!("{} {key}", boot()?);
    if let Some(seen) = MEMO.lock().unwrap().get(&key) {
        return Some(seen.clone());
    }
    let seen = saved(&fs::read_to_string(table()?).ok()?, &key)?;
    MEMO.lock().unwrap().insert(key, seen.clone());
    Some(seen)
}

/// Keeps `values` for `key` this boot, when its first value beats the one kept.
pub(crate) fn keep(key: &str, values: &[f64]) {
    let first = values.first().copied().unwrap_or(0.0);
    if !(first.is_finite() && first > 0.0) || known(key).is_some_and(|k| k[0] >= first) {
        return;
    }
    let Some((boot, path)) = boot().zip(table()) else { return };
    let key = format!("{boot} {key}");
    MEMO.lock().unwrap().insert(key.clone(), values.to_vec());
    let line: String = values.iter().map(|v| format!("\t{v}")).collect();
    let _ = fs::create_dir_all(path.parent().expect("the table has a directory"));
    // Windows locks need read or write access, which append alone does not grant.
    if let Ok(mut f) = OpenOptions::new().read(true).create(true).append(true).open(path) {
        if f.lock().is_ok() {
            let _ = f.write_all(format!("{key}{line}\n").as_bytes());
        }
    }
}

static MEMO: Mutex<BTreeMap<String, Vec<f64>>> = Mutex::new(BTreeMap::new());

fn boot() -> Option<String> {
    platform::process::boot_id().ok().filter(|s| !s.is_empty())
}

static STORE: OnceLock<PathBuf> = OnceLock::new();

/// Keeps this process's ceilings in the store at `root` (the one `tfs --store` names), not the
/// home one; the first root named holds.
pub fn keep_ceilings_in(root: &Path) {
    let _ = STORE.set(root.to_path_buf());
}

/// `tmp/ceilings/table` in the TensorFS store. A directory in `tmp/` is its maker's: GC leaves it.
fn table() -> Option<PathBuf> {
    let store = match STORE.get() {
        Some(root) => root.clone(),
        None => crate::home::resolve(None).ok()?,
    };
    Some(store.join("tmp").join("ceilings").join("table"))
}

/// The values of `key`'s line with the highest positive, finite first value.
fn saved(text: &str, key: &str) -> Option<Vec<f64>> {
    let lines = text.lines().filter_map(|l| l.strip_prefix(key)?.strip_prefix('\t'));
    let values = lines.filter_map(|l| l.split('\t').map(|v| v.parse::<f64>().ok()).collect::<Option<Vec<f64>>>());
    let valid = values.filter(|v| v.first().is_some_and(|f| f.is_finite() && *f > 0.0));
    valid.max_by(|a, b| a[0].total_cmp(&b[0]))
}

fn files(path: &Path, volume: u64, out: &mut Vec<(u64, PathBuf)>) -> io::Result<()> {
    if durable::device(path).map_err(io::Error::other)? != volume {
        return Ok(());
    }
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
            files(&entry.path(), volume, out)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::saved;

    #[test]
    fn a_kept_ceiling_is_the_highest_positive_rate_for_the_exact_key() {
        let key = "boot-a disk 42";
        assert_eq!(saved("boot-a disk 420\t99\t1\nboot-b disk 42\t88\t1\n", key), None);
        let text = "boot-a disk 42\t3.25\t1\nboot-a disk 42\t0.58\t1\nboot-a disk 42\t2.5\t0\n";
        assert_eq!(saved(text, key), Some(vec![3.25, 1.0]));
        for value in ["NaN\t1", "inf\t1", "0\t1", "-1\t1", "no rate"] {
            assert_eq!(saved(&format!("{key}\t{value}\n"), key), None, "{value}");
        }
    }
}
