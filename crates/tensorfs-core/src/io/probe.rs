//! The hardware ceiling copies are measured against (design §13). Here, the disk: a volume's
//! sequential read rate, measured once per volume and boot, kept in `~/.cache/tensorfs-ceilings`
//! (tab-separated `key value…` lines; the last for a key wins).

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Mutex;
use std::time::Instant;

use platform::durable;

use super::host::{read_at, Bounce};

/// Bytes a probe reads at most.
const PROBE: u64 = 1 << 30;
/// Bytes a probe must read for its rate to stand for the volume (and be kept).
const KEEP: u64 = 256 << 20;
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
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "the H2D ceiling needs the CUDA engine",
        ));
    }
    let (disk_gbps, disk_direct) = disk(path)?;
    Ok(Ceiling {
        disk_gbps,
        disk_direct,
    })
}

/// GB/s reading the volume holding `path` (a file, or a folder of them): its largest files in
/// whole blocks, up to 1 GiB, on 8 threads, O_DIRECT where the filesystem allows (else through
/// the page cache, and `false`). A probe of under 256 MiB is returned, not kept.
pub fn disk(path: &Path) -> io::Result<(f64, bool)> {
    static SEEN: Mutex<BTreeMap<u64, (f64, bool)>> = Mutex::new(BTreeMap::new());
    let volume = durable::device(path).map_err(io::Error::other)?;
    let key = platform::process::boot_id()
        .ok()
        .filter(|s| !s.is_empty())
        .map(|boot| format!("disk {boot} {volume}"));
    if let Some(&seen) = SEEN.lock().unwrap().get(&volume) {
        return Ok(seen);
    }
    if let Some(seen) = key.as_deref().and_then(kept) {
        SEEN.lock().unwrap().insert(volume, seen);
        return Ok(seen);
    }
    let mut found = Vec::new();
    files(path, volume, &mut found)?;
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
        SEEN.lock().unwrap().insert(volume, seen);
        if let Some(key) = key {
            keep(&key, seen);
        }
    }
    Ok(seen)
}

fn ceilings() -> Option<PathBuf> {
    Some(platform::paths::login_home()?.join(".cache/tensorfs-ceilings"))
}

fn kept(key: &str) -> Option<(f64, bool)> {
    let text = fs::read_to_string(ceilings()?).ok()?;
    saved(&text, key)
}

fn saved(text: &str, key: &str) -> Option<(f64, bool)> {
    let line = text
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix(key)?.strip_prefix('\t'))?;
    let (rate, direct) = line.split_once('\t')?;
    let rate: f64 = rate.parse().ok()?;
    let direct = match direct {
        "0" => false,
        "1" => true,
        _ => return None,
    };
    (rate.is_finite() && rate > 0.0).then_some((rate, direct))
}

fn keep(key: &str, (rate, direct): (f64, bool)) {
    let Some(path) = ceilings() else { return };
    let _ = fs::create_dir_all(path.parent().unwrap());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(path) {
        if f.lock().is_ok() {
            let _ = f.write_all(format!("{key}\t{rate}\t{}\n", u8::from(direct)).as_bytes());
        }
    }
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
    fn persisted_rates_are_positive_finite_and_match_the_exact_volume_and_boot() {
        let key = "disk boot-a 42";
        assert_eq!(
            saved("disk boot-a 420\t99\t1\ndisk boot-b 42\t88\t1\n", key),
            None
        );
        assert_eq!(
            saved("disk boot-a 42\t2.5\t0\ndisk boot-a 42\t3.25\t1\n", key),
            Some((3.25, true))
        );
        for value in [
            "NaN\t1", "inf\t1", "0\t1", "-1\t1", "2\t3", "2\t1\t9", "no rate",
        ] {
            assert_eq!(saved(&format!("{key}\t{value}\n"), key), None, "{value}");
        }
    }
}
