//! What copies are measured against (memory-varena §5.8): a GPU's pinned host→device bandwidth
//! (best of 1, 2 and 4 streams) and PCIe link, and a store volume's O_DIRECT sequential read.
//! Each ceiling is measured once per machine and kept in `~/.cache/varena/ceilings`, keyed by
//! GPU UUID and driver or by the volume's device. Every copy run lands in one log (`take`).

use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use platform::durable;

use crate::cuda::{self, CUstream};
use crate::Result;

const H2D_PROBE: usize = 128 << 20;
const DISK_PROBE: u64 = 1 << 30;
const DISK_BLOCK: usize = 8 << 20;
const DISK_THREADS: usize = 8;
const LOG_KEEP: usize = 256;

/// One run of copies: a GPU's copy stream busy from its first job to its last landing
/// (`h2d`), or a host layout filled from a store's files (`disk`).
#[derive(Debug, Clone, Default)]
pub struct Copy {
    pub kind: &'static str,
    /// -1 for a disk fill.
    pub gpu: i32,
    pub bytes: u64,
    pub ns: u64,
    /// GB/s; 0 when unmeasured.
    pub ceiling: f64,
    /// The PCIe link as the run ended (`Gen4 x16 of Gen5 x16`), "" if unknown or a disk fill.
    pub link: String,
    /// h2d: GPU time in copies; the rest of `ns` the host left the stream idle.
    pub busy_ns: u64,
    /// h2d: host time page-locking source ranges.
    pub pin_ns: u64,
    /// h2d: bytes through the pinned staging buffers.
    pub staged: u64,
    /// disk: bytes the page cache held.
    pub cached: u64,
}

static LOG: Mutex<Vec<Copy>> = Mutex::new(Vec::new());

pub(crate) fn record(c: Copy) {
    let mut log = LOG.lock().unwrap();
    if log.len() >= LOG_KEEP {
        log.remove(0);
    }
    log.push(c);
}

/// Every copy run since the last call.
pub fn take() -> Vec<Copy> {
    std::mem::take(&mut LOG.lock().unwrap())
}

fn path() -> Option<PathBuf> {
    Some(platform::paths::login_home()?.join(".cache/varena/ceilings"))
}

fn cached(key: &str) -> Option<Vec<f64>> {
    let text = std::fs::read_to_string(path()?).ok()?;
    let line = text.lines().rev().find_map(|l| l.strip_prefix(key)?.strip_prefix('\t'))?;
    line.split('\t').map(|v| v.parse().ok()).collect()
}

fn keep(key: &str, values: &[f64]) {
    let Some(p) = path() else { return };
    let line: String = values.iter().map(|v| format!("\t{v}")).collect();
    let _ = std::fs::create_dir_all(p.parent().unwrap());
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(p) {
        let _ = f.write_all(format!("{key}{line}\n").as_bytes());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link {
    pub gen: u32,
    pub width: u32,
    pub max_gen: u32,
    pub max_width: u32,
}

impl std::fmt::Display for Link {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Gen{} x{} of Gen{} x{}", self.gen, self.width, self.max_gen, self.max_width)
    }
}

/// The link of PCI device `pci` (`0000:65:00.0`) from sysfs. An idle GPU trains its link down,
/// so read it right after a copy.
pub fn link(pci: &str) -> Option<Link> {
    let read = |f: &str| std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/{f}")).ok();
    let gen = |f: &str| -> Option<u32> {
        let gts: f64 = read(f)?.split_whitespace().next()?.parse().ok()?;
        [2.5, 5.0, 8.0, 16.0, 32.0, 64.0].iter().position(|&g| g == gts).map(|i| i as u32 + 1)
    };
    let width = |f: &str| read(f)?.trim().parse().ok();
    Some(Link { gen: gen("current_link_speed")?, width: width("current_link_width")?, max_gen: gen("max_link_speed")?, max_width: width("max_link_width")? })
}

/// Pinned host→device GB/s on CUDA device `device`, and the streams that reached it: measured
/// once per GPU and driver, on its primary context.
pub fn h2d(device: i32) -> Option<(f64, u32)> {
    let (dev, ctx) = cuda::primary_context(device).ok()?;
    let _c = cuda::CtxGuard::enter(ctx).ok()?;
    let driver = std::fs::read_to_string("/sys/module/nvidia/version").unwrap_or_default();
    let key = format!("h2d {} {}", cuda::uuid(dev).ok()?, driver.trim());
    if let Some(&[gbps, streams]) = cached(&key).as_deref() {
        return Some((gbps, streams as u32));
    }
    let (gbps, streams) = measure_h2d().inspect_err(|e| tracing::warn!("varena: no H2D ceiling: {e}")).ok()?;
    keep(&key, &[gbps, streams as f64]);
    Some((gbps, streams))
}

/// Buffers and streams that free themselves however the probe ends.
struct Probe {
    host: *mut c_void,
    dev: u64,
    streams: Vec<CUstream>,
}

impl Drop for Probe {
    fn drop(&mut self) {
        let Ok(d) = cuda::driver() else { return };
        // SAFETY: ours, and every copy into them synced or never enqueued.
        unsafe {
            for &s in &self.streams {
                (d.stream_synchronize)(s);
                (d.stream_destroy)(s);
            }
            if self.dev != 0 {
                (d.mem_free)(self.dev);
            }
            if !self.host.is_null() {
                (d.mem_free_host)(self.host);
            }
        }
    }
}

fn measure_h2d() -> Result<(f64, u32)> {
    let d = cuda::driver()?;
    let mut p = Probe { host: std::ptr::null_mut(), dev: 0, streams: Vec::new() };
    // SAFETY: out-params; the current context's device and host memory.
    unsafe {
        cuda::check("cuMemHostAlloc", (d.mem_host_alloc)(&mut p.host, H2D_PROBE, 0))?;
        cuda::check("cuMemAlloc", (d.mem_alloc)(&mut p.dev, H2D_PROBE))?;
        for _ in 0..4 {
            let mut s: CUstream = std::ptr::null_mut();
            cuda::check("cuStreamCreate", (d.stream_create)(&mut s, cuda::STREAM_NON_BLOCKING))?;
            p.streams.push(s);
        }
    }
    let mut best = (0.0, 1);
    for n in [1, 2, 4, 1, 2, 4, 1, 2, 4] {
        let piece = H2D_PROBE / n;
        let t = Instant::now();
        for (i, &s) in p.streams[..n].iter().enumerate() {
            // SAFETY: both ranges lie inside the probe's buffers.
            cuda::check("cuMemcpyHtoDAsync", unsafe { (d.memcpy_htod_async)(p.dev + (i * piece) as u64, (p.host as *const u8).add(i * piece) as *const c_void, piece, s) })?;
        }
        for &s in &p.streams[..n] {
            // SAFETY: our stream.
            cuda::check("cuStreamSynchronize", unsafe { (d.stream_synchronize)(s) })?;
        }
        let gbps = H2D_PROBE as f64 / t.elapsed().as_nanos() as f64;
        if gbps > best.0 {
            best = (gbps, n as u32);
        }
    }
    Ok(best)
}

/// The O_DIRECT sequential read GB/s of the volume holding `files`: whole 8 MiB blocks of
/// them, up to 1 GiB, on 8 threads. Measured once per volume; None where O_DIRECT is refused.
pub fn disk(files: &[&File]) -> Option<f64> {
    let key = format!("disk {}", files.first()?.metadata().ok()?.dev());
    if let Some(&[gbps]) = cached(&key).as_deref() {
        return Some(gbps);
    }
    let direct: Vec<File> = files.iter().filter_map(|f| durable::open_direct(f).ok().flatten()).collect();
    let mut blocks = Vec::new();
    for (i, f) in direct.iter().enumerate() {
        let len = f.metadata().ok()?.len();
        let fit = (len / DISK_BLOCK as u64).min((DISK_PROBE / DISK_BLOCK as u64).saturating_sub(blocks.len() as u64));
        blocks.extend((0..fit).map(|b| (i, b * DISK_BLOCK as u64)));
    }
    if blocks.is_empty() {
        return None;
    }
    let (next, failed) = (AtomicUsize::new(0), AtomicUsize::new(0));
    let t = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..DISK_THREADS.min(blocks.len()) {
            s.spawn(|| {
                // SAFETY: a fresh private anonymous mapping, page aligned as O_DIRECT needs.
                let buf = unsafe { libc::mmap(std::ptr::null_mut(), DISK_BLOCK, libc::PROT_READ | libc::PROT_WRITE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, -1, 0) };
                if buf == libc::MAP_FAILED {
                    failed.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                while let Some(&(i, at)) = blocks.get(next.fetch_add(1, Ordering::Relaxed)) {
                    // SAFETY: an aligned read into our own buffer.
                    if unsafe { libc::pread(direct[i].as_raw_fd(), buf, DISK_BLOCK, at as libc::off_t) } != DISK_BLOCK as isize {
                        failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
                // SAFETY: our mapping.
                unsafe { libc::munmap(buf, DISK_BLOCK) };
            });
        }
    });
    if failed.into_inner() > 0 {
        return None;
    }
    let gbps = (blocks.len() * DISK_BLOCK) as f64 / t.elapsed().as_nanos() as f64;
    keep(&key, &[gbps]);
    Some(gbps)
}
