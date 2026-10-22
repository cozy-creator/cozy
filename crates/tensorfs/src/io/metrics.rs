//! What TensorFS copies are measured against: a GPU's pinned host→device bandwidth (best of 1,
//! 2 and 4 streams) and PCIe link, its readers' page cache → pinned copy (`device`), and a
//! volume's O_DIRECT read (`probe`). Ceilings are kept per boot (`probe::keep`), a GPU's by UUID
//! and driver. Reads never measure one (`io::ceiling` does); a run faster than a ceiling raises
//! it. Every copy run lands in one log (`take`).

use std::ffi::c_void;
use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::sync::Mutex;
use std::time::Instant;

use super::{cuda, probe};
use cuda::Result;

const H2D_PROBE: usize = 128 << 20;
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
    /// h2d: the stage that sets `ceiling`: `disk` (bytes read past the page cache), `copy` (page
    /// cache → pinned staging) or `h2d`; "" when unmeasured.
    pub bound: &'static str,
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
    /// h2d: staging readers the run ended with.
    pub readers: u64,
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

/// A run of reads from `files`, logged against its volume's kept ceiling. Only a run the page
/// cache had no part in measures the volume: one of a probe's size seeds the ceiling, any that
/// beats it raises it. A run that read cached bytes is logged unmeasured (ceiling 0). Returns the
/// volume's ceiling after the run (0 when unmeasured).
pub(crate) fn disk_run(files: &[&File], bytes: u64, ns: u64, direct: u64, cached: u64) -> f64 {
    let Some(volume) = files.first().and_then(|f| f.metadata().ok()).map(|m| volume(m.dev())) else {
        return 0.0;
    };
    let rate = direct as f64 / ns.max(1) as f64;
    let kept = probe::disk_known(volume).map(|(gbps, _)| gbps);
    let measures = cached == 0 && (direct >= probe::KEEP || kept.is_some_and(|k| rate > k));
    if measures {
        probe::disk_keep(volume, (rate, true));
    }
    let ceiling = match (cached, kept) {
        (0, Some(k)) => k.max(rate),
        (0, None) if measures => rate,
        _ => 0.0,
    };
    record(Copy { kind: "disk", gpu: -1, bytes, ns, cached, ceiling, ..Default::default() });
    kept.unwrap_or(0.0).max(if measures { rate } else { 0.0 })
}

/// A run of copies to GPU `c.gpu`, held against its slowest stage: the volume for the `direct`
/// bytes read past the page cache (`disk`, GB/s), page cache → pinned staging for the cached
/// ones, and the link for all. The stages overlap, so the run can take no less than its slowest
/// stage's share. A stage not yet measured leaves the run unmeasured (ceiling 0); a run faster
/// than a stage's kept ceiling raises it.
pub(crate) fn h2d_run(mut c: Copy, direct: u64, disk: f64) {
    let ns = c.ns.max(1) as f64;
    let mut stages = vec![("h2d", c.bytes, raised(gpu_key("h2d", c.gpu), c.bytes as f64 / ns))];
    if c.cached > 0 {
        stages.push(("copy", c.cached, raised(gpu_key("copy", c.gpu), c.cached as f64 / ns)));
    }
    if direct > 0 {
        stages.push(("disk", direct, disk));
    }
    if stages.iter().all(|s| s.2 > 0.0) {
        let (bound, bytes, gbps) = stages
            .into_iter()
            .max_by(|a, b| (a.1 as f64 / a.2).total_cmp(&(b.1 as f64 / b.2)))
            .expect("the link stage");
        c.ceiling = c.bytes as f64 * gbps / bytes as f64;
        c.bound = bound;
    }
    record(c);
}

/// `key`'s kept GB/s, raised to `rate` when the run beat it; 0 when unmeasured.
fn raised(key: Option<String>, rate: f64) -> f64 {
    let Some(kept) = key.as_deref().and_then(probe::known) else { return 0.0 };
    if rate > kept[0] {
        probe::keep(key.as_deref().unwrap(), &[rate, kept.get(1).copied().unwrap_or(1.0)]);
    }
    kept[0].max(rate)
}

/// A `st_dev` as `durable::device` names its volume.
#[cfg(target_os = "linux")]
fn volume(dev: u64) -> u64 {
    u64::from(libc::major(dev)) << 32 | u64::from(libc::minor(dev))
}

#[cfg(not(target_os = "linux"))]
fn volume(dev: u64) -> u64 {
    dev
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
        write!(
            f,
            "Gen{} x{} of Gen{} x{}",
            self.gen, self.width, self.max_gen, self.max_width
        )
    }
}

/// The link of PCI device `pci` (`0000:65:00.0`) from sysfs. An idle GPU trains its link down,
/// so read it right after a copy.
pub fn link(pci: &str) -> Option<Link> {
    let read = |f: &str| std::fs::read_to_string(format!("/sys/bus/pci/devices/{pci}/{f}")).ok();
    let gen = |f: &str| -> Option<u32> {
        let gts: f64 = read(f)?.split_whitespace().next()?.parse().ok()?;
        [2.5, 5.0, 8.0, 16.0, 32.0, 64.0]
            .iter()
            .position(|&g| g == gts)
            .map(|i| i as u32 + 1)
    };
    let width = |f: &str| read(f)?.trim().parse().ok();
    Some(Link {
        gen: gen("current_link_speed")?,
        width: width("current_link_width")?,
        max_gen: gen("max_link_speed")?,
        max_width: width("max_link_width")?,
    })
}

/// The current PCIe link of CUDA device `device`, without creating a second context.
pub fn device_link(device: i32) -> Option<Link> {
    link(&pci_address(device)?)
}

/// CUDA device `device`'s PCI address as sysfs names it (`0000:65:00.0`).
pub(crate) fn pci_address(device: i32) -> Option<String> {
    let d = cuda::driver().ok()?;
    let mut dev = 0;
    // SAFETY: driver out-param; an invalid ordinal returns an error.
    cuda::check("cuDeviceGet", unsafe { (d.device_get)(&mut dev, device) }).ok()?;
    let attr = |a| cuda::attribute(dev, a).ok();
    Some(format!(
        "{:04x}:{:02x}:{:02x}.0",
        attr(cuda::ATTR_PCI_DOMAIN_ID)?,
        attr(cuda::ATTR_PCI_BUS_ID)?,
        attr(cuda::ATTR_PCI_DEVICE_ID)?
    ))
}

/// Pinned host→device GB/s on CUDA device `device`, and the streams that reached it: kept this
/// boot, else measured on its primary context.
pub fn h2d(device: i32) -> Option<(f64, u32)> {
    let key = gpu_key("h2d", device)?;
    if let Some(&[gbps, streams]) = probe::known(&key).as_deref() {
        return Some((gbps, streams as u32));
    }
    let (dev, ctx) = cuda::primary_context(device).ok()?;
    let retained = PrimaryContext(dev);
    let _c = cuda::CtxGuard::enter(ctx).ok()?;
    let (gbps, streams) = measure_h2d(device, ctx, retained)
        .inspect_err(|e| tracing::warn!("tensorfs: no H2D ceiling: {e}"))
        .ok()?;
    probe::keep(&key, &[gbps, streams as f64]);
    Some((gbps, streams))
}

/// Page cache → pinned staging GB/s for GPU `device`'s readers, and the readers that reached it:
/// kept this boot, else measured (`device::copy_ceiling`).
pub fn copy(device: i32) -> Option<(f64, u32)> {
    let key = gpu_key("copy", device)?;
    if let Some(&[gbps, readers]) = probe::known(&key).as_deref() {
        return Some((gbps, readers as u32));
    }
    let (gbps, readers) = super::device::copy_ceiling(device)
        .inspect_err(|e| tracing::warn!("tensorfs: no host copy ceiling: {e}"))
        .ok()?;
    probe::keep(&key, &[gbps, readers as f64]);
    Some((gbps, readers))
}

/// GPU `device`'s kept H2D GB/s this boot, without measuring.
pub(crate) fn h2d_kept(device: i32) -> Option<f64> {
    probe::known(&gpu_key("h2d", device)?).map(|kept| kept[0])
}

/// A GPU's ceiling key for `kind`: its UUID and the driver.
fn gpu_key(kind: &str, device: i32) -> Option<String> {
    let d = cuda::driver().ok()?;
    let mut dev = 0;
    // SAFETY: driver out-param; an invalid ordinal returns an error.
    cuda::check("cuDeviceGet", unsafe { (d.device_get)(&mut dev, device) }).ok()?;
    let driver = std::fs::read_to_string("/sys/module/nvidia/version").unwrap_or_default();
    Some(format!(
        "{kind} {} {}",
        cuda::uuid(dev).ok()?,
        driver.trim()
    ))
}

struct PrimaryContext(i32);
impl Drop for PrimaryContext {
    fn drop(&mut self) {
        if let Ok(driver) = cuda::driver() {
            // Balances this probe's successful primary-context retain.
            unsafe { (driver.primary_ctx_release)(self.0) };
        }
    }
}

/// Actual probe owners, including the context lease. A failed completion check retains the
/// entire owner box for process lifetime; no failed stream or allocation is ever reused.
struct ProbeResources {
    _context: PrimaryContext,
    ctx: cuda::CUcontext,
    host: *mut c_void,
    dev: u64,
    streams: Vec<cuda::Stream>,
}

impl ProbeResources {
    fn drain(&self) -> Result<()> {
        let mut first = None;
        for stream in &self.streams {
            if let Err(error) = stream.drain() {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), Err)
    }
}

struct Probe(Option<Box<ProbeResources>>);
impl Drop for Probe {
    fn drop(&mut self) {
        let Some(owners) = self.0.take() else { return };
        let drained = owners
            .drain()
            .and_then(|()| cuda::CtxGuard::enter(owners.ctx));
        let Ok(_context) = drained else {
            // Keep the owned streams, both allocations AND the primary-context lease.
            // Dropping raw pointer identities alone would not preserve their ownership.
            std::mem::forget(owners);
            return;
        };
        let Ok(driver) = cuda::driver() else {
            std::mem::forget(owners);
            return;
        };
        // Every private stream has a checked completion proof before either allocation frees.
        unsafe {
            if owners.dev != 0 {
                (driver.mem_free)(owners.dev);
            }
            if !owners.host.is_null() {
                (driver.mem_free_host)(owners.host);
            }
        }
        // Stream owners and the retained primary context release after the buffers.
        drop(owners);
    }
}

fn measure_h2d(device: i32, ctx: cuda::CUcontext, retained: PrimaryContext) -> Result<(f64, u32)> {
    let driver = cuda::driver()?;
    let mut probe = Probe(Some(Box::new(ProbeResources {
        _context: retained,
        ctx,
        host: std::ptr::null_mut(),
        dev: 0,
        streams: Vec::new(),
    })));
    let p = probe.0.as_mut().expect("probe owners live");
    unsafe {
        cuda::check(
            "cuMemHostAlloc",
            (driver.mem_host_alloc)(&mut p.host, H2D_PROBE, 0),
        )?;
        cuda::check("cuMemAlloc", (driver.mem_alloc)(&mut p.dev, H2D_PROBE))?;
    }
    for _ in 0..4 {
        p.streams.push(cuda::Stream::new(device)?);
    }
    let mut best = (0.0, 1);
    for n in [1, 2, 4, 1, 2, 4, 1, 2, 4] {
        let piece = H2D_PROBE / n;
        let t = Instant::now();
        for (i, stream) in p.streams[..n].iter().enumerate() {
            // Both ranges lie inside the owned probe allocations.
            cuda::check("cuMemcpyHtoDAsync", unsafe {
                (driver.memcpy_htod_async)(
                    p.dev + (i * piece) as u64,
                    (p.host as *const u8).add(i * piece).cast(),
                    piece,
                    stream.raw() as _,
                )
            })?;
        }
        for stream in &p.streams[..n] {
            stream.drain()?;
        }
        let gbps = H2D_PROBE as f64 / t.elapsed().as_nanos() as f64;
        if gbps > best.0 {
            best = (gbps, n as u32);
        }
    }
    Ok(best)
}
