//! Shared plumbing for the G-series GPU experiments (g1_vmm, g2_overlap): flags, stats, a JSON
//! line, driver symbols the plane does not carry, host buffers, VMM ranges and a cuBLAS GEMM load.
#![allow(dead_code, unused_macros, unused_imports)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::fmt::{self, Display};
use std::path::Path;
use std::ptr::null_mut;
use std::str::FromStr;
use std::sync::OnceLock;
use std::time::Instant;

use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_NOW};
use libloading::Library;
pub use tensorfs_plane::cuda::{
    self, check, AccessDesc, AllocationProp, CUcontext, CUdevice, CUdeviceptr, CUevent, CUresult,
    CUstream, CtxGuard, Driver, Handle, MemLocation,
};
pub use tensorfs_plane::{Error, Result};

pub const MIB: usize = 1 << 20;
pub const GIB: usize = 1 << 30;
pub const GRANULARITY_RECOMMENDED: u32 = 1;
pub const DEFAULT_CUBLAS: &str =
    "/home/fidika/cozy_v2/packages/.venv/lib/python3.13/site-packages/nvidia/cu13/lib/libcublas.so.13";

/// `cu!(d.mem_map(..))`: a checked call through a driver table bound to the ident `d`.
macro_rules! cu {
    ($t:ident . $f:ident ( $($a:expr),* $(,)? )) => {
        $crate::gpu_bench::check(stringify!($f), unsafe { ($t.$f)($($a),*) })
    };
}

macro_rules! obj {
    ($($k:expr => $v:expr),* $(,)?) => {
        $crate::gpu_bench::J::O(vec![$(($k.to_string(), $crate::gpu_bench::J::from($v))),*])
    };
}

macro_rules! row {
    ($($c:expr),* $(,)?) => { println!("| {} |", [$($c.to_string()),*].join(" | ")) };
}

pub fn table(head: &[&str]) {
    println!("| {} |\n|{}", head.join(" | "), "---|".repeat(head.len()));
}

pub fn us(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e6
}

/// Three significant digits, no exponent.
pub fn n3(x: f64) -> String {
    let a = x.abs();
    let p = if a >= 100.0 || a == 0.0 {
        0
    } else if a >= 10.0 {
        1
    } else if a >= 1.0 {
        2
    } else {
        (2 - a.log10().floor() as i32).clamp(3, 7) as usize
    };
    format!("{x:.p$}")
}

pub fn gbps(bytes: usize, ms: f64) -> f64 {
    bytes as f64 / (ms * 1e6)
}

// ---- flags ----

pub struct Args {
    kv: BTreeMap<String, String>,
    help: bool,
    known: RefCell<Vec<(String, String)>>,
}

impl Args {
    pub fn parse() -> Args {
        let (mut kv, mut help) = (BTreeMap::new(), false);
        let mut it = std::env::args().skip(1).peekable();
        while let Some(a) = it.next() {
            match a.strip_prefix("--") {
                Some("help") => help = true,
                Some(k) => {
                    let (k, v) = match k.split_once('=') {
                        Some((k, v)) => (k.to_string(), v.to_string()),
                        None => {
                            let v = it.next_if(|n| !n.starts_with("--"));
                            (k.to_string(), v.unwrap_or_else(|| "true".into()))
                        }
                    };
                    kv.insert(k, v);
                }
                None => eprintln!("warning: ignoring argument {a:?}"),
            }
        }
        Args { kv, help, known: RefCell::default() }
    }

    pub fn get<T: FromStr + Display>(&self, k: &str, default: T) -> T {
        self.known.borrow_mut().push((k.into(), default.to_string()));
        match self.kv.get(k) {
            None => default,
            Some(v) => v.parse().unwrap_or_else(|_| die(k, v)),
        }
    }

    /// Comma-separated list, e.g. `--sizes-mib 2,64`.
    pub fn list<T: FromStr>(&self, k: &str, default: &str) -> Vec<T> {
        let v = self.get(k, default.to_string());
        v.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap_or_else(|_| die(k, s)))
            .collect()
    }

    /// `--help` prints the flags with defaults and exits; unknown flags warn.
    pub fn done(&self, about: &str) {
        let known = self.known.borrow();
        if self.help {
            println!("{about}\n\nflags (default):");
            known.iter().for_each(|(k, v)| println!("  --{k} {v}"));
            std::process::exit(0);
        }
        for k in self.kv.keys().filter(|k| !known.iter().any(|(n, _)| n == *k)) {
            eprintln!("warning: unknown flag --{k} ignored");
        }
    }
}

fn die(k: &str, v: &str) -> ! {
    eprintln!("--{k}: cannot parse {v:?}");
    std::process::exit(2)
}

// ---- stats and JSON ----

#[derive(Clone, Copy, Default)]
pub struct Stats {
    pub n: usize,
    pub median: f64,
    pub p90: f64,
    pub max: f64,
    pub mean: f64,
}

pub fn stats(v: &[f64]) -> Stats {
    if v.is_empty() {
        return Stats::default();
    }
    let mut s = v.to_vec();
    s.sort_by(f64::total_cmp);
    let q = |p: f64| s[((s.len() - 1) as f64 * p).round() as usize];
    Stats {
        n: s.len(),
        median: q(0.5),
        p90: q(0.9),
        max: s[s.len() - 1],
        mean: s.iter().sum::<f64>() / s.len() as f64,
    }
}

/// Named sample series in insertion order.
#[derive(Default, Clone)]
pub struct Samples(pub Vec<(&'static str, Vec<f64>)>);

impl Samples {
    pub fn add(&mut self, k: &'static str, v: f64) {
        match self.0.iter_mut().find(|(n, _)| *n == k) {
            Some((_, xs)) => xs.push(v),
            None => self.0.push((k, vec![v])),
        }
    }

    /// Runs `f`, recording its wall time in µs under `k`.
    pub fn time<T>(&mut self, k: &'static str, f: impl FnOnce() -> Result<T>) -> Result<T> {
        let t = Instant::now();
        let r = f();
        self.add(k, us(t));
        r
    }

    pub fn get(&self, k: &str) -> &[f64] {
        self.0.iter().find(|(n, _)| *n == k).map_or(&[][..], |(_, v)| v.as_slice())
    }

    pub fn stats(&self, k: &str) -> Stats {
        stats(self.get(k))
    }

    pub fn merge(&mut self, o: &Samples) {
        for (k, v) in &o.0 {
            v.iter().for_each(|&x| self.add(k, x));
        }
    }

    pub fn json(&self) -> J {
        J::O(self.0.iter().map(|(k, v)| (k.to_string(), stats(v).into())).collect())
    }
}

pub enum J {
    N(f64),
    B(bool),
    S(String),
    A(Vec<J>),
    O(Vec<(String, J)>),
}

impl Display for J {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // Five significant digits.
            J::N(x) if x.is_finite() => write!(f, "{}", format!("{x:.4e}").parse::<f64>().unwrap_or(*x)),
            J::N(_) => f.write_str("null"),
            J::B(b) => write!(f, "{b}"),
            J::S(s) => write!(f, "{s:?}"),
            J::A(v) => {
                f.write_str("[")?;
                for (i, x) in v.iter().enumerate() {
                    write!(f, "{}{x}", if i > 0 { "," } else { "" })?;
                }
                f.write_str("]")
            }
            J::O(v) => {
                f.write_str("{")?;
                for (i, (k, x)) in v.iter().enumerate() {
                    write!(f, "{}{k:?}:{x}", if i > 0 { "," } else { "" })?;
                }
                f.write_str("}")
            }
        }
    }
}

macro_rules! j_num {
    ($($t:ty),*) => { $(impl From<$t> for J { fn from(x: $t) -> J { J::N(x as f64) } })* };
}
j_num!(f64, f32, usize, u64, i32);

macro_rules! j_from {
    ($($t:ty => |$x:ident| $e:expr),* $(,)?) => { $(impl From<$t> for J { fn from($x: $t) -> J { $e } })* };
}
j_from!(bool => |b| J::B(b), &str => |s| J::S(s.into()), String => |s| J::S(s), Vec<J> => |v| J::A(v),
    Option<f64> => |v| J::N(v.unwrap_or(f64::NAN)));

impl From<Stats> for J {
    fn from(s: Stats) -> J {
        obj! {"n" => s.n, "median" => s.median, "p90" => s.p90, "max" => s.max, "mean" => s.mean}
    }
}

// ---- driver ----

/// Driver symbols the plane's table does not carry.
pub struct Extra {
    _lib: Library,
    pub mem_alloc: unsafe extern "C" fn(*mut CUdeviceptr, usize) -> CUresult,
    pub mem_free: unsafe extern "C" fn(CUdeviceptr) -> CUresult,
    pub mem_get_info: unsafe extern "C" fn(*mut usize, *mut usize) -> CUresult,
    pub device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CUdevice) -> CUresult,
    pub driver_get_version: unsafe extern "C" fn(*mut c_int) -> CUresult,
}

fn sym<T: Copy>(lib: &Library, name: &str) -> Result<T> {
    // SAFETY: every caller names the documented ABI of `name` as `T`.
    unsafe { lib.get::<T>(name) }
        .map(|s| *s)
        .map_err(|e| Error::CudaUnavailable(format!("{name}: {e}")))
}

static EXTRA: OnceLock<Extra> = OnceLock::new();

pub fn extra() -> Result<&'static Extra> {
    if let Some(x) = EXTRA.get() {
        return Ok(x);
    }
    cuda::driver()?; // cuInit
    // SAFETY: dlopen of the NVIDIA driver, already loaded by the plane's table.
    let lib = unsafe { Library::new("libcuda.so.1") }
        .map_err(|e| Error::CudaUnavailable(format!("dlopen libcuda: {e}")))?;
    let x = Extra {
        mem_alloc: sym(&lib, "cuMemAlloc_v2")?,
        mem_free: sym(&lib, "cuMemFree_v2")?,
        mem_get_info: sym(&lib, "cuMemGetInfo_v2")?,
        device_get_name: sym(&lib, "cuDeviceGetName")?,
        driver_get_version: sym(&lib, "cuDriverGetVersion")?,
        _lib: lib,
    };
    Ok(EXTRA.get_or_init(|| x))
}

/// Device `ordinal`'s primary context, current on this thread for the value's life.
pub struct Gpu {
    pub d: &'static Driver,
    pub x: &'static Extra,
    pub ordinal: c_int,
    pub dev: CUdevice,
    pub ctx: CUcontext,
    _bound: CtxGuard,
}

impl Gpu {
    pub fn open(ordinal: c_int) -> Result<Gpu> {
        let d = cuda::driver()?;
        let (dev, ctx) = cuda::primary_context(ordinal)?;
        let _bound = CtxGuard::enter(ctx)?;
        Ok(Gpu { d, x: extra()?, ordinal, dev, ctx, _bound })
    }

    pub fn free_mib(&self) -> Result<usize> {
        let x = self.x;
        let (mut free, mut total) = (0, 0);
        cu!(x.mem_get_info(&mut free, &mut total))?;
        Ok(free / MIB)
    }

    /// Prints the device header and returns it as JSON.
    pub fn info(&self) -> Result<J> {
        let x = self.x;
        let mut name = [0 as c_char; 256];
        cu!(x.device_get_name(name.as_mut_ptr(), 256, self.dev))?;
        // SAFETY: the driver wrote a NUL-terminated string into `name`.
        let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let (mut ver, mut free, mut total) = (0, 0, 0);
        cu!(x.driver_get_version(&mut ver))?;
        cu!(x.mem_get_info(&mut free, &mut total))?;
        let vmm = cuda::attribute(self.dev, cuda::ATTR_VMM_SUPPORTED)?;
        let fd = cuda::attribute(self.dev, cuda::ATTR_POSIX_FD_SUPPORTED)?;
        let uuid = cuda::uuid(self.dev)?;
        println!("- device {}: {name} ({uuid}), driver API {}.{}", self.ordinal, ver / 1000, ver % 1000 / 10);
        println!("- memory: {} MiB free of {} MiB", free / MIB, total / MIB);
        println!("- VMM supported: {vmm}; POSIX-fd export: {fd}\n");
        Ok(obj! {"name" => name, "uuid" => uuid, "driver_api" => ver, "free_mib" => free / MIB,
            "total_mib" => total / MIB, "vmm" => vmm, "posix_fd" => fd})
    }

    pub fn stream(&self) -> Result<CUstream> {
        let d = self.d;
        let mut s = null_mut();
        cu!(d.stream_create(&mut s, cuda::STREAM_NON_BLOCKING))?;
        Ok(s)
    }
}

/// A pool of timing events, grown on demand.
pub struct Events {
    pub e: Vec<CUevent>,
}

impl Events {
    pub fn new(n: usize) -> Result<Events> {
        let mut ev = Events { e: Vec::with_capacity(n) };
        ev.grow(n)?;
        Ok(ev)
    }

    fn grow(&mut self, n: usize) -> Result<()> {
        let d = cuda::driver()?;
        while self.e.len() < n {
            let mut e = null_mut();
            cu!(d.event_create(&mut e, cuda::EVENT_DEFAULT))?;
            self.e.push(e);
        }
        Ok(())
    }

    pub fn record(&mut self, i: usize, s: CUstream) -> Result<()> {
        self.grow(i + 1)?;
        let d = cuda::driver()?;
        cu!(d.event_record(self.e[i], s))
    }

    pub fn sync(&self, i: usize) -> Result<()> {
        let d = cuda::driver()?;
        cu!(d.event_synchronize(self.e[i]))
    }

    pub fn ms(&self, a: usize, b: usize) -> Result<f64> {
        elapsed(self.e[a], self.e[b])
    }
}

impl Drop for Events {
    fn drop(&mut self) {
        if let Ok(d) = cuda::driver() {
            for &e in &self.e {
                // SAFETY: an event created by this pool.
                unsafe { (d.event_destroy)(e) };
            }
        }
    }
}

/// Device ms from `a` to `b` (completed events, any streams).
pub fn elapsed(a: CUevent, b: CUevent) -> Result<f64> {
    let d = cuda::driver()?;
    let mut t = 0f32;
    cu!(d.event_elapsed_time(&mut t, a, b))?;
    Ok(t as f64)
}

pub fn event_done(e: CUevent) -> Result<bool> {
    let d = cuda::driver()?;
    // SAFETY: a live event.
    match unsafe { (d.event_query)(e) } {
        cuda::SUCCESS => Ok(true),
        cuda::ERROR_NOT_READY => Ok(false),
        rc => check("event_query", rc).map(|_| false),
    }
}

// ---- device memory ----

/// `cuMemAlloc_v2` memory.
pub struct DevBuf(pub CUdeviceptr);

impl DevBuf {
    pub fn new(len: usize) -> Result<DevBuf> {
        let x = extra()?;
        let mut p = 0;
        cu!(x.mem_alloc(&mut p, len))?;
        Ok(DevBuf(p))
    }
}

impl Drop for DevBuf {
    fn drop(&mut self) {
        if let Ok(x) = extra() {
            // SAFETY: allocated by `new`.
            unsafe { (x.mem_free)(self.0) };
        }
    }
}

pub fn access(dev: CUdevice) -> AccessDesc {
    AccessDesc {
        location: MemLocation { kind: cuda::MEM_LOCATION_TYPE_DEVICE, id: dev },
        flags: cuda::MEM_ACCESS_PROT_READWRITE,
    }
}

pub fn reserve(len: usize) -> Result<CUdeviceptr> {
    let d = cuda::driver()?;
    let mut va = 0;
    cu!(d.mem_address_reserve(&mut va, len, 0, 0, 0))?;
    Ok(va)
}

/// `len` bytes of VA backed by `handle`-byte `cuMemCreate` handles, mapped contiguously.
pub struct VmmRange {
    pub va: CUdeviceptr,
    pub len: usize,
    pub handle: usize,
    pub hs: Vec<Handle>,
}

impl VmmRange {
    pub fn new(prop: &AllocationProp, len: usize, handle: usize) -> Result<VmmRange> {
        let d = cuda::driver()?;
        let mut r = VmmRange { va: reserve(len)?, len, handle, hs: vec![] };
        for off in (0..len).step_by(handle) {
            let mut h = 0;
            cu!(d.mem_create(&mut h, handle, prop, 0))?;
            r.hs.push(h);
            cu!(d.mem_map(r.va + off as u64, handle, 0, h, 0))?;
        }
        cu!(d.mem_set_access(r.va, len, &access(prop.location.id), 1))?;
        Ok(r)
    }
}

impl Drop for VmmRange {
    fn drop(&mut self) {
        let Ok(d) = cuda::driver() else { return };
        // SAFETY: tearing down exactly what `new` built; errors are moot at teardown.
        unsafe {
            for (i, &h) in self.hs.iter().enumerate() {
                (d.mem_unmap)(self.va + (i * self.handle) as u64, self.handle);
                (d.mem_release)(h);
            }
            (d.mem_address_free)(self.va, self.len);
        }
    }
}

// ---- host memory ----

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum HostKind {
    /// `cuMemHostAlloc(PORTABLE)`, i.e. `cudaHostAlloc`.
    HostAlloc,
    /// memfd + mmap(MAP_SHARED), touched, then `cuMemHostRegister(PORTABLE)` by the caller.
    Memfd { huge: bool },
    /// A touched `Vec<u8>`.
    Pageable,
}

impl HostKind {
    pub fn parse(s: &str) -> Option<HostKind> {
        Some(match s {
            "hostalloc" => HostKind::HostAlloc,
            "memfd" => HostKind::Memfd { huge: false },
            "memfd_huge" => HostKind::Memfd { huge: true },
            "pageable" => HostKind::Pageable,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            HostKind::HostAlloc => "hostalloc",
            HostKind::Memfd { huge: false } => "memfd",
            HostKind::Memfd { huge: true } => "memfd_huge",
            HostKind::Pageable => "pageable",
        }
    }
}

pub struct HostBuf {
    pub kind: HostKind,
    pub ptr: *mut u8,
    pub len: usize,
    fd: c_int,
    vec: Vec<u8>,
    registered: bool,
    /// µs: `alloc` (hostalloc), `map` (memfd create+truncate+mmap+madvise), `touch`.
    pub setup: Samples,
    /// Shmem mapped by PMD (2 MiB) pages after the touch, from smaps_rollup.
    pub pmd_kib: u64,
}

impl HostBuf {
    pub fn new(kind: HostKind, len: usize) -> Result<HostBuf> {
        let mut b = HostBuf {
            kind,
            ptr: null_mut(),
            len,
            fd: -1,
            vec: Vec::new(),
            registered: false,
            setup: Samples::default(),
            pmd_kib: 0,
        };
        let mut s = Samples::default();
        match kind {
            HostKind::HostAlloc => {
                let d = cuda::driver()?;
                let mut p = null_mut();
                s.time("alloc", || cu!(d.mem_host_alloc(&mut p, len, cuda::MEMHOSTALLOC_PORTABLE)))?;
                b.ptr = p.cast();
            }
            HostKind::Memfd { huge } => s.time("map", || b.map_memfd(huge))?,
            HostKind::Pageable => {
                s.time("touch", || {
                    b.vec = vec![0x5a; len];
                    Ok(())
                })?;
                b.ptr = b.vec.as_mut_ptr();
            }
        }
        if kind != HostKind::Pageable {
            let before = shmem_pmd_kib();
            // SAFETY: `ptr` maps `len` writable bytes owned by `b`.
            s.time("touch", || {
                unsafe { std::ptr::write_bytes(b.ptr, 0x5a, len) };
                Ok(())
            })?;
            b.pmd_kib = shmem_pmd_kib().saturating_sub(before);
        }
        b.setup = s;
        Ok(b)
    }

    fn map_memfd(&mut self, huge: bool) -> Result<()> {
        let io = |op: &str| Error::Io(format!("{op}: {}", std::io::Error::last_os_error()));
        // SAFETY: syscalls on a fresh fd and a fresh mapping of exactly `len` bytes.
        unsafe {
            self.fd = libc::memfd_create(c"g-bench".as_ptr(), libc::MFD_CLOEXEC);
            if self.fd < 0 {
                return Err(io("memfd_create"));
            }
            if libc::ftruncate(self.fd, self.len as libc::off_t) != 0 {
                return Err(io("ftruncate"));
            }
            let p = libc::mmap(
                null_mut(),
                self.len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                self.fd,
                0,
            );
            if p == libc::MAP_FAILED {
                return Err(io("mmap"));
            }
            self.ptr = p.cast();
            if huge && libc::madvise(p, self.len, libc::MADV_HUGEPAGE) != 0 {
                eprintln!("warning: {}", io("madvise(MADV_HUGEPAGE)"));
            }
        }
        Ok(())
    }

    pub fn register(&mut self) -> Result<()> {
        let d = cuda::driver()?;
        cu!(d.mem_host_register(self.ptr.cast(), self.len, cuda::MEMHOSTREGISTER_PORTABLE))?;
        self.registered = true;
        Ok(())
    }

    pub fn unregister(&mut self) -> Result<()> {
        let d = cuda::driver()?;
        cu!(d.mem_host_unregister(self.ptr.cast()))?;
        self.registered = false;
        Ok(())
    }
}

impl Drop for HostBuf {
    fn drop(&mut self) {
        // SAFETY: releasing exactly what `new` acquired.
        unsafe {
            if let Ok(d) = cuda::driver() {
                if self.registered {
                    (d.mem_host_unregister)(self.ptr.cast());
                }
                if self.kind == HostKind::HostAlloc && !self.ptr.is_null() {
                    (d.mem_free_host)(self.ptr.cast());
                }
            }
            if matches!(self.kind, HostKind::Memfd { .. }) {
                if !self.ptr.is_null() {
                    libc::munmap(self.ptr.cast(), self.len);
                }
                if self.fd >= 0 {
                    libc::close(self.fd);
                }
            }
        }
    }
}

pub fn shmem_pmd_kib() -> u64 {
    let s = std::fs::read_to_string("/proc/self/smaps_rollup").unwrap_or_default();
    s.lines()
        .find_map(|l| l.strip_prefix("ShmemPmdMapped:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

pub fn thp_shmem_setting() -> String {
    std::fs::read_to_string("/sys/kernel/mm/transparent_hugepage/shmem_enabled")
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| "unknown".into())
}

// ---- cuBLAS GEMM load ----

const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLAS_GEMM_DEFAULT: c_int = -1;
const WORKSPACE: usize = 32 * MIB;

/// cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta,
/// C, Ctype, ldc, computeType, algo).
#[rustfmt::skip]
type GemmEx = unsafe extern "C" fn(
    *mut c_void, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int,
    *const c_void, c_int, c_int, *const c_void, *mut c_void, c_int, c_int, c_int, c_int,
) -> c_int;

/// C = A·B, bf16 m×m×m with fp32 accumulation, enqueued on `stream`.
pub struct Gemm {
    h: *mut c_void,
    gemm_ex: GemmEx,
    destroy: unsafe extern "C" fn(*mut c_void) -> c_int,
    bufs: Vec<DevBuf>, // A, B, C, workspace
    pub m: usize,
    pub stream: CUstream,
    _libs: Vec<Library>, // last: unloaded after the handle is destroyed
}

fn blas(op: &str, status: c_int) -> Result<()> {
    match status {
        0 => Ok(()),
        s => Err(Error::Invalid(format!("{op}: cuBLAS status {s}"))),
    }
}

fn open_global(p: &Path) -> Result<Library> {
    // SAFETY: dlopen of NVIDIA's cuBLAS libraries.
    unsafe { UnixLibrary::open(Some(p), RTLD_NOW | RTLD_GLOBAL) }
        .map(Library::from)
        .map_err(|e| Error::CudaUnavailable(format!("dlopen {}: {e}", p.display())))
}

impl Gemm {
    pub fn new(cublas: &str, m: usize, stream: CUstream) -> Result<Gemm> {
        let p = Path::new(cublas);
        let mut libs = Vec::new();
        // Preload libcublasLt globally in case libcublas's RUNPATH does not reach it.
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or_default();
        let lt = p.with_file_name(name.replacen("libcublas.", "libcublasLt.", 1));
        if lt != p && lt.exists() {
            libs.push(open_global(&lt)?);
        }
        let lib = open_global(p)?;
        let create: unsafe extern "C" fn(*mut *mut c_void) -> c_int = sym(&lib, "cublasCreate_v2")?;
        let set_stream: unsafe extern "C" fn(*mut c_void, CUstream) -> c_int =
            sym(&lib, "cublasSetStream_v2")?;
        let set_workspace: unsafe extern "C" fn(*mut c_void, *mut c_void, usize) -> c_int =
            sym(&lib, "cublasSetWorkspace_v2")?;
        let (gemm_ex, destroy) = (sym(&lib, "cublasGemmEx")?, sym(&lib, "cublasDestroy_v2")?);
        libs.push(lib);

        let (d, n) = (cuda::driver()?, m * m * 2);
        let bufs = [n, n, n, WORKSPACE].into_iter().map(DevBuf::new).collect::<Result<Vec<_>>>()?;
        for b in &bufs[..2] {
            cu!(d.memset_d8_async(b.0, 0x3c, n, stream))?; // bf16 0x3c3c ≈ 0.0115
        }
        let mut h = null_mut();
        // SAFETY: out-param.
        blas("cublasCreate_v2", unsafe { create(&mut h) })?;
        let ws = bufs[3].0 as *mut c_void;
        let g = Gemm { h, gemm_ex, destroy, bufs, m, stream, _libs: libs };
        // SAFETY: a live handle, stream and workspace of the stated size.
        blas("cublasSetStream_v2", unsafe { set_stream(h, stream) })?;
        blas("cublasSetWorkspace_v2", unsafe { set_workspace(h, ws, WORKSPACE) })?;
        g.enqueue()?; // warm-up: module load and heuristics
        cu!(d.stream_synchronize(stream))?;
        Ok(g)
    }

    pub fn enqueue(&self) -> Result<()> {
        let (alpha, beta, m) = (1f32, 0f32, self.m as c_int);
        let [a, b, c] = [0, 1, 2].map(|i| self.bufs[i].0 as *mut c_void);
        let (pa, pb) = ((&alpha as *const f32).cast(), (&beta as *const f32).cast());
        // SAFETY: m×m bf16 device buffers owned by self; host scalars outlive the call.
        blas("cublasGemmEx", unsafe {
            (self.gemm_ex)(
                self.h, 0, 0, m, m, m, pa, a, CUDA_R_16BF, m, b, CUDA_R_16BF, m, pb, c,
                CUDA_R_16BF, m, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT,
            )
        })
    }

    /// Records `ev[0]`, then `n` GEMMs each followed by `ev[i + 1]`.
    pub fn batch(&self, ev: &mut Events, n: usize) -> Result<()> {
        ev.record(0, self.stream)?;
        for i in 0..n {
            self.enqueue()?;
            ev.record(i + 1, self.stream)?;
        }
        Ok(())
    }

    pub fn times(&self, ev: &Events, n: usize) -> Result<Vec<f64>> {
        (0..n).map(|i| ev.ms(i, i + 1)).collect()
    }

    pub fn device_bytes(&self) -> usize {
        3 * self.m * self.m * 2 + WORKSPACE
    }

    pub fn tflops(&self, ms: f64) -> f64 {
        2.0 * (self.m as f64).powi(3) / (ms * 1e-3) / 1e12
    }
}

impl Drop for Gemm {
    fn drop(&mut self) {
        // SAFETY: the handle from cublasCreate_v2; buffers and libraries drop after it.
        unsafe { (self.destroy)(self.h) };
    }
}
