//! `tensorfs.varena`: the weight plane's Python surface (`crates/varena/API.md`).
//!
//! Marshal only. Every call that can wait, do I/O or touch the driver runs detached from the
//! GIL. Stats and events cross as plain dicts and lists; device bytes cross as uint8 DLPack
//! views that keep the VA (never the pages) alive.

use std::ffi::{c_void, CStr};
use std::os::fd::{BorrowedFd, IntoRawFd};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{create_exception, ffi};
use varena::layout::Layout;
use varena::plane::{Event, LeaseView, Stats};
use varena::{Cursor, Error, Host, Lease, Plane, PlaneConfig, Ticket, WsId};

use crate::v2::PyCheckpoint;

create_exception!(tensorfs.varena, PlaneError, PyException, "A weight-plane error; `.code` names its class.");
create_exception!(tensorfs.varena, DeviceFull, PlaneError, "The driver refused memory or the floor would be crossed; `.deficit` bytes must be freed.");
create_exception!(tensorfs.varena, Floor, PlaneError, "`.need` bytes exceed what the device can ever give (`.capacity`).");
create_exception!(tensorfs.varena, CudaError, PlaneError);
create_exception!(tensorfs.varena, Closed, PlaneError);
create_exception!(tensorfs.varena, Invalid, PlaneError);
create_exception!(tensorfs.varena, Poisoned, PlaneError);
create_exception!(tensorfs.varena, IoError, PlaneError);

/// The one conversion.
fn err(py: Python<'_>, e: Error) -> PyErr {
    let msg = e.to_string();
    let x = match &e {
        Error::DeviceFull { .. } => DeviceFull::new_err(msg),
        Error::Floor { .. } => Floor::new_err(msg),
        Error::Cuda { .. } => CudaError::new_err(msg),
        Error::Closed(_) => Closed::new_err(msg),
        Error::Invalid(_) => Invalid::new_err(msg),
        Error::Poisoned(_) => Poisoned::new_err(msg),
        Error::Io(_) => IoError::new_err(msg),
    };
    let v = x.value(py);
    let _ = v.setattr("code", e.code());
    match e {
        Error::DeviceFull { deficit } => {
            let _ = v.setattr("deficit", deficit);
        }
        Error::Floor { need, capacity } => {
            let _ = v.setattr("need", need);
            let _ = v.setattr("capacity", capacity);
        }
        _ => {}
    }
    x
}

trait OrRaise<T> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T>;
}

impl<T> OrRaise<T> for varena::Result<T> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T> {
        self.map_err(|e| err(py, e))
    }
}

macro_rules! dict {
    ($py:expr; $($k:literal => $v:expr),* $(,)?) => {{
        let d = PyDict::new($py);
        $(d.set_item($k, $v)?;)*
        d
    }};
}

fn stats<'p>(py: Python<'p>, s: Stats) -> PyResult<Bound<'p, PyDict>> {
    let devices = PyList::empty(py);
    for d in &s.devices {
        let c = &d.counters;
        devices.append(dict!(py;
            "ordinal" => d.ordinal, "uuid" => &d.uuid, "granularity" => d.granularity,
            "exportable" => d.exportable, "mapped" => d.mapped, "leased" => d.leased,
            "shared_bytes" => d.shared_bytes, "copies" => c.copies, "misses" => c.misses,
            "host_copy_bytes" => c.host_copy_bytes, "host_copy_ns" => c.host_copy_ns,
            "bounce_copy_bytes" => c.bounce_copy_bytes, "bounce_copy_ns" => c.bounce_copy_ns,
            "disk_copy_bytes" => c.disk_copy_bytes, "disk_copy_ns" => c.disk_copy_ns,
        ))?;
    }
    let regions = PyList::empty(py);
    for r in &s.regions {
        regions.append(dict!(py;
            "ws" => r.ws, "device" => r.device, "region" => r.region, "bytes" => r.bytes,
            "state" => r.state, "live" => r.live, "last_use" => r.last_use,
        ))?;
    }
    let io = &s.io;
    let io = dict!(py;
        "direct_bytes" => io.direct_bytes, "cached_bytes" => io.cached_bytes,
        "buffered_bytes" => io.buffered_bytes, "direct_refused" => io.direct_refused,
        "register_failed" => io.register_failed,
    );
    Ok(dict!(py;
        "devices" => devices, "regions" => regions, "io" => io, "now" => s.now,
        "events_dropped" => s.events_dropped, "poisoned" => s.poisoned,
    ))
}

fn event<'p>(py: Python<'p>, e: Event) -> PyResult<Bound<'p, PyDict>> {
    Ok(dict!(py;
        "kind" => e.kind, "ws" => e.ws, "region" => e.region, "device" => e.device,
        "bytes" => e.bytes, "detail" => e.detail,
    ))
}

// ---------------------------------------------------------------- host sources

/// Host bytes from the chunk files through the plane's staging slots. `direct`: a chunk the
/// page cache does not hold reads O_DIRECT; otherwise every read is buffered.
#[pyclass(name = "PageCache", module = "tensorfs.varena", frozen)]
pub struct PyPageCache {
    #[pyo3(get)]
    direct: bool,
}

#[pymethods]
impl PyPageCache {
    #[new]
    #[pyo3(signature = (direct = true))]
    fn new(direct: bool) -> Self {
        PyPageCache { direct }
    }
}

/// A sealed host layout of the weight set (its owner's memfd; the fd stays the caller's).
#[pyclass(name = "SealedLayout", module = "tensorfs.varena", frozen)]
pub struct PySealedLayout {
    #[pyo3(get)]
    fd: i32,
}

#[pymethods]
impl PySealedLayout {
    #[new]
    fn new(fd: i32) -> Self {
        PySealedLayout { fd }
    }
}

#[derive(FromPyObject)]
enum HostArg<'py> {
    PageCache(PyRef<'py, PyPageCache>),
    Sealed(PyRef<'py, PySealedLayout>),
}

// ---------------------------------------------------------------- the plane

#[pyclass(name = "Plane", module = "tensorfs.varena", frozen)]
pub struct PyPlane {
    plane: Arc<Plane>,
}

impl PyPlane {
    fn id(&self, py: Python<'_>, ws: &PyWeightSet) -> PyResult<WsId> {
        if Arc::ptr_eq(&self.plane, &ws.plane) {
            return Ok(ws.id);
        }
        Err(err(py, Error::Invalid(format!("weight set {} belongs to another plane", ws.id.0))))
    }
}

#[pymethods]
impl PyPlane {
    #[new]
    #[pyo3(signature = (devices, readers = 16, copy_streams = 1, staging_buffers = 16, staging_bytes = 16 << 20, floor_bytes = 256 << 20))]
    fn new(
        py: Python<'_>,
        devices: Vec<i32>,
        readers: usize,
        copy_streams: usize,
        staging_buffers: usize,
        staging_bytes: u64,
        floor_bytes: u64,
    ) -> PyResult<Self> {
        let cfg = PlaneConfig { devices, readers, copy_streams, staging_buffers, staging_bytes, floor_bytes };
        let plane = py.detach(|| Plane::open(cfg)).or_raise(py)?;
        Ok(PyPlane { plane: Arc::new(plane) })
    }

    #[getter]
    fn devices(&self) -> Vec<(i32, String)> {
        self.plane.devices()
    }

    /// One weight set: `regions` (`tensorfs.v2.regions`) of `ck`, host bytes from `host`
    /// (default `PageCache()`).
    #[pyo3(signature = (ck, regions, host = None))]
    fn register(&self, py: Python<'_>, ck: PyRef<'_, PyCheckpoint>, regions: Vec<Vec<u32>>, host: Option<HostArg<'_>>) -> PyResult<PyWeightSet> {
        let host = match host {
            None => Host::PageCache { direct: true },
            Some(HostArg::PageCache(p)) => Host::PageCache { direct: p.direct },
            Some(HostArg::Sealed(s)) => Host::Sealed(s.fd),
        };
        let (plane, ck) = (&self.plane, &ck.ck);
        let (id, layout) = py
            .detach(|| {
                let id = plane.register(ck, &regions, host)?;
                Ok((id, plane.layout(id)?))
            })
            .or_raise(py)?;
        Ok(PyWeightSet { plane: self.plane.clone(), id, layout })
    }

    /// Map and fill `regions` (all when None) not resident on `device`; raises `DeviceFull`
    /// or `Floor` (nothing mapped) when they do not fit.
    #[pyo3(signature = (ws, device, regions = None))]
    fn want(&self, py: Python<'_>, ws: PyRef<'_, PyWeightSet>, device: i32, regions: Option<Vec<u32>>) -> PyResult<PyTicket> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let ticket = py.detach(|| plane.want(id, device, regions.as_deref())).or_raise(py)?;
        Ok(PyTicket { ticket })
    }

    /// Unmap `regions` (all when None) and release their memory; leased regions stay.
    /// Returns the bytes freed.
    #[pyo3(name = "drop", signature = (ws, device, regions = None))]
    fn drop_regions(&self, py: Python<'_>, ws: PyRef<'_, PyWeightSet>, device: i32, regions: Option<Vec<u32>>) -> PyResult<u64> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        py.detach(|| plane.drop_regions(id, device, regions.as_deref())).or_raise(py)
    }

    /// This device can export and attach shared regions.
    fn device_sharing(&self, py: Python<'_>, device: i32) -> PyResult<bool> {
        self.plane.device_sharing(device).or_raise(py)
    }

    /// Share the Ready, unleased regions (all when None) on `device`: they become read-only.
    /// Returns `[{"region": int, "chunks": [(bytes, fd), ...]}]`; the caller owns every fd.
    #[pyo3(signature = (ws, device, regions = None))]
    fn export_device<'p>(&self, py: Python<'p>, ws: PyRef<'_, PyWeightSet>, device: i32, regions: Option<Vec<u32>>) -> PyResult<Bound<'p, PyList>> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let exported = py.detach(|| plane.export_device(id, device, regions.as_deref())).or_raise(py)?;
        let out = PyList::empty(py);
        for e in exported {
            let chunks: Vec<(u64, i32)> = e.chunks.into_iter().map(|(n, fd)| (n, fd.into_raw_fd())).collect();
            out.append(dict!(py; "region" => e.region, "chunks" => chunks))?;
        }
        Ok(out)
    }

    /// Map another process's shared `region` read-only, without a copy: `chunks` are
    /// `(bytes, fd)` in VA order, from `export_device`; `digest` must equal `ws.digest`. The fds
    /// stay the caller's to close.
    fn attach_device(&self, py: Python<'_>, ws: PyRef<'_, PyWeightSet>, device: i32, region: u32, chunks: Vec<(u64, i32)>, digest: String) -> PyResult<()> {
        let id = self.id(py, &ws)?;
        if chunks.iter().any(|(_, fd)| *fd < 0) {
            return Err(err(py, Error::Invalid("a negative fd".into())));
        }
        // SAFETY: the caller keeps these fds open for the call.
        let borrowed: Vec<(u64, BorrowedFd<'_>)> = chunks.iter().map(|(n, fd)| (*n, unsafe { BorrowedFd::borrow_raw(*fd) })).collect();
        let plane = &self.plane;
        py.detach(|| plane.attach_device(id, device, region, &borrowed, &digest)).or_raise(py)
    }

    /// A lease on `region` in place; an unmapped region is filled first (a miss).
    fn acquire(&self, py: Python<'_>, ws: PyRef<'_, PyWeightSet>, device: i32, region: u32, stream: u64) -> PyResult<PyPlaneLease> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let lease = py.detach(|| plane.acquire(id, device, region, stream)).or_raise(py)?;
        Ok(PyPlaneLease::new(lease, stream))
    }

    /// Stream `regions` in order (`repeat` times; 0 until close) through a ring of `ring_bytes`
    /// (default: twice the largest streamed region). Regions mapped now are served in place.
    #[pyo3(signature = (ws, device, regions, ring_bytes = None, repeat = 1))]
    fn stream(&self, py: Python<'_>, ws: PyRef<'_, PyWeightSet>, device: i32, regions: Vec<u32>, ring_bytes: Option<u64>, repeat: u32) -> PyResult<PyCursor> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let cur = py.detach(|| plane.stream(id, device, &regions, repeat, ring_bytes)).or_raise(py)?;
        let (home, ring_ptr, ring_nbytes) = (cur.home.clone(), cur.ring_ptr, cur.ring_nbytes);
        Ok(PyCursor { home, ring_ptr, ring_nbytes, cur })
    }

    fn stats<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyDict>> {
        let s = py.detach(|| self.plane.stats());
        stats(py, s)
    }

    /// Drains the event ring.
    fn events<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let out = PyList::empty(py);
        for e in py.detach(|| self.plane.events()) {
            out.append(event(py, e)?)?;
        }
        Ok(out)
    }

    /// Closes every weight set; refuses while leases, cursors or views live.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.plane.close()).or_raise(py)
    }
}

// ---------------------------------------------------------------- weight sets

#[pyclass(name = "WeightSet", module = "tensorfs.varena", frozen)]
pub struct PyWeightSet {
    plane: Arc<Plane>,
    id: WsId,
    layout: Arc<Layout>,
}

#[pymethods]
impl PyWeightSet {
    /// The layout span (a multiple of 2 MiB).
    #[getter]
    fn nbytes(&self) -> u64 {
        self.layout.nbytes
    }

    /// Identity of the byte layout: equal digests place the same chunks at the same offsets.
    #[getter]
    fn digest(&self) -> String {
        self.layout.digest.clone()
    }

    /// `[{"index", "offset", "nbytes", "span", "chunks"}]`, `chunks` the region's chunk indexes.
    #[getter]
    fn regions<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let out = PyList::empty(py);
        for (i, r) in self.layout.regions.iter().enumerate() {
            let chunks: Vec<u32> = r.chunks.iter().map(|c| c.index).collect();
            out.append(dict!(py; "index" => i, "offset" => r.offset, "nbytes" => r.nbytes, "span" => r.span, "chunks" => chunks))?;
        }
        Ok(out)
    }

    /// `[{"component", "key", "role", "region", "offset", "nbytes"}]`: every non-empty part, at
    /// its layout offset (a region's bytes start at the region's offset).
    #[getter]
    fn parts<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let out = PyList::empty(py);
        for p in &self.layout.parts {
            out.append(dict!(py;
                "component" => &p.component, "key" => &p.key, "role" => &p.role,
                "region" => p.region, "offset" => p.offset, "nbytes" => p.nbytes,
            ))?;
        }
        Ok(out)
    }

    /// Refuses while leases, cursors or views remain.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.plane.close_ws(self.id)).or_raise(py)
    }

    fn __repr__(&self) -> String {
        format!("<tensorfs.varena.WeightSet {} B in {} regions>", self.layout.nbytes, self.layout.regions.len())
    }
}

#[pyclass(name = "Ticket", module = "tensorfs.varena", frozen)]
pub struct PyTicket {
    ticket: Ticket,
}

#[pymethods]
impl PyTicket {
    fn done(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| self.ticket.done()).or_raise(py)
    }

    fn wait(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.ticket.wait()).or_raise(py)
    }
}

// ---------------------------------------------------------------- leases and cursors

/// Use of one region's bytes on one stream: the only pin. Release exactly once; a lease
/// collected unreleased poisons the plane.
#[pyclass(name = "Lease", module = "tensorfs.varena", frozen)]
pub struct PyPlaneLease {
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    region: u32,
    /// Where a streamed lease's bytes sit in the cursor's ring; None for a region in place.
    #[pyo3(get)]
    ring_offset: Option<u64>,
    /// The acquire stream: `release()` records there unless told otherwise.
    stream: u64,
    released: AtomicBool,
    lease: Lease,
}

impl PyPlaneLease {
    fn new(lease: Lease, stream: u64) -> Self {
        let (ptr, nbytes, region, ring_offset) = (lease.ptr, lease.nbytes, lease.region, lease.ring_offset);
        PyPlaneLease { ptr, nbytes, region, ring_offset, stream, released: AtomicBool::new(false), lease }
    }
}

#[pymethods]
impl PyPlaneLease {
    /// One uint8 DLPack view of the region's bytes, at `ptr`.
    fn view(&self, py: Python<'_>) -> PyResult<PyDeviceView> {
        let v = py.detach(|| self.lease.view()).or_raise(py)?;
        Ok(PyDeviceView { held: Arc::new(v) })
    }

    #[pyo3(signature = (stream = None))]
    fn release(&self, py: Python<'_>, stream: Option<u64>) -> PyResult<()> {
        self.released.store(true, Ordering::SeqCst);
        let s = stream.unwrap_or(self.stream);
        py.detach(|| self.lease.release(s)).or_raise(py)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        if !self.released.load(Ordering::SeqCst) {
            self.release(py, None)?;
        }
        Ok(false)
    }
}

#[pyclass(name = "Cursor", module = "tensorfs.varena", frozen)]
pub struct PyCursor {
    /// Per order index: served in place (mapped at `stream()`).
    #[pyo3(get)]
    home: Vec<bool>,
    #[pyo3(get)]
    ring_ptr: u64,
    #[pyo3(get)]
    ring_nbytes: u64,
    cur: Cursor,
}

#[pymethods]
impl PyCursor {
    /// Schedule positions consumed so far.
    #[getter]
    fn position(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.cur.position())
    }

    /// Completed cycles over the order.
    #[getter]
    fn passes(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.cur.passes())
    }

    /// Lease `region`: the next scheduled one, a later one (skipping ahead), or a demand fill.
    fn acquire(&self, py: Python<'_>, region: u32, stream: u64) -> PyResult<PyPlaneLease> {
        let lease = py.detach(|| self.cur.acquire(region, stream)).or_raise(py)?;
        Ok(PyPlaneLease::new(lease, stream))
    }

    /// Refuses while a lease or view is live; waits for copies in flight.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.cur.close()).or_raise(py)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

// ---------------------------------------------------------------- DLPack views

/// Device bytes as a 1-D uint8 DLPack tensor. Consumers' tensors keep the view alive.
#[pyclass(name = "DeviceView", module = "tensorfs.varena", frozen)]
pub struct PyDeviceView {
    held: Arc<LeaseView>,
}

const KDL_CUDA: i32 = 2;
const KDL_UINT: u8 = 1;
const DLTENSOR: &CStr = c"dltensor";

#[repr(C)]
struct DLDevice {
    device_type: i32,
    device_id: i32,
}

#[repr(C)]
struct DLDataType {
    code: u8,
    bits: u8,
    lanes: u16,
}

#[repr(C)]
struct DLTensor {
    data: *mut c_void,
    device: DLDevice,
    ndim: i32,
    dtype: DLDataType,
    shape: *mut i64,
    strides: *mut i64,
    byte_offset: u64,
}

#[repr(C)]
struct DLManagedTensor {
    dl_tensor: DLTensor,
    manager_ctx: *mut c_void,
    deleter: Option<unsafe extern "C" fn(*mut DLManagedTensor)>,
}

/// What a consumer's tensor owns: the shape and stride cells, and the view.
struct Held {
    dims: [i64; 2],
    _view: Arc<LeaseView>,
}

unsafe extern "C" fn deleter(t: *mut DLManagedTensor) {
    if t.is_null() {
        return;
    }
    // SAFETY: both boxes were leaked by `__dlpack__` and are freed exactly once, here.
    let m = unsafe { Box::from_raw(t) };
    drop(unsafe { Box::from_raw(m.manager_ctx as *mut Held) });
}

/// Runs only for an unconsumed capsule: a consumer renames it `used_dltensor` and owns the
/// deleter from then on.
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: a valid "dltensor" capsule holds the pointer `__dlpack__` stored.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, DLTENSOR.as_ptr()) == 1 {
            deleter(ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr()) as *mut DLManagedTensor);
        }
    }
}

#[pymethods]
impl PyDeviceView {
    /// True if a region of the weight set was unmapped since the view was made (a ring view
    /// never is: the ring stays mapped for the cursor's life).
    #[getter]
    fn stale(&self, py: Python<'_>) -> bool {
        py.detach(|| self.held.stale())
    }

    // Mapping is host-synchronous: no producer work is queued for the consumer's stream.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'p>(
        &self,
        py: Python<'p>,
        stream: Option<Bound<'p, PyAny>>,
        max_version: Option<Bound<'p, PyAny>>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let _ = (stream, max_version);
        let (ptr, nbytes, device) = (self.held.ptr, self.held.nbytes, self.held.device);
        if copy == Some(true) {
            return Err(PyBufferError::new_err("a plane view is never copied"));
        }
        if dl_device.is_some_and(|d| d != (KDL_CUDA, device)) {
            return Err(PyBufferError::new_err(format!("a plane view lives on (2, {device})")));
        }
        let held = Box::into_raw(Box::new(Held { dims: [nbytes as i64, 1], _view: self.held.clone() }));
        // SAFETY: `held` is live until the deleter runs.
        let dims = unsafe { (*held).dims.as_mut_ptr() };
        let t = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: ptr as *mut c_void,
                device: DLDevice { device_type: KDL_CUDA, device_id: device },
                ndim: 1,
                dtype: DLDataType { code: KDL_UINT, bits: 8, lanes: 1 },
                shape: dims,
                strides: unsafe { dims.add(1) },
                byte_offset: 0,
            },
            manager_ctx: held as *mut c_void,
            deleter: Some(deleter),
        }));
        // SAFETY: the capsule takes the leaked tensor; on failure it is freed here.
        unsafe {
            let cap = ffi::PyCapsule_New(t as *mut c_void, DLTENSOR.as_ptr(), Some(capsule_destructor));
            if cap.is_null() {
                deleter(t);
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, cap))
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (KDL_CUDA, self.held.device)
    }
}

// ---------------------------------------------------------------- the module

macro_rules! add_errors {
    ($m:expr; $($e:ident),*) => { $($m.add(stringify!($e), $m.py().get_type::<$e>())?;)* };
}

/// `tensorfs._ext.varena`, importable as such (the facade module `tensorfs.varena` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PyPageCache>()?;
    m.add_class::<PySealedLayout>()?;
    m.add_class::<PyWeightSet>()?;
    m.add_class::<PyTicket>()?;
    m.add_class::<PyPlaneLease>()?;
    m.add_class::<PyCursor>()?;
    m.add_class::<PyDeviceView>()?;
    add_errors!(m; PlaneError, DeviceFull, Floor, CudaError, Closed, Invalid, Poisoned, IoError);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}
