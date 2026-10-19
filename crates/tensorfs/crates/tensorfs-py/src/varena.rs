//! `tensorfs.varena`: the memory core's Python surface, typed in `tensorfs/varena/__init__.pyi`.
//!
//! Marshal only. Every call that can wait, read or touch the driver runs detached from the GIL.
//! A bound region crosses as a uint8 DLPack tensor whose deleter releases the pin.

use std::ffi::{c_void, CStr};
#[cfg(feature = "store")]
use std::fs::File;
#[cfg(feature = "store")]
use std::os::fd::BorrowedFd;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{create_exception, ffi};
#[cfg(feature = "store")]
use tensorfs_core::ids::ObjectRef;
use varena::memory::{Counters, Part, SetStats};
use varena::{Error, Gpu, GpuStats, Host, HostPinned, HostSet, Pinned, WeightSet};

use crate::errors::to_py;

/// What varena reads of a `tensorfs.Checkpoint`: its header, and its chunk paths when a store
/// holds it (a parsed header reads only a grant's files or a sealed layout).
#[cfg(feature = "store")]
fn ck_of(ck: &crate::store::PyCheckpoint) -> varena::checkpoint::Ck<'_> {
    varena::checkpoint::Ck { header: &ck.header, paths: ck.stored.as_ref().map(|(c, _)| c.chunk_paths()) }
}

create_exception!(tensorfs.varena, VarenaError, PyException, "A varena error; `.code` names its class.");
create_exception!(tensorfs.varena, DeviceFull, VarenaError);
create_exception!(tensorfs.varena, CudaError, VarenaError);
create_exception!(tensorfs.varena, CudaUnavailable, VarenaError);
create_exception!(tensorfs.varena, Invalid, VarenaError);
create_exception!(tensorfs.varena, IoError, VarenaError);
create_exception!(tensorfs.varena, Closed, VarenaError);
create_exception!(tensorfs.varena, Unsupported, VarenaError);

/// The one conversion. Source refusals cross as the usual `tensorfs.errors` classes.
fn err(py: Python<'_>, e: Error) -> PyErr {
    let msg = e.to_string();
    let x = match &e {
        Error::Source(r) => return to_py(py, r),
        Error::DeviceFull { .. } => DeviceFull::new_err(msg),
        Error::Cuda { .. } => CudaError::new_err(msg),
        Error::CudaUnavailable(_) => CudaUnavailable::new_err(msg),
        Error::Invalid(_) => Invalid::new_err(msg),
        Error::Io(_) => IoError::new_err(msg),
        Error::Closed(_) => Closed::new_err(msg),
        Error::Unsupported(_) => Unsupported::new_err(msg),
    };
    let v = x.value(py);
    let _ = v.setattr("code", e.code());
    if let Error::DeviceFull { need, deficit } = e {
        let _ = v.setattr("need", need);
        let _ = v.setattr("deficit", deficit);
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

// ---------------------------------------------------------------- records

#[pyclass(name = "Part", module = "tensorfs.varena", frozen)]
pub struct PyPart {
    #[pyo3(get)]
    component: String,
    #[pyo3(get)]
    key: String,
    #[pyo3(get)]
    role: String,
    #[pyo3(get)]
    dtype: String,
    #[pyo3(get)]
    region: u32,
    #[pyo3(get)]
    offset: u64,
    #[pyo3(get)]
    nbytes: u64,
    dims: Vec<u64>,
}

#[pymethods]
impl PyPart {
    #[getter]
    fn shape<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyTuple>> {
        PyTuple::new(py, &self.dims)
    }
}

impl From<&Part> for PyPart {
    fn from(p: &Part) -> Self {
        PyPart { component: p.component.clone(), key: p.key.clone(), role: p.role.clone(), dtype: p.dtype.clone(), region: p.region, offset: p.offset, nbytes: p.nbytes, dims: p.shape.clone() }
    }
}

#[pyclass(name = "Counters", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyCounters {
    binds: u64,
    faults: u64,
    paged_in: u64,
    streamed_bytes: u64,
    evicted: u64,
    copy_ns: u64,
    stall_ns: u64,
    map_ns: u64,
}

impl From<Counters> for PyCounters {
    fn from(c: Counters) -> Self {
        PyCounters { binds: c.binds, faults: c.faults, paged_in: c.paged_in, streamed_bytes: c.streamed_bytes, evicted: c.evicted, copy_ns: c.copy_ns, stall_ns: c.stall_ns, map_ns: c.map_ns }
    }
}

#[pyclass(name = "SetStats", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PySetStats {
    name: String,
    nbytes: u64,
    regions: u32,
    home: u64,
    resident: u64,
    pinned: u64,
    pins: u32,
    slot_bytes: u64,
    step: PyCounters,
    last: PyCounters,
    total: PyCounters,
}

impl From<SetStats> for PySetStats {
    fn from(s: SetStats) -> Self {
        PySetStats { name: s.name, nbytes: s.nbytes, regions: s.regions, home: s.home, resident: s.resident, pinned: s.pinned, pins: s.pins, slot_bytes: s.slot_bytes, step: s.step.into(), last: s.last.into(), total: s.total.into() }
    }
}

#[pyclass(name = "GpuStats", module = "tensorfs.varena", frozen, get_all)]
pub struct PyGpuStats {
    device: i32,
    chunk: u64,
    steps: u64,
    mapped: u64,
    idle: u64,
    slots: u32,
    slot_bytes: u64,
    slot_span: u64,
    driver_free: u64,
    driver_total: u64,
    sets: Vec<PySetStats>,
}

impl From<GpuStats> for PyGpuStats {
    fn from(s: GpuStats) -> Self {
        PyGpuStats {
            device: s.device,
            chunk: s.chunk,
            steps: s.steps,
            mapped: s.mapped,
            idle: s.idle,
            slots: s.slots,
            slot_bytes: s.slot_bytes,
            slot_span: s.slot_span,
            driver_free: s.driver_free,
            driver_total: s.driver_total,
            sets: s.sets.into_iter().map(Into::into).collect(),
        }
    }
}

// ---------------------------------------------------------------- cores

/// A grant's chunk files, `(sha256, length, fd)` each: every fd duplicated, the caller's kept.
#[cfg(feature = "store")]
fn granted(py: Python<'_>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<Option<Vec<(ObjectRef, File)>>> {
    let Some(files) = files else { return Ok(None) };
    let mut out = Vec::with_capacity(files.len());
    for (sha256, length, fd) in files {
        // SAFETY: a descriptor the caller holds for the duration of this call.
        let own = unsafe { BorrowedFd::borrow_raw(fd) }.try_clone_to_owned().map_err(|e| err(py, Error::Io(format!("dup chunk fd {fd}: {e}"))))?;
        out.push((ObjectRef { sha256, length }, File::from(own)));
    }
    Ok(Some(out))
}

#[pyclass(name = "Gpu", module = "tensorfs.varena", frozen)]
pub struct PyGpu {
    gpu: Arc<Gpu>,
}

#[pymethods]
impl PyGpu {
    #[new]
    #[pyo3(signature = (device, chunk = 2 << 20))]
    fn new(py: Python<'_>, device: i32, chunk: u64) -> PyResult<Self> {
        Ok(PyGpu { gpu: py.detach(|| Gpu::open(device, chunk)).or_raise(py)? })
    }

    #[getter]
    fn device(&self) -> i32 {
        self.gpu.device()
    }

    #[getter]
    fn chunk(&self) -> u64 {
        self.gpu.chunk()
    }

    #[cfg(feature = "store")]
    #[pyo3(signature = (name, ck, regions, host_fd = None, files = None))]
    fn register(&self, py: Python<'_>, name: String, ck: PyRef<'_, crate::store::PyCheckpoint>, regions: Vec<Vec<String>>, host_fd: Option<i32>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (gpu, ck, files) = (&self.gpu, ck_of(&ck), granted(py, files)?);
        let ws = py.detach(|| gpu.register(&name, ck, &regions, host_fd, files)).or_raise(py)?;
        Ok(PyWeightSet::new(SetRef::Gpu(ws)))
    }

    fn trim(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.gpu.trim()).or_raise(py)
    }

    /// Answers other processes' asks of `row` from now on, on a Rust thread of this process with
    /// no Python (`Gpu::give`): idle chunks, then unpinned pages of idle sets before warm ones.
    fn serve(&self, py: Python<'_>, row: PyRef<'_, crate::varena_ledger::PyRow>) -> PyResult<()> {
        let gpu = self.gpu.clone();
        row.0.serve(move |want| gpu.give(want).unwrap_or(0)).map(drop).or_raise(py)
    }

    fn step(&self, py: Python<'_>) {
        py.detach(|| self.gpu.step())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyGpuStats> {
        Ok(py.detach(|| self.gpu.stats()).or_raise(py)?.into())
    }
}

#[pyclass(name = "Host", module = "tensorfs.varena", frozen)]
pub struct PyHost;

#[pymethods]
impl PyHost {
    #[new]
    fn new() -> Self {
        PyHost
    }

    #[cfg(feature = "store")]
    #[pyo3(signature = (name, ck, regions, host_fd = None, files = None))]
    fn register(&self, py: Python<'_>, name: String, ck: PyRef<'_, crate::store::PyCheckpoint>, regions: Vec<Vec<String>>, host_fd: Option<i32>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (ck, files) = (ck_of(&ck), granted(py, files)?);
        let ws = py.detach(|| Host.register(&name, ck, &regions, host_fd, files)).or_raise(py)?;
        Ok(PyWeightSet::new(SetRef::Host(ws)))
    }
}

// ---------------------------------------------------------------- weight sets

#[derive(Clone)]
enum SetRef {
    Gpu(Arc<WeightSet>),
    Host(Arc<HostSet>),
}

#[pyclass(name = "WeightSet", module = "tensorfs.varena", frozen)]
pub struct PyWeightSet {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    spans: Vec<u64>,
    #[pyo3(get)]
    device: Option<i32>,
    parts: Vec<Part>,
    set: Mutex<Option<SetRef>>,
}

impl PyWeightSet {
    fn new(set: SetRef) -> Self {
        let (name, layout, parts, device) = match &set {
            SetRef::Gpu(w) => (w.name(), w.layout(), w.parts(), Some(w.device())),
            SetRef::Host(h) => (h.name(), h.layout(), h.parts(), None),
        };
        PyWeightSet {
            name: name.into(),
            nbytes: layout.nbytes,
            spans: layout.regions.iter().map(|r| r.span).collect(),
            device,
            parts: parts.to_vec(),
            set: Mutex::new(Some(set)),
        }
    }

    fn set(&self, py: Python<'_>) -> PyResult<SetRef> {
        let set = self.set.lock().unwrap().clone();
        set.ok_or_else(|| err(py, Error::Closed(format!("weight set {:?}", self.name))))
    }
}

#[pymethods]
impl PyWeightSet {
    #[getter]
    fn parts(&self) -> Vec<PyPart> {
        self.parts.iter().map(Into::into).collect()
    }

    /// The watermark (GPU): these regions live at home, the rest stream through slots.
    fn home(&self, py: Python<'_>, regions: Vec<u32>) -> PyResult<()> {
        match self.set(py)? {
            SetRef::Gpu(w) => py.detach(|| w.home(&regions)).or_raise(py),
            SetRef::Host(_) => Ok(()),
        }
    }

    #[pyo3(signature = (region, stream = 0))]
    fn bind(&self, py: Python<'_>, region: u32, stream: u64) -> PyResult<PyPinned> {
        Ok(match self.set(py)? {
            SetRef::Gpu(w) => PyPinned::gpu(py.detach(|| w.bind(region, stream)).or_raise(py)?),
            SetRef::Host(h) => PyPinned::host(py.detach(|| h.bind(region)).or_raise(py)?),
        })
    }

    fn prefetch(&self, py: Python<'_>, region: u32) -> PyResult<()> {
        match self.set(py)? {
            SetRef::Gpu(w) => py.detach(|| w.prefetch(region)).or_raise(py),
            SetRef::Host(_) => Ok(()),
        }
    }

    /// Whether its model is warm (loaded at GPU level): an ask takes idle sets first.
    fn warm(&self, py: Python<'_>, warm: bool) -> PyResult<()> {
        if let SetRef::Gpu(w) = self.set(py)? {
            w.warm(warm);
        }
        Ok(())
    }

    /// Whether a forward of its model is in flight: an ask never takes a busy set's pages.
    fn busy(&self, py: Python<'_>, busy: bool) -> PyResult<()> {
        if let SetRef::Gpu(w) = self.set(py)? {
            w.busy(busy);
        }
        Ok(())
    }

    #[pyo3(signature = (regions = None))]
    fn evict(&self, py: Python<'_>, regions: Option<Vec<u32>>) -> PyResult<u64> {
        match self.set(py)? {
            SetRef::Gpu(w) => py.detach(|| w.evict(regions.as_deref())).or_raise(py),
            SetRef::Host(_) => Ok(0),
        }
    }

    /// Lets the set go: its VA and pages are released now, or when its last pin dies.
    fn close(&self, py: Python<'_>) {
        let set = self.set.lock().unwrap().take();
        py.detach(|| drop(set));
    }

    fn __repr__(&self) -> String {
        format!("<tensorfs.varena.WeightSet {:?} {} B in {} regions>", self.name, self.nbytes, self.spans.len())
    }
}

// ---------------------------------------------------------------- pins and DLPack

/// The pin a Python handle and every tensor made from it share.
struct Pin {
    gpu: Option<Pinned>,
    _host: Option<HostPinned>,
}

/// A bound region: a uint8 DLPack tensor of `nbytes` at `ptr`. It and every tensor made from it
/// hold the pin.
#[pyclass(name = "Pinned", module = "tensorfs.varena", frozen)]
pub struct PyPinned {
    #[pyo3(get)]
    region: u32,
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    home: bool,
    #[pyo3(get)]
    fresh: bool,
    #[pyo3(get)]
    writable: bool,
    dl: (i32, i32),
    pin: Arc<Pin>,
}

impl PyPinned {
    fn gpu(p: Pinned) -> Self {
        PyPinned { region: p.region, ptr: p.ptr, nbytes: p.nbytes, home: p.home(), fresh: p.fresh, writable: true, dl: (KDL_CUDA, p.device()), pin: Arc::new(Pin { gpu: Some(p), _host: None }) }
    }
    fn host(p: HostPinned) -> Self {
        PyPinned { region: p.region, ptr: p.ptr, nbytes: p.nbytes, home: true, fresh: p.fresh, writable: p.writable, dl: (KDL_CPU, 0), pin: Arc::new(Pin { gpu: None, _host: Some(p) }) }
    }
}

const KDL_CPU: i32 = 1;
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

/// What a consumer's tensor owns: the shape and stride cells, and the pin.
struct Held {
    dims: [i64; 2],
    _pin: Arc<Pin>,
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
impl PyPinned {
    /// `stream`: the consumer's, which also waits for the bytes (the bind stream already does).
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'p>(
        &self,
        py: Python<'p>,
        stream: Option<i64>,
        max_version: Option<Bound<'p, PyAny>>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let _ = max_version;
        if copy == Some(true) {
            return Err(PyBufferError::new_err("a bound region is never copied"));
        }
        if dl_device.is_some_and(|d| d != self.dl) {
            return Err(PyBufferError::new_err(format!("a bound region lives on {:?}", self.dl)));
        }
        if let (Some(p), Some(s)) = (&self.pin.gpu, stream.filter(|&s| s > 0)) {
            py.detach(|| p.wait_on(s as u64)).or_raise(py)?;
        }
        let held = Box::into_raw(Box::new(Held { dims: [self.nbytes as i64, 1], _pin: self.pin.clone() }));
        // SAFETY: `held` is live until the deleter runs.
        let dims = unsafe { (*held).dims.as_mut_ptr() };
        let t = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: self.ptr as *mut c_void,
                device: DLDevice { device_type: self.dl.0, device_id: self.dl.1 },
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
        self.dl
    }
}

// ---------------------------------------------------------------- the host tier

/// A whole-tier claim on a host-tier memfd for an owner that maps none of it (TensorD): a new
/// descriptor the caller owns; closing it ends the claim.
#[pyfunction]
fn hold_tier(py: Python<'_>, fd: i32) -> PyResult<i32> {
    varena::host::hold(fd).map(std::os::fd::IntoRawFd::into_raw_fd).or_raise(py)
}

/// End a `hold_tier` claim: punches every Ready region nobody claims, closes `hold`, and
/// returns the bytes punched.
#[pyfunction]
fn release_tier(py: Python<'_>, hold: i32) -> PyResult<u64> {
    // SAFETY: the caller hands over the descriptor `hold_tier` returned.
    let hold = unsafe { <std::os::fd::OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(hold) };
    varena::host::release_hold(hold).or_raise(py)
}

/// `regions` of a v2 checkpoint in a sealed host layout, filled in this process from its chunk
/// files (`files`, else the checkpoint's paths): what `register(host_fd=)` adopts. Returns the
/// memfd (the caller's).
#[cfg(feature = "store")]
#[pyfunction]
#[pyo3(signature = (ck, regions, name, threads = 8, files = None))]
fn seal(py: Python<'_>, ck: PyRef<'_, crate::store::PyCheckpoint>, regions: Vec<Vec<String>>, name: String, threads: usize, files: Option<Vec<(String, u64, i32)>>) -> PyResult<i32> {
    let (ck, files) = (ck_of(&ck), granted(py, files)?);
    py.detach(|| varena::memory::seal(ck, &regions, &name, threads, files)).map(std::os::fd::IntoRawFd::into_raw_fd).or_raise(py)
}

// ---------------------------------------------------------------- the module

macro_rules! add_errors {
    ($m:expr; $($e:ident),*) => { $($m.add(stringify!($e), $m.py().get_type::<$e>())?;)* };
}

/// `tensorfs._ext.varena`, importable as such (the package `tensorfs.varena` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<PyGpu>()?;
    m.add_class::<PyHost>()?;
    m.add_class::<PyWeightSet>()?;
    m.add_class::<PyPinned>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PyCounters>()?;
    m.add_class::<PySetStats>()?;
    m.add_class::<PyGpuStats>()?;
    m.add_function(wrap_pyfunction!(hold_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(release_tier, &m)?)?;
    #[cfg(feature = "store")]
    m.add_function(wrap_pyfunction!(seal, &m)?)?;
    add_errors!(m; VarenaError, DeviceFull, CudaError, CudaUnavailable, Invalid, IoError, Closed, Unsupported);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}
