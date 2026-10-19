//! `tensorfs.varena`: varena's Python surface, typed in `tensorfs/varena/__init__.pyi`. Marshal
//! only: every call that can wait, read or touch the driver runs detached from the GIL. A bound
//! region crosses as a uint8 DLPack tensor whose deleter releases the pin.

use std::ffi::{c_void, CStr};
#[cfg(feature = "store")]
use std::fs::File;
use std::os::fd::{AsRawFd, BorrowedFd, FromRawFd, IntoRawFd, OwnedFd};
use std::sync::{Arc, Mutex};

use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use pyo3::{create_exception, ffi};
#[cfg(feature = "store")]
use tensorfs_core::ids::ObjectRef;
use varena::ledger::{Admit, Facts, Ledger, Row, RowFacts, State, Victim};
use varena::ceiling::Copy;
use varena::memory::{Counters, GpuStats, Part, SetStats};
use varena::{Error, Gpu, Host, HostPinned, HostSet, Pinned, WeightSet};

use crate::errors::to_py;
#[cfg(feature = "store")]
use crate::store::PyCheckpoint;

/// What varena reads of a `tensorfs.Checkpoint`: its header, and its chunk paths when a store
/// holds it (a parsed header reads only a grant's files or a sealed layout).
#[cfg(feature = "store")]
fn ck_of(ck: &PyCheckpoint) -> varena::checkpoint::Ck<'_> {
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
create_exception!(tensorfs.varena, LedgerError, PyException, "A ledger error; `.code` names its class.");

trait OrRaise<T> {
    /// The core's errors: a class per kind, and source refusals as the usual `tensorfs.errors`.
    fn or_raise(self, py: Python<'_>) -> PyResult<T>;
    /// The ledger's: `LedgerError`.
    fn or_ledger(self, py: Python<'_>) -> PyResult<T>;
}

impl<T> OrRaise<T> for varena::Result<T> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T> {
        self.map_err(|e| {
            let x = match &e {
                Error::Source(r) => return to_py(py, r),
                Error::DeviceFull { .. } => DeviceFull::new_err(e.to_string()),
                Error::Cuda { .. } => CudaError::new_err(e.to_string()),
                Error::CudaUnavailable(_) => CudaUnavailable::new_err(e.to_string()),
                Error::Invalid(_) => Invalid::new_err(e.to_string()),
                Error::Io(_) => IoError::new_err(e.to_string()),
                Error::Closed(_) => Closed::new_err(e.to_string()),
                Error::Unsupported(_) => Unsupported::new_err(e.to_string()),
            };
            let v = x.value(py);
            let _ = v.setattr("code", e.code());
            if let Error::DeviceFull { need, deficit } = e {
                let _ = (v.setattr("need", need), v.setattr("deficit", deficit));
            }
            x
        })
    }

    fn or_ledger(self, py: Python<'_>) -> PyResult<T> {
        self.map_err(|e| {
            let x = LedgerError::new_err(e.to_string());
            let _ = x.value(py).setattr("code", e.code());
            x
        })
    }
}

// ---------------------------------------------------------------- records

/// A frozen class with a property per field, from varena's record of the same fields; the
/// field after `;` is a list whose items convert.
macro_rules! record {
    ($py:ident = $name:literal from $rs:ty { $($f:ident: $t:ty),* $(; $v:ident: $vt:ty)? }) => {
        #[pyclass(name = $name, module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
        #[derive(Clone)]
        pub struct $py {
            $($f: $t,)*
            $($v: Vec<$vt>)?
        }

        impl From<$rs> for $py {
            fn from(x: $rs) -> Self {
                $py { $($f: x.$f.into(),)* $($v: x.$v.into_iter().map(Into::into).collect())? }
            }
        }
    };
}

record!(PyCounters = "Counters" from Counters {
    binds: u64, faults: u64, paged_in: u64, streamed_bytes: u64, evicted: u64, copy_ns: u64, stall_ns: u64, map_ns: u64
});
record!(PySetStats = "SetStats" from SetStats {
    name: String, nbytes: u64, regions: u32, home: u64, resident: u64, pinned: u64, pins: u32, slot_bytes: u64,
    step: PyCounters, last: PyCounters, total: PyCounters
});
record!(PyGpuStats = "GpuStats" from GpuStats {
    device: i32, chunk: u64, steps: u64, mapped: u64, idle: u64, slots: u32, slot_bytes: u64, slot_span: u64,
    driver_free: u64, driver_total: u64; sets: PySetStats
});

record!(PyCopy = "Copy" from Copy {
    kind: String, gpu: i32, bytes: u64, ns: u64, ceiling: f64, link: String, busy_ns: u64, pin_ns: u64, staged: u64, cached: u64
});

/// Every run of weight copies since the last call (`h2d`: a GPU's copy stream from its first job
/// to its last landing; `disk`: a host layout filled from chunk files), with its ceiling in GB/s.
#[pyfunction]
fn copies() -> Vec<PyCopy> {
    varena::ceiling::take().into_iter().map(Into::into).collect()
}

#[pyclass(name = "Part", module = "tensorfs.varena", frozen, get_all)]
pub struct PyPart {
    component: String,
    key: String,
    role: String,
    dtype: String,
    shape: Py<PyTuple>,
    region: u32,
    offset: u64,
    nbytes: u64,
}

// ---------------------------------------------------------------- cores

/// A grant's chunk files, `(sha256, length, fd)` each: every fd duplicated, the caller's kept.
#[cfg(feature = "store")]
fn granted(py: Python<'_>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<Option<Vec<(ObjectRef, File)>>> {
    let dup = |(sha256, length, fd): (String, u64, i32)| -> varena::Result<(ObjectRef, File)> {
        // SAFETY: a descriptor the caller holds for the duration of this call.
        let own = unsafe { BorrowedFd::borrow_raw(fd) }.try_clone_to_owned().map_err(|e| Error::Io(format!("dup chunk fd {fd}: {e}")));
        Ok((ObjectRef { sha256, length }, File::from(own?)))
    };
    files.map(|f| f.into_iter().map(dup).collect::<varena::Result<_>>()).transpose().or_raise(py)
}

#[pyclass(name = "Gpu", module = "tensorfs.varena", frozen)]
pub struct PyGpu(Arc<Gpu>);

#[pymethods]
impl PyGpu {
    #[new]
    #[pyo3(signature = (device, chunk = 64 << 20))]
    fn new(py: Python<'_>, device: i32, chunk: u64) -> PyResult<Self> {
        Ok(PyGpu(py.detach(|| Gpu::open(device, chunk)).or_raise(py)?))
    }

    #[getter]
    fn device(&self) -> i32 {
        self.0.device()
    }

    #[getter]
    fn chunk(&self) -> u64 {
        self.0.chunk()
    }

    #[cfg(feature = "store")]
    #[pyo3(signature = (name, ck, regions, host_fd = None, files = None))]
    fn register(&self, py: Python<'_>, name: String, ck: PyRef<'_, PyCheckpoint>, regions: Vec<Vec<String>>, host_fd: Option<i32>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (ck, files) = (ck_of(&ck), granted(py, files)?);
        Ok(PyWeightSet::new(SetRef::Gpu(py.detach(|| self.0.register(&name, ck, &regions, host_fd, files)).or_raise(py)?)))
    }

    /// The sealed host layout `host_fd` registered from the regions it carries, before any module.
    #[cfg(feature = "store")]
    #[pyo3(signature = (name, ck, host_fd, files = None))]
    fn adopt(&self, py: Python<'_>, name: String, ck: PyRef<'_, PyCheckpoint>, host_fd: i32, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (ck, files) = (ck_of(&ck), granted(py, files)?);
        Ok(PyWeightSet::new(SetRef::Gpu(py.detach(|| self.0.adopt(&name, ck, host_fd, files)).or_raise(py)?)))
    }

    fn trim(&self, py: Python<'_>) -> PyResult<u64> {
        py.detach(|| self.0.trim()).or_raise(py)
    }

    /// Frees up to `nbytes` as an ask of this process's row would; what each set gave.
    fn give(&self, py: Python<'_>, nbytes: u64) -> PyResult<Vec<(String, u64)>> {
        py.detach(|| self.0.give(nbytes)).or_raise(py)
    }

    /// Answers other processes' asks of `row` from now on with `give`, on a Rust thread.
    fn serve(&self, py: Python<'_>, row: PyRef<'_, PyRow>) -> PyResult<()> {
        let gpu = self.0.clone();
        row.0.serve(move |want| gpu.give(want).map_or(0, |gave| gave.iter().map(|g| g.1).sum())).map(drop).or_raise(py)
    }

    fn step(&self, py: Python<'_>) {
        py.detach(|| self.0.step())
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyGpuStats> {
        Ok(py.detach(|| self.0.stats()).or_raise(py)?.into())
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
    fn register(&self, py: Python<'_>, name: String, ck: PyRef<'_, PyCheckpoint>, regions: Vec<Vec<String>>, host_fd: Option<i32>, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (ck, files) = (ck_of(&ck), granted(py, files)?);
        Ok(PyWeightSet::new(SetRef::Host(py.detach(|| Host.register(&name, ck, &regions, host_fd, files)).or_raise(py)?)))
    }

    #[cfg(feature = "store")]
    #[pyo3(signature = (name, ck, host_fd, files = None))]
    fn adopt(&self, py: Python<'_>, name: String, ck: PyRef<'_, PyCheckpoint>, host_fd: i32, files: Option<Vec<(String, u64, i32)>>) -> PyResult<PyWeightSet> {
        let (ck, files) = (ck_of(&ck), granted(py, files)?);
        Ok(PyWeightSet::new(SetRef::Host(py.detach(|| Host.adopt(&name, ck, host_fd, files)).or_raise(py)?)))
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
    regions: Vec<Vec<String>>,
    #[pyo3(get)]
    device: Option<i32>,
    parts: Vec<Part>,
    set: Mutex<Option<SetRef>>,
}

impl PyWeightSet {
    fn new(set: SetRef) -> Self {
        let (w, device) = match &set {
            SetRef::Gpu(ws) => (&*ws.weights, Some(ws.device())),
            SetRef::Host(hs) => (&hs.weights, None),
        };
        let spans = w.layout.regions.iter().map(|r| r.span).collect();
        let regions = w.layout.names.clone();
        PyWeightSet { name: w.name.clone(), nbytes: w.layout.nbytes, spans, regions, device, parts: w.parts.clone(), set: Mutex::new(Some(set)) }
    }

    fn set(&self, py: Python<'_>) -> PyResult<SetRef> {
        let set = self.set.lock().unwrap().clone();
        set.ok_or_else(|| Error::Closed(format!("weight set {:?}", self.name))).or_raise(py)
    }

    /// The GPU set; None on the host, where the watermark, prefetch and eviction do nothing.
    fn gpu(&self, py: Python<'_>) -> PyResult<Option<Arc<WeightSet>>> {
        Ok(match self.set(py)? {
            SetRef::Gpu(w) => Some(w),
            SetRef::Host(_) => None,
        })
    }
}

#[pymethods]
impl PyWeightSet {
    #[getter]
    fn parts(&self, py: Python<'_>) -> PyResult<Vec<PyPart>> {
        let part = |p: &Part| {
            let shape = PyTuple::new(py, &p.shape)?.unbind();
            let (component, key, role, dtype) = (p.component.clone(), p.key.clone(), p.role.clone(), p.dtype.clone());
            Ok(PyPart { component, key, role, dtype, shape, region: p.region, offset: p.offset, nbytes: p.nbytes })
        };
        self.parts.iter().map(part).collect()
    }

    fn home(&self, py: Python<'_>, regions: Vec<u32>) -> PyResult<()> {
        self.gpu(py)?.map_or(Ok(()), |w| py.detach(|| w.home(&regions)).or_raise(py))
    }

    #[pyo3(signature = (region, stream = 0))]
    fn bind(&self, py: Python<'_>, region: u32, stream: u64) -> PyResult<PyPinned> {
        Ok(match self.set(py)? {
            SetRef::Gpu(w) => PyPinned::gpu(py.detach(|| w.bind(region, stream)).or_raise(py)?),
            SetRef::Host(h) => PyPinned::host(py.detach(|| h.bind(region)).or_raise(py)?),
        })
    }

    fn prefetch(&self, py: Python<'_>, region: u32) -> PyResult<()> {
        self.gpu(py)?.map_or(Ok(()), |w| py.detach(|| w.prefetch(region)).or_raise(py))
    }

    /// GPU: mapped at home with bytes there or on their way. Host: always.
    fn resident(&self, py: Python<'_>, region: u32) -> PyResult<bool> {
        match self.gpu(py)? {
            Some(w) => w.resident(region).or_raise(py),
            None if (region as usize) < self.spans.len() => Ok(true),
            None => Err(Error::Invalid(format!("region {region} of {}", self.spans.len()))).or_raise(py),
        }
    }

    fn warm(&self, py: Python<'_>, warm: bool) -> PyResult<()> {
        if let Some(w) = self.gpu(py)? {
            w.warm(warm)
        }
        Ok(())
    }

    fn busy(&self, py: Python<'_>, busy: bool) -> PyResult<()> {
        if let Some(w) = self.gpu(py)? {
            w.busy(busy)
        }
        Ok(())
    }

    #[pyo3(signature = (regions = None))]
    fn evict(&self, py: Python<'_>, regions: Option<Vec<u32>>) -> PyResult<u64> {
        self.gpu(py)?.map_or(Ok(0), |w| py.detach(|| w.evict(regions.as_deref())).or_raise(py))
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
        let (home, dl) = (p.home(), (KDL_CUDA, p.device()));
        PyPinned { region: p.region, ptr: p.ptr, nbytes: p.nbytes, home, fresh: p.fresh, writable: true, dl, pin: Arc::new(Pin { gpu: Some(p), _host: None }) }
    }

    fn host(p: HostPinned) -> Self {
        let (region, ptr, nbytes, fresh, writable) = (p.region, p.ptr, p.nbytes, p.fresh, p.writable);
        PyPinned { region, ptr, nbytes, home: true, fresh, writable, dl: (KDL_CPU, 0), pin: Arc::new(Pin { gpu: None, _host: Some(p) }) }
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
    if !t.is_null() {
        // SAFETY: both boxes were leaked by `__dlpack__` and are freed exactly once, here.
        let m = unsafe { Box::from_raw(t) };
        drop(unsafe { Box::from_raw(m.manager_ctx as *mut Held) });
    }
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
    fn __dlpack__<'p>(&self, py: Python<'p>, stream: Option<i64>, max_version: Option<Bound<'p, PyAny>>, dl_device: Option<(i32, i32)>, copy: Option<bool>) -> PyResult<Bound<'p, PyAny>> {
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
        let (device, dtype) = (DLDevice { device_type: self.dl.0, device_id: self.dl.1 }, DLDataType { code: KDL_UINT, bits: 8, lanes: 1 });
        let dl_tensor = DLTensor { data: self.ptr as *mut c_void, device, ndim: 1, dtype, shape: dims, strides: unsafe { dims.add(1) }, byte_offset: 0 };
        let t = Box::into_raw(Box::new(DLManagedTensor { dl_tensor, manager_ctx: held as *mut c_void, deleter: Some(deleter) }));
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
    varena::host::hold(fd).map(IntoRawFd::into_raw_fd).or_raise(py)
}

/// Ends a `hold_tier` claim: punches every Ready region nobody claims, closes `hold`, and
/// returns the bytes punched.
#[pyfunction]
fn release_tier(py: Python<'_>, hold: i32) -> PyResult<u64> {
    // SAFETY: the caller hands over the descriptor `hold_tier` returned.
    varena::host::release_hold(unsafe { OwnedFd::from_raw_fd(hold) }).or_raise(py)
}

/// `regions` of a v2 checkpoint in a sealed host layout, filled in this process from its chunk
/// files (`files`, else the checkpoint's paths): what `register(host_fd=)` adopts. Returns the
/// memfd (the caller's).
#[cfg(feature = "store")]
#[pyfunction]
#[pyo3(signature = (ck, regions, name, threads = 8, files = None))]
fn seal(py: Python<'_>, ck: PyRef<'_, PyCheckpoint>, regions: Vec<Vec<String>>, name: String, threads: usize, files: Option<Vec<(String, u64, i32)>>) -> PyResult<i32> {
    let (ck, files) = (ck_of(&ck), granted(py, files)?);
    py.detach(|| varena::memory::seal(ck, &regions, &name, threads, files)).map(IntoRawFd::into_raw_fd).or_raise(py)
}

// ---------------------------------------------------------------- the ledger

#[pyclass(name = "Ledger", module = "tensorfs.varena", frozen)]
pub struct PyLedger(Ledger);

#[pymethods]
impl PyLedger {
    #[staticmethod]
    fn create(py: Python<'_>, gpu_uuid: &str, capacity: u64) -> PyResult<Self> {
        Ok(Self(Ledger::create(gpu_uuid, capacity).or_ledger(py)?.0))
    }

    /// The table behind `fd`, and the eventfd beside it (`freed` signals it).
    #[staticmethod]
    #[pyo3(signature = (fd, events_fd = None))]
    fn adopt(py: Python<'_>, fd: i32, events_fd: Option<i32>) -> PyResult<Self> {
        // SAFETY: the caller holds `fd` and `events_fd` open for the call.
        let (fd, events) = unsafe { (BorrowedFd::borrow_raw(fd), events_fd.map(|e| BorrowedFd::borrow_raw(e))) };
        Ok(Self(Ledger::adopt(fd, events).or_ledger(py)?))
    }

    #[getter]
    fn fd(&self) -> i32 {
        self.0.fd().as_raw_fd()
    }

    #[getter]
    fn events_fd(&self) -> Option<i32> {
        self.0.events().map(|e| e.as_raw_fd())
    }

    #[getter]
    fn gpu_uuid(&self) -> String {
        self.0.gpu_uuid()
    }

    #[getter]
    fn capacity(&self) -> u64 {
        self.0.capacity()
    }

    /// A row for this process.
    fn claim(&self, py: Python<'_>) -> PyResult<PyRow> {
        Ok(PyRow(Arc::new(py.detach(|| self.0.claim(std::process::id())).or_ledger(py)?)))
    }

    #[pyo3(signature = (need, asker = None))]
    fn ask(&self, py: Python<'_>, need: u64, asker: Option<u32>) -> PyResult<Vec<PyVictim>> {
        Ok(victims(py.detach(|| self.0.ask(asker, need)).or_ledger(py)?))
    }

    fn answered(&self, py: Python<'_>, victim: &PyVictim) -> PyResult<bool> {
        self.0.answered(&victim.v).or_ledger(py)
    }

    fn wait(&self, py: Python<'_>, victim: &PyVictim) -> PyResult<()> {
        py.detach(|| self.0.wait(&victim.v)).or_ledger(py)
    }

    fn admit(&self, py: Python<'_>, need: u64) -> PyResult<PyAdmit> {
        Ok(match py.detach(|| self.0.admit(need)).or_ledger(py)? {
            Admit::Now => PyAdmit { now: true, victims: vec![], short: 0 },
            Admit::After { victims: v, short } => PyAdmit { now: false, victims: victims(v), short },
        })
    }

    fn reap(&self, py: Python<'_>, pid: u32) -> PyResult<u64> {
        py.detach(|| self.0.reap(pid)).or_ledger(py)
    }

    fn returned(&self, py: Python<'_>, pid: u32) -> PyResult<()> {
        self.0.returned(pid).or_ledger(py)
    }

    fn stalled(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
        py.detach(|| self.0.stalled()).or_ledger(py)
    }

    fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<PyRowFacts>> {
        Ok(py.detach(|| self.0.snapshot()).or_ledger(py)?.into_iter().map(Into::into).collect())
    }

    /// Bytes freed here: the free generation moves and the eventfd is signalled.
    fn freed(&self) {
        self.0.freed()
    }

    fn freed_gen(&self) -> u64 {
        self.0.freed_gen()
    }

    fn wait_freed(&self, py: Python<'_>, seen: u64) -> u64 {
        py.detach(|| self.0.wait_freed(seen))
    }

    fn peak(&self, key: &str) -> Option<u64> {
        self.0.peak(key)
    }

    fn set_peak(&self, py: Python<'_>, key: &str, nbytes: u64) -> PyResult<()> {
        self.0.set_peak(key, nbytes).or_ledger(py)
    }

    fn dispatched(&self, request_id: &str) -> Option<String> {
        self.0.dispatched(request_id)
    }

    fn dispatch(&self, py: Python<'_>, request_id: &str, key: &str) -> PyResult<()> {
        self.0.dispatch(request_id, key).or_ledger(py)
    }

    fn undispatch(&self, request_id: &str) {
        self.0.undispatch(request_id)
    }

    #[pyo3(signature = (package, shape = None, spawn = false))]
    fn fits(&self, py: Python<'_>, package: &str, shape: Option<&str>, spawn: bool) -> PyResult<bool> {
        py.detach(|| self.0.fits(package, shape, spawn)).or_ledger(py)
    }
}

#[pyclass(name = "Victim", module = "tensorfs.varena", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyVictim {
    #[pyo3(get)]
    slot: u32,
    #[pyo3(get)]
    pid: u32,
    #[pyo3(get)]
    bytes: u64,
    v: Victim,
}

fn victims(vs: Vec<Victim>) -> Vec<PyVictim> {
    vs.into_iter().map(|v| PyVictim { slot: v.slot, pid: v.pid, bytes: v.bytes, v }).collect()
}

#[pyclass(name = "Admit", module = "tensorfs.varena", frozen, get_all)]
pub struct PyAdmit {
    now: bool,
    victims: Vec<PyVictim>,
    short: u64,
}

/// `Row.publish`'s keywords and `RowFacts`' properties: the ledger's `Facts`, field for field.
macro_rules! facts {
    ($($f:ident),*) => {
        #[pyclass(name = "Row", module = "tensorfs.varena", frozen)]
        pub struct PyRow(Arc<Row>);

        #[pymethods]
        impl PyRow {
            #[getter]
            fn id(&self) -> u32 {
                self.0.id()
            }

            #[getter]
            fn pid(&self) -> u32 {
                self.0.pid()
            }

            #[pyo3(signature = (*, $($f = 0),*))]
            #[allow(clippy::too_many_arguments)]
            fn publish(&self, $($f: u64),*) {
                self.0.publish(&Facts { $($f),* })
            }

            fn want(&self) -> u64 {
                self.0.want()
            }

            fn wait_want(&self, py: Python<'_>) -> u64 {
                py.detach(|| self.0.wait_want())
            }

            fn ack(&self, freed: u64) {
                self.0.ack(freed)
            }

            /// Its requests running now: (request id, remaining measured peak; -1 never measured).
            fn running(&self, py: Python<'_>, requests: Vec<(String, i64)>) -> PyResult<()> {
                self.0.running(&requests).or_ledger(py)
            }
        }

        #[pyclass(name = "RowFacts", module = "tensorfs.varena", frozen, get_all)]
        pub struct PyRowFacts {
            slot: u32,
            gen: u64,
            pid: u32,
            state: &'static str,
            $($f: u64,)*
            want: u64,
            acked: u64,
            progress: u64,
            running: Vec<(String, i64)>,
            held: u64,
            unevictable: u64,
            reclaimable: u64,
        }

        impl From<RowFacts> for PyRowFacts {
            fn from(r: RowFacts) -> Self {
                let state = match r.state {
                    State::Live => "live",
                    State::Dead => "dead",
                    State::Reaping => "reaping",
                };
                let (held, unevictable, reclaimable) = (r.held(), r.unevictable(), r.reclaimable());
                let (slot, gen, pid, want, acked, progress) = (r.slot, r.gen, r.pid, r.want, r.acked, r.progress);
                PyRowFacts { slot, gen, pid, state, $($f: r.facts.$f,)* want, acked, progress, running: r.running, held, unevictable, reclaimable }
            }
        }
    };
}

facts!(context, resident, cold, kept, pinned, torch_reserved, torch_allocated, cache, idle, active_peak, last_use, waiting);

// ---------------------------------------------------------------- the module

/// `tensorfs._ext.varena`, importable as such (the package `tensorfs.varena` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    macro_rules! add {
        ($($c:ty),*; $($f:ident),*; $($e:ident),*) => {
            $(m.add_class::<$c>()?;)*
            $(m.add_function(wrap_pyfunction!($f, &m)?)?;)*
            $(m.add(stringify!($e), py.get_type::<$e>())?;)*
        };
    }
    add!(PyGpu, PyHost, PyWeightSet, PyPinned, PyPart, PyCounters, PySetStats, PyGpuStats, PyCopy, PyLedger, PyRow, PyVictim, PyAdmit, PyRowFacts;
        copies, hold_tier, release_tier;
        VarenaError, DeviceFull, CudaError, CudaUnavailable, Invalid, IoError, Closed, Unsupported, LedgerError);
    #[cfg(feature = "store")]
    add!(; seal;);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}
