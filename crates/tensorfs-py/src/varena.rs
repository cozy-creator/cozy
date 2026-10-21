//! `tensorfs.varena`: varena's [`Memory`] for Python, typed in `tensorfs/varena/__init__.pyi`.
//! Conversion only: every call runs detached from the GIL with panics caught (`VarenaError`,
//! code `INTERNAL`). A pin crosses as a uint8 DLPack tensor, and every tensor made from it
//! holds it.

use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

use platform::handles::Owned;
use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::ffi;
use pyo3::prelude::*;
use tensorfs::format::Body;
use varena::layout::Layout;
use varena::{Backend, CudaBackend, Error, Fill, Fingerprint, GpuId, Grant, HostBackend, Level, Measured, Memory, Next, Origin, Pass, Pin, Plan, Request, SetId, SetSpec, Site, Step, Torch};

use crate::store::PyCheckpoint;

create_exception!(tensorfs.varena, VarenaError, PyException, "A varena error: `.code` names its class; `.sticky` means the device context is gone.");
create_exception!(tensorfs.varena, Exhausted, VarenaError, "The one memory error: `.gpu`, `.need`, `.site`, `.stats`.");

fn error(py: Python<'_>, gpu: &str, e: Error) -> PyErr {
    let err = match &e {
        Error::Exhausted(_) => Exhausted::new_err(e.to_string()),
        _ => VarenaError::new_err(e.to_string()),
    };
    let v = err.value(py);
    let _ = v.setattr("code", e.code());
    let _ = v.setattr("sticky", e.is_sticky());
    if let Error::Exhausted(x) = e {
        let _ = (v.setattr("gpu", gpu), v.setattr("need", x.need), v.setattr("site", x.site));
        let _ = Py::new(py, PyStats::from(x.stats)).and_then(|s| v.setattr("stats", s));
    }
    err
}

fn caught<T>(f: impl FnOnce() -> varena::Result<T>) -> varena::Result<T> {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|p| {
        let msg = p.downcast_ref::<&str>().map(|s| s.to_string()).or_else(|| p.downcast_ref::<String>().cloned());
        Err(Error::Internal(format!("panic: {}", msg.unwrap_or_default())))
    })
}

/// `f` off the GIL, its panic an `INTERNAL` error.
fn detached<T: Send>(py: Python<'_>, gpu: &str, f: impl Send + FnOnce() -> varena::Result<T>) -> PyResult<T> {
    py.detach(|| caught(f)).map_err(|e| error(py, gpu, e))
}

/// A Python callback's byte count, from varena's thread; its error is reported, never raised.
fn call<'py>(py: Python<'py>, f: &Py<PyAny>, args: impl pyo3::call::PyCallArgs<'py>) -> u64 {
    f.call1(py, args).and_then(|r| r.extract::<u64>(py)).unwrap_or_else(|e| {
        e.write_unraisable(py, None);
        0
    })
}

/// `detached` for a call with nothing to return.
fn done(py: Python<'_>, gpu: &str, f: impl Send + FnOnce()) -> PyResult<()> {
    detached(py, gpu, || {
        f();
        Ok(())
    })
}

/// Drops `x` off the GIL: a native drop may wait for varena's lock, whose holder may call back
/// into Python (a cache's `shrink`).
fn release<T: Send>(x: T) {
    Python::attach(|py| py.detach(|| drop(catch_unwind(AssertUnwindSafe(|| drop(x))))));
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

/// A descriptor (Unix) or handle (Windows) the caller hands over.
fn owned(raw: i64) -> Owned {
    // SAFETY: the caller transfers ownership of an open descriptor or handle.
    #[cfg(unix)]
    return unsafe { std::os::fd::FromRawFd::from_raw_fd(raw as i32) };
    #[cfg(windows)]
    return unsafe { std::os::windows::io::FromRawHandle::from_raw_handle(raw as isize as *mut c_void) };
}

// ---------------------------------------------------------------- records

#[pyclass(name = "GpuId", module = "tensorfs.varena", frozen, get_all, from_py_object)]
#[derive(Clone)]
pub struct PyGpuId {
    uuid: String,
    model: String,
    memory: u64,
}

#[pymethods]
impl PyGpuId {
    #[new]
    fn new(uuid: String, model: String, memory: u64) -> Self {
        PyGpuId { uuid, model, memory }
    }
}

#[pyclass(name = "Fills", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyFills {
    bytes: u64,
    ns: u64,
}

#[pyclass(name = "Moved", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyMoved {
    kept: PyFills,
    tier: PyFills,
    store: PyFills,
    speculative: u64,
}

#[pyclass(name = "Ref", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyRef_ {
    step: u32,
    set: String,
    block: u32,
    bytes: u64,
    origin: &'static str,
    speculative: bool,
}

#[pyclass(name = "StepStats", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyStepStats {
    capacity: u64,
    w: u64,
    evicted: Vec<u64>,
}

#[pyclass(name = "Stats", module = "tensorfs.varena", frozen, skip_from_py_object)]
#[derive(Clone)]
pub struct PyStats {
    raw: varena::Stats,
    #[pyo3(get)]
    request: String,
    #[pyo3(get)]
    moved: PyMoved,
    #[pyo3(get)]
    trace: Vec<PyRef_>,
    #[pyo3(get)]
    steps: Vec<PyStepStats>,
    #[pyo3(get)]
    stall_ns: u64,
    #[pyo3(get)]
    evicted: Vec<u64>,
    #[pyo3(get)]
    evicted_lowered: u64,
    #[pyo3(get)]
    speculative_evicted_d: u64,
    #[pyo3(get)]
    retries: u32,
    #[pyo3(get)]
    cuts: u32,
    #[pyo3(get)]
    parks: u32,
    #[pyo3(get)]
    parked: u64,
    #[pyo3(get)]
    lowered: bool,
    #[pyo3(get)]
    peak: u64,
    #[pyo3(get)]
    leftover: u64,
}

impl From<varena::Fills> for PyFills {
    fn from(f: varena::Fills) -> Self {
        PyFills { bytes: f.bytes, ns: f.ns }
    }
}

impl From<varena::Stats> for PyStats {
    fn from(s: varena::Stats) -> Self {
        let raw = s.clone();
        let m = s.moved;
        let origin = |o: Origin| match o {
            Origin::Hit => "hit",
            Origin::Kept => "kept",
            Origin::Tier => "tier",
            Origin::Store => "store",
        };
        PyStats {
            raw,
            request: s.request,
            moved: PyMoved { kept: m.kept.into(), tier: m.tier.into(), store: m.store.into(), speculative: m.speculative },
            trace: s.trace.into_iter().map(|r| PyRef_ { step: r.step, set: r.set, block: r.block, bytes: r.bytes, origin: origin(r.origin), speculative: r.speculative }).collect(),
            steps: s.steps.into_iter().map(|t| PyStepStats { capacity: t.capacity, w: t.w, evicted: t.evicted.into() }).collect(),
            stall_ns: s.stall_ns,
            evicted: s.evicted.into(),
            evicted_lowered: s.evicted_lowered,
            speculative_evicted_d: s.speculative_evicted_d,
            retries: s.retries,
            cuts: s.cuts,
            parks: s.parks,
            parked: s.parked,
            lowered: s.lowered,
            peak: s.peak,
            leftover: s.leftover,
        }
    }
}

#[pymethods]
impl PyStats {
    /// The record as JSON, field for field (what a Runtime relays).
    fn json(&self) -> PyResult<String> {
        serde_json::to_string(&self.raw).map_err(|e| VarenaError::new_err(e.to_string()))
    }
}

/// What turns a page's stored bytes into its weights (varena's `Fill`).
#[pyclass(name = "Fill", module = "tensorfs.varena", frozen)]
pub enum PyFill {
    Stored(),
    Declared { identity: String },
    Undeclared(),
}

#[pyclass(name = "Level", module = "tensorfs.varena", frozen, eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
pub enum PyLevel {
    #[pyo3(name = "RAM")]
    Ram,
    #[pyo3(name = "GPU")]
    Gpu,
}

/// `r.oom`'s answer.
#[pyclass(name = "Next", module = "tensorfs.varena", frozen)]
pub enum PyNext {
    Retry(),
    Cut { level: u32 },
    /// Move idle activations, up to `budget` bytes, to the host; report them with `parked`.
    Park { budget: u64 },
    /// `error` is the `Exhausted` to raise.
    Exhausted { error: Py<PyAny> },
}

/// TensorD's descriptors for this process's memory; `Memory.open` takes them.
#[pyclass(name = "Grant", module = "tensorfs.varena", frozen)]
pub struct PyGrant(Mutex<Option<Grant>>);

#[pymethods]
impl PyGrant {
    /// Takes ownership of the table's and the events' descriptors (handles on Windows).
    #[new]
    fn new(table: i64, events: i64) -> Self {
        PyGrant(Mutex::new(Some(Grant { table: owned(table), events: owned(events) })))
    }

    /// A standalone engine's grant: `gpu`'s table by name; `limit` lowers its capacity (the first
    /// opener sets it, a later one can only lower it).
    #[staticmethod]
    #[pyo3(signature = (gpu, limit = None))]
    fn open(py: Python<'_>, gpu: PyGpuId, limit: Option<u64>) -> PyResult<Self> {
        let id = GpuId { uuid: gpu.uuid, model: gpu.model, memory: gpu.memory };
        let grant = detached(py, &id.uuid, || Grant::open(&id, limit))?;
        Ok(PyGrant(Mutex::new(Some(grant))))
    }
}

/// The fill a `Fill` names.
fn fill_of(fill: &PyFill) -> Fill {
    match fill {
        PyFill::Stored() => Fill::Stored,
        PyFill::Declared { identity } => Fill::Declared(identity.clone()),
        PyFill::Undeclared() => Fill::Undeclared,
    }
}

// ---------------------------------------------------------------- sets

#[pyclass(name = "Part", module = "tensorfs.varena", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyPart {
    component: String,
    key: String,
    role: String,
    dtype: String,
    shape: Vec<u64>,
    region: u32,
    /// Within its region.
    offset: u64,
    nbytes: u64,
}

/// A weight set: its layout (one page per region) from TensorFS, its source and fingerprint.
#[pyclass(name = "Set", module = "tensorfs.varena", frozen)]
pub struct PySet {
    spec: SetSpec,
    #[pyo3(get)]
    parts: Vec<PyPart>,
}

#[pymethods]
impl PySet {
    /// `regions[r]` names region `r`'s tensors (`component/key`); `gpu` is the device's UUID,
    /// `release` the package release, `shard` the rank's slice (`0/1` whole).
    #[new]
    #[pyo3(signature = (name, ck, regions, *, fill, release, shard, gpu))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, name: String, ck: PyRef<'_, PyCheckpoint>, regions: Vec<Vec<String>>, fill: PyRef<'_, PyFill>, release: &str, shard: &str, gpu: &str) -> PyResult<Self> {
        let ck = ck.core(py)?;
        let spans = py.detach(|| ck.spans(&regions)).map_err(|f| crate::errors::raise(py, f))?;
        let invalid = |e: Error| error(py, gpu, e);
        let layout = Layout::build(&spans, &regions).map_err(invalid)?;
        let mut ids = Vec::with_capacity(layout.parts.len());
        let mut parts = Vec::with_capacity(layout.parts.len());
        for p in &layout.parts {
            let stored = ck.stored_part(&p.what).ok_or_else(|| invalid(Error::Invalid(format!("{} is not stored", p.what))))?;
            let encoding = match ck.header().tensor(&stored.component, &stored.key).map(|t| &t.body) {
                Some(Body::Encoded { encoding, .. }) => encoding.to_string(),
                _ => String::new(),
            };
            let region = layout.region(p.region).map_err(invalid)?;
            ids.push(varena::PartId { name: p.what.clone(), encoding, dtype: stored.dtype.clone() });
            let role = if stored.role.is_empty() { "value".into() } else { stored.role };
            let (component, key, dtype, shape) = (stored.component, stored.key, stored.dtype, stored.shape);
            parts.push(PyPart { component, key, role, dtype, shape, region: p.region, offset: p.offset - region.offset, nbytes: p.nbytes });
        }
        let fingerprint = Fingerprint::new(&layout, &ids, fill_of(&fill), release, shard, gpu);
        let source = source(&ck).map_err(invalid)?;
        Ok(PySet { spec: SetSpec { name, layout: Arc::new(layout), source, fingerprint }, parts })
    }

    #[getter]
    fn name(&self) -> &str {
        &self.spec.name
    }

    #[getter]
    fn fingerprint(&self) -> &str {
        &self.spec.fingerprint.digest
    }

    /// Each region's tensor names, as given.
    #[getter]
    fn regions(&self) -> Vec<Vec<String>> {
        self.spec.layout.names.clone()
    }

    #[getter]
    fn nbytes(&self) -> u64 {
        self.spec.layout.nbytes
    }

    /// Each region's mapped bytes.
    #[getter]
    fn spans(&self) -> Vec<u64> {
        self.spec.layout.regions.iter().map(|r| r.span).collect()
    }
}

#[cfg(unix)]
fn source(ck: &tensorfs::Checkpoint) -> varena::Result<Arc<varena::table::Source>> {
    Ok(Arc::new(ck.source(true)))
}

#[cfg(windows)]
fn source(_: &tensorfs::Checkpoint) -> varena::Result<Arc<varena::table::Source>> {
    Err(Error::Invalid("TensorFS has no read source on Windows yet".into()))
}

// ---------------------------------------------------------------- memory

/// One process's memory on one device.
#[pyclass(name = "Memory", module = "tensorfs.varena", frozen)]
pub struct PyMemory {
    memory: Option<Arc<Memory>>,
    #[pyo3(get)]
    gpu: PyGpuId,
    /// The DLPack device its pins live on.
    dl: (i32, i32),
}

impl Drop for PyMemory {
    fn drop(&mut self) {
        release(self.memory.take());
    }
}

impl PyMemory {
    fn memory(&self) -> &Arc<Memory> {
        // Only `drop` takes it.
        self.memory.as_ref().unwrap_or_else(|| unreachable!())
    }

    /// `grant`, else `gpu`'s table by name with `limit`; the backend opens first.
    fn open(py: Python<'_>, grant: Option<PyRef<'_, PyGrant>>, limit: Option<u64>, dl: (i32, i32), backend: impl Send + FnOnce() -> varena::Result<(GpuId, Box<dyn Backend>)>) -> PyResult<Self> {
        let grant = match grant {
            Some(g) => Some(lock(&g.0).take().ok_or_else(|| error(py, "", Error::Closed("grant already used".into())))?),
            None => None,
        };
        let (gpu, memory) = detached(py, "", || {
            let (gpu, backend) = backend()?;
            let grant = match grant {
                Some(g) => g,
                None => Grant::open(&gpu, limit)?,
            };
            Ok((gpu.clone(), Memory::open(gpu, Some(grant), backend)?))
        })?;
        let gpu = PyGpuId { uuid: gpu.uuid, model: gpu.model, memory: gpu.memory };
        Ok(PyMemory { memory: Some(Arc::new(memory)), gpu, dl })
    }
}

#[pymethods]
impl PyMemory {
    /// CUDA device `ordinal`'s table through TensorD's `grant`, or by name without one
    /// (standalone, its capacity lowered to `limit`). Open it before torch initializes CUDA.
    #[staticmethod]
    #[pyo3(signature = (ordinal, grant = None, limit = None))]
    fn cuda(py: Python<'_>, ordinal: i32, grant: Option<PyRef<'_, PyGrant>>, limit: Option<u64>) -> PyResult<Self> {
        PyMemory::open(py, grant, limit, (KDL_CUDA, ordinal), || {
            let backend = CudaBackend::open(ordinal)?;
            Ok((backend.gpu()?, Box::new(backend) as Box<dyn Backend>))
        })
    }

    /// Host pages of `gpu.memory` bytes (CPU, MPS), on the table `gpu.uuid` names.
    #[staticmethod]
    #[pyo3(signature = (gpu, grant = None, limit = None))]
    fn host(py: Python<'_>, gpu: PyGpuId, grant: Option<PyRef<'_, PyGrant>>, limit: Option<u64>) -> PyResult<Self> {
        let id = GpuId { uuid: gpu.uuid, model: gpu.model, memory: gpu.memory };
        PyMemory::open(py, grant, limit, (KDL_CPU, 0), || Ok((id.clone(), Box::new(HostBackend::new(id.memory)) as Box<dyn Backend>)))
    }

    /// Registers `set` and brings it to `level`; returns its id.
    fn load(&self, py: Python<'_>, set: PyRef<'_, PySet>, level: PyLevel) -> PyResult<u32> {
        let (spec, level) = (set.spec.clone(), if level == PyLevel::Gpu { Level::Gpu } else { Level::Ram });
        detached(py, &self.gpu.uuid, || self.memory().load(spec, level)).map(|s| s.0)
    }

    fn unload(&self, py: Python<'_>, set: u32) -> PyResult<()> {
        done(py, &self.gpu.uuid, || self.memory().unload(SetId(set)))
    }

    /// The allocator's trim (torch's `empty_cache`), returning the bytes it freed: a demand
    /// trims before it evicts anything.
    fn trim_with(&self, py: Python<'_>, trim: Py<PyAny>) -> PyResult<()> {
        let trim = Box::new(move || Python::try_attach(|py| call(py, &trim, ())).unwrap_or(0));
        done(py, &self.gpu.uuid, || self.memory().trim_with(trim))
    }

    /// `set`'s current fill (a no-op when unchanged): pages filled under another come back fresh.
    fn fill(&self, py: Python<'_>, set: u32, fill: PyRef<'_, PyFill>) -> PyResult<()> {
        let fill = fill_of(&fill);
        done(py, &self.gpu.uuid, || self.memory().fill(SetId(set), fill))
    }

    /// Bytes the caller holds that `shrink(want) -> freed` gives back; returns its id.
    fn cache(&self, py: Python<'_>, owner: &str, shrink: Py<PyAny>) -> PyResult<u32> {
        let shrink = Box::new(move |want: u64| Python::try_attach(|py| call(py, &shrink, (want,))).unwrap_or(0));
        detached(py, &self.gpu.uuid, || Ok(self.memory().cache(owner, shrink))).map(|c| c.0)
    }

    fn cached(&self, py: Python<'_>, cache: u32, nbytes: u64) -> PyResult<()> {
        done(py, &self.gpu.uuid, || self.memory().cached(varena::CacheId(cache), nbytes))
    }

    fn uncache(&self, py: Python<'_>, cache: u32) -> PyResult<()> {
        done(py, &self.gpu.uuid, || self.memory().uncache(varena::CacheId(cache)))
    }

    /// Opens request `id` with its plan: `(set names, passes)` per step.
    fn request(&self, py: Python<'_>, id: &str, plan: Vec<(Vec<String>, u32)>) -> PyResult<PyRequest> {
        let r = detached(py, &self.gpu.uuid, || self.memory().request(id, steps(plan)))?;
        Ok(PyRequest { request: Mutex::new(Some(Arc::new(r))), gpu: self.gpu.uuid.clone() })
    }

    /// Block `block` of `set`, mapped and filled; `stream` (a `CUstream`, 0 the default) waits
    /// on the fill and is the only stream a tensor made from it may be consumed on.
    #[pyo3(signature = (set, block, stream = 0))]
    fn bind(&self, py: Python<'_>, set: u32, block: u32, stream: usize) -> PyResult<PyPin> {
        let pin = detached(py, &self.gpu.uuid, || self.memory().bind(SetId(set), block, stream))?;
        Ok(PyPin { ptr: pin.ptr, len: pin.len, fresh: pin.fresh, stream, dl: self.dl, gpu: self.gpu.uuid.clone(), pin: Arc::new(Held(Some(pin))) })
    }

    /// Bytes torch's allocator may hold now.
    fn torch_limit(&self, py: Python<'_>) -> PyResult<u64> {
        detached(py, &self.gpu.uuid, || Ok(self.memory().torch_limit()))
    }

    /// The record of the request that ended last.
    fn stats(&self, py: Python<'_>) -> PyResult<Option<PyStats>> {
        detached(py, &self.gpu.uuid, || Ok(self.memory().stats())).map(|s| s.map(Into::into))
    }
}

// ---------------------------------------------------------------- requests and passes

fn steps(plan: Vec<(Vec<String>, u32)>) -> Plan {
    Plan { steps: plan.into_iter().map(|(sets, passes)| Step { sets, passes }).collect() }
}

/// A request in flight; `end` (or leaving its `with`) ends it.
#[pyclass(name = "Request", module = "tensorfs.varena", frozen)]
pub struct PyRequest {
    request: Mutex<Option<Arc<Request>>>,
    gpu: String,
}

impl Drop for PyRequest {
    fn drop(&mut self) {
        release(lock(&self.request).take());
    }
}

impl PyRequest {
    fn get(&self, py: Python<'_>) -> PyResult<Arc<Request>> {
        lock(&self.request).clone().ok_or_else(|| error(py, &self.gpu, Error::Closed("request ended".into())))
    }
}

#[pymethods]
impl PyRequest {
    /// Starts a pass of `set` at `shape` (dims joined by `x`) with torch's bytes now; returns
    /// once its measured room is free.
    #[pyo3(name = "pass_")]
    fn pass_(&self, py: Python<'_>, set: u32, shape: &str, allocated: u64, reserved: u64) -> PyResult<PyPass> {
        let r = self.get(py)?;
        let pass = detached(py, &self.gpu, || r.pass(SetId(set), shape, Torch { allocated, reserved }))?;
        Ok(PyPass(Mutex::new(Some(pass))))
    }

    /// Replaces the plan: `(set names, passes)` per step.
    fn plan(&self, py: Python<'_>, plan: Vec<(Vec<String>, u32)>) -> PyResult<()> {
        let r = self.get(py)?;
        done(py, &self.gpu, || r.plan(steps(plan)))
    }

    /// An op ran out of memory, `need` more bytes when known, at cut site `site` (None: the op
    /// cannot be cut); a `collective` site's finer level waits for the next pass's agreement.
    #[pyo3(signature = (allocated, reserved, need = None, site = None, collective = false))]
    fn oom(&self, py: Python<'_>, allocated: u64, reserved: u64, need: Option<u64>, site: Option<&str>, collective: bool) -> PyResult<PyNext> {
        let r = self.get(py)?;
        let site = site.map(|name| Site { name, collective });
        Ok(match detached(py, &self.gpu, || Ok(r.oom(need, site, Torch { allocated, reserved })))? {
            Next::Retry => PyNext::Retry(),
            Next::Cut(level) => PyNext::Cut { level },
            Next::Park(budget) => PyNext::Park { budget },
            Next::Exhausted(x) => PyNext::Exhausted { error: error(py, &self.gpu, Error::Exhausted(x)).into_value(py).into_any() },
        })
    }

    /// The bytes a `Park` answer moved to the host (0: none, and the next `oom` is Exhausted).
    fn parked(&self, py: Python<'_>, nbytes: u64) -> PyResult<()> {
        let r = self.get(py)?;
        done(py, &self.gpu, || r.parked(nbytes))
    }

    /// The current pass's collective sites' proposed `(site, level)`: the level in effect, or a
    /// finer one an OOM proposed.
    fn levels(&self, py: Python<'_>) -> PyResult<Vec<(String, u32)>> {
        let r = self.get(py)?;
        detached(py, &self.gpu, || Ok(r.levels()))
    }

    /// The levels the ranks agreed (the maximum of their proposals), in effect for the current pass.
    fn agree(&self, py: Python<'_>, levels: Vec<(String, u32)>) -> PyResult<()> {
        let r = self.get(py)?;
        done(py, &self.gpu, || r.agree(levels))
    }

    /// The current pass's cut level for op site `site` (0: whole).
    fn cut(&self, py: Python<'_>, site: &str) -> PyResult<u32> {
        let r = self.get(py)?;
        detached(py, &self.gpu, || Ok(r.cut(site)))
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyStats> {
        let r = self.get(py)?;
        detached(py, &self.gpu, || Ok(r.stats())).map(Into::into)
    }

    /// Ends the request: its pins, rooms and caches go. Refused while another call is in it.
    fn end(&self, py: Python<'_>) -> PyResult<PyStats> {
        let r = lock(&self.request).take().ok_or_else(|| error(py, &self.gpu, Error::Closed("request ended".into())))?;
        match Arc::try_unwrap(r) {
            Ok(r) => detached(py, &self.gpu, || Ok(r.end())).map(Into::into),
            Err(r) => {
                *lock(&self.request) = Some(r);
                Err(error(py, &self.gpu, Error::Invalid("a call is still running in the request".into())))
            }
        }
    }
}

/// A pass whose room is made; `end` records what it measured, dropping it records nothing.
#[pyclass(name = "Pass", module = "tensorfs.varena", frozen)]
pub struct PyPass(Mutex<Option<Pass>>);

impl Drop for PyPass {
    fn drop(&mut self) {
        release(lock(&self.0).take());
    }
}

#[pymethods]
impl PyPass {
    /// torch's allocated peak (absolute), the allocator's slack at it and each cut site's own
    /// allocated peak; then torch's bytes now.
    fn end(&self, py: Python<'_>, allocated_peak: u64, slack: u64, sites: Vec<(String, u64)>, allocated: u64, reserved: u64) -> PyResult<()> {
        let Some(pass) = lock(&self.0).take() else { return Ok(()) };
        done(py, "", || pass.end(Measured { allocated_peak, slack, sites }, Torch { allocated, reserved }))
    }
}

// ---------------------------------------------------------------- pins and DLPack

/// The pin a `Pin` and every tensor made from it share; its drop releases it.
struct Held(Option<Pin>);

impl Drop for Held {
    /// Also run by the DLPack deleter, maybe on a thread without the GIL: never unwinds.
    fn drop(&mut self) {
        let pin = self.0.take();
        let _ = catch_unwind(AssertUnwindSafe(|| drop(pin)));
    }
}

/// A bound block: `len` bytes at `ptr`, mapped and filled (a fresh one awaits its fill).
#[pyclass(name = "Pin", module = "tensorfs.varena", frozen)]
pub struct PyPin {
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    len: u64,
    /// Its bytes were just copied: a set with a fill runs it, then `filled()`.
    #[pyo3(get)]
    fresh: bool,
    stream: usize,
    dl: (i32, i32),
    gpu: String,
    pin: Arc<Held>,
}

const KDL_CPU: i32 = 1;
const KDL_CUDA: i32 = 2;
const KDL_UINT: u8 = 1;
const DLTENSOR: &CStr = c"dltensor";

/// DLPack's consumer stream as a `CUstream`: None and 1 are the legacy default (0), 2 the
/// per-thread default; 0 and negative values (no sync) cannot order the consumer.
fn consumer(stream: Option<i64>) -> Option<usize> {
    match stream {
        None | Some(1) => Some(0),
        Some(s) if s > 1 => Some(s as usize),
        Some(_) => None,
    }
}

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

/// What a consumer's tensor owns: its shape and stride cells, and the pin.
struct Tensor {
    dims: [i64; 2],
    _pin: Arc<Held>,
}

unsafe extern "C" fn deleter(t: *mut DLManagedTensor) {
    if !t.is_null() {
        // SAFETY: both boxes were leaked by `__dlpack__` and are freed exactly once, here.
        let m = unsafe { Box::from_raw(t) };
        drop(unsafe { Box::from_raw(m.manager_ctx as *mut Tensor) });
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
impl PyPin {
    /// The caller's fill (a LoRA bake) of a fresh pin completed: its pages may now be kept.
    fn filled(&self, py: Python<'_>) -> PyResult<()> {
        let pin = self.pin.clone();
        done(py, &self.gpu, || pin.0.iter().for_each(Pin::filled))
    }

    /// A uint8 tensor of `len` bytes, consumed on the stream the pin was bound on.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'p>(&self, py: Python<'p>, stream: Option<i64>, max_version: Option<Bound<'p, PyAny>>, dl_device: Option<(i32, i32)>, copy: Option<bool>) -> PyResult<Bound<'p, PyAny>> {
        let _ = max_version;
        if copy == Some(true) {
            return Err(PyBufferError::new_err("a pin is never copied"));
        }
        if dl_device.is_some_and(|d| d != self.dl) {
            return Err(PyBufferError::new_err(format!("a pin lives on {:?}", self.dl)));
        }
        let ordered = match self.dl.0 {
            KDL_CUDA => consumer(stream) == consumer(Some(self.stream as i64).filter(|s| *s != 0)),
            _ => stream.is_none(),
        };
        if !ordered {
            return Err(PyBufferError::new_err(format!("a pin is consumed on the stream it was bound on ({}), not {stream:?}", self.stream)));
        }
        let held = Box::into_raw(Box::new(Tensor { dims: [self.len as i64, 1], _pin: self.pin.clone() }));
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

// ---------------------------------------------------------------- the module

/// `tensorfs._ext.varena`, importable as such (the package `tensorfs.varena` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<PyGpuId>()?;
    m.add_class::<PyGrant>()?;
    m.add_class::<PyFill>()?;
    m.add_class::<PyLevel>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PySet>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRequest>()?;
    m.add_class::<PyPass>()?;
    m.add_class::<PyPin>()?;
    m.add_class::<PyNext>()?;
    m.add_class::<PyFills>()?;
    m.add_class::<PyMoved>()?;
    m.add_class::<PyRef_>()?;
    m.add_class::<PyStepStats>()?;
    m.add_class::<PyStats>()?;
    m.add("VarenaError", py.get_type::<VarenaError>())?;
    m.add("Exhausted", py.get_type::<Exhausted>())?;
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}

#[cfg(test)]
mod tests {
    use super::consumer;

    #[test]
    fn dlpack_consumer_streams() {
        assert_eq!(consumer(None), Some(0));
        assert_eq!(consumer(Some(1)), Some(0));
        assert_eq!(consumer(Some(2)), Some(2));
        assert_eq!(consumer(Some(99)), Some(99));
        for s in [0, -1, -2, i64::MIN] {
            assert_eq!(consumer(Some(s)), None);
        }
    }
}
