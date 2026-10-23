//! `tensorfs.varena`: varena's [`Memory`] for Python, typed in `tensorfs/varena/__init__.pyi`.
//! Conversion only: every call runs detached from the GIL with panics caught (`VarenaError`,
//! code `INTERNAL`). A pin crosses as a uint8 DLPack tensor, and every tensor made from it
//! holds it.

use std::ffi::{c_void, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, MutexGuard};

use pyo3::create_exception;
use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::ffi;
use pyo3::prelude::*;
use tensorfs::format::Body;
use varena::layout::Layout;
use varena::{Backend, CudaBackend, Error, Fill, Fingerprint, GpuId, HostBackend, Level, Memory, Origin, Pass, Pin, Plan, RegionKind, Request, Retained, SetId, SetSpec, Step};

use crate::store::PyCheckpoint;

create_exception!(tensorfs.varena, VarenaError, PyException, "A varena error: `.code` names its class; `.cuda_code` is the driver's CUresult for a device error; `.sticky` means the executor must retire (device failure or unproven ownership).");
create_exception!(tensorfs.varena, StaleBinding, VarenaError, "The retained dependency identity changed; discard this graph before recapture.");
create_exception!(tensorfs.varena, Exhausted, VarenaError, "The one memory error: `.gpu`, `.need`, `.site`, `.stats`.");

fn error(py: Python<'_>, gpu: &str, e: Error) -> PyErr {
    let err = match &e {
        Error::Exhausted(_) => Exhausted::new_err(e.to_string()),
        Error::StaleBinding(_) => StaleBinding::new_err(e.to_string()),
        _ => VarenaError::new_err(e.to_string()),
    };
    let v = err.value(py);
    let _ = v.setattr("code", e.code());
    let _ = v.setattr("sticky", e.is_sticky());
    let cuda = match &e {
        Error::Device { code, .. } => Some(*code),
        _ => None,
    };
    let _ = v.setattr("cuda_code", cuda);
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
    peak: u64,
    #[pyo3(get)]
    compute: u64,
    #[pyo3(get)]
    leftover: u64,
    /// Each make_room that took bytes: (step, trigger, need, freed, release events waited on).
    #[pyo3(get)]
    make_room: Vec<(u32, &'static str, u64, u64, u32)>,
    /// Its torch blocks still live at its end (ptr, bytes, stream): the request-end audit.
    #[pyo3(get)]
    outlived: Vec<Outlived>,
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
            Origin::Store => "store",
        };
        PyStats {
            raw,
            request: s.request,
            moved: PyMoved { kept: m.kept.into(), store: m.store.into(), speculative: m.speculative },
            trace: s.trace.into_iter().map(|r| PyRef_ { step: r.step, set: r.set, block: r.block, bytes: r.bytes, origin: origin(r.origin), speculative: r.speculative }).collect(),
            steps: s.steps.into_iter().map(|t| PyStepStats { capacity: t.capacity, evicted: t.evicted.into() }).collect(),
            stall_ns: s.stall_ns,
            evicted: s.evicted.into(),
            peak: s.peak,
            compute: s.compute,
            leftover: s.leftover,
            make_room: s.make_room.into_iter().map(|m| (m.step, m.trigger, m.need, m.freed, m.waited)).collect(),
            outlived: s.outlived.into_iter().map(|o| (o.ptr, o.bytes, o.stream)).collect(),
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

    /// `gpu`'s memory; the backend opens first.
    fn open(py: Python<'_>, dl: (i32, i32), backend: impl Send + FnOnce() -> varena::Result<(GpuId, Box<dyn Backend>)>) -> PyResult<Self> {
        let (gpu, memory) = detached(py, "", || {
            let (gpu, backend) = backend()?;
            Ok((gpu.clone(), Memory::open(gpu, backend)?))
        })?;
        let gpu = PyGpuId { uuid: gpu.uuid, model: gpu.model, memory: gpu.memory };
        Ok(PyMemory { memory: Some(Arc::new(memory)), gpu, dl })
    }
}

#[pymethods]
impl PyMemory {
    /// CUDA device `ordinal`'s memory. Open it before torch initializes CUDA.
    #[staticmethod]
    fn cuda(py: Python<'_>, ordinal: i32) -> PyResult<Self> {
        let memory = PyMemory::open(py, (KDL_CUDA, ordinal), || {
            let backend = CudaBackend::open(ordinal)?;
            Ok((backend.gpu()?, Box::new(backend) as Box<dyn Backend>))
        })?;
        // torch's allocations on this device are this Memory's from now on (the shim's agent).
        crate::agent::serve(ordinal, memory.memory().clone());
        Ok(memory)
    }

    /// varena's torch figures here: (allocated, mapped, peak, fragmentation (mapped less live),
    /// per open request (id, allocated, peak), per pool (pool, mapped, allocated)). torch's own memory statistics don't see a
    /// pluggable allocator.
    fn torch_stats(&self, py: Python<'_>) -> PyResult<TorchFigures> {
        detached(py, &self.gpu.uuid, || Ok(torch_stats(self.memory())))
    }

    /// An event on `stream` after torch pool `pool`'s use there (a replay and its readers): the
    /// pool's chunks leave, once released, only after every fence completed.
    fn fence_pool(&self, py: Python<'_>, pool: (u64, u64), stream: usize) -> PyResult<()> {
        detached(py, &self.gpu.uuid, || self.memory().pool_fence(varena::PoolId(pool.0, pool.1), stream))
    }

    /// A long-lived intermediate's region (§6): `kind` is kv, latent, conditioning, derived or
    /// workspace; `sets` the components it serves; `spill` false drops it for recompute on revoke.
    #[pyo3(signature = (owner, kind, sets = Vec::new(), spill = true))]
    fn region(&self, py: Python<'_>, owner: &str, kind: &str, sets: Vec<u32>, spill: bool) -> PyResult<PyRegion> {
        let kind = match kind {
            "kv" => RegionKind::Kv,
            "latent" => RegionKind::Latent,
            "conditioning" => RegionKind::Conditioning,
            "derived" => RegionKind::Derived,
            "workspace" => RegionKind::Workspace,
            other => return Err(error(py, &self.gpu.uuid, Error::Invalid(format!("no region kind {other:?}")))),
        };
        let sets: Vec<SetId> = sets.into_iter().map(SetId).collect();
        let region = detached(py, &self.gpu.uuid, || self.memory().region(owner, kind, &sets, spill))?;
        Ok(PyRegion { region: Mutex::new(Some(region)), gpu: self.gpu.uuid.clone() })
    }

    /// A component's graph region on this device (§5): every graph of it captures into `pool`,
    /// the device's one capture pool token.
    #[pyo3(signature = (pool, sets = Vec::new(), retained = Vec::new()))]
    fn graph_region(&self, py: Python<'_>, pool: (u64, u64), sets: Vec<u32>, retained: Vec<u32>) -> PyResult<PyGraphRegion> {
        let (sets, retained): (Vec<SetId>, Vec<SetId>) = (sets.into_iter().map(SetId).collect(), retained.into_iter().map(SetId).collect());
        let graph = detached(py, &self.gpu.uuid, || self.memory().graph_region(&sets, &retained, varena::PoolId(pool.0, pool.1)))?;
        Ok(PyGraphRegion { graph: Mutex::new(Some(graph)), gpu: self.gpu.uuid.clone() })
    }

    /// The room a cut sizes from at plan step `step` (the request's cursor when None): (free,
    /// cold, warm, granularity, floor). Lock-free: read with the GIL held.
    #[pyo3(signature = (step = None))]
    fn room(&self, step: Option<u32>) -> (u64, u64, u64, u64, u64) {
        let r = self.memory().room(step);
        (r.free, r.cold, r.warm, r.granularity, r.floor)
    }

    /// Before a capture into torch pool `pool` (a `graph_pool_handle()`) begins: maps `expected`
    /// bytes into it, making room now, so the capture never needs to.
    fn prepare_pool(&self, py: Python<'_>, pool: (u64, u64), expected: u64) -> PyResult<()> {
        detached(py, &self.gpu.uuid, || self.memory().pool_prepare(varena::PoolId(pool.0, pool.1), expected))
    }

    /// Leases this GPU's weight pages from the manager TensorD serves in namespace `ns` (its
    /// uid), right after open: None when leased, else (code, why). `NotRegistered`: TensorD has
    /// not registered this executor yet; `Incompatible`: its manager speaks another protocol;
    /// `Absent`: no manager serves this GPU.
    fn lease(&self, py: Python<'_>, ns: u32) -> PyResult<Option<(String, String)>> {
        detached(py, &self.gpu.uuid, || lease(self.memory(), ns, &self.gpu.uuid))
    }

    /// Serves torch's allocations standalone, before any is made: from the device's own free
    /// bytes less the backend's reserve (no compatible manager, or none asked for). A leased
    /// Memory's are served by its lease.
    fn serve_standalone(&self, py: Python<'_>) -> PyResult<()> {
        detached(py, &self.gpu.uuid, || self.memory().serve_standalone_torch().map(drop))
    }

    /// Host pages of `gpu.memory` bytes (CPU, MPS), on the table `gpu.uuid` names.
    #[staticmethod]
    fn host(py: Python<'_>, gpu: PyGpuId) -> PyResult<Self> {
        let id = GpuId { uuid: gpu.uuid, model: gpu.model, memory: gpu.memory };
        PyMemory::open(py, (KDL_CPU, 0), || Ok((id.clone(), Box::new(HostBackend::new(id.memory)) as Box<dyn Backend>)))
    }

    /// Registers `set` and brings it to `level`; returns its id.
    fn load(&self, py: Python<'_>, set: PyRef<'_, PySet>, level: PyLevel) -> PyResult<u32> {
        let (spec, level) = (set.spec.clone(), if level == PyLevel::Gpu { Level::Gpu } else { Level::Ram });
        detached(py, &self.gpu.uuid, || self.memory().load(spec, level)).map(|s| s.0)
    }

    fn unload(&self, py: Python<'_>, set: u32) -> PyResult<()> {
        done(py, &self.gpu.uuid, || self.memory().unload(SetId(set)))
    }

    /// `set`'s current fill (a no-op when unchanged): pages filled under another come back fresh.
    fn fill(&self, py: Python<'_>, set: u32, fill: PyRef<'_, PyFill>) -> PyResult<()> {
        let fill = fill_of(&fill);
        done(py, &self.gpu.uuid, || self.memory().fill(SetId(set), fill))
    }

    /// Opens request `id` with its plan: `(set names, passes)` per step. An `idle` one (a
    /// capture or warm-up beside others' work) takes free bytes only: it evicts nothing.
    #[pyo3(signature = (id, plan, idle = false))]
    fn request(&self, py: Python<'_>, id: &str, plan: Vec<(Vec<String>, u32)>, idle: bool) -> PyResult<PyRequest> {
        let m = self.memory();
        let r = detached(py, &self.gpu.uuid, || if idle { m.request_idle(id, steps(plan)) } else { m.request(id, steps(plan)) })?;
        Ok(PyRequest { request: Mutex::new(Some(Arc::new(r))), gpu: self.gpu.uuid.clone() })
    }

    /// Block `block` of `set`, mapped and filled; `stream` (a `CUstream`, 0 the default) waits
    /// on the fill and is the only stream a tensor made from it may be consumed on. With
    /// `evict` false it maps into free bytes only, `Exhausted` at once when they are short.
    #[pyo3(signature = (set, block, stream = 0, evict = true))]
    fn bind(&self, py: Python<'_>, set: u32, block: u32, stream: usize, evict: bool) -> PyResult<PyPin> {
        let m = self.memory();
        let pin = detached(py, &self.gpu.uuid, || if evict { m.bind(SetId(set), block, stream) } else { m.bind_free(SetId(set), block, stream) })?;
        Ok(PyPin { ptr: pin.ptr, len: pin.len, fresh: pin.fresh, stream, dl: self.dl, gpu: self.gpu.uuid.clone(), pin: Arc::new(Held::new(pin)) })
    }

    /// Graph address retention: no physical pin until `prepare`.
    fn retain(&self, py: Python<'_>, set: u32, blocks: Vec<u32>) -> PyResult<PyRetained> {
        let retained = detached(py, &self.gpu.uuid, || self.memory().retain(SetId(set), &blocks))?;
        let identity = retained.identity().to_string();
        Ok(PyRetained { retained: Mutex::new(Some(Arc::new(retained))), identity, gpu: self.gpu.uuid.clone(), dl: self.dl })
    }

    fn context_measurable(&self, py: Python<'_>) -> PyResult<bool> {
        detached(py, &self.gpu.uuid, || Ok(self.memory().context_measurable()))
    }

    /// Mapped bytes by class (F, O, S, N, D, W, P).
    fn classes(&self) -> [u64; 7] {
        self.memory().classes()
    }

    /// The record of the request that ended last.
    fn stats(&self, py: Python<'_>) -> PyResult<Option<PyStats>> {
        detached(py, &self.gpu.uuid, || Ok(self.memory().stats())).map(|s| s.map(Into::into))
    }
}

/// A graph's address references; physical leases are acquired separately for each use.
#[pyclass(name = "Retained", module = "tensorfs.varena", frozen)]
pub struct PyRetained {
    retained: Mutex<Option<Arc<Retained>>>,
    #[pyo3(get)]
    identity: String,
    gpu: String,
    dl: (i32, i32),
}

#[pymethods]
impl PyRetained {
    fn valid(&self, py: Python<'_>) -> PyResult<bool> {
        let r = lock(&self.retained).clone();
        detached(py, &self.gpu, || Ok(r.is_some_and(|r| r.valid())))
    }

    /// Every block mapped and filled now: a read (no pin, no fill, no room made).
    fn resident(&self, py: Python<'_>) -> PyResult<bool> {
        let r = lock(&self.retained).clone();
        detached(py, &self.gpu, || Ok(r.is_some_and(|r| r.resident())))
    }

    #[pyo3(signature = (stream = 0, request = None))]
    fn prepare(&self, py: Python<'_>, stream: usize, request: Option<PyRef<'_, PyRequest>>) -> PyResult<Vec<PyPin>> {
        let r = lock(&self.retained).clone().ok_or_else(|| error(py, &self.gpu, Error::Closed("retained range".into())))?;
        let request = request.map(|r| r.get(py)).transpose()?;
        let pins = detached(py, &self.gpu, || r.prepare_for(stream, request.as_deref()))?;
        Ok(pins.into_iter().map(|pin| PyPin { ptr: pin.ptr, len: pin.len, fresh: pin.fresh, stream, dl: self.dl, gpu: self.gpu.clone(), pin: Arc::new(Held::new(pin)) }).collect())
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let r = lock(&self.retained).take();
        done(py, &self.gpu, || drop(r))
    }
}

impl Drop for PyRetained {
    fn drop(&mut self) { release(lock(&self.retained).take()); }
}

// ---------------------------------------------------------------- requests and passes

#[cfg(unix)]
fn lease(memory: &Memory, ns: u32, uuid: &str) -> varena::Result<Option<(String, String)>> {
    let Some(stream) = varena::lease::connect(ns, uuid)? else {
        return Ok(Some(("Absent".into(), format!("no lease manager serves GPU {uuid}"))));
    };
    Ok(memory.lease_over(stream)?.map(|(code, why)| (format!("{code:?}"), why)))
}

#[cfg(not(unix))]
fn lease(_: &Memory, _: u32, _: &str) -> varena::Result<Option<(String, String)>> {
    Ok(Some(("Absent".into(), "TensorD leases Linux GPUs only so far".into())))
}

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
    /// Starts a pass of `set` keyed `key` (`entry|dims`): the plan's cursor moves to it.
    #[pyo3(name = "pass_")]
    fn pass_(&self, py: Python<'_>, set: u32, key: &str) -> PyResult<PyPass> {
        let r = self.get(py)?;
        let pass = detached(py, &self.gpu, || r.pass(SetId(set), key))?;
        Ok(PyPass(Mutex::new(Some(pass))))
    }

    /// Replaces the plan: `(set names, passes)` per step.
    fn plan(&self, py: Python<'_>, plan: Vec<(Vec<String>, u32)>) -> PyResult<()> {
        let r = self.get(py)?;
        done(py, &self.gpu, || r.plan(steps(plan)))
    }

    /// The request waits on a child call until the guard ends: its step's pages are a later use,
    /// taken before the child's own.
    fn yielding(&self, py: Python<'_>) -> PyResult<PyYielded> {
        let r = self.get(py)?;
        let y = detached(py, &self.gpu, || Ok(r.yielding()))?;
        Ok(PyYielded { yielded: Mutex::new(Some(y)), gpu: self.gpu.clone() })
    }

    /// Its torch arena's high-water since the last call or pass start, which restarts here (a
    /// section of a pass, such as a split's window); the pass's own peak still covers it.
    fn reset_peak(&self, py: Python<'_>) -> PyResult<u64> {
        let r = self.get(py)?;
        detached(py, &self.gpu, || Ok(r.section_peak()))
    }

    /// This thread's torch blocks go to this request's arena in `memory` until the guard ends
    /// (on this thread).
    fn allocating(&self, py: Python<'_>, memory: PyRef<'_, PyMemory>) -> PyResult<PyAllocating> {
        let r = self.get(py)?;
        let m = memory.memory().clone();
        let guard = detached(py, &self.gpu, || Ok(m.allocating(&r)))?;
        Ok(PyAllocating(Mutex::new(Some(guard))))
    }

    fn stats(&self, py: Python<'_>) -> PyResult<PyStats> {
        let r = self.get(py)?;
        detached(py, &self.gpu, || Ok(r.stats())).map(Into::into)
    }

    /// Ends the request: its pins and caches go. Refused while another call is in it.
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

/// A request waiting on a child call (`Request.yielding`); `end` ends the wait.
#[pyclass(name = "Yielded", module = "tensorfs.varena", frozen)]
pub struct PyYielded {
    yielded: Mutex<Option<varena::Yielded>>,
    gpu: String,
}

#[pymethods]
impl PyYielded {
    fn end(&self, py: Python<'_>) -> PyResult<()> {
        let Some(y) = lock(&self.yielded).take() else { return Ok(()) };
        detached(py, &self.gpu, || y.end())
    }
}

/// A long-lived intermediate (`Memory.region`); `close` (or its drop) releases its arena.
#[pyclass(name = "Region", module = "tensorfs.varena", frozen)]
pub struct PyRegion {
    region: Mutex<Option<varena::Region>>,
    gpu: String,
}

impl PyRegion {
    fn with<T: Send>(&self, py: Python<'_>, f: impl Send + FnOnce(&varena::Region) -> varena::Result<T>) -> PyResult<T> {
        let guard = lock(&self.region);
        let region = guard.as_ref().ok_or_else(|| error(py, &self.gpu, Error::Closed("region closed".into())))?;
        detached(py, &self.gpu, || f(region))
    }
}

#[pymethods]
impl PyRegion {
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<u64> {
        self.with(py, |r| Ok(r.id()))
    }

    /// Pins it while in use, mapping it again first if it was reclaimed (which can raise
    /// Exhausted); the pin's `fresh` says it came back empty (recompute it).
    fn pin(&self, py: Python<'_>) -> PyResult<PyRegionPin> {
        let pin = self.with(py, |r| r.pin())?;
        Ok(PyRegionPin { fresh: pin.fresh(), pin: Mutex::new(Some(pin)) })
    }

    /// As `pin`, and this thread's torch blocks go to the region's own arena until it ends.
    fn allocating(&self, py: Python<'_>) -> PyResult<PyRegionPin> {
        let pin = self.with(py, |r| r.allocating())?;
        Ok(PyRegionPin { fresh: pin.fresh(), pin: Mutex::new(Some(pin)) })
    }

    /// Its production time in nanoseconds, for the spill-versus-drop choice.
    fn produced(&self, py: Python<'_>, ns: u64) -> PyResult<()> {
        self.with(py, |r| {
            r.produced(ns);
            Ok(())
        })
    }

    /// The plan step of its next use; None: no use declared.
    #[pyo3(signature = (step = None))]
    fn next_use(&self, py: Python<'_>, step: Option<u32>) -> PyResult<()> {
        self.with(py, |r| {
            r.next_use(step);
            Ok(())
        })
    }

    /// Releases it: (allocated, peak, its blocks still live (ptr, bytes, stream)).
    fn close(&self, py: Python<'_>) -> PyResult<(u64, u64, Vec<Outlived>)> {
        let Some(region) = lock(&self.region).take() else { return Ok((0, 0, Vec::new())) };
        detached(py, &self.gpu, || {
            let e = region.close();
            Ok((e.allocated, e.peak, e.outlived.into_iter().map(|o| (o.ptr, o.bytes, o.stream)).collect()))
        })
    }
}

/// A region pinned for use (`Region.pin`); `end` unpins it.
#[pyclass(name = "RegionPin", module = "tensorfs.varena", frozen)]
pub struct PyRegionPin {
    pin: Mutex<Option<varena::RegionPin>>,
    /// It was remapped empty since its last pin: the owner recomputes it.
    #[pyo3(get)]
    fresh: bool,
}

#[pymethods]
impl PyRegionPin {
    fn end(&self) {
        lock(&self.pin).take();
    }
}

/// A component's graphs on one device (`Memory.graph_region`).
#[pyclass(name = "GraphRegion", module = "tensorfs.varena", frozen)]
pub struct PyGraphRegion {
    graph: Mutex<Option<varena::GraphRegion>>,
    gpu: String,
}

impl PyGraphRegion {
    fn with<T: Send>(&self, py: Python<'_>, f: impl Send + FnOnce(&varena::GraphRegion) -> varena::Result<T>) -> PyResult<T> {
        let guard = lock(&self.graph);
        let graph = guard.as_ref().ok_or_else(|| error(py, &self.gpu, Error::Closed("graph region closed".into())))?;
        detached(py, &self.gpu, || f(graph))
    }
}

#[pymethods]
impl PyGraphRegion {
    #[getter]
    fn id(&self, py: Python<'_>) -> PyResult<u64> {
        self.with(py, |g| Ok(g.id()))
    }

    /// What capturing its graphs took, in nanoseconds.
    fn captured(&self, py: Python<'_>, ns: u64) -> PyResult<()> {
        self.with(py, |g| {
            g.captured(ns);
            Ok(())
        })
    }

    /// The capture pool token its graphs capture into.
    #[getter]
    fn pool(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        self.with(py, |g| Ok((g.pool().0, g.pool().1)))
    }

    /// Maps `expected` bytes into the pool before capture begins (a shape's first capture: None).
    #[pyo3(signature = (expected = None))]
    fn prepare(&self, py: Python<'_>, expected: Option<u64>) -> PyResult<()> {
        self.with(py, |g| g.prepare(expected))
    }

    /// Before each replay: False while it is being dropped (run eager, recapture off-path). A
    /// True one is released exactly once, even when the replay raises.
    fn acquire(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |g| Ok(g.acquire()))
    }

    /// After the replay's launch on `stream`: an event recorded there releases it.
    fn release(&self, py: Python<'_>, stream: usize) -> PyResult<()> {
        self.with(py, |g| g.release(stream))
    }

    #[getter]
    fn dropping(&self, py: Python<'_>) -> PyResult<bool> {
        self.with(py, |g| Ok(g.dropping()))
    }

    /// Drops its graphs' pins and its use of the pool.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let Some(graph) = lock(&self.graph).take() else { return Ok(()) };
        done(py, &self.gpu, || graph.close())
    }
}

/// A thread allocating into a request's arena (`Request.allocating`); `end` (on that thread)
/// ends it.
#[pyclass(name = "Allocating", module = "tensorfs.varena", frozen)]
pub struct PyAllocating(Mutex<Option<varena::Allocating>>);

#[pymethods]
impl PyAllocating {
    fn end(&self) {
        lock(&self.0).take();
    }
}

/// A pass: the plan's cursor at its set; `end` records torch's allocated peak.
#[pyclass(name = "Pass", module = "tensorfs.varena", frozen)]
pub struct PyPass(Mutex<Option<Pass>>);

impl Drop for PyPass {
    fn drop(&mut self) {
        release(lock(&self.0).take());
    }
}

#[pymethods]
impl PyPass {
    /// torch's allocated peak over it (absolute).
    fn end(&self, py: Python<'_>, peak: u64) -> PyResult<()> {
        let Some(pass) = lock(&self.0).take() else { return Ok(()) };
        done(py, "", || pass.end(peak))
    }
}

// ---------------------------------------------------------------- pins and DLPack

/// The pin a `Pin` and every tensor made from it share; its drop releases it.
struct Held(Option<Pin>);

/// Pins the binding holds now: Python's pin objects and the tensors that keep one.
static HELD: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Held {
    fn new(pin: Pin) -> Self {
        HELD.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self(Some(pin))
    }
}

impl Drop for Held {
    /// Also run by the DLPack deleter, maybe on a thread without the GIL: never unwinds.
    fn drop(&mut self) {
        let pin = self.0.take();
        if pin.is_some() {
            HELD.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
        }
        let _ = catch_unwind(AssertUnwindSafe(|| drop(pin)));
    }
}

/// Pins the binding holds now (Python's pin objects and the tensors that keep one).
#[pyfunction]
fn pins_held() -> u64 {
    HELD.load(std::sync::atomic::Ordering::Relaxed)
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

/// A pin a window's tensors may come to hold (`Window.keep`).
type Keeper = Arc<Mutex<Option<Arc<Held>>>>;

/// What a consumer's tensor owns: its shape and stride cells, and its pin: a pin's own, or the
/// one its window was handed (none until then).
struct Tensor {
    dims: [i64; 2],
    _pin: Option<Arc<Held>>,
    _kept: Option<Keeper>,
}

/// A uint8 DLPack capsule of `len` bytes at `ptr` on `dl`, owning `pin` (or what `kept` comes to
/// hold) while any tensor of it lives.
fn capsule<'p>(py: Python<'p>, ptr: u64, len: u64, dl: (i32, i32), pin: Option<Arc<Held>>, kept: Option<Keeper>) -> PyResult<Bound<'p, PyAny>> {
    let held = Box::into_raw(Box::new(Tensor { dims: [len as i64, 1], _pin: pin, _kept: kept }));
    // SAFETY: `held` is live until the deleter runs.
    let dims = unsafe { (*held).dims.as_mut_ptr() };
    let (device, dtype) = (DLDevice { device_type: dl.0, device_id: dl.1 }, DLDataType { code: KDL_UINT, bits: 8, lanes: 1 });
    let dl_tensor = DLTensor { data: ptr as *mut c_void, device, ndim: 1, dtype, shape: dims, strides: unsafe { dims.add(1) }, byte_offset: 0 };
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
        capsule(py, self.ptr, self.len, self.dl, Some(self.pin.clone()), None)
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        self.dl
    }

    /// The same bytes as a window that holds no pin: tensors made from it stay valid only while
    /// the block is bound (an adapter keeps them across binds of one mapping and installs them
    /// only under a pin).
    fn window(&self) -> PyWindow {
        PyWindow { ptr: self.ptr, len: self.len, dl: self.dl, kept: Arc::new(Mutex::new(None)) }
    }
}

/// A block's bytes without a pin (`Pin.window`): its tensors are valid only while it is bound,
/// unless `keep` hands them a pin.
#[pyclass(name = "Window", module = "tensorfs.varena", frozen)]
pub struct PyWindow {
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    len: u64,
    dl: (i32, i32),
    kept: Keeper,
}

#[pymethods]
impl PyWindow {
    /// A uint8 tensor of `len` bytes; the pin that binds the block orders its consumers.
    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'p>(&self, py: Python<'p>, stream: Option<i64>, max_version: Option<Bound<'p, PyAny>>, dl_device: Option<(i32, i32)>, copy: Option<bool>) -> PyResult<Bound<'p, PyAny>> {
        let _ = (stream, max_version);
        if copy == Some(true) {
            return Err(PyBufferError::new_err("a window is never copied"));
        }
        if dl_device.is_some_and(|d| d != self.dl) {
            return Err(PyBufferError::new_err(format!("a window lives on {:?}", self.dl)));
        }
        capsule(py, self.ptr, self.len, self.dl, None, Some(self.kept.clone()))
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        self.dl
    }

    /// Its tensors hold `pin` from now on: the page stays bound while any of them lives (an
    /// escaped view's), released with the last.
    fn keep(&self, pin: PyRef<'_, PyPin>) {
        *lock(&self.kept) = Some(pin.pin.clone());
    }
}


/// varena's torch allocator figures: (allocated, mapped, peak, per open request (id, allocated,
/// peak)).
/// A torch block still live at its arena's end: (ptr, bytes, stream).
type Outlived = (u64, u64, usize);

type TorchFigures = (u64, u64, u64, u64, Vec<(String, u64, u64)>, Vec<((u64, u64), u64, u64)>);

fn torch_stats(memory: &Memory) -> TorchFigures {
    let s = memory.torch_stats();
    (s.allocated, s.mapped, s.peak, s.fragmentation, s.requests, s.pools.into_iter().map(|(p, m, a)| ((p.0, p.1), m, a)).collect())
}

// ---------------------------------------------------------------- the module

/// `tensorfs._ext.varena`, importable as such (the package `tensorfs.varena` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<PyGpuId>()?;
    m.add_class::<PyFill>()?;
    m.add_class::<PyLevel>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PySet>()?;
    m.add_class::<PyMemory>()?;
    m.add_class::<PyRequest>()?;
    m.add_class::<PyAllocating>()?;
    m.add_class::<PyRegion>()?;
    m.add_class::<PyYielded>()?;
    m.add_class::<PyRegionPin>()?;
    m.add_class::<PyGraphRegion>()?;
    m.add_class::<PyPass>()?;
    m.add_class::<PyPin>()?;
    m.add_class::<PyRetained>()?;
    m.add_class::<PyWindow>()?;
    m.add_class::<PyFills>()?;
    m.add_class::<PyMoved>()?;
    m.add_class::<PyRef_>()?;
    m.add_class::<PyStepStats>()?;
    m.add_class::<PyStats>()?;
    m.add_function(wrap_pyfunction!(pins_held, &m)?)?;
    m.add("VarenaError", py.get_type::<VarenaError>())?;
    m.add("Exhausted", py.get_type::<Exhausted>())?;
    m.add("StaleBinding", py.get_type::<StaleBinding>())?;
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.varena", &m)
}
