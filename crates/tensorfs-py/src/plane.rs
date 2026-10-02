//! `tensorfs.plane`: the weight plane's Python surface, exactly `crates/tensorfs-plane/API.md`.
//!
//! Marshal only. Every call that can wait, do I/O or touch the driver runs detached from the
//! GIL. Stats, events and trims cross as plain dicts and lists; device bytes cross as uint8
//! DLPack views that keep the VA (never the pages) alive.

use std::ffi::{c_void, CStr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::{create_exception, ffi};
use tensorfs_plane::io::Source;
use tensorfs_plane::layout::Layout;
use tensorfs_plane::plane::{Event, LeaseView, Stats};
use tensorfs_plane::{
    Cursor, Error, Lease, Plane, PlaneConfig, RingView, Ticket, Tier, Trim, View, WsId,
};

use crate::errors::to_py;
use crate::store::{PyLease, PyPlan, PyStore};

create_exception!(tensorfs.plane, PlaneError, PyException, "A weight-plane error; `.code` names its class.");
create_exception!(tensorfs.plane, BudgetExceeded, PlaneError);
create_exception!(tensorfs.plane, Shortfall, PlaneError);
create_exception!(tensorfs.plane, BelowFloor, PlaneError);
create_exception!(tensorfs.plane, LeaseViolation, PlaneError);
create_exception!(tensorfs.plane, Poisoned, PlaneError);
create_exception!(tensorfs.plane, CudaError, PlaneError);
create_exception!(tensorfs.plane, CudaUnavailable, PlaneError);
create_exception!(tensorfs.plane, Invalid, PlaneError);
create_exception!(tensorfs.plane, IoError, PlaneError);
create_exception!(tensorfs.plane, Closed, PlaneError);

/// The one conversion. Source refusals cross as the usual `tensorfs.errors` classes.
fn err(py: Python<'_>, e: Error) -> PyErr {
    let msg = e.to_string();
    let x = match &e {
        Error::Source(r) => return to_py(py, r),
        Error::BudgetExceeded { .. } => BudgetExceeded::new_err(msg),
        Error::Shortfall { .. } => Shortfall::new_err(msg),
        Error::BelowFloor { .. } => BelowFloor::new_err(msg),
        Error::LeaseViolation { .. } => LeaseViolation::new_err(msg),
        Error::Poisoned(_) => Poisoned::new_err(msg),
        Error::Cuda { .. } => CudaError::new_err(msg),
        Error::CudaUnavailable(_) => CudaUnavailable::new_err(msg),
        Error::Invalid(_) => Invalid::new_err(msg),
        Error::Io(_) => IoError::new_err(msg),
        Error::Closed => Closed::new_err(msg),
    };
    let v = x.value(py);
    let _ = v.setattr("code", e.code());
    if let Error::Shortfall { pool, need, available, blockers } = &e {
        let _ = v.setattr("pool", *pool);
        let _ = v.setattr("need", *need);
        let _ = v.setattr("available", *available);
        let _ = v.setattr("blockers", blockers.clone());
    }
    if let Error::BudgetExceeded { pool, requested, available } = &e {
        let _ = v.setattr("pool", *pool);
        let _ = v.setattr("requested", *requested);
        let _ = v.setattr("available", *available);
    }
    if let Error::BelowFloor { pool, need, budget } = &e {
        let _ = v.setattr("pool", *pool);
        let _ = v.setattr("need", *need);
        let _ = v.setattr("budget", *budget);
    }
    x
}

trait OrRaise<T> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T>;
}

impl<T> OrRaise<T> for tensorfs_plane::Result<T> {
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

/// `"pinned"` or a CUDA ordinal.
#[derive(FromPyObject)]
enum TierArg {
    Device(i32),
    Name(String),
}

fn tier(py: Python<'_>, t: TierArg) -> PyResult<Tier> {
    match t {
        TierArg::Device(o) => Ok(Tier::Device(o)),
        TierArg::Name(s) if s == "pinned" => Ok(Tier::Pinned),
        TierArg::Name(s) => Err(err(py, Error::Invalid(format!("tier {s:?}: \"pinned\" or a CUDA ordinal")))),
    }
}

fn trim<'p>(py: Python<'p>, t: Trim) -> PyResult<Bound<'p, PyDict>> {
    Ok(dict!(py;
        "freed" => t.freed,
        "evicted" => t.evicted,
        "over_budget_unreleasable" => t.over_budget_unreleasable,
        "blockers" => t.blockers,
    ))
}

fn stats<'p>(py: Python<'p>, s: Stats) -> PyResult<Bound<'p, PyDict>> {
    let (h, c) = (&s.host, &s.host.counters);
    let host = dict!(py;
        "budget" => h.budget, "used" => h.used, "ready_bytes" => h.ready_bytes,
        "registered_bytes" => h.registered_bytes, "memfd_bytes" => h.memfd_bytes,
        "cached_bytes" => h.cached_bytes, "direct_bytes" => h.direct_bytes,
        "buffered_bytes" => h.buffered_bytes, "inline_bytes" => h.inline_bytes,
        "direct_refused" => h.direct_refused, "map_refused" => h.map_refused,
        "disk_read_bytes" => h.disk_read_bytes, "fills" => c.fills, "fill_bytes" => c.fill_bytes,
        "fill_ns" => c.fill_ns, "evictions" => c.evictions, "evicted_bytes" => c.evicted_bytes,
        "register_failed" => c.register_failed, "trims" => c.trims,
    );
    let devices = PyList::empty(py);
    for d in &s.devices {
        let c = &d.counters;
        devices.append(dict!(py;
            "ordinal" => d.ordinal, "uuid" => &d.uuid, "granularity" => d.granularity,
            "exportable" => d.exportable, "budget" => d.budget, "committed" => d.committed,
            "mapped" => d.mapped, "idle" => d.idle, "leased_bytes" => d.leased_bytes,
            "held_bytes" => d.held_bytes, "over_budget_unreleasable" => d.over_budget_unreleasable,
            "reserved_va" => d.reserved_va, "copies" => c.copies,
            "host_copy_bytes" => c.host_copy_bytes, "host_copy_ns" => c.host_copy_ns,
            "disk_copy_bytes" => c.disk_copy_bytes, "disk_copy_ns" => c.disk_copy_ns,
            "mapped_copy_bytes" => c.mapped_copy_bytes, "mapped_copy_ns" => c.mapped_copy_ns,
            "misses" => c.misses, "evictions" => c.evictions, "evicted_bytes" => c.evicted_bytes,
            "trims" => c.trims,
        ))?;
    }
    let sets = PyList::empty(py);
    for w in &s.sets {
        let devs = PyList::empty(py);
        for (ordinal, ready) in &w.devices {
            devs.append(dict!(py; "ordinal" => ordinal, "ready_bytes" => ready))?;
        }
        sets.append(dict!(py;
            "id" => w.id, "name" => &w.name, "nbytes" => w.nbytes, "regions" => w.regions,
            "host_ready_bytes" => w.host_ready_bytes, "host_memfd_bytes" => w.host_memfd_bytes,
            "evictions" => w.evictions, "evicted_bytes" => w.evicted_bytes, "devices" => devs,
        ))?;
    }
    let stages = PyList::empty(py);
    for (ws, device, tag, t) in &s.tags {
        stages.append(dict!(py;
            "ws" => ws, "device" => device, "tag" => tag, "acquires" => t.acquires,
            "skipped" => t.skipped, "misses" => t.misses, "late" => t.late,
            "late_bytes" => t.late_bytes, "stall_ns" => t.stall_ns,
            "streamed_bytes" => t.streamed_bytes,
        ))?;
    }
    let regions = PyList::empty(py);
    for (ws, device, tag, region, r) in &s.regions {
        regions.append(dict!(py;
            "ws" => ws, "device" => device, "tag" => tag, "region" => region, "uses" => r.uses,
            "compute_ns" => r.compute_ns, "stall_ns" => r.stall_ns, "late" => r.late,
        ))?;
    }
    Ok(dict!(py;
        "host" => host, "devices" => devices, "sets" => sets, "stages" => stages,
        "regions" => regions, "events_dropped" => s.events_dropped, "poisoned" => s.poisoned,
    ))
}

fn event<'p>(py: Python<'p>, e: Event) -> PyResult<Bound<'p, PyDict>> {
    Ok(dict!(py;
        "kind" => e.kind, "ws" => e.ws, "region" => e.region, "device" => e.device,
        "bytes" => e.bytes, "ns" => e.ns, "detail" => e.detail,
    ))
}

// ---------------------------------------------------------------- the plane

#[pyclass(name = "Plane", module = "tensorfs.plane", frozen)]
pub struct PyPlane {
    plane: Arc<Plane>,
}

impl PyPlane {
    fn id(&self, py: Python<'_>, ws: &PyWeightSet) -> PyResult<WsId> {
        if Arc::ptr_eq(&self.plane, &ws.plane) {
            return Ok(ws.id);
        }
        Err(err(py, Error::Invalid(format!("weight set {:?} belongs to another plane", ws.name))))
    }
}

#[pymethods]
impl PyPlane {
    #[new]
    #[pyo3(signature = (devices, readers = 8, copy_streams = 1, slab_bytes = 64 << 20, staging_buffers = 4, staging_bytes = 64 << 20, direct_io = true))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        devices: Vec<i32>,
        readers: usize,
        copy_streams: usize,
        slab_bytes: u64,
        staging_buffers: usize,
        staging_bytes: u64,
        direct_io: bool,
    ) -> PyResult<Self> {
        let cfg = PlaneConfig { devices, readers, copy_streams, slab_bytes, staging_buffers, staging_bytes, direct_io };
        let plane = py.detach(|| Plane::open(cfg)).or_raise(py)?;
        Ok(PyPlane { plane: Arc::new(plane) })
    }

    #[getter]
    fn devices(&self) -> Vec<(i32, String)> {
        self.plane.devices()
    }

    /// Wrap a TensorFS lease (CONSUMED) as a byte source for a manifest's weight sets. The
    /// lease ends when the last weight set using it closes and the source is dropped.
    fn source(&self, py: Python<'_>, store: PyRef<'_, PyStore>, lease: PyRef<'_, PyLease>) -> PyResult<PySource> {
        let lease = lease.take_for_plane(py)?;
        Ok(PySource { source: self.plane.source(store.store.clone(), store.meta.clone(), lease) })
    }

    #[pyo3(signature = (name, source, plan, regions, host_fd = None))]
    fn register(
        &self,
        py: Python<'_>,
        name: String,
        source: PyRef<'_, PySource>,
        plan: PyRef<'_, PyPlan>,
        regions: Vec<Vec<String>>,
        host_fd: Option<i32>,
    ) -> PyResult<PyWeightSet> {
        let (src, p, plane) = (source.source.clone(), &plan.plan, &self.plane);
        let (id, layout) = py
            .detach(|| {
                let id = plane.register(&name, src, p, &regions, host_fd)?;
                Ok((id, plane.layout(id)?))
            })
            .or_raise(py)?;
        Ok(PyWeightSet { plane: self.plane.clone(), id, name, layout })
    }

    fn set_vram_budget<'p>(&self, py: Python<'p>, device: i32, nbytes: u64) -> PyResult<Bound<'p, PyDict>> {
        let t = py.detach(|| self.plane.set_vram_budget(device, nbytes)).or_raise(py)?;
        trim(py, t)
    }

    fn set_pinned_budget<'p>(&self, py: Python<'p>, nbytes: u64) -> PyResult<Bound<'p, PyDict>> {
        let t = py.detach(|| self.plane.set_pinned_budget(nbytes)).or_raise(py)?;
        trim(py, t)
    }

    #[pyo3(signature = (ws, tier, regions = None, priority = 0, pin = false))]
    fn want(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        tier: TierArg,
        regions: Option<Vec<u32>>,
        priority: i64,
        pin: bool,
    ) -> PyResult<PyTicket> {
        let (id, t) = (self.id(py, &ws)?, self::tier(py, tier)?);
        let plane = &self.plane;
        let ticket = py.detach(|| plane.want(id, t, regions.as_deref(), priority, pin)).or_raise(py)?;
        Ok(PyTicket { ticket })
    }

    /// Bytes freed; refuses on live leases or holds.
    #[pyo3(name = "drop", signature = (ws, tier, regions = None))]
    fn drop_regions(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        tier: TierArg,
        regions: Option<Vec<u32>>,
    ) -> PyResult<u64> {
        let (id, t) = (self.id(py, &ws)?, self::tier(py, tier)?);
        let plane = &self.plane;
        py.detach(|| plane.drop_regions(id, t, regions.as_deref())).or_raise(py)
    }

    /// Set priority (and pin, when given) without moving a byte.
    #[pyo3(signature = (ws, tier, regions = None, priority = 0, pin = None))]
    fn prioritise(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        tier: TierArg,
        regions: Option<Vec<u32>>,
        priority: i64,
        pin: Option<bool>,
    ) -> PyResult<()> {
        let (id, t) = (self.id(py, &ws)?, self::tier(py, tier)?);
        let plane = &self.plane;
        py.detach(|| plane.prioritise(id, t, regions.as_deref(), priority, pin)).or_raise(py)
    }

    #[pyo3(signature = (ws, devices, regions = None, priority = 0, pin = false))]
    fn replicate(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        devices: Vec<i32>,
        regions: Option<Vec<u32>>,
        priority: i64,
        pin: bool,
    ) -> PyResult<Vec<PyTicket>> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let tickets = py
            .detach(|| plane.replicate(id, &devices, regions.as_deref(), priority, pin))
            .or_raise(py)?;
        Ok(tickets.into_iter().map(|ticket| PyTicket { ticket }).collect())
    }

    /// A region not resident is demand-filled at `priority` (when given, it becomes the
    /// region's priority first).
    #[pyo3(signature = (ws, device, region, stream, priority = None))]
    fn acquire(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        device: i32,
        region: u32,
        stream: u64,
        priority: Option<i64>,
    ) -> PyResult<PyPlaneLease> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let lease = py.detach(|| plane.acquire(id, device, region, stream, priority)).or_raise(py)?;
        Ok(PyPlaneLease::new(lease, stream))
    }

    #[pyo3(signature = (ws, device, order, window, repeat = 1, priority = 0, tag = "denoise", ring_bytes = None))]
    #[allow(clippy::too_many_arguments)]
    fn stream(
        &self,
        py: Python<'_>,
        ws: PyRef<'_, PyWeightSet>,
        device: i32,
        order: Vec<u32>,
        window: u32,
        repeat: u32,
        priority: i64,
        tag: &str,
        ring_bytes: Option<u64>,
    ) -> PyResult<PyCursor> {
        let (id, plane) = (self.id(py, &ws)?, &self.plane);
        let cur = py
            .detach(|| plane.stream(id, device, &order, window, repeat, priority, tag, ring_bytes))
            .or_raise(py)?;
        let (home, window, ring_ptr, ring_nbytes) = (cur.home.clone(), cur.window, cur.ring_ptr, cur.ring_nbytes);
        Ok(PyCursor { home, window, ring_ptr, ring_nbytes, cur })
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

#[pyclass(name = "WeightSet", module = "tensorfs.plane", frozen)]
pub struct PyWeightSet {
    plane: Arc<Plane>,
    id: WsId,
    #[pyo3(get)]
    name: String,
    layout: Arc<Layout>,
}

#[pymethods]
impl PyWeightSet {
    /// The layout span (a multiple of 2 MiB).
    #[getter]
    fn nbytes(&self) -> u64 {
        self.layout.nbytes
    }

    #[getter]
    fn regions<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let out = PyList::empty(py);
        for (i, r) in self.layout.regions.iter().enumerate() {
            out.append(dict!(py; "index" => i, "offset" => r.offset, "nbytes" => r.nbytes, "span" => r.span))?;
        }
        Ok(out)
    }

    #[getter]
    fn parts<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let out = PyList::empty(py);
        for p in &self.layout.parts {
            out.append(dict!(py; "what" => &p.what, "region" => p.region, "offset" => p.offset, "nbytes" => p.nbytes))?;
        }
        Ok(out)
    }

    /// The plane-owned memfd; pass it to another process to share the pinned tier.
    #[getter]
    fn host_fd(&self, py: Python<'_>) -> PyResult<i32> {
        py.detach(|| self.plane.host_fd(self.id)).or_raise(py)
    }

    /// The stable device VA as a uint8 DLPack view of `nbytes`.
    fn view(&self, py: Python<'_>, device: i32) -> PyResult<PyDeviceView> {
        let v = py.detach(|| self.plane.view(self.id, device)).or_raise(py)?;
        Ok(PyDeviceView { held: Arc::new(Viewed::Ws(v)) })
    }

    /// Driver truth: every chunk the books call mapped is the driver's mapping there. Returns
    /// chunks checked; a mismatch poisons the plane.
    fn verify(&self, py: Python<'_>, device: i32) -> PyResult<u64> {
        py.detach(|| self.plane.verify(self.id, device)).or_raise(py)
    }

    /// Refuses while leases, holds, cursors or views remain. The source's lease ends when its
    /// last weight set closes and the Source handle is gone.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.plane.close_ws(self.id)).or_raise(py)
    }

    fn __repr__(&self) -> String {
        format!("<tensorfs.plane.WeightSet {:?} {} B in {} regions>", self.name, self.layout.nbytes, self.layout.regions.len())
    }
}

/// A manifest's byte source: one TensorFS lease shared by its weight sets.
#[pyclass(name = "Source", module = "tensorfs.plane", frozen)]
pub struct PySource {
    source: Arc<Source>,
}

#[pyclass(name = "Ticket", module = "tensorfs.plane", frozen)]
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

/// Use of one region's bytes on one stream. Release exactly once; a lease collected
/// unreleased poisons the plane.
#[pyclass(name = "Lease", module = "tensorfs.plane", frozen)]
pub struct PyPlaneLease {
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    region: u32,
    /// Where a streamed lease's bytes sit in the cursor's ring; None for a resident one.
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
        Ok(PyDeviceView { held: Arc::new(Viewed::Lease(v)) })
    }

    /// Diagnostic: parts whose device bytes differ from the source (waits for the copy and
    /// reads the store; debugging runs only).
    fn verify_bytes(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.lease.verify_bytes()).or_raise(py)
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

#[pyclass(name = "Cursor", module = "tensorfs.plane", frozen)]
pub struct PyCursor {
    /// Per order index: served in place (resident at `stream()`).
    #[pyo3(get)]
    home: Vec<bool>,
    #[pyo3(get)]
    window: u32,
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

    /// Completed cycles over `order`.
    #[getter]
    fn passes(&self, py: Python<'_>) -> u64 {
        py.detach(|| self.cur.passes())
    }


    /// Lease `region`: the next scheduled one, a later one (skipping ahead), or a demand fill.
    fn acquire(&self, py: Python<'_>, region: u32, stream: u64) -> PyResult<PyPlaneLease> {
        let lease = py.detach(|| self.cur.acquire(region, stream)).or_raise(py)?;
        Ok(PyPlaneLease::new(lease, stream))
    }

    /// A uint8 DLPack view of the whole ring; it keeps the cursor from closing.
    fn ring_view(&self, py: Python<'_>) -> PyResult<PyDeviceView> {
        let v = py.detach(|| self.cur.ring_view()).or_raise(py)?;
        Ok(PyDeviceView { held: Arc::new(Viewed::Ring(v)) })
    }

    /// Refuses while a lease or ring view is live; waits for copies in flight.
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

enum Viewed {
    Ws(View),
    Ring(RingView),
    Lease(LeaseView),
}

impl Viewed {
    /// (ptr, nbytes, device ordinal)
    fn span(&self) -> (u64, u64, i32) {
        match self {
            Viewed::Ws(v) => (v.ptr, v.nbytes, v.device()),
            Viewed::Ring(s) => (s.ptr, s.nbytes, s.device),
            Viewed::Lease(l) => (l.ptr, l.nbytes, l.device),
        }
    }
}

/// Device bytes as a 1-D uint8 DLPack tensor. Consumers' tensors keep the view alive.
#[pyclass(name = "DeviceView", module = "tensorfs.plane", frozen)]
pub struct PyDeviceView {
    held: Arc<Viewed>,
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
    _view: Arc<Viewed>,
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
    /// True if a region of the weight set was unmapped since the view was made (ring views:
    /// False, the ring stays mapped for the cursor's life).
    #[getter]
    fn stale(&self, py: Python<'_>) -> bool {
        py.detach(|| match &*self.held {
            Viewed::Ws(v) => v.stale(),
            Viewed::Lease(l) => l.stale(),
            Viewed::Ring(_) => false,
        })
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
        let (ptr, nbytes, device) = self.held.span();
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
        (KDL_CUDA, self.held.span().2)
    }
}

// ---------------------------------------------------------------- the module

macro_rules! add_errors {
    ($m:expr; $($e:ident),*) => { $($m.add(stringify!($e), $m.py().get_type::<$e>())?;)* };
}

/// A whole-tier claim on a pinned-tier memfd for a process without a plane (the Worker): a
/// new descriptor the caller owns; closing it ends the claim.
#[pyfunction]
fn hold_tier(py: Python<'_>, fd: i32) -> PyResult<i32> {
    tensorfs_plane::host::hold(fd).map(std::os::fd::IntoRawFd::into_raw_fd).or_raise(py)
}

/// End a `hold_tier` claim: punches every Ready region no plane claims, closes `hold`, and
/// returns the bytes punched.
#[pyfunction]
fn release_tier(py: Python<'_>, hold: i32) -> PyResult<u64> {
    // SAFETY: the caller hands over the descriptor `hold_tier` returned.
    let hold = unsafe { <std::os::fd::OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(hold) };
    tensorfs_plane::host::release_hold(hold).or_raise(py)
}

/// `tensorfs._ext.plane`, importable as such (the facade module `tensorfs.plane` re-exports it).
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "plane")?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PySource>()?;
    m.add_class::<PyWeightSet>()?;
    m.add_class::<PyTicket>()?;
    m.add_class::<PyPlaneLease>()?;
    m.add_class::<PyCursor>()?;
    m.add_class::<PyDeviceView>()?;
    m.add_function(wrap_pyfunction!(hold_tier, &m)?)?;
    m.add_function(wrap_pyfunction!(release_tier, &m)?)?;
    add_errors!(m; PlaneError, BudgetExceeded, Shortfall, BelowFloor, LeaseViolation, Poisoned, CudaError,
        CudaUnavailable, Invalid, IoError, Closed);
    parent.add_submodule(&m)?;
    py.import("sys")?.getattr("modules")?.set_item("tensorfs._ext.plane", &m)
}
