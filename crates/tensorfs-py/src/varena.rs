//! Python marshaling for the one native placement controller. TensorFS describes and reads
//! bytes; Placement decides residency. Every native wait runs without the GIL.
use std::collections::BTreeMap;
use std::ffi::{c_void, CStr};
use std::os::fd::{FromRawFd, OwnedFd};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

#[cfg(feature = "store")]
use crate::store::PyCheckpoint;
use pyo3::exceptions::{PyBufferError, PyException};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use pyo3::{create_exception, ffi};
use varena::gpu::Pin;
use varena::placement::{Placement, Relief, Stats};
use varena::plan::Tier;
use varena::{Error, Result};

create_exception!(
    tensorfs.varena,
    VarenaError,
    PyException,
    "A placement failure; code and sticky describe recovery."
);
create_exception!(tensorfs.varena, Unsupported, VarenaError);

fn raise(py: Python<'_>, error: Error) -> PyErr {
    let result = if matches!(error, Error::Unsupported(_)) {
        Unsupported::new_err(error.to_string())
    } else {
        VarenaError::new_err(error.to_string())
    };
    let value = result.value(py);
    let _ = value.setattr("code", error.code());
    let _ = value.setattr("sticky", error.is_sticky());
    if let Error::DeviceFull { need, deficit } = error {
        let _ = value.setattr("need", need);
        let _ = value.setattr("deficit", deficit);
    }
    result
}
fn native<T: Send>(py: Python<'_>, call: impl FnOnce() -> Result<T> + Send) -> PyResult<T> {
    py.detach(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(call))
            .unwrap_or(Err(Error::Internal("native placement panicked")))
    })
    .map_err(|error| raise(py, error))
}
fn duplicate(fd: i32) -> Result<OwnedFd> {
    // SAFETY: fcntl validates the integer descriptor; a successful result is newly owned.
    let copy = unsafe { libc_dup(fd) };
    if copy < 0 {
        return Err(Error::Io(format!(
            "duplicate grant descriptor: {}",
            std::io::Error::last_os_error()
        )));
    }
    Ok(unsafe { OwnedFd::from_raw_fd(copy) })
}
unsafe fn libc_dup(fd: i32) -> i32 {
    // platform owns descriptor mechanics; the raw integer may be invalid and must be refused.
    unsafe { libc::fcntl(fd, libc::F_DUPFD_CLOEXEC, 0) }
}
fn relief(value: Relief) -> (&'static str, u64) {
    match value {
        Relief::Complete => ("complete", 0),
        Relief::Cut(n) => ("cut", n),
        Relief::Lower(n) => ("lower", n),
        Relief::Exhausted(n) => ("exhausted", n),
    }
}
fn callback_error(error: PyErr) -> Error {
    Error::Io(format!("Python memory callback: {error}"))
}
fn shrink(callback: Py<PyAny>) -> Arc<varena::placement::Shrink> {
    Arc::new(move |need| {
        Python::attach(|py| {
            callback
                .bind(py)
                .call1((need,))
                .and_then(|value| value.extract())
                .map_err(callback_error)
        })
    })
}

#[derive(Clone)]
struct Model {
    id: u64,
    name: String,
    nbytes: u64,
    spans: Vec<u64>,
    regions: Vec<Vec<String>>,
    pages: Vec<u64>,
    parts: Vec<Part>,
}
#[derive(Clone)]
struct Part {
    component: String,
    key: String,
    role: String,
    dtype: String,
    shape: Vec<u64>,
    region: u32,
    offset: u64,
    nbytes: u64,
}

#[pyclass(name = "Part", module = "tensorfs.varena", frozen, get_all)]
struct PyPart {
    component: String,
    key: String,
    role: String,
    dtype: String,
    shape: Vec<u64>,
    region: u32,
    offset: u64,
    nbytes: u64,
}
impl From<Part> for PyPart {
    fn from(p: Part) -> Self {
        Self {
            component: p.component,
            key: p.key,
            role: p.role,
            dtype: p.dtype,
            shape: p.shape,
            region: p.region,
            offset: p.offset,
            nbytes: p.nbytes,
        }
    }
}
#[pyclass(name = "WeightSet", module = "tensorfs.varena", frozen)]
struct PyWeightSet(Arc<Model>);
#[pymethods]
impl PyWeightSet {
    #[getter]
    fn id(&self) -> u64 {
        self.0.id
    }
    #[getter]
    fn name(&self) -> &str {
        &self.0.name
    }
    #[getter]
    fn nbytes(&self) -> u64 {
        self.0.nbytes
    }
    #[getter]
    fn spans(&self) -> Vec<u64> {
        self.0.spans.clone()
    }
    #[getter]
    fn regions(&self) -> Vec<Vec<String>> {
        self.0.regions.clone()
    }
    #[getter]
    fn parts(&self) -> Vec<PyPart> {
        self.0.parts.clone().into_iter().map(Into::into).collect()
    }
}

#[pyclass(name = "Plane", module = "tensorfs.varena", frozen)]
struct PyPlane {
    core: Mutex<Option<Arc<Placement>>>,
    models: Mutex<BTreeMap<u64, Arc<Model>>>,
    next: AtomicU64,
}
impl PyPlane {
    fn core(&self, py: Python<'_>) -> PyResult<Arc<Placement>> {
        self.core
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .ok_or_else(|| raise(py, Error::Closed("placement".into())))
    }
    fn model(&self, py: Python<'_>, id: u64) -> PyResult<Arc<Model>> {
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&id)
            .cloned()
            .ok_or_else(|| raise(py, Error::Closed("weight set".into())))
    }
}
impl Drop for PyPlane {
    fn drop(&mut self) {
        if let Some(core) = self
            .core
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            crate::cleanup::defer(core);
        }
    }
}
#[pymethods]
impl PyPlane {
    #[new]
    #[pyo3(signature=(device,grant=None))]
    fn new(py: Python<'_>, device: i32, grant: Option<(i32, i32, i32)>) -> PyResult<Self> {
        let core = native(py, || {
            let grant = grant
                .map(|(table, event, channel)| {
                    Ok((duplicate(table)?, duplicate(event)?, duplicate(channel)?))
                })
                .transpose()?;
            Placement::open(device, grant)
        })?;
        Ok(Self {
            core: Mutex::new(Some(core)),
            models: Mutex::default(),
            next: AtomicU64::new(1),
        })
    }
    #[getter]
    fn device(&self, py: Python<'_>) -> PyResult<i32> {
        Ok(self.core(py)?.device())
    }
    #[cfg(feature = "store")]
    #[pyo3(signature=(name,ck,regions,fingerprint,fill=None))]
    fn register(
        &self,
        py: Python<'_>,
        name: String,
        ck: PyRef<'_, PyCheckpoint>,
        regions: Vec<Vec<String>>,
        fingerprint: String,
        fill: Option<Py<PyAny>>,
    ) -> PyResult<PyWeightSet> {
        let ck = ck.core(py)?;
        let core = self.core(py)?;
        let id = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .map_err(|_| raise(py, Error::Invalid("model ids exhausted".into())))?;
        // The GPU-owned writable callback guard is wired with Placement's fill seam.
        if fill.is_some() {
            return Err(raise(
                py,
                Error::Unsupported("writable fill guard is being integrated".into()),
            ));
        }
        let model = native(py, || {
            let layout = varena::layout::Layout::build(&ck.spans(&regions)?, &regions)?;
            let identity = serde_json::to_vec(&(&layout.digest, &fingerprint, core.gpu_uuid()?))
                .map_err(|e| Error::Io(e.to_string()))?;
            let fingerprint = tensorfs::sha256::digest(&identity);
            let mut parts = Vec::new();
            for part in &layout.parts {
                let info = ck
                    .stored_part(&part.what)
                    .ok_or_else(|| Error::Invalid(format!("missing part {}", part.what)))?;
                let region = layout.region(part.region)?;
                parts.push(Part {
                    component: info.component,
                    key: info.key,
                    role: info.role,
                    dtype: info.dtype,
                    shape: info.shape,
                    region: part.region,
                    offset: part.offset - region.offset,
                    nbytes: part.nbytes,
                });
            }
            let pages = core.register(
                id,
                name.clone(),
                fingerprint,
                &layout,
                Arc::new(ck.source(true)),
                None,
            )?;
            Ok(Arc::new(Model {
                id,
                name,
                nbytes: layout.nbytes,
                spans: layout.regions.iter().map(|r| r.span).collect(),
                regions,
                pages,
                parts,
            }))
        })?;
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(id, model.clone());
        Ok(PyWeightSet(model))
    }
    fn begin(
        &self,
        py: Python<'_>,
        id: u64,
        plan: Vec<Vec<u64>>,
        package: String,
        shape: String,
    ) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.begin(id, plan, package, shape))
    }
    fn declare(&self, py: Python<'_>, id: u64, plan: Vec<Vec<u64>>) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.declare(id, plan))
    }
    fn enter(&self, py: Python<'_>, id: u64, model: u64) -> PyResult<(&'static str, u64)> {
        let c = self.core(py)?;
        native(py, || c.enter(id, model)).map(relief)
    }
    fn bind(
        &self,
        py: Python<'_>,
        id: u64,
        model: u64,
        region: u32,
        stream: usize,
    ) -> PyResult<PyPinned> {
        let c = self.core(py)?;
        let m = self.model(py, model)?;
        let page = m
            .pages
            .get(region as usize)
            .copied()
            .ok_or_else(|| raise(py, Error::Invalid("region outside weight set".into())))?;
        let device = c.device();
        let pin = native(py, || c.bind(id, page, stream))?;
        Ok(PyPinned::new(pin, region, device))
    }
    fn leave(&self, py: Python<'_>, id: u64, peak: u64) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.leave(id, peak))
    }
    fn end(&self, py: Python<'_>, id: u64) -> PyResult<Py<PyDict>> {
        let c = self.core(py)?;
        stats(py, native(py, || c.end(id))?)
    }
    fn relief(
        &self,
        py: Python<'_>,
        id: u64,
        need: u64,
        lower: bool,
    ) -> PyResult<(&'static str, u64)> {
        let c = self.core(py)?;
        native(py, || c.relief(id, need, lower)).map(relief)
    }
    fn load(&self, py: Python<'_>, model: u64, level: &str) -> PyResult<(&'static str, u64)> {
        let level = match level {
            "ram" => Tier::Ram,
            "gpu" => Tier::Gpu,
            _ => {
                return Err(raise(
                    py,
                    Error::Invalid("load level must be ram or gpu".into()),
                ))
            }
        };
        let c = self.core(py)?;
        native(py, || c.load(model, level)).map(relief)
    }
    fn resident(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let c = self.core(py)?;
        let values = native(py, || c.resident())?;
        let out = PyDict::new(py);
        for (k, v) in values {
            out.set_item(k, v)?;
        }
        Ok(out.unbind())
    }
    fn allocator(
        &self,
        py: Python<'_>,
        context: u64,
        allocated: u64,
        reserved: u64,
    ) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.allocator(context, allocated, reserved))
    }
    fn allocator_budget(&self, py: Python<'_>) -> PyResult<u64> {
        let c = self.core(py)?;
        native(py, || c.allocator_budget())
    }
    fn available(&self, py: Python<'_>) -> PyResult<u64> {
        let c = self.core(py)?;
        native(py, || c.available())
    }
    fn trim_with(&self, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || {
            c.trim_with(shrink(callback));
            Ok(())
        })
    }
    fn cache(&self, py: Python<'_>, bytes: u64, callback: Py<PyAny>) -> PyResult<u64> {
        let c = self.core(py)?;
        native(py, || c.cache(bytes, shrink(callback)))
    }
    fn cache_bytes(&self, py: Python<'_>, id: u64, bytes: u64) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.cache_bytes(id, bytes))
    }
    fn close_cache(&self, py: Python<'_>, id: u64) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || {
            c.close_cache(id);
            Ok(())
        })
    }
    fn close_model(&self, py: Python<'_>, id: u64) -> PyResult<()> {
        let c = self.core(py)?;
        native(py, || c.close_model(id))?;
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&id);
        Ok(())
    }
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        let core = self
            .core
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take();
        self.models
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        if let Some(c) = core {
            native(py, || c.close())?;
            crate::cleanup::defer(c);
        }
        Ok(())
    }
}

fn stats(py: Python<'_>, value: Stats) -> PyResult<Py<PyDict>> {
    let out = PyDict::new(py);
    let mut events = Vec::new();
    let mut copies = Vec::new();
    for e in value.events {
        let item = PyDict::new(py);
        item.set_item("sequence", e.sequence)?;
        item.set_item("request", e.request)?;
        item.set_item("step", e.step)?;
        item.set_item("model", e.model)?;
        item.set_item("page", e.page)?;
        item.set_item("class", format!("{:?}", e.class).to_ascii_lowercase())?;
        item.set_item("kind", e.kind)?;
        item.set_item("reason", e.reason)?;
        item.set_item("bytes", e.bytes)?;
        item.set_item("capacity", e.capacity)?;
        item.set_item("working_set", e.working_set)?;
        item.set_item("resident", e.resident)?;
        item.set_item("orphan_bytes", e.orphan_bytes)?;
        item.set_item("speculative_budget", e.speculative_budget)?;
        events.push(item.unbind());
    }
    for c in value.copies {
        let item = PyDict::new(py);
        item.set_item("request", c.request)?;
        item.set_item("page", c.page)?;
        item.set_item("tier", format!("{:?}", c.tier).to_ascii_lowercase())?;
        item.set_item("speculative", c.speculative)?;
        item.set_item("bytes", c.stats.bytes)?;
        item.set_item("ns", c.stats.ns)?;
        item.set_item("direct", c.stats.direct)?;
        item.set_item("cached", c.stats.cached)?;
        item.set_item("bounced", c.stats.bounced)?;
        item.set_item("pin_ns", c.stats.pin_ns)?;
        item.set_item("staged", c.stats.staged)?;
        item.set_item("busy_ns", c.stats.busy_ns)?;
        copies.push(item.unbind());
    }
    out.set_item("trace", events)?;
    out.set_item("copies", copies)?;
    out.set_item("peak", value.peak)?;
    Ok(out.unbind())
}

struct HeldPin(Option<Pin>);
impl Drop for HeldPin {
    fn drop(&mut self) {
        if let Some(pin) = self.0.take() {
            crate::cleanup::defer(pin);
        }
    }
}
#[pyclass(name = "Pinned", module = "tensorfs.varena", frozen)]
struct PyPinned {
    #[pyo3(get)]
    region: u32,
    #[pyo3(get)]
    ptr: u64,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    writable: bool,
    device: i32,
    pin: Arc<HeldPin>,
}
impl PyPinned {
    fn new(pin: Pin, region: u32, device: i32) -> Self {
        Self {
            region,
            ptr: pin.ptr,
            nbytes: pin.bytes,
            writable: false,
            device,
            pin: Arc::new(HeldPin(Some(pin))),
        }
    }
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
struct Held {
    dims: [i64; 2],
    _pin: Arc<HeldPin>,
}
unsafe extern "C" fn deleter(t: *mut DLManagedTensor) {
    if !t.is_null() {
        let m = unsafe { Box::from_raw(t) };
        drop(unsafe { Box::from_raw(m.manager_ctx as *mut Held) });
    }
}
unsafe extern "C" fn capsule_destructor(capsule: *mut ffi::PyObject) {
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, DLTENSOR.as_ptr()) == 1 {
            deleter(ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr()) as *mut DLManagedTensor);
        }
    }
}
fn stream(stream: Option<i64>) -> PyResult<usize> {
    match stream {
        None => Ok(1),
        Some(n) if n > 0 => {
            usize::try_from(n).map_err(|_| PyBufferError::new_err("stream overflows"))
        }
        _ => Err(PyBufferError::new_err(
            "CUDA stream must be positive or None",
        )),
    }
}
#[pymethods]
impl PyPinned {
    #[pyo3(signature=(*,stream=None,max_version=None,dl_device=None,copy=None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<i64>,
        max_version: Option<Bound<'py, PyAny>>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = max_version;
        if copy == Some(true) {
            return Err(PyBufferError::new_err("a pinned region is never copied"));
        }
        if dl_device.is_some_and(|d| d != (KDL_CUDA, self.device)) {
            return Err(PyBufferError::new_err("pinned region is on another device"));
        }
        let stream = crate::varena::stream(stream)?;
        if let Some(pin) = &self.pin.0 {
            native(py, || pin.wait_on(stream))?;
        }
        let len = i64::try_from(self.nbytes)
            .map_err(|_| PyBufferError::new_err("region is too large for DLPack"))?;
        let held = Box::into_raw(Box::new(Held {
            dims: [len, 1],
            _pin: self.pin.clone(),
        }));
        let dims = unsafe { (*held).dims.as_mut_ptr() };
        let managed = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: self.ptr as *mut c_void,
                device: DLDevice {
                    device_type: KDL_CUDA,
                    device_id: self.device,
                },
                ndim: 1,
                dtype: DLDataType {
                    code: KDL_UINT,
                    bits: 8,
                    lanes: 1,
                },
                shape: dims,
                strides: unsafe { dims.add(1) },
                byte_offset: 0,
            },
            manager_ctx: held.cast(),
            deleter: Some(deleter),
        }));
        unsafe {
            let capsule =
                ffi::PyCapsule_New(managed.cast(), DLTENSOR.as_ptr(), Some(capsule_destructor));
            if capsule.is_null() {
                deleter(managed);
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, capsule))
        }
    }
    fn __dlpack_device__(&self) -> (i32, i32) {
        (KDL_CUDA, self.device)
    }
}

#[pyclass(name = "Copy", module = "tensorfs.varena", frozen, get_all)]
struct PyCopy {
    kind: String,
    gpu: i32,
    bytes: u64,
    ns: u64,
    ceiling: f64,
    link: String,
    busy_ns: u64,
    pin_ns: u64,
    staged: u64,
    cached: u64,
}
#[pyfunction]
fn copies(py: Python<'_>) -> PyResult<Vec<PyCopy>> {
    let values = native(py, || Ok(tensorfs::io::metrics::take()))?;
    Ok(values
        .into_iter()
        .map(|c| PyCopy {
            kind: c.kind.into(),
            gpu: c.gpu,
            bytes: c.bytes,
            ns: c.ns,
            ceiling: c.ceiling,
            link: c.link,
            busy_ns: c.busy_ns,
            pin_ns: c.pin_ns,
            staged: c.staged,
            cached: c.cached,
        })
        .collect())
}
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let m = PyModule::new(py, "varena")?;
    m.add_class::<PyPlane>()?;
    m.add_class::<PyWeightSet>()?;
    m.add_class::<PyPinned>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PyCopy>()?;
    m.add_function(wrap_pyfunction!(copies, &m)?)?;
    m.add("VarenaError", py.get_type::<VarenaError>())?;
    m.add("Unsupported", py.get_type::<Unsupported>())?;
    parent.add_submodule(&m)?;
    py.import("sys")?
        .getattr("modules")?
        .set_item("tensorfs._ext.varena", &m)
}
