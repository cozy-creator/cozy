//! `tensorfs`: the store, checkpoints, the writer and the error families (design §1), typed in
//! `_ext.pyi`. Marshal only: every call that moves bytes runs detached from the GIL.

use std::ffi::{c_void, CStr};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString, PyTuple};
use pyo3::{create_exception, ffi};
use tensorfs_core::client::{self, ImportOptions, Named, PullFrom};
use tensorfs_core::export;
use tensorfs_core::failure::{Failure, Kind};
use tensorfs_core::format::{Body, Dtype, Encoding, Header, Metadata, Value};
use tensorfs_core::store2::{Checkpoint, Store};
use tensorfs_core::writer::{Data, PartData, Source, Writer};

create_exception!(
    tensorfs,
    Error,
    PyException,
    "A TensorFS error: `.code` names its family, `.detail` its cause."
);
create_exception!(tensorfs, InvalidError, Error);
create_exception!(tensorfs, FormatError, Error);
create_exception!(tensorfs, NotFoundError, Error);
create_exception!(tensorfs, IntegrityError, Error);
create_exception!(tensorfs, UnknownSchemeError, Error);
create_exception!(tensorfs, DecodeError, Error);
create_exception!(tensorfs, StorageError, Error);
create_exception!(tensorfs, DiskFullError, Error);
create_exception!(tensorfs, NetworkError, Error);
create_exception!(tensorfs, DeniedError, Error);
create_exception!(tensorfs, UnsupportedError, Error);

fn family(py: Python<'_>, kind: Kind) -> Bound<'_, pyo3::types::PyType> {
    match kind {
        Kind::Invalid => py.get_type::<InvalidError>(),
        Kind::Format => py.get_type::<FormatError>(),
        Kind::NotFound => py.get_type::<NotFoundError>(),
        Kind::Integrity => py.get_type::<IntegrityError>(),
        Kind::UnknownScheme => py.get_type::<UnknownSchemeError>(),
        Kind::Decode => py.get_type::<DecodeError>(),
        Kind::Io => py.get_type::<StorageError>(),
        Kind::Full => py.get_type::<DiskFullError>(),
        Kind::Network => py.get_type::<NetworkError>(),
        Kind::Denied => py.get_type::<DeniedError>(),
        Kind::Unsupported => py.get_type::<UnsupportedError>(),
    }
}

/// The one conversion: a failure becomes its family's exception, with `.code` and `.detail`.
fn raise(py: Python<'_>, f: Failure) -> PyErr {
    let err = PyErr::from_type(family(py, f.kind), f.to_string());
    let value = err.value(py);
    let _ = value.setattr("code", f.kind.code());
    let _ = value.setattr("detail", f.detail);
    err
}

fn invalid(py: Python<'_>, detail: impl Into<String>) -> PyErr {
    raise(
        py,
        Failure {
            kind: Kind::Invalid,
            detail: detail.into(),
        },
    )
}

trait OrRaise<T> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T>;
}

impl<T, E: Into<Failure>> OrRaise<T> for Result<T, E> {
    fn or_raise(self, py: Python<'_>) -> PyResult<T> {
        self.map_err(|e| raise(py, e.into()))
    }
}

// ---------------------------------------------------------------- values

fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Int(i) => match i64::try_from(*i) {
            Ok(n) => n.into_pyobject(py)?.into_any(),
            Err(_) => (*i as u64).into_pyobject(py)?.into_any(),
        },
        Value::Str(s) => PyString::new(py, s).into_any(),
        Value::List(items) => PyList::new(
            py,
            items
                .iter()
                .map(|v| to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?,
        )?
        .into_any(),
        Value::Map(map) => metadata(py, map)?.into_any(),
    })
}

fn metadata<'py>(py: Python<'py>, map: &Metadata) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, to_py(py, v)?)?;
    }
    Ok(dict)
}

fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if let Ok(n) = obj.extract::<i64>() {
        return Ok(Value::Int(n.into()));
    }
    if let Ok(n) = obj.extract::<u64>() {
        return Ok(Value::Int(n.into()));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::Str(s));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut map = Metadata::new();
        for (k, v) in d.iter() {
            map.insert(k.extract()?, from_py(&v)?);
        }
        return Ok(Value::Map(map));
    }
    if let Ok(items) = obj.extract::<Vec<Bound<'_, PyAny>>>() {
        return Ok(Value::List(
            items.iter().map(from_py).collect::<PyResult<_>>()?,
        ));
    }
    Err(invalid(
        obj.py(),
        format!("{obj}: metadata is None, bool, int, str, list or dict"),
    ))
}

/// Any C-contiguous buffer (bytes, bytearray, memoryview, a numpy array) as bytes.
struct Bytes(PyBuffer<u8>);

impl Bytes {
    fn of(obj: &Bound<'_, PyAny>) -> PyResult<Bytes> {
        let py = obj.py();
        let view = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((obj,))?;
        if !view.getattr("c_contiguous")?.extract::<bool>()? {
            return Err(invalid(py, "the data is not C-contiguous"));
        }
        Ok(Bytes(PyBuffer::get(&view.call_method1("cast", ("B",))?)?))
    }

    fn slice(&self) -> &[u8] {
        // SAFETY: the buffer is held (and its exporter alive) for as long as `self`.
        unsafe { std::slice::from_raw_parts(self.0.buf_ptr() as *const u8, self.0.len_bytes()) }
    }
}

// ---------------------------------------------------------------- references

/// What a ref-taking call was handed: a `Manifest`, or a reference string.
fn named(store: &Store, reference: &Bound<'_, PyAny>, lane: Option<&str>) -> PyResult<Named> {
    let py = reference.py();
    if let Ok(m) = reference.cast::<PyManifest>() {
        return Ok(m.get().named.clone());
    }
    match reference.extract::<String>() {
        Ok(text) => py.detach(|| store.resolve(&text, lane)).or_raise(py),
        Err(_) => Err(invalid(
            py,
            "a ref is a Manifest or a string (org/name[@release], sha256:<hex>)",
        )),
    }
}

/// A checkpoint's manifest, and the ref, release and lane naming it when one does.
#[pyclass(name = "Manifest", module = "tensorfs", frozen)]
pub struct PyManifest {
    named: Named,
}

#[pymethods]
impl PyManifest {
    #[getter]
    fn sha256(&self) -> &str {
        &self.named.manifest.sha256
    }
    #[getter]
    fn length(&self) -> u64 {
        self.named.manifest.length
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.named.name.as_deref()
    }
    #[getter]
    fn release(&self) -> Option<&str> {
        self.named.release.as_deref()
    }
    #[getter]
    fn lane(&self) -> Option<&str> {
        self.named.lane.as_deref()
    }
    fn __str__(&self) -> String {
        format!("sha256:{}", self.named.manifest.sha256)
    }
    fn __repr__(&self) -> String {
        match &self.named.name {
            Some(name) => format!(
                "Manifest({name}@{}, sha256:{})",
                self.named.release.as_deref().unwrap_or(""),
                self.named.manifest.sha256
            ),
            None => format!("Manifest(sha256:{})", self.named.manifest.sha256),
        }
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyManifest>()
            .is_ok_and(|o| o.get().named.manifest == self.named.manifest)
    }
    fn __hash__(&self) -> u64 {
        u64::from_str_radix(&self.named.manifest.sha256[..16], 16).unwrap_or(0)
    }
}

// ---------------------------------------------------------------- the store

/// A store at `root`, created when absent.
#[pyclass(name = "Store", module = "tensorfs", frozen)]
pub struct PyStore {
    store: Store,
}

#[pymethods]
impl PyStore {
    #[new]
    fn new(py: Python<'_>, root: PathBuf) -> PyResult<PyStore> {
        let root = match (root.strip_prefix("~"), std::env::var_os("HOME")) {
            (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
            _ => root,
        };
        let store = py.detach(|| Store::open(root)).or_raise(py)?;
        Ok(PyStore { store })
    }

    #[getter]
    fn root(&self) -> PathBuf {
        self.store.root().to_path_buf()
    }

    /// Imports `src` (a file, directory, `hf://`, `civitai://` or URL; or several) and refs it
    /// as `name` (`local/<its name>` by default).
    #[pyo3(signature = (src, name = None, converter = None, variant = None))]
    fn import_(
        &self,
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        name: Option<String>,
        converter: Option<String>,
        variant: Option<String>,
    ) -> PyResult<PyManifest> {
        let sources: Vec<String> = match src.extract::<String>() {
            Ok(one) => vec![one],
            Err(_) => src.extract()?,
        };
        let opts = ImportOptions {
            name: name.as_deref(),
            converter: converter.as_deref(),
            variant: variant.as_deref(),
            progress: &|_: &str, _: u64, _: u64| {},
        };
        let named = py
            .detach(|| self.store.import_named(&sources, &opts))
            .or_raise(py)?;
        Ok(PyManifest { named })
    }

    /// Pulls `ref` (`org/name[@release]`) from Tensorhub (`hub`, else tensorhub.com; `token`
    /// reads private checkpoints), or from a static `origin`'s URL or directory.
    #[pyo3(signature = (r#ref, lane = None, origin = None, hub = None, token = None))]
    fn pull(
        &self,
        py: Python<'_>,
        r#ref: String,
        lane: Option<String>,
        origin: Option<String>,
        hub: Option<String>,
        token: Option<String>,
    ) -> PyResult<PyManifest> {
        let from = match (&origin, &hub, &token) {
            (Some(origin), None, None) => PullFrom::Origin(origin),
            (None, hub, token) => PullFrom::Hub {
                base: hub.as_deref(),
                token: token.as_deref(),
            },
            _ => return Err(invalid(py, "a static origin takes no hub or token")),
        };
        let pulled = py
            .detach(|| self.store.pull(from, &r#ref, lane.as_deref()))
            .or_raise(py)?;
        Ok(PyManifest {
            named: pulled.named,
        })
    }

    #[pyo3(signature = (r#ref, lane = None))]
    fn checkpoint(
        &self,
        py: Python<'_>,
        r#ref: &Bound<'_, PyAny>,
        lane: Option<String>,
    ) -> PyResult<PyCheckpoint> {
        let named = named(&self.store, r#ref, lane.as_deref())?;
        let ck = py
            .detach(|| self.store.checkpoint(&named.manifest))
            .or_raise(py)?;
        Ok(PyCheckpoint {
            header: ck.header().clone(),
            stored: Some((ck, named)),
            tensors: OnceLock::new(),
        })
    }

    /// A writer of a new checkpoint, carrying `base`'s tensors and files; with `key`, it resumes
    /// what an interrupted writer of that key wrote.
    #[pyo3(signature = (base = None, key = None))]
    fn writer(
        &self,
        py: Python<'_>,
        base: Option<&Bound<'_, PyAny>>,
        key: Option<String>,
    ) -> PyResult<PyWriter> {
        let base = base
            .map(|b| named(&self.store, b, None))
            .transpose()?
            .map(|n| n.manifest);
        let w = py
            .detach(|| self.store.writer(base.as_ref(), key.as_deref()))
            .or_raise(py)?;
        Ok(PyWriter {
            w: Mutex::new(Some(w)),
        })
    }

    /// Writes `ref` into `dir` as a Diffusers/Transformers folder (encoded tensors keep their
    /// parts unless `dequantize` names F32, BF16 or F16), or with `origin=True` as the static
    /// origin layout instead. Returns the paths written.
    #[pyo3(signature = (r#ref, dir, safetensors = true, dequantize = None, *, origin = false, lane = None))]
    #[allow(clippy::too_many_arguments)]
    fn export(
        &self,
        py: Python<'_>,
        r#ref: &Bound<'_, PyAny>,
        dir: PathBuf,
        safetensors: bool,
        dequantize: Option<String>,
        origin: bool,
        lane: Option<String>,
    ) -> PyResult<Vec<String>> {
        let named = named(&self.store, r#ref, lane.as_deref())?;
        if origin && dequantize.is_some() {
            return Err(invalid(
                py,
                "an origin keeps objects as they are: no dequantize",
            ));
        }
        if !origin && !safetensors {
            return Err(invalid(
                py,
                "export writes safetensors, or with origin=True an origin",
            ));
        }
        let dtype = dequantize.map(|d| Dtype::parse(&d.to_ascii_uppercase()));
        py.detach(|| match origin {
            true => client::export_origin(&self.store, &named, &dir),
            false => export::safetensors(
                &self.store.checkpoint(&named.manifest)?,
                &dir,
                dtype.as_ref(),
            ),
        })
        .or_raise(py)
    }

    /// Sweeps unreferenced objects past the grace; with `older_than` (seconds), first drops
    /// every pin unused that long. Returns `{"objects", "bytes", "pins"}`.
    #[pyo3(signature = (older_than = None))]
    fn gc<'py>(&self, py: Python<'py>, older_than: Option<f64>) -> PyResult<Bound<'py, PyDict>> {
        let collected = py
            .detach(|| match older_than {
                None => self.store.gc(),
                Some(s) => self
                    .store
                    .collect(u64::MAX, Duration::from_secs_f64(s.max(0.0))),
            })
            .or_raise(py)?;
        let out = PyDict::new(py);
        out.set_item("objects", collected.objects)?;
        out.set_item("bytes", collected.bytes)?;
        out.set_item("pins", collected.pins)?;
        Ok(out)
    }
}

// ---------------------------------------------------------------- checkpoints

/// A checkpoint: its tensors, files and metadata, read from a store; or only its header's
/// tensors (`Checkpoint.parse`), with no chunks to read.
#[pyclass(name = "Checkpoint", module = "tensorfs", frozen)]
pub struct PyCheckpoint {
    pub(crate) header: Header,
    pub(crate) stored: Option<(Checkpoint, Named)>,
    tensors: OnceLock<Py<PyDict>>,
}

/// One part as stored: a plain tensor's data (role `""`) or an encoded tensor's part.
#[pyclass(name = "Part", module = "tensorfs", frozen, get_all)]
pub struct PyPart {
    dtype: String,
    shape: Py<PyTuple>,
    chunk: u64,
    offset: u64,
    length: u64,
}

/// A scheme and its complete params; `str()` is `scheme {key: value, ...}`.
#[pyclass(name = "Encoding", module = "tensorfs", frozen, get_all)]
pub struct PyEncoding {
    scheme: String,
    params: Py<PyDict>,
}

#[pymethods]
impl PyEncoding {
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let params = match from_py(self.params.bind(py).as_any())? {
            Value::Map(map) => map,
            _ => unreachable!("a dict converts to a map"),
        };
        let e = Encoding { scheme: self.scheme.clone(), params };
        Ok(e.to_string())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Encoding({})", self.__str__(py)?))
    }
}

/// A tensor's logical dtype and shape, its encoding (`None` when plain) and its stored parts.
#[pyclass(name = "Info", module = "tensorfs", frozen, get_all)]
pub struct PyInfo {
    dtype: String,
    shape: Py<PyTuple>,
    encoding: Option<Py<PyEncoding>>,
    parts: Py<PyDict>,
}

#[pymethods]
impl PyInfo {
    fn __repr__(&self, py: Python<'_>) -> String {
        let scheme = self
            .encoding
            .as_ref()
            .map_or(String::new(), |e| format!(", {}", e.get().scheme));
        format!("Info({}, {}{scheme})", self.dtype, self.shape.bind(py))
    }
}

impl PyCheckpoint {
    /// The store's checkpoint behind it; a parsed header has none.
    pub(crate) fn stored(&self, py: Python<'_>) -> PyResult<&Checkpoint> {
        match &self.stored {
            Some((ck, _)) => Ok(ck),
            None => Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "a parsed header has no store: its chunks and files are not here".into(),
                },
            )),
        }
    }

    fn build(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let out = PyDict::new(py);
        // Dicts keep insertion order: components and tensors iterate in header order.
        for c in self.header.components() {
            let tensors = PyDict::new(py);
            for t in &c.tensors {
                let parts = PyDict::new(py);
                let part = |p: PyPart| Py::new(py, p);
                let encoding = match &t.body {
                    Body::Data(at) => {
                        let p = PyPart {
                            dtype: t.dtype.name().into(),
                            shape: PyTuple::new(py, &t.shape)?.unbind(),
                            chunk: at.chunk,
                            offset: at.offset,
                            length: at.length,
                        };
                        parts.set_item("", part(p)?)?;
                        None
                    }
                    Body::Encoded {
                        encoding,
                        parts: stored,
                    } => {
                        for p in stored {
                            let role = &p.role;
                            let p = PyPart {
                                dtype: p.dtype.name().into(),
                                shape: PyTuple::new(py, &p.shape)?.unbind(),
                                chunk: p.chunk,
                                offset: p.offset,
                                length: p.length,
                            };
                            parts.set_item(role, part(p)?)?;
                        }
                        let params = metadata(py, &encoding.params)?.unbind();
                        Some(Py::new(
                            py,
                            PyEncoding {
                                scheme: encoding.scheme.clone(),
                                params,
                            },
                        )?)
                    }
                };
                let info = PyInfo {
                    dtype: t.dtype.name().into(),
                    shape: PyTuple::new(py, &t.shape)?.unbind(),
                    encoding,
                    parts: parts.unbind(),
                };
                tensors.set_item(&t.key, Py::new(py, info)?)?;
            }
            out.set_item(&c.name, tensors)?;
        }
        Ok(out.unbind())
    }
}

#[pymethods]
impl PyCheckpoint {
    /// A `cozytensors/1` header document's tensors and metadata, with no store behind them.
    #[staticmethod]
    fn parse(py: Python<'_>, header: &[u8]) -> PyResult<PyCheckpoint> {
        let header = Header::from_cbor(header).or_raise(py)?;
        Ok(PyCheckpoint {
            header,
            stored: None,
            tensors: OnceLock::new(),
        })
    }

    /// The manifest naming it; None for a parsed header.
    #[getter]
    fn manifest(&self) -> Option<PyManifest> {
        self.stored.as_ref().map(|(_, named)| PyManifest {
            named: named.clone(),
        })
    }

    /// The header's chunks, in order: `(sha256, length)` (`Part.chunk` indexes them).
    #[getter]
    fn chunks(&self) -> Vec<(String, u64)> {
        (self.header.chunks.iter())
            .map(|c| (c.sha256.clone(), c.length))
            .collect()
    }

    /// Each chunk's file in the store, in header order; a parsed header has none.
    #[getter]
    fn chunk_paths(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        let ck = self.stored(py)?;
        Ok((0..ck.header().chunks.len()).map(|i| ck.chunk_path(i).display().to_string()).collect())
    }

    /// `tensors[component][key] -> Info`.
    #[getter]
    fn tensors(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        if let Some(t) = self.tensors.get() {
            return Ok(t.clone_ref(py));
        }
        let built = self.build(py)?;
        Ok(self.tensors.get_or_init(|| built).clone_ref(py))
    }

    /// The manifest's file paths (configs, tokenizers, the header).
    #[getter]
    fn files(&self) -> Vec<String> {
        (self.stored.iter())
            .flat_map(|(ck, _)| ck.manifest().files.keys().cloned())
            .collect()
    }

    /// The header's metadata.
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata(py, &self.header.metadata)
    }

    fn file<'py>(&self, py: Python<'py>, path: String) -> PyResult<Bound<'py, PyBytes>> {
        let ck = self.stored(py)?;
        let bytes = py.detach(|| ck.file(&path)).or_raise(py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// A tensor's logical bytes on the CPU, as a DLPack-able `Array`; an encoded tensor goes
    /// through the reference decode.
    fn read(&self, py: Python<'_>, component: String, key: String) -> PyResult<PyArray> {
        let ck = self.stored(py)?;
        let t = ck.tensor(&component, &key).or_raise(py)?;
        let length = match &t.body {
            Body::Data(at) => at.length,
            Body::Encoded { .. } => t.dtype.byte_len(&t.shape).ok_or_else(|| {
                raise(
                    py,
                    Failure {
                        kind: Kind::Unsupported,
                        detail: format!("{component}/{key}: no size for {} {:?}", t.dtype, t.shape),
                    },
                )
            })?,
        };
        let (dtype, shape) = (t.dtype.clone(), t.shape.clone());
        let mut buf = Aligned::new(length as usize);
        py.detach(|| ck.read_into(&component, &key, buf.bytes_mut()))
            .or_raise(py)?;
        Ok(PyArray {
            buf: Arc::new(buf),
            dtype,
            shape,
        })
    }
}

/// Groups `ck`'s tensors into `<component>/<key>` regions for a weight plane: tensors sharing a
/// chunk stay together, and neighbours merge while their chunks total at most `max_bytes`,
/// never across a `<component>/<key prefix>` boundary (a block) unless a shared chunk does.
/// `only` limits the work and the result to the tensors a load wants.
#[pyfunction]
#[pyo3(signature = (ck, max_bytes = 64 << 20, boundaries = None, only = None))]
fn regions(ck: &PyCheckpoint, max_bytes: u64, boundaries: Option<Vec<String>>, only: Option<Vec<String>>) -> Vec<Vec<String>> {
    client::regions(&ck.header, max_bytes, &boundaries.unwrap_or_default(), &only.unwrap_or_default())
}

// ---------------------------------------------------------------- arrays

/// Bytes aligned for every dtype (8).
struct Aligned {
    words: Vec<u64>,
    len: usize,
}

impl Aligned {
    fn new(len: usize) -> Aligned {
        Aligned {
            words: vec![0; len.div_ceil(8)],
            len,
        }
    }
    fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds at least `len` initialized bytes; u8 has no alignment.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }
    fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `self` is borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.len) }
    }
}

/// A tensor's bytes on the CPU. `torch.from_dlpack(a)` and `numpy.from_dlpack(a)` view them
/// without a copy; an unknown dtype reads (`tobytes`) but has no DLPack type.
#[pyclass(name = "Array", module = "tensorfs", frozen)]
pub struct PyArray {
    buf: Arc<Aligned>,
    dtype: Dtype,
    shape: Vec<u64>,
}

const KDL_CPU: i32 = 1;
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

/// What a consumer's tensor owns: its shape cells and the bytes.
struct Held {
    shape: Vec<i64>,
    _buf: Arc<Aligned>,
}

/// DLPack 1.1's type code and bits for `dtype`.
fn dl_type(dtype: &Dtype) -> Option<(u8, u8)> {
    use Dtype::*;
    Some(match dtype {
        BOOL => (6, 8),
        U8 => (1, 8),
        U16 => (1, 16),
        U32 => (1, 32),
        U64 => (1, 64),
        I8 => (0, 8),
        I16 => (0, 16),
        I32 => (0, 32),
        I64 => (0, 64),
        F16 => (2, 16),
        F32 => (2, 32),
        F64 => (2, 64),
        BF16 => (4, 16),
        C64 => (5, 64),
        F8_E4M3 => (10, 8),
        F8_E4M3FNUZ => (11, 8),
        F8_E5M2 => (12, 8),
        F8_E5M2FNUZ => (13, 8),
        F8_E8M0 => (14, 8),
        F6_E2M3 => (15, 6),
        F6_E3M2 => (16, 6),
        F4 => (17, 4),
        Other(_) => return None,
    })
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
impl PyArray {
    #[getter]
    fn dtype(&self) -> &str {
        self.dtype.name()
    }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.shape)
    }
    #[getter]
    fn nbytes(&self) -> usize {
        self.buf.len
    }
    fn tobytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.buf.bytes())
    }

    #[pyo3(signature = (*, stream = None, max_version = None, dl_device = None, copy = None))]
    fn __dlpack__<'py>(
        &self,
        py: Python<'py>,
        stream: Option<Bound<'py, PyAny>>,
        max_version: Option<Bound<'py, PyAny>>,
        dl_device: Option<(i32, i32)>,
        copy: Option<bool>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let _ = (stream, max_version, copy);
        if dl_device.is_some_and(|d| d != (KDL_CPU, 0)) {
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "an Array lives on the CPU (1, 0)".into(),
                },
            ));
        }
        let Some((code, bits)) = dl_type(&self.dtype) else {
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: format!("DLPack has no type for {}", self.dtype),
                },
            ));
        };
        let held = Box::into_raw(Box::new(Held {
            shape: self.shape.iter().map(|&d| d as i64).collect(),
            _buf: self.buf.clone(),
        }));
        // SAFETY: `held` is live until the deleter runs; the consumer may write its own copy.
        let t = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data: self.buf.words.as_ptr() as *mut c_void,
                device: DLDevice {
                    device_type: KDL_CPU,
                    device_id: 0,
                },
                ndim: self.shape.len() as i32,
                dtype: DLDataType {
                    code,
                    bits,
                    lanes: 1,
                },
                shape: unsafe { (*held).shape.as_mut_ptr() },
                strides: std::ptr::null_mut(),
                byte_offset: 0,
            },
            manager_ctx: held as *mut c_void,
            deleter: Some(deleter),
        }));
        // SAFETY: the capsule takes the leaked tensor; on failure it is freed here.
        unsafe {
            let cap = ffi::PyCapsule_New(
                t as *mut c_void,
                DLTENSOR.as_ptr(),
                Some(capsule_destructor),
            );
            if cap.is_null() {
                deleter(t);
                return Err(PyErr::fetch(py));
            }
            Ok(Bound::from_owned_ptr(py, cap))
        }
    }

    fn __dlpack_device__(&self) -> (i32, i32) {
        (KDL_CPU, 0)
    }

    fn __repr__(&self) -> String {
        format!("Array({}, {:?})", self.dtype, self.shape)
    }
}

// ---------------------------------------------------------------- the writer

/// Tensors and files in, one committed checkpoint out (`commit`).
#[pyclass(name = "Writer", module = "tensorfs", frozen)]
pub struct PyWriter {
    w: Mutex<Option<Writer>>,
}

impl PyWriter {
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut Writer) -> Result<T, Failure> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_mut() {
                Some(w) => f(w),
                None => Err(Failure {
                    kind: Kind::Invalid,
                    detail: "the writer has committed or been abandoned".into(),
                }),
            }
        })
        .or_raise(py)
    }
}

#[pymethods]
impl PyWriter {
    /// A plain tensor: `data` is any C-contiguous buffer of exactly its bytes.
    fn put(
        &self,
        py: Python<'_>,
        component: String,
        key: String,
        dtype: String,
        shape: Vec<u64>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        let dtype = Dtype::parse(&dtype);
        self.with(py, |w| {
            let mut src = bytes.slice();
            Ok(w.put(&component, &key, &dtype, &shape, Data::Plain(Source::Read(&mut src)))?)
        })
    }

    /// An encoded tensor: `parts` maps each role to `(dtype, shape, data)`.
    #[pyo3(signature = (component, key, dtype, shape, scheme, parts, params = None))]
    #[allow(clippy::too_many_arguments)]
    fn put_encoded(
        &self,
        py: Python<'_>,
        component: String,
        key: String,
        dtype: String,
        shape: Vec<u64>,
        scheme: String,
        parts: &Bound<'_, PyDict>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let params = match params {
            Some(p) => match from_py(p.as_any())? {
                Value::Map(map) => map,
                _ => unreachable!("a dict converts to a map"),
            },
            None => Metadata::new(),
        };
        let mut held = Vec::new();
        for (role, part) in parts.iter() {
            let (dtype, shape, data): (String, Vec<u64>, Bound<'_, PyAny>) = part.extract()?;
            held.push((
                role.extract::<String>()?,
                Dtype::parse(&dtype),
                shape,
                Bytes::of(&data)?,
            ));
        }
        let dtype = Dtype::parse(&dtype);
        self.with(py, |w| {
            let mut slices: Vec<&[u8]> = held.iter().map(|h| h.3.slice()).collect();
            let parts = held
                .iter()
                .zip(slices.iter_mut())
                .map(|((role, dtype, shape, _), data)| PartData {
                    role: role.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    data: Source::Read(data),
                })
                .collect();
            let encoding = Encoding { scheme, params };
            Ok(w.put(
                &component,
                &key,
                &dtype,
                &shape,
                Data::Encoded { encoding, parts },
            )?)
        })
    }

    /// Takes `component/key` from `source`, a checkpoint of this store; large parts keep their
    /// objects.
    fn copy(
        &self,
        py: Python<'_>,
        component: String,
        key: String,
        source: &PyCheckpoint,
    ) -> PyResult<()> {
        let from = source.stored(py)?;
        self.with(py, |w| Ok(w.copy(&component, &key, from)?))
    }

    /// A file of the checkpoint's tree: a config, a tokenizer.
    fn file(&self, py: Python<'_>, path: String, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        self.with(py, |w| Ok(w.file(&path, bytes.slice())?))
    }

    /// Written already (or by an interrupted writer of this key).
    fn has(&self, component: &str, key: &str) -> bool {
        let guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
        guard.as_ref().is_some_and(|w| w.has(component, key))
    }

    /// Lays out the chunks and commits; the new checkpoint is pinned under the writer's key, or
    /// its digest.
    fn commit(&self, py: Python<'_>) -> PyResult<PyManifest> {
        let manifest = self.with(py, |w| Ok(w.commit()?))?;
        self.w.lock().unwrap_or_else(|e| e.into_inner()).take();
        Ok(PyManifest {
            named: Named {
                manifest,
                name: None,
                release: None,
                lane: None,
            },
        })
    }

    /// Ends the writer without a checkpoint.
    fn abandon(&self) {
        if let Some(w) = self.w.lock().unwrap_or_else(|e| e.into_inner()).take() {
            w.abandon();
        }
    }
}

// ---------------------------------------------------------------- the module

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<PyStore>()?;
    m.add_class::<PyManifest>()?;
    m.add_class::<PyCheckpoint>()?;
    m.add_class::<PyInfo>()?;
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyWriter>()?;
    m.add_function(wrap_pyfunction!(regions, m)?)?;
    m.add("Error", py.get_type::<Error>())?;
    for kind in Kind::ALL {
        let ty = family(py, kind);
        ty.setattr("code", kind.code())?;
        m.add(ty.name()?, ty)?;
    }
    Ok(())
}
