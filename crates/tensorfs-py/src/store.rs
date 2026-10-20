//! `tensorfs`: the store, checkpoints, the component writer and the error families, typed in
//! `_ext.pyi`. Marshal only: every call that moves bytes runs detached from the GIL.

use std::ffi::{c_void, CStr};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::buffer::PyBuffer;
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyList, PyString, PyTuple};
use tensorfs::format::{
    Body, Config, ConfigValue, Dtype, Encoding, Header, Metadata, ObjectRef, Tensor, Value,
};
use tensorfs::raw::block_on;
use tensorfs::{
    Checkpoint, Client, Convert, Failure, Download, Format, Kind, Open, Ref, Selection, Writer,
    WriterOptions,
};

pub(crate) use crate::errors::raise;
use crate::tensors::{PyEncoded, PyTensors};

pub(crate) fn invalid(py: Python<'_>, detail: impl Into<String>) -> PyErr {
    raise(
        py,
        Failure {
            kind: Kind::Invalid,
            detail: detail.into(),
        },
    )
}

pub(crate) trait OrRaise<T> {
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

pub(crate) fn metadata<'py>(py: Python<'py>, map: &Metadata) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, to_py(py, v)?)?;
    }
    Ok(dict)
}

pub(crate) fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
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

// ---------------------------------------------------------------- configs

pub(crate) fn config_to_py<'py>(py: Python<'py>, v: &ConfigValue) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        ConfigValue::Null => py.None().into_bound(py),
        ConfigValue::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        ConfigValue::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => i.into_pyobject(py)?.into_any(),
            (_, Some(u), _) => u.into_pyobject(py)?.into_any(),
            (_, _, f) => PyFloat::new(py, f.unwrap_or(f64::NAN)).into_any(),
        },
        ConfigValue::String(s) => PyString::new(py, s).into_any(),
        ConfigValue::Array(items) => {
            let items = items
                .iter()
                .map(|v| config_to_py(py, v))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        ConfigValue::Object(map) => config_dict(py, map)?.into_any(),
    })
}

fn config_dict<'py>(py: Python<'py>, map: &Config) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (k, v) in map {
        dict.set_item(k, config_to_py(py, v)?)?;
    }
    Ok(dict)
}

pub(crate) fn config_from_py(obj: &Bound<'_, PyAny>) -> PyResult<ConfigValue> {
    let py = obj.py();
    if obj.is_none() {
        return Ok(ConfigValue::Null);
    }
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(ConfigValue::Bool(b.is_true()));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return match f.value() {
            v if v.is_finite() => Ok(ConfigValue::from(v)),
            _ => Err(invalid(py, format!("{f}: configs hold finite floats"))),
        };
    }
    if let Ok(n) = obj.extract::<i64>() {
        return Ok(ConfigValue::from(n));
    }
    if let Ok(n) = obj.extract::<u64>() {
        return Ok(ConfigValue::from(n));
    }
    if let Ok(s) = obj.extract::<String>() {
        return Ok(ConfigValue::String(s));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        return Ok(ConfigValue::Object(config_map(d)?));
    }
    match obj.extract::<Vec<Bound<'_, PyAny>>>() {
        Ok(items) => Ok(ConfigValue::Array(
            items.iter().map(config_from_py).collect::<PyResult<_>>()?,
        )),
        Err(_) => Err(invalid(
            py,
            format!("{obj}: a config holds None, bool, int, float, str, list or dict"),
        )),
    }
}

fn config_map(d: &Bound<'_, PyDict>) -> PyResult<Config> {
    let mut map = Config::new();
    for (k, v) in d.iter() {
        map.insert(k.extract()?, config_from_py(&v)?);
    }
    Ok(map)
}

// ---------------------------------------------------------------- references

/// What a ref-taking call was handed: a stored `Checkpoint`, or a reference string.
/// An internal language adapter; cache storage is owned by the public Rust Client.
#[pyclass(name = "_Client", module = "tensorfs._ext", frozen)]
pub struct PyClient {
    client: Client,
}

fn reference(
    repo_id: &str,
    revision: Option<String>,
    variant: Option<String>,
) -> tensorfs::Result<Ref> {
    let mut reference: Ref = repo_id.parse()?;
    if revision.is_some() {
        reference.revision = revision;
    }
    if variant.is_some() {
        reference.variant = variant;
    }
    Ok(reference)
}

#[pymethods]
impl PyClient {
    #[new]
    #[pyo3(signature = (root, *, endpoint = None, token = None, ca_der = None))]
    fn new(
        py: Python<'_>,
        root: PathBuf,
        endpoint: Option<String>,
        token: Option<&Bound<'_, PyAny>>,
        ca_der: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let config = crate::credentials::config(endpoint, Some(root), token, ca_der)?;
        let client = py.detach(|| Client::new(config)).or_raise(py)?;
        Ok(Self { client })
    }

    #[pyo3(signature = (name_or_path, *, revision = None, variant = None, local_files_only = false, lazy = true))]
    fn open(
        &self,
        py: Python<'_>,
        name_or_path: &str,
        revision: Option<String>,
        variant: Option<String>,
        local_files_only: bool,
        lazy: bool,
    ) -> PyResult<PyCheckpoint> {
        let options = Open {
            revision,
            variant,
            local_files_only,
            lazy,
            ..Open::default()
        };
        let checkpoint = py
            .detach(|| block_on(self.client.open(name_or_path, &options)))
            .or_raise(py)?;
        Ok(PyCheckpoint::from_core(checkpoint))
    }

    #[pyo3(signature = (identity, *, components = None, local_dir = None, local_files_only = false))]
    fn materialize(
        &self,
        py: Python<'_>,
        identity: (String, Option<String>, Option<String>),
        components: Option<Vec<String>>,
        local_dir: Option<PathBuf>,
        local_files_only: bool,
    ) -> PyResult<PathBuf> {
        let reference = reference(&identity.0, identity.1, identity.2).or_raise(py)?;
        let selection = components.map(Selection::Components).unwrap_or_default();
        let want = Download {
            selection,
            ..Download::default()
        };
        py.detach(|| {
            block_on(self.client.materialize(
                &reference,
                &want,
                local_dir.as_deref(),
                local_files_only,
            ))
        })
        .or_raise(py)
    }

    #[pyo3(signature = (repo_id, filename, *, revision = None, local_dir = None, local_files_only = false))]
    fn download_file(
        &self,
        py: Python<'_>,
        repo_id: &str,
        filename: &str,
        revision: Option<String>,
        local_dir: Option<PathBuf>,
        local_files_only: bool,
    ) -> PyResult<PathBuf> {
        let reference = reference(repo_id, revision, None).or_raise(py)?;
        py.detach(|| {
            block_on(self.client.download_file(
                &reference,
                filename,
                local_dir.as_deref(),
                local_files_only,
            ))
        })
        .or_raise(py)
    }

    #[pyo3(signature = (folder_path, repo_id, revision = None))]
    fn upload_folder(
        &self,
        py: Python<'_>,
        folder_path: PathBuf,
        repo_id: &str,
        revision: Option<String>,
    ) -> PyResult<String> {
        let reference = reference(repo_id, revision, None).or_raise(py)?;
        py.detach(|| block_on(self.client.upload_folder(&reference, &folder_path, None)))
            .map(|tree| tree.id())
            .or_raise(py)
    }

    #[pyo3(signature = (source, *, converter = None, variant = None, progress = None))]
    fn convert_to_cozytensors(
        &self,
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
        converter: Option<String>,
        variant: Option<String>,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<PyCheckpoint> {
        let sources: Vec<String> = source
            .extract::<String>()
            .map(|s| vec![s])
            .or_else(|_| source.extract())?;
        let callback_error = Mutex::new(None);
        let notify = |stage: &str, done, total| -> tensorfs::Result<()> {
            if let Some(callback) = &progress {
                if let Err(error) = Python::attach(|py| callback.call1(py, (stage, done, total))) {
                    *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    return Err(Failure {
                        kind: Kind::Invalid,
                        detail: "the conversion progress callback failed".into(),
                    });
                }
            }
            Ok(())
        };
        let options = Convert {
            converter: converter.as_deref(),
            variant: variant.as_deref(),
            progress: progress
                .as_ref()
                .map(|_| &notify as &tensorfs::ConvertProgress<'_>),
        };
        let result = py.detach(|| block_on(self.client.convert_to_cozytensors(&sources, &options)));
        if let Some(error) = callback_error
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            return Err(error);
        }
        Ok(PyCheckpoint::from_core(result.or_raise(py)?))
    }

}

#[pyfunction]
#[pyo3(signature = (checkpoint, directory, *, format = "cozytensors", dequantize = None))]
fn _materialize(
    py: Python<'_>,
    checkpoint: &PyCheckpoint,
    directory: PathBuf,
    format: &str,
    dequantize: Option<String>,
) -> PyResult<PathBuf> {
    let checkpoint = checkpoint.core(py)?;
    let format = match format {
        "safetensors" => Format::Safetensors,
        "cozytensors" => Format::Cozytensors,
        _ => return Err(invalid(py, "format must be safetensors or cozytensors")),
    };
    let dtype = dequantize.map(|d| Dtype::parse(&d.to_ascii_uppercase()));
    py.detach(move || block_on(checkpoint.materialize(&directory, format, dtype)))
        .or_raise(py)
}

// ---------------------------------------------------------------- checkpoints

/// One cozytensors: its pipeline, components and metadata, read from a store.
#[pyclass(name = "Checkpoint", module = "tensorfs", frozen)]
pub struct PyCheckpoint {
    header: Header,
    id: ObjectRef,
    tree: Option<ObjectRef>,
    named: Option<Ref>,
    variant: Option<String>,
    state: Mutex<CheckpointState>,
}

struct CheckpointState {
    access: Option<Arc<CheckpointAccess>>,
    components: Option<Py<PyDict>>,
    closed: bool,
}

pub(crate) struct CheckpointAccess(Option<Checkpoint>);

impl std::ops::Deref for CheckpointAccess {
    type Target = Checkpoint;
    fn deref(&self) -> &Checkpoint {
        self.0.as_ref().expect("live checkpoint source")
    }
}

impl Drop for CheckpointAccess {
    fn drop(&mut self) {
        crate::cleanup::defer(self.0.take());
    }
}

/// What `model_index.json` held: `cls` (`diffusers.<Class>`) and `config`.
#[pyclass(name = "Pipeline", module = "tensorfs", frozen, get_all)]
pub struct PyPipeline {
    cls: String,
    config: Py<PyDict>,
}

/// One component: its class, config, tensors and assets.
#[pyclass(name = "Component", module = "tensorfs", frozen)]
pub struct PyComponent {
    #[pyo3(get)]
    name: String,
    /// `<library>.<Class>`, `""` when unknown.
    #[pyo3(get)]
    cls: String,
    #[pyo3(get)]
    config: Py<PyDict>,
    /// The file name `config` exports as: `config.json` unless the writer named another.
    #[pyo3(get)]
    config_file: String,
    /// `key -> TensorMetadata`, in construction order; reads go through the engine.
    #[pyo3(get)]
    tensors: Py<PyTensors>,
    /// `name -> (sha256, length)`: run data in the component's folder.
    #[pyo3(get)]
    assets: Py<PyDict>,
    ck: Option<Arc<CheckpointAccess>>,
}

#[pymethods]
impl PyComponent {
    /// Export config and supporting files to `directory`, or by default to a folder the cache
    /// manages and evicts like its other entries.
    #[pyo3(signature = (directory = None))]
    fn export_files(&self, py: Python<'_>, directory: Option<PathBuf>) -> PyResult<PathBuf> {
        let checkpoint = self
            .ck
            .as_ref()
            .ok_or_else(|| invalid(py, "a parsed header has no support files here"))?;
        checkpoint.fetch_assets(py, &self.name, None)?;
        py.detach(|| {
            checkpoint
                .component(&self.name)?
                .export_files(directory.as_deref())
        })
        .or_raise(py)
    }
    /// Asset `name`'s bytes; a parsed header has none.
    fn asset<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let Some(ck) = &self.ck else {
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "a parsed header has no assets here".into(),
                },
            ));
        };
        let checkpoint = ck;
        checkpoint.fetch_assets(py, &self.name, Some(&name))?;
        let bytes = py
            .detach(|| checkpoint.component(&self.name)?.asset(&name))
            .or_raise(py)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn __repr__(&self) -> String {
        format!("Component({}, {})", self.name, self.cls)
    }
}

/// One part as stored: a plain tensor's data (role `""`) or an encoded tensor's part.
#[pyclass(name = "Part", module = "tensorfs.raw", frozen, get_all)]
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
        let e = Encoding {
            scheme: self.scheme.clone(),
            params,
        };
        Ok(e.to_string())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("Encoding({})", self.__str__(py)?))
    }
}

/// A tensor's logical dtype, shape and encoding (`None` when plain).
#[pyclass(name = "TensorMetadata", module = "tensorfs", frozen, get_all)]
pub struct PyTensorMetadata {
    dtype: String,
    shape: Py<PyTuple>,
    encoding: Option<Py<PyEncoding>>,
}

#[pymethods]
impl PyTensorMetadata {
    fn __repr__(&self, py: Python<'_>) -> String {
        let scheme = self
            .encoding
            .as_ref()
            .map_or(String::new(), |e| format!(", {}", e.get().scheme));
        format!(
            "TensorMetadata({}, {}{scheme})",
            self.dtype,
            self.shape.bind(py)
        )
    }
}

fn info(py: Python<'_>, tensor: &Tensor) -> PyResult<PyTensorMetadata> {
    let encoding = match &tensor.body {
        Body::Data(_) => None,
        Body::Encoded { encoding, .. } => Some(Py::new(
            py,
            PyEncoding {
                scheme: encoding.scheme.clone(),
                params: metadata(py, &encoding.params)?.unbind(),
            },
        )?),
    };
    Ok(PyTensorMetadata {
        dtype: tensor.dtype.name().into(),
        shape: PyTuple::new(py, &tensor.shape)?.unbind(),
        encoding,
    })
}

/// Inspect physical stored ranges without fetching or loading array bytes.
#[pyfunction(name = "parts")]
fn raw_parts<'py>(
    py: Python<'py>,
    checkpoint: &PyCheckpoint,
    component: &str,
    key: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let component = checkpoint
        .header
        .components()
        .iter()
        .find(|c| c.name == component)
        .ok_or_else(|| {
            raise(
                py,
                Failure {
                    kind: Kind::NotFound,
                    detail: format!("no component {component:?}"),
                },
            )
        })?;
    let tensor = component
        .tensors
        .iter()
        .find(|t| t.key == key)
        .ok_or_else(|| {
            raise(
                py,
                Failure {
                    kind: Kind::NotFound,
                    detail: format!("no tensor {key:?}"),
                },
            )
        })?;
    let parts = PyDict::new(py);
    let part = |dtype: &Dtype, shape: &[u64], chunk: u64, offset: u64, length: u64| {
        Py::new(
            py,
            PyPart {
                dtype: dtype.name().into(),
                shape: PyTuple::new(py, shape)?.unbind(),
                chunk,
                offset,
                length,
            },
        )
    };
    match &tensor.body {
        Body::Data(at) => parts.set_item(
            "",
            part(&tensor.dtype, &tensor.shape, at.chunk, at.offset, at.length)?,
        )?,
        Body::Encoded { parts: stored, .. } => {
            for p in stored {
                parts.set_item(
                    &p.role,
                    part(&p.dtype, &p.shape, p.chunk, p.offset, p.length)?,
                )?;
            }
        }
    }
    Ok(parts)
}

pub(crate) struct FetchTensor<'a> {
    pub key: &'a str,
    pub role: Option<&'a str>,
    pub range: Option<Range<u64>>,
}

impl PyCheckpoint {
    fn from_core(checkpoint: Checkpoint) -> Self {
        let header = checkpoint.header().clone();
        let id = checkpoint.id().clone();
        let tree = checkpoint.tree().cloned();
        let named = checkpoint.named().cloned();
        let variant = checkpoint.variant().map(str::to_owned);
        Self {
            header,
            id,
            tree,
            named,
            variant,
            state: Mutex::new(CheckpointState {
                access: Some(Arc::new(CheckpointAccess(Some(checkpoint)))),
                components: None,
                closed: false,
            }),
        }
    }

    pub(crate) fn access(&self, py: Python<'_>) -> PyResult<Arc<CheckpointAccess>> {
        let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return Err(invalid(py, "the checkpoint is closed"));
        }
        state.access.clone().ok_or_else(|| invalid(py, "the checkpoint is closed"))
    }

    fn ensure_open(&self, py: Python<'_>) -> PyResult<()> {
        let closed = self.state.lock().unwrap_or_else(|e| e.into_inner()).closed;
        if closed {
            Err(invalid(py, "the checkpoint is closed"))
        } else {
            Ok(())
        }
    }

    pub(crate) fn core(&self, py: Python<'_>) -> PyResult<Checkpoint> {
        Ok((**self.access(py)?).clone())
    }

    fn tensor_dict(
        &self,
        py: Python<'_>,
        component: &tensorfs::format::Component,
    ) -> PyResult<Py<PyDict>> {
        let tensors = PyDict::new(py);
        for tensor in &component.tensors {
            tensors.set_item(&tensor.key, Py::new(py, info(py, tensor)?)?)?;
        }
        Ok(tensors.unbind())
    }
}

impl CheckpointAccess {
    pub(crate) fn fetch_tensors(
        &self,
        py: Python<'_>,
        component: &str,
        reads: &[FetchTensor<'_>],
    ) -> PyResult<()> {
        let selections = if reads
            .iter()
            .all(|read| read.role.is_none() && read.range.is_none())
        {
            vec![Selection::Tensors {
                component: component.to_owned(),
                keys: reads.iter().map(|read| read.key.to_owned()).collect(),
            }]
        } else {
            reads
                .iter()
                .map(|read| Selection::Part {
                    component: component.to_owned(),
                    key: read.key.to_owned(),
                    role: read.role.map(str::to_owned),
                    range: read.range.clone(),
                })
                .collect()
        };
        py.detach(|| {
            // Validate the entire request before starting any selected payload I/O.
            for selection in &selections {
                self.missing(selection)?;
            }
            block_on(async {
                for selection in selections {
                    self.download(&Download {
                        selection,
                        ..Download::default()
                    })
                    .await?;
                }
                Ok(())
            })
        })
        .or_raise(py)
    }

    fn fetch_assets(&self, py: Python<'_>, component: &str, name: Option<&str>) -> PyResult<()> {
        let want = Download {
            selection: Selection::Assets {
                component: component.to_owned(),
                names: name.map(|n| vec![n.to_owned()]),
            },
            ..Download::default()
        };
        py.detach(|| block_on(self.download(&want)))
            .or_raise(py)
            .map(drop)
    }
}

#[pymethods]
impl PyCheckpoint {
    fn close(&self) {
        let released = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.closed = true;
            (state.access.take(), state.components.take())
        };
        drop(released);
    }

    fn __enter__(slf: &Bound<'_, Self>) -> PyResult<Py<PyCheckpoint>> {
        slf.get().ensure_open(slf.py())?;
        Ok(slf.clone().unbind())
    }

    fn __exit__(
        &self,
        exc_type: &Bound<'_, PyAny>,
        exc_value: &Bound<'_, PyAny>,
        traceback: &Bound<'_, PyAny>,
    ) {
        let _ = (exc_type, exc_value, traceback);
        self.close();
    }

    #[pyo3(signature = (components=None))]
    fn verify(&self, py: Python<'_>, components: Option<Vec<String>>) -> PyResult<crate::verification::PyVerificationReport> {
        let checkpoint = self.core(py)?;
        let selection = components.map(Selection::Components).unwrap_or_default();
        let report = py.detach(move || checkpoint.verify(&selection)).or_raise(py)?;
        Ok(report.into())
    }

    /// `sha256:<hex>` of the header: the cozytensors' id.
    #[getter]
    fn id(&self) -> String {
        self.id.id()
    }

    /// The tree, repo, revision and variant naming it, when one does.
    #[getter]
    fn tree(&self) -> Option<String> {
        self.tree.as_ref().map(ObjectRef::id)
    }
    #[getter]
    fn repo_id(&self) -> Option<&str> {
        self.named.as_ref().map(|named| named.repo_id.as_str())
    }
    #[getter]
    fn revision(&self) -> Option<&str> {
        self.named.as_ref()?.revision.as_deref()
    }
    #[getter]
    fn variant(&self) -> Option<&str> {
        self.variant.as_deref()
    }
    #[getter]
    fn pipeline(&self, py: Python<'_>) -> PyResult<Option<PyPipeline>> {
        (self.header.pipeline())
            .map(|p| {
                Ok(PyPipeline {
                    cls: p.class.clone(),
                    config: config_dict(py, &p.config)?.unbind(),
                })
            })
            .transpose()
    }

    /// `components[name] -> Component`, in construction order.
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let access = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.closed {
                drop(state);
                return Err(invalid(py, "the checkpoint is closed"));
            }
            if let Some(components) = &state.components {
                return Ok(components.clone_ref(py));
            }
            state.access.clone()
        };
        let out = PyDict::new(py);
        for c in self.header.components() {
            let assets = PyDict::new(py);
            for (name, o) in &c.assets {
                assets.set_item(name, (o.sha256.clone(), o.length))?;
            }
            let component = PyComponent {
                name: c.name.clone(),
                cls: c.class.clone(),
                config: config_dict(py, &c.config)?.unbind(),
                config_file: c.config_name().to_owned(),
                tensors: Py::new(
                    py,
                    PyTensors::new(&c.name, self.tensor_dict(py, c)?, access.clone()),
                )?,
                assets: assets.unbind(),
                ck: access.clone(),
            };
            out.set_item(&c.name, Py::new(py, component)?)?;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            drop(state);
            return Err(invalid(py, "the checkpoint is closed"));
        }
        let components = state.components.get_or_insert_with(|| out.unbind());
        Ok(components.clone_ref(py))
    }

    /// The header's chunks, in order: `(sha256, length)`.
    #[getter]
    fn chunks(&self) -> Vec<(String, u64)> {
        (self.header.chunks.iter())
            .map(|c| (c.sha256.clone(), c.length))
            .collect()
    }

    /// The chunks and assets of `components` (all when None) not here: `(sha256, length)`.
    #[pyo3(signature = (components = None))]
    fn missing(
        &self,
        py: Python<'_>,
        components: Option<Vec<String>>,
    ) -> PyResult<Vec<(String, u64)>> {
        let ck = self.core(py)?;
        let missing = py
            .detach(|| ck.missing(&components.map(Selection::Components).unwrap_or_default()))
            .or_raise(py)?;
        Ok(missing.into_iter().map(|o| (o.sha256, o.length)).collect())
    }

    #[pyo3(signature = (components = None, *, priority = 0, progress = None))]
    fn download(
        &self,
        py: Python<'_>,
        components: Option<Vec<String>>,
        priority: u64,
        progress: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let access = self.access(py)?;
        let callback_error = Mutex::new(None);
        let report = |held, total| -> tensorfs::Result<()> {
            if let Some(callback) = &progress {
                if let Err(error) = Python::attach(|py| callback.call1(py, ("fetch", held, total)))
                {
                    *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    return Err(Failure {
                        kind: Kind::Invalid,
                        detail: "the download progress callback failed".into(),
                    });
                }
            }
            Ok(())
        };
        let want = Download {
            selection: components.map(Selection::Components).unwrap_or_default(),
            priority,
            progress: progress
                .as_ref()
                .map(|_| &report as &tensorfs::Progress<'_>),
        };
        let result = py.detach(|| block_on(access.download(&want))).map(drop);
        if let Some(error) = callback_error
            .into_inner()
            .unwrap_or_else(|e| e.into_inner())
        {
            return Err(error);
        }
        result.or_raise(py)
    }

    /// The header's loading metadata (`source`, `filled`, `safetensors_metadata`).
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata(py, &self.header.metadata)
    }
}

// ---------------------------------------------------------------- arrays

/// Bytes on a page boundary: aligned for every dtype, and an O_DIRECT read lands in place.
/// Zeroed lazily by the allocator (calloc), so a large one costs no pass over its pages.
pub(crate) struct Aligned {
    words: Vec<u64>,
    at: usize,
    pub(crate) len: usize,
}

const PAGE: usize = 4096;

impl Aligned {
    pub(crate) fn new(len: usize) -> Aligned {
        let words = vec![0u64; (len + PAGE).div_ceil(8)];
        let at = (PAGE - words.as_ptr() as usize % PAGE) % PAGE;
        Aligned { words, at, len }
    }
    fn ptr(&self) -> *const u8 {
        (self.words.as_ptr() as *const u8).wrapping_add(self.at)
    }
    pub(crate) fn as_mut_ptr(&mut self) -> *mut u8 {
        (self.words.as_mut_ptr() as *mut u8).wrapping_add(self.at)
    }
    pub(crate) fn bytes(&self) -> &[u8] {
        // SAFETY: `words` holds `at + len` initialized bytes; u8 has no alignment.
        unsafe { std::slice::from_raw_parts(self.ptr(), self.len) }
    }
    pub(crate) fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above, and `self` is borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }
}

/// An owned host or device allocation, also retained by every exported DLPack capsule.
pub(crate) struct DeviceArray(Option<tensorfs::io::DeviceBuffer>);

impl DeviceArray {
    pub(crate) fn new(buffer: tensorfs::io::DeviceBuffer) -> Arc<Self> {
        Arc::new(Self(Some(buffer)))
    }
}

impl std::ops::Deref for DeviceArray {
    type Target = tensorfs::io::DeviceBuffer;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().expect("live device allocation")
    }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        // cuMemFree may synchronize pending work. Move its actual final owner to the
        // native cleanup worker, also when a foreign DLPack consumer calls the deleter.
        crate::cleanup::defer(self.0.take());
    }
}

pub(crate) struct MappedArray {
    buffer: Option<tensorfs::io::MappedBuffer>,
    pub(crate) access: Mutex<()>,
}

impl MappedArray {
    pub(crate) fn new(buffer: tensorfs::io::MappedBuffer) -> Arc<Self> {
        Arc::new(Self { buffer: Some(buffer), access: Mutex::new(()) })
    }
}

impl std::ops::Deref for MappedArray {
    type Target = tensorfs::io::MappedBuffer;
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().expect("live mapped host allocation")
    }
}

impl Drop for MappedArray {
    fn drop(&mut self) {
        // The actual host allocation and admission lease move together. The existing
        // cleanup worker performs the buffer's checked consumer completion off Python.
        crate::cleanup::defer(self.buffer.take());
    }
}

/// An unsuccessful CUDA operation may still reference these actual allocations. Cleanup
/// runs after caller locks have ended; the shared engine owns the completion proof.
struct FailedCuda<T> {
    device: i32,
    owner: Option<T>,
}

impl<T> Drop for FailedCuda<T> {
    fn drop(&mut self) {
        if tensorfs::io::drain(self.device).is_err() {
            // Keep every allocation if the engine cannot prove all work has stopped.
            std::mem::forget(self.owner.take());
        }
    }
}

pub(crate) fn defer_failed_cuda<T: Send + 'static>(device: i32, owner: T) {
    crate::cleanup::defer(FailedCuda {
        device,
        owner: Some(owner),
    });
}

#[derive(Clone)]
pub(crate) enum ArrayBuffer {
    Host(Arc<Aligned>),
    Device(Arc<DeviceArray>),
    Mapped(Arc<MappedArray>),
}

impl ArrayBuffer {
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Host(b) => b.len,
            Self::Device(b) => b.len(),
            Self::Mapped(b) => b.len(),
        }
    }
    pub(crate) fn host(&self) -> Option<&[u8]> {
        match self {
            Self::Host(b) => Some(b.bytes()),
            Self::Device(_) | Self::Mapped(_) => None,
        }
    }
    fn ptr(&self) -> *mut c_void {
        match self {
            Self::Host(b) => b.ptr() as _,
            Self::Device(b) => b.ptr() as usize as _,
            Self::Mapped(b) => b.ptr() as usize as _,
        }
    }
    fn device(&self) -> (i32, i32) {
        match self {
            Self::Host(_) => (KDL_CPU, 0),
            Self::Device(b) => (KDL_CUDA, b.device()),
            Self::Mapped(b) => (KDL_CUDA, b.device()),
        }
    }
    fn duplicate(&self) -> Result<Self, Failure> {
        use tensorfs::io::{self, Copy, Dst, ReadOpts, Src};
        Ok(match self {
            Self::Host(b) => {
                let mut copy = Aligned::new(b.len);
                copy.bytes_mut().copy_from_slice(b.bytes());
                Self::Host(Arc::new(copy))
            }
            Self::Device(b) => {
                let copy = io::DeviceBuffer::new(b.device(), b.len())?;
                let request = Copy {
                    src: Src::Device {
                        device: b.device(),
                        ptr: b.ptr(),
                        stream: 0,
                    },
                    dst: Dst::Device {
                        device: b.device(),
                        ptr: copy.ptr(),
                        len: copy.len(),
                        stream: 0,
                    },
                };
                // SAFETY: both allocations remain held until the copy completes.
                if let Err(error) = unsafe { io::read(&[request], &ReadOpts::default()) } {
                    defer_failed_cuda(b.device(), (Arc::clone(b), copy));
                    return Err(error.source.into());
                }
                Self::Device(DeviceArray::new(copy))
            }
            Self::Mapped(_) => return Err(Failure {
                kind: Kind::Unsupported,
                detail: "mapped output copies require the owning framework allocator".into(),
            }),
        })
    }
}

/// A tensor's bytes. Consumers view them through DLPack without a framework dependency.
#[pyclass(name = "Array", module = "tensorfs", frozen)]
pub struct PyArray {
    pub(crate) buf: ArrayBuffer,
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<u64>,
}

const KDL_CPU: i32 = 1;
const KDL_CUDA: i32 = 2;
const DLTENSOR: &CStr = c"dltensor";
const USED_DLTENSOR: &CStr = c"used_dltensor";

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
    _buf: ArrayBuffer,
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
        self.buf.len()
    }
    fn tobytes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        if let Some(bytes) = self.buf.host() {
            return Ok(PyBytes::new(py, bytes));
        }
        let buffer = self.buf.clone();
        let device = buffer.device().1;
        let bytes = py
            .detach(|| -> Result<Vec<u8>, Failure> {
                use tensorfs::io::{self, Copy, Dst, ReadOpts, Src};
                let _access = match &buffer {
                    ArrayBuffer::Mapped(b) => Some(b.access.lock().unwrap_or_else(|e| e.into_inner())),
                    _ => None,
                };
                let mut bytes = vec![0; buffer.len()];
                let request = Copy {
                    src: match &buffer {
                        ArrayBuffer::Device(b) => Src::Device {
                            device, ptr: b.ptr(), stream: 0,
                        },
                        ArrayBuffer::Mapped(b) => {
                            b.synchronize()?;
                            Src::Host(b.host_ptr())
                        }
                        ArrayBuffer::Host(_) => unreachable!(),
                    },
                    dst: Dst::host(&mut bytes),
                };
                if let Err(error) = unsafe { io::read(&[request], &ReadOpts::default()) } {
                    defer_failed_cuda(device, (buffer.clone(), bytes));
                    return Err(error.source.into());
                }
                Ok(bytes)
            })
            .or_raise(py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[pyo3(signature = (source, axis, offset, *, stream))]
    fn _write_band(
        &self, py: Python<'_>, source: &Bound<'_, PyAny>, axis: usize, offset: u64, stream: usize,
    ) -> PyResult<()> {
        crate::tensors::write_band(py, self, source, axis, offset, stream)
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
        let _ = max_version;
        let device = self.buf.device();
        if copy == Some(true) && matches!(&self.buf, ArrayBuffer::Mapped(_)) {
            return Err(pyo3::exceptions::PyBufferError::new_err(
                "mapped output copy=True requires the owning framework allocator",
            ));
        }
        if let Some(stream) = stream {
            let value = stream.extract::<i64>()?;
            if device.0 == KDL_CPU || value == 0 || value < -1 {
                return Err(invalid(
                    py,
                    "DLPack stream must be a CUDA stream, 1/2 for default streams, or -1",
                ));
            }
            if device.0 == KDL_CUDA && copy == Some(true) {
                return Err(pyo3::exceptions::PyBufferError::new_err(
                    "CUDA DLPack copy=True with an explicit consumer stream is unsupported",
                ));
            }
            if let ArrayBuffer::Mapped(buffer) = &self.buf {
                if value != -1 && py.detach(|| buffer.is_capturing(value as usize)).or_raise(py)? {
                    return Err(pyo3::exceptions::PyBufferError::new_err(
                        "mapped host DLPack export during CUDA capture is unsupported",
                    ));
                }
            }
            // Every TensorFS read/launch drains before returning. The allocation is ready
            // on every consumer stream, including the DLPack default-stream sentinels.
        }
        if dl_device.is_some_and(|d| d != device) {
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: format!("this Array lives on DLPack device {device:?}"),
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
        let shape = self
            .shape
            .iter()
            .map(|&dimension| {
                i64::try_from(dimension).map_err(|_| invalid(py, "DLPack dimension exceeds i64"))
            })
            .collect::<PyResult<Vec<_>>>()?;
        if self.shape.len() > i32::MAX as usize {
            return Err(invalid(py, "DLPack rank exceeds i32"));
        }
        let buf = if copy == Some(true) {
            py.detach(|| self.buf.duplicate()).or_raise(py)?
        } else {
            self.buf.clone()
        };
        let data = buf.ptr();
        let held = Box::into_raw(Box::new(Held { shape, _buf: buf }));
        // SAFETY: `held` is live until the deleter runs; the consumer may write its own copy.
        let t = Box::into_raw(Box::new(DLManagedTensor {
            dl_tensor: DLTensor {
                data,
                device: DLDevice {
                    device_type: device.0,
                    device_id: device.1,
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
        self.buf.device()
    }

    fn __repr__(&self) -> String {
        format!("Array({}, {:?})", self.dtype, self.shape)
    }
}

/// The transferred DLPack lifetime. Final release always runs on the existing cleanup
/// worker, after native operations and Writer locks have ended.
struct DlpackRelease {
    stream: Option<tensorfs::io::cuda::Stream>,
    producer: Option<Py<PyAny>>,
    capsule: Option<Py<PyAny>>,
    managed: Option<usize>,
    engine_error: bool,
    quiescent: bool,
}

impl DlpackRelease {
    fn retain(&mut self) {
        // Failed completion is not permission to free producer memory. Preserve every
        // actual owner, including the stream/context, rather than an address-only guard.
        std::mem::forget(self.stream.take());
        std::mem::forget(self.producer.take());
        std::mem::forget(self.capsule.take());
        let _ = self.managed.take();
    }

    fn drain(&mut self) -> tensorfs::Result<()> {
        if self.quiescent {
            return Ok(());
        }
        if let Some(stream) = &self.stream {
            let checked = if self.engine_error {
                tensorfs::io::drain(stream.device()).and_then(|_| stream.drain())
            } else {
                stream.drain()
            };
            if let Err(error) = checked {
                self.retain();
                return Err(error.into());
            }
        }
        self.quiescent = true;
        Ok(())
    }
}

impl Drop for DlpackRelease {
    fn drop(&mut self) {
        if self.drain().is_err() {
            return;
        }
        // A checked drain has proved quiescence. Destroy the stream/context before
        // calling a foreign producer, whose deleter may acquire Python or native locks.
        drop(self.stream.take());
        if Python::try_attach(|_| {
            if let Some(managed) = self.managed.take() {
                let managed = managed as *mut DLManagedTensor;
                // SAFETY: successful capsule rename transferred this unique managed
                // tensor to us; all uses have completed and it is released exactly once.
                if let Some(deleter) = unsafe { (*managed).deleter } {
                    unsafe { deleter(managed) };
                }
            }
            drop(self.capsule.take());
            drop(self.producer.take());
        })
        .is_none()
        {
            // Interpreter shutdown cannot safely run a Python-aware foreign deleter.
            self.retain();
        }
    }
}

struct DlpackLifetime(Option<DlpackRelease>);

impl DlpackLifetime {
    fn record_cuda_error(&mut self) {
        if let Some(release) = &mut self.0 {
            release.engine_error = true;
            release.quiescent = false;
        }
    }

    fn finish(&mut self) -> tensorfs::Result<()> {
        if let Some(release) = &mut self.0 {
            release.drain()?;
        }
        crate::cleanup::defer(self.0.take());
        Ok(())
    }
}

impl Drop for DlpackLifetime {
    fn drop(&mut self) {
        // Never acquire Python or invoke a producer while a caller's Writer/GPU mutex
        // is held. This also covers validation errors and a never-read device source.
        if let Some(release) = self.0.take() {
            crate::cleanup::defer(release);
        }
    }
}

/// The one shared DLPack parser for Writer sources and read destinations. Contiguous by
/// default; mapped-output band sources additionally permit validated nonnegative strides.
/// Validated legacy capsules are consumed once and retain a unique managed deleter.
pub(crate) struct Dlpack {
    lifetime: DlpackLifetime,
    pub(crate) device: (i32, i32),
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<u64>,
    pub(crate) len: usize,
    pub(crate) address: usize,
    pub(crate) strides: Vec<u64>,
}

impl Dlpack {
    pub(crate) fn of(
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        dtype: Option<String>,
        shape: Option<Vec<u64>>,
    ) -> PyResult<Self> {
        Self::parse(py, data, dtype, shape, false)
    }

    pub(crate) fn strided(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        Self::parse(py, data, None, None, true)
    }

    fn parse(
        py: Python<'_>, data: &Bound<'_, PyAny>, dtype: Option<String>,
        shape: Option<Vec<u64>>, strided: bool,
    ) -> PyResult<Self> {
        let device: (i32, i32) = data.call_method0("__dlpack_device__")?.extract()?;
        let stream = match device {
            (KDL_CPU, _) => None,
            (KDL_CUDA, device) if device >= 0 => Some(
                py.detach(|| tensorfs::io::cuda::Stream::new(device))
                    .or_raise(py)?,
            ),
            _ => {
                return Err(raise(
                    py,
                    Failure {
                        kind: Kind::Unsupported,
                        detail: "expected a CPU or CUDA DLPack array".into(),
                    },
                ))
            }
        };
        let mut lifetime = DlpackLifetime(Some(DlpackRelease {
            stream,
            producer: Some(data.clone().unbind()),
            capsule: None,
            managed: None,
            engine_error: false,
            quiescent: false,
        }));
        let exported = if let Some(stream) = &lifetime.0.as_ref().unwrap().stream {
            let kwargs = PyDict::new(py);
            kwargs.set_item("stream", stream.raw())?;
            data.call_method("__dlpack__", (), Some(&kwargs))
        } else {
            data.call_method0("__dlpack__")
        }?;
        lifetime.0.as_mut().unwrap().capsule = Some(exported.unbind());
        let parsed = (|| {
            let capsule = lifetime
                .0
                .as_ref()
                .unwrap()
                .capsule
                .as_ref()
                .unwrap()
                .bind(py);
            if unsafe { ffi::PyCapsule_IsValid(capsule.as_ptr(), DLTENSOR.as_ptr()) } != 1 {
                return Err(invalid(py, "__dlpack__ did not return a dltensor capsule"));
            }
            let managed = unsafe {
                ffi::PyCapsule_GetPointer(capsule.as_ptr(), DLTENSOR.as_ptr())
                    as *mut DLManagedTensor
            };
            if managed.is_null() {
                return Err(PyErr::fetch(py));
            }
            let tensor = unsafe { &(*managed).dl_tensor };
            if tensor.device.device_type != device.0
                || (device.0 == KDL_CUDA && tensor.device.device_id != device.1)
                || !(0..=32).contains(&tensor.ndim)
                || tensor.dtype.lanes != 1
            {
                return Err(invalid(
                    py,
                    "DLPack device or scalar-lane tensor metadata disagrees",
                ));
            }
            if tensor.ndim != 0 && tensor.shape.is_null() {
                return Err(invalid(py, "DLPack tensor has no shape"));
            }
            let dims = if tensor.ndim == 0 {
                &[][..]
            } else {
                unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) }
            };
            let inferred_shape: Vec<u64> = dims
                .iter()
                .map(|d| u64::try_from(*d).map_err(|_| invalid(py, "DLPack shape is negative")))
                .collect::<PyResult<_>>()?;
            let mut stride = 1i64;
            let mut strides = vec![0; dims.len()];
            for i in (0..dims.len()).rev() {
                let actual = if tensor.strides.is_null() { stride } else {
                    unsafe { *tensor.strides.add(i) }
                };
                strides[i] = u64::try_from(if strided { actual } else { stride })
                    .map_err(|_| invalid(py, "negative DLPack strides are unsupported"))?;
                if !strided
                    && dims[i] > 1
                    && actual != stride
                {
                    return Err(invalid(py, "expected a contiguous DLPack array"));
                }
                stride = stride
                    .checked_mul(dims[i])
                    .ok_or_else(|| invalid(py, "DLPack shape overflows"))?;
            }
            let inferred_dtype = Dtype::known()
                .find(|d| dl_type(d) == Some((tensor.dtype.code, tensor.dtype.bits)))
                .ok_or_else(|| invalid(py, "DLPack dtype is unsupported"))?;
            let dtype = dtype
                .map(|d| Dtype::parse(&d))
                .unwrap_or_else(|| inferred_dtype.clone());
            let shape = shape.unwrap_or_else(|| inferred_shape.clone());
            if dtype != inferred_dtype || shape != inferred_shape {
                return Err(invalid(
                    py,
                    "explicit dtype or shape disagrees with the DLPack array",
                ));
            }
            let len = dtype
                .byte_len(&shape)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| invalid(py, "DLPack byte size overflows"))?;
            let offset = usize::try_from(tensor.byte_offset)
                .map_err(|_| invalid(py, "DLPack byte offset overflows"))?;
            let address = (tensor.data as usize)
                .checked_add(offset)
                .ok_or_else(|| invalid(py, "DLPack address overflows"))?;
            if len != 0 && address == 0 {
                return Err(invalid(py, "DLPack tensor has no data"));
            }
            if address.checked_add(len).is_none() {
                return Err(invalid(py, "DLPack address range overflows"));
            }
            if strided && len != 0 {
                let width = dtype.byte_len(&[1])
                    .ok_or_else(|| invalid(py, "DLPack element width is unsupported"))?;
                let end = inferred_shape.iter().zip(&strides).try_fold(0u64, |end, (&dim, &step)| {
                    dim.saturating_sub(1).checked_mul(step).and_then(|part| end.checked_add(part))
                }).and_then(|end| end.checked_add(1)).and_then(|end| end.checked_mul(width))
                    .and_then(|end| usize::try_from(end).ok())
                    .and_then(|span| address.checked_add(span));
                if end.is_none() {
                    return Err(invalid(py, "strided DLPack address range overflows"));
                }
            }
            Ok((dtype, shape, len, address, managed as usize, strides))
        })();
        let (dtype, shape, len, address, managed, strides) = parsed?;
        let release = lifetime.0.as_mut().unwrap();
        let capsule = release.capsule.as_ref().unwrap().bind(py);
        // SAFETY: both names have static storage. Validation completed before this
        // single ownership transfer; no fallible work lies between rename and guard.
        if unsafe { ffi::PyCapsule_SetName(capsule.as_ptr(), USED_DLTENSOR.as_ptr()) } != 0 {
            return Err(PyErr::fetch(py));
        }
        release.managed = Some(managed);
        Ok(Self {
            lifetime,
            device,
            dtype,
            shape,
            len,
            address,
            strides,
        })
    }

    pub(crate) fn stream(&self) -> Option<&tensorfs::io::cuda::Stream> {
        self.lifetime
            .0
            .as_ref()
            .and_then(|release| release.stream.as_ref())
    }

    /// Check completion without releasing any member of a borrowed decode group.
    pub(crate) fn checked_drain(&mut self) -> tensorfs::Result<()> {
        match self.lifetime.0.as_mut() {
            Some(release) => release.drain(),
            None => Ok(()),
        }
    }

    /// Preserve the actual transferred owner after another group member loses proof.
    pub(crate) fn retain(&mut self) {
        if let Some(release) = self.lifetime.0.as_mut() {
            release.retain();
        }
    }

    pub(crate) fn record_cuda_error(&mut self) {
        if self.device.0 == KDL_CUDA {
            self.lifetime.record_cuda_error();
        }
    }

    pub(crate) fn finish(&mut self) -> tensorfs::Result<()> {
        self.lifetime.finish()
    }

    pub(crate) fn destination(&self) -> tensorfs::io::Dst {
        match self.stream() {
            Some(stream) => tensorfs::io::Dst::Device {
                device: self.device.1,
                ptr: self.address as u64,
                len: self.len,
                stream: stream.raw(),
            },
            None => tensorfs::io::Dst::Host {
                ptr: self.address as *mut u8,
                len: self.len,
            },
        }
    }
}

// ---------------------------------------------------------------- the writer

/// One cozytensors out, component by component (`component`), sealed by `commit`.
#[pyclass(name = "Writer", module = "tensorfs", frozen)]
pub struct PyWriter {
    w: Mutex<Option<Writer>>,
}

impl PyWriter {
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut Writer) -> tensorfs::Result<T> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
            match guard.as_mut() {
                Some(writer) => f(writer),
                None => Err(Failure {
                    kind: Kind::Invalid,
                    detail: "the writer has committed or been abandoned".into(),
                }),
            }
        })
        .or_raise(py)
    }
}

impl Drop for PyWriter {
    fn drop(&mut self) {
        let writer = self.w.get_mut().unwrap_or_else(|e| e.into_inner()).take();
        crate::cleanup::defer(writer);
    }
}

#[pymethods]
impl PyWriter {
    #[new]
    #[pyo3(signature = (dest = None, *, variant = None, base = None, resume = None, cache_dir = None, repo_id = None, endpoint = None, token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        dest: Option<PathBuf>,
        variant: Option<String>,
        base: Option<&PyCheckpoint>,
        resume: Option<String>,
        cache_dir: Option<PathBuf>,
        repo_id: Option<String>,
        endpoint: Option<String>,
        token: Option<&Bound<'_, PyAny>>,
        ca_der: Option<Vec<u8>>,
    ) -> PyResult<Self> {
        let base = base.map(|base| base.core(py)).transpose()?;
        let config = crate::credentials::config(endpoint, cache_dir, token, ca_der)?;
        let writer = py
            .detach(move || {
                let client = Client::new(config)?;
                let options = WriterOptions {
                    dest: dest.as_deref(),
                    variant: variant.as_deref(),
                    base: base.as_ref(),
                    resume: resume.as_deref(),
                    repo_id: repo_id.as_deref(),
                    ..WriterOptions::default()
                };
                block_on(async { client.writer(options) })
            })
            .or_raise(py)?;
        Ok(Self {
            w: Mutex::new(Some(writer)),
        })
    }

    fn pipeline(&self, py: Python<'_>, cls: String, config: &Bound<'_, PyDict>) -> PyResult<()> {
        let config = config_map(config)?;
        self.with(py, |writer| writer.pipeline(&cls, config))
    }

    #[pyo3(signature = (name, cls = "", config = None, config_file = None))]
    fn component(
        slf: &Bound<'_, Self>,
        name: String,
        cls: &str,
        config: Option<&Bound<'_, PyDict>>,
        config_file: Option<&str>,
    ) -> PyResult<PyComponentWriter> {
        let py = slf.py();
        let config = config.map(config_map).transpose()?.unwrap_or_default();
        slf.get()
            .with(py, |writer| writer.component(&name, cls, config, config_file).map(drop))?;
        Ok(PyComponentWriter {
            w: slf.clone().unbind(),
            name,
        })
    }

    fn set_metadata(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = from_py(value)?;
        self.with(py, |writer| writer.set_metadata(&key, value))
    }

    fn has(&self, py: Python<'_>, component: &str, key: &str) -> PyResult<bool> {
        self.with(py, |writer| writer.has(component, key))
    }

    fn commit(&self, py: Python<'_>) -> PyResult<PyCheckpoint> {
        let (result, retained) = py.detach(|| {
            let mut guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
            let Some(writer) = guard.as_mut() else {
                return (
                    Err(Failure {
                        kind: Kind::Invalid,
                        detail: "the writer has committed or been abandoned".into(),
                    }),
                    None,
                );
            };
            let result = block_on(writer.commit());
            let retained = writer.committed();
            if result.is_ok() {
                guard.take();
            }
            (result, retained)
        });
        match result {
            Ok(checkpoint) => Ok(PyCheckpoint::from_core(checkpoint)),
            Err(failure) => {
                let error = raise(py, failure);
                if let Some(checkpoint) = retained {
                    error.value(py).setattr(
                        "checkpoint",
                        Py::new(py, PyCheckpoint::from_core(checkpoint))?,
                    )?;
                }
                Err(error)
            }
        }
    }

    fn abandon(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let writer = self.w.lock().unwrap_or_else(|e| e.into_inner()).take();
            match writer {
                Some(writer) => writer.abandon(),
                None => Ok(()),
            }
        })
        .or_raise(py)
    }
}

#[pyclass(name = "ComponentWriter", module = "tensorfs", frozen)]
pub struct PyComponentWriter {
    w: Py<PyWriter>,
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl PyComponentWriter {
    #[pyo3(signature = (key, source, as_key = None))]
    fn copy_component(
        &self,
        py: Python<'_>,
        key: String,
        source: &PyComponent,
        as_key: Option<String>,
    ) -> PyResult<()> {
        let access = source
            .ck
            .as_ref()
            .ok_or_else(|| invalid(py, "a parsed component has no bytes"))?;
        access.fetch_tensors(
            py,
            &source.name,
            &[FetchTensor {
                key: &key,
                role: None,
                range: None,
            }],
        )?;
        let source = access.component(&source.name).or_raise(py)?.clone();
        self.w.get().with(py, |writer| {
            writer
                .component_writer(&self.name)?
                .copy(&key, &source, as_key.as_deref())
        })
    }

    #[pyo3(signature = (key, data, *, dtype = None, shape = None))]
    fn put_array(
        &self,
        py: Python<'_>,
        key: String,
        data: &Bound<'_, PyAny>,
        dtype: Option<String>,
        shape: Option<Vec<u64>>,
    ) -> PyResult<()> {
        if !data.hasattr("__dlpack__")? {
            let dtype = dtype.ok_or_else(|| invalid(py, "a buffer needs dtype"))?;
            let shape = shape.ok_or_else(|| invalid(py, "a buffer needs shape"))?;
            return self.put(py, key, dtype, shape, data);
        }
        let mut view = Dlpack::of(py, data, dtype, shape)?;
        let dtype = view.dtype.clone();
        let shape = view.shape.clone();
        if view.stream().is_some() {
            let mut source = crate::dlpack_source::DeviceSource::new(view).or_raise(py)?;
            let result = self.w.get().with(py, |writer| {
                writer
                    .component_writer(&self.name)?
                    .put(&key, dtype, &shape, &mut source)
            });
            // The Writer mutex has been released before drain or foreign cleanup.
            let finished = py.detach(|| source.finish()).or_raise(py);
            return result.and(finished);
        }
        let result = self.w.get().with(py, |writer| {
            let mut bytes = if view.len == 0 {
                &[][..]
            } else {
                // SAFETY: the consumed managed owner remains in view until after put.
                unsafe { std::slice::from_raw_parts(view.address as *const u8, view.len) }
            };
            writer
                .component_writer(&self.name)?
                .put(&key, dtype, &shape, &mut bytes)
        });
        let finished = py.detach(|| view.finish()).or_raise(py);
        result.and(finished)
    }

    fn put(
        &self,
        py: Python<'_>,
        key: String,
        dtype: String,
        shape: Vec<u64>,
        data: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        let dtype = Dtype::parse(&dtype);
        self.w.get().with(py, |writer| {
            let mut source = bytes.slice();
            writer
                .component_writer(&self.name)?
                .put(&key, dtype, &shape, &mut source)
        })
    }

    #[pyo3(signature = (key, dtype, shape, scheme, parts, params = None))]
    #[allow(clippy::too_many_arguments)]
    fn put_encoded(
        &self,
        py: Python<'_>,
        key: String,
        dtype: String,
        shape: Vec<u64>,
        scheme: String,
        parts: &Bound<'_, PyDict>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let params = match params {
            Some(params) => match from_py(params.as_any())? {
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
        self.w.get().with(py, |writer| {
            let mut slices: Vec<&[u8]> = held.iter().map(|held| held.3.slice()).collect();
            let parts = held
                .iter()
                .zip(slices.iter_mut())
                .map(|((role, dtype, shape, _), data)| tensorfs::raw::PartData {
                    role: role.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    data,
                })
                .collect();
            writer.component_writer(&self.name)?.put_encoded(
                &key,
                dtype,
                &shape,
                Encoding { scheme, params },
                parts,
            )
        })
    }

    #[pyo3(signature = (key, source, from_component = None, from_key = None))]
    fn copy(
        &self,
        py: Python<'_>,
        key: String,
        source: &PyCheckpoint,
        from_component: Option<String>,
        from_key: Option<String>,
    ) -> PyResult<()> {
        let access = source.access(py)?;
        let component = from_component.unwrap_or_else(|| self.name.clone());
        let from_key = from_key.unwrap_or_else(|| key.clone());
        access.fetch_tensors(
            py,
            &component,
            &[FetchTensor {
                key: &from_key,
                role: None,
                range: None,
            }],
        )?;
        let source = access.component(&component).or_raise(py)?.clone();
        self.w.get().with(py, |writer| {
            writer
                .component_writer(&self.name)?
                .copy(&from_key, &source, Some(&key))
        })
    }

    fn asset(&self, py: Python<'_>, name: String, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        self.w.get().with(py, |writer| {
            writer
                .component_writer(&self.name)?
                .asset(&name, bytes.slice())
        })
    }

    fn order(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        self.w.get().with(py, |writer| {
            writer.component_writer(&self.name)?.order(&keys)
        })
    }

    fn has(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        self.w
            .get()
            .with(py, |writer| writer.component_writer(&self.name)?.has(key))
    }
}

// ---------------------------------------------------------------- the module

#[pyfunction]
fn _kernel_binary<'py>(
    py: Python<'py>,
    family: u32,
    core: &str,
    dtype: &str,
) -> PyResult<(Bound<'py, PyBytes>, &'static str, u32, u32)> {
    let binary = tensorfs::kernels::BINARIES
        .iter()
        .find(|b| b.family == family && b.kind == core && b.dtype == dtype)
        .ok_or_else(|| {
            raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: format!("no prebuilt sm{family}/{core}/{dtype} kernel"),
                },
            )
        })?;
    Ok((
        PyBytes::new(py, binary.image),
        binary.symbol,
        binary.shared,
        binary.warps,
    ))
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::verification::PyVerificationReport>()?;
    m.add_class::<PyClient>()?;
    m.add_class::<PyCheckpoint>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyComponent>()?;
    m.add_class::<PyComponentWriter>()?;
    m.add_class::<PyTensorMetadata>()?;
    m.add_class::<PyEncoding>()?;
    let raw = PyModule::new(m.py(), "raw")?;
    raw.add_class::<PyPart>()?;
    raw.add_function(wrap_pyfunction!(raw_parts, &raw)?)?;
    crate::resident_decode::install(&raw)?;
    m.add_submodule(&raw)?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyTensors>()?;
    m.add_class::<PyEncoded>()?;
    m.add_class::<PyWriter>()?;
    m.add_function(wrap_pyfunction!(_kernel_binary, m)?)?;
    m.add_function(wrap_pyfunction!(_materialize, m)?)?;
    Ok(())
}
