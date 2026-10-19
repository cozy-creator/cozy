//! `tensorfs`: the store, checkpoints, the component writer and the error families, typed in
//! `_ext.pyi`. Marshal only: every call that moves bytes runs detached from the GIL.

use std::collections::BTreeMap;
use std::ffi::{c_void, CStr};
use std::ops::Range;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyList, PyString, PyTuple};
use pyo3::ffi;
use tensorfs_core::client::{self, ImportOptions, Named, PullFrom};
use tensorfs_core::export;
use tensorfs_core::fetch::{self, Client as FetchClient, Endpoint, Fetcher, ObjectStore, Origin, Want};
use tensorfs_core::failure::{Failure, Kind};
use tensorfs_core::format::{Body, Config, ConfigValue, Dtype, Encoding, Header, Metadata, ObjectRef, Tensor, Tree, Value};
use tensorfs_core::store2::{Checkpoint, Store};
use tensorfs_core::writer::{Data, PartData, Source, Writer};

use crate::tensors::{PyEncoded, PyTensors};
pub(crate) use crate::errors::raise;

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
            let items = items.iter().map(|v| config_to_py(py, v)).collect::<PyResult<Vec<_>>>()?;
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
        Ok(items) => Ok(ConfigValue::Array(items.iter().map(config_from_py).collect::<PyResult<_>>()?)),
        Err(_) => Err(invalid(py, format!("{obj}: a config holds None, bool, int, float, str, list or dict"))),
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
fn named(store: &Store, reference: &Bound<'_, PyAny>, variant: Option<&str>) -> PyResult<Named> {
    let py = reference.py();
    if let Ok(ck) = reference.cast::<PyCheckpoint>() {
        let stored = ck.get().stored(py)?;
        if let Some(named) = &ck.get().named {
            return Ok(named.clone());
        }
        return Ok(Named { header: stored.id().clone(), tree: None, name: None, release: None, variant: None });
    }
    match reference.extract::<String>() {
        Ok(text) => py.detach(|| store.resolve(&text, variant)).or_raise(py),
        Err(_) => Err(invalid(
            py,
            "a ref is a stored Checkpoint or a string (org/name[@release], sha256:<hex>)",
        )),
    }
}

fn opened(py: Python<'_>, store: &Store, named: Named) -> PyResult<PyCheckpoint> {
    let ck = py.detach(|| store.checkpoint(&named.header)).or_raise(py)?;
    Ok(PyCheckpoint::new(ck.header().clone(), Some(ck), Some(named)))
}

// ---------------------------------------------------------------- the store

/// A store at `root`, created when absent.
#[pyclass(name = "Store", module = "tensorfs", frozen)]
pub struct PyStore {
    store: Store,
}

#[derive(Clone)]
struct Remote {
    store: Store,
    client: FetchClient,
    endpoint: Arc<Endpoint>,
    offline: bool,
}

impl Remote {
    fn fetch(&self, objects: &[ObjectRef], priority: u64) -> Result<(), Failure> {
        self.fetch_progress(objects, priority, &|_, _| Ok(()))
    }

    fn fetch_progress(&self, objects: &[ObjectRef], priority: u64, report: &(dyn Fn(u64, u64) -> Result<(), Failure> + Sync)) -> Result<(), Failure> {
        let _hold = self.store.cache_hold(objects)?;
        let total = objects.iter().map(|object| object.length).sum();
        report(0, total)?;
        if self.offline {
            let mut held = 0;
            for object in objects {
                if !self.store.reuse(object)? {
                    return Err(Failure { kind: Kind::NotFound, detail: format!("{} is not cached", object.id()) });
                }
                held += object.length;
                report(held, total)?;
            }
            return Ok(());
        }
        let origin: Arc<dyn Origin> = self.endpoint.clone();
        let store: Arc<dyn ObjectStore> = Arc::new(self.store.clone());
        fetch::runtime()?.block_on(async {
            let fetcher = Fetcher::new(store, self.client.clone(), None, fetch::Config::default());
            let counters = fetch::Progress::default();
            let want = Want { priority, progress: Some(&counters), ..Want::default() };
            let mut work = std::pin::pin!(fetcher.objects(&origin, objects, &want));
            let mut tick = tokio::time::interval(Duration::from_millis(250));
            loop {
                tokio::select! {
                    biased;
                    result = &mut work => {
                        report(counters.held.load(std::sync::atomic::Ordering::Relaxed), total)?;
                        result?;
                        break;
                    }
                    _ = tick.tick() => report(counters.held.load(std::sync::atomic::Ordering::Relaxed), total)?,
                }
            }
            fetcher.settled().await;
            Ok(())
        })
    }

    fn root(&self, name: &str, revision: Option<&str>) -> Result<(ObjectRef, Option<String>), Failure> {
        if let Some(digest) = revision.filter(|r| r.starts_with("sha256:")).or_else(|| name.starts_with("sha256:").then_some(name)) {
            let hex = digest.trim_start_matches("sha256:");
            let path = self.store.path(&ObjectRef { sha256: hex.to_owned(), length: 0 })?;
            if let Ok(meta) = std::fs::metadata(path) {
                return Ok((ObjectRef { sha256: hex.to_owned(), length: meta.len() }, None));
            }
            if self.offline { return Err(Failure { kind: Kind::NotFound, detail: format!("{digest} is not cached") }); }
            let object = fetch::runtime()?.block_on(self.endpoint.object(hex))?
                .ok_or_else(|| Failure { kind: Kind::NotFound, detail: format!("no object {digest}") })?;
            return Ok((object, None));
        }
        if revision.is_some() || self.offline {
            if let Some(repo) = self.store.get_ref(name)? {
                if let Ok((version, tree)) = fetch::release_of(&repo, revision) { return Ok((tree, Some(version))); }
            }
        }
        if self.offline { return Err(Failure { kind: Kind::NotFound, detail: format!("{name} is not cached") }); }
        let resolved = fetch::runtime()?.block_on(self.endpoint.repo(name));
        let (repo, _) = match resolved {
            Ok(Some(repo)) => repo,
            Ok(None) => return Err(Failure { kind: Kind::NotFound, detail: format!("no repo {name}") }),
            Err(error) => {
                // The last named resolution is useful without a network. A refusal,
                // invalid response or absent repo must still be visible to the caller.
                if matches!(error.kind, Kind::Network) {
                    if let Some(repo) = self.store.get_ref(name)? {
                        if let Ok((version, tree)) = fetch::release_of(&repo, revision) {
                            if self.store.reuse(&tree)? {
                                return Ok((tree, Some(version)));
                            }
                        }
                    }
                }
                return Err(error);
            }
        };
        let (version, tree) = fetch::release_of(&repo, revision)?;
        Ok((tree, Some(version)))
    }
}

#[pymethods]
impl PyStore {
    #[new]
    fn new(py: Python<'_>, root: PathBuf) -> PyResult<PyStore> {
        let root = match (root.strip_prefix("~"), std::env::var_os("HOME")) {
            (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
            _ => root,
        };
        let store = py.detach(|| Store::open(root).map(Store::with_cache)).or_raise(py)?;
        Ok(PyStore { store })
    }

    #[getter]
    fn root(&self) -> PathBuf {
        self.store.root().to_path_buf()
    }

    #[pyo3(signature = (path, variant = None))]
    fn open_path(&self, py: Python<'_>, path: PathBuf, variant: Option<String>) -> PyResult<PyCheckpoint> {
        let ck = py.detach(|| Checkpoint::folder(&path, variant.as_deref(), Some(&self.store))).or_raise(py)?;
        Ok(PyCheckpoint::new(ck.header().clone(), Some(ck), None))
    }

    #[pyo3(signature = (repo_id, revision = None, variant = None, endpoint = "https://tensorhub.com", token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn open_remote(&self, py: Python<'_>, repo_id: String, revision: Option<String>, variant: Option<String>, endpoint: &str, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<PyCheckpoint> {
        let client = crate::credentials::client(endpoint, ca_der.as_deref()).or_raise(py)?;
        let endpoint = Arc::new(Endpoint::new(&client, endpoint, crate::credentials::provider(token)?).or_raise(py)?);
        let remote = Remote { store: self.store.clone(), client, endpoint, offline: false };
        let named = py.detach(|| -> Result<Named, Failure> {
            let (root, version) = remote.root(&repo_id, revision.as_deref())?;
            let _hold = self.store.cache_hold(std::slice::from_ref(&root))?;
            remote.fetch(std::slice::from_ref(&root), 0)?;
            let bytes = self.store.read(&root)?;
            let (tree, header, chosen) = match Tree::from_json(&bytes) {
                Ok(tree) => {
                    let (v, header) = tree.variant(variant.as_deref())?;
                    remote.fetch(std::slice::from_ref(header), 0)?;
                    (Some(root), header.clone(), Some(v.to_owned()))
                }
                Err(_) => {
                    Header::from_cbor(&bytes)?;
                    (None, root, variant)
                }
            };
            if let (Some(tree), Some(version)) = (&tree, &version) {
                self.store.update_ref(&repo_id, |old| Some(fetch::with_tree(old, &repo_id, tree, Some(version))))?;
                self.store.cache_ref(&repo_id)?;
            }
            Ok(Named { header, tree, name: Some(repo_id), release: version, variant: chosen })
        }).or_raise(py)?;
        let mut ck = opened(py, &self.store, named)?;
        ck.set_remote(remote);
        Ok(ck)
    }

    #[pyo3(signature = (folder_path, repo_id, revision = None, endpoint = "https://tensorhub.com", token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn upload_folder(&self, py: Python<'_>, folder_path: PathBuf, repo_id: String, revision: Option<String>, endpoint: &str, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<String> {
        let client = crate::credentials::client(endpoint, ca_der.as_deref()).or_raise(py)?;
        let endpoint = Endpoint::new(&client, endpoint, crate::credentials::provider(token)?).or_raise(py)?;
        py.detach(|| -> Result<String, Failure> {
            let tree = self.store.admit_folder(&folder_path)?;
            fetch::runtime()?.block_on(fetch::upload_tree(&endpoint, &repo_id, &tree, &self.store, revision.as_deref()))?;
            Ok(tree.id())
        }).or_raise(py)
    }

    #[pyo3(signature = (folder_path, repo_id, revision = None))]
    fn upload_local(&self, py: Python<'_>, folder_path: PathBuf, repo_id: String, revision: Option<String>) -> PyResult<String> {
        py.detach(|| -> Result<String, Failure> {
            let tree = self.store.admit_folder(&folder_path)?;
            self.store.update_ref(&repo_id, |old| {
                let mut repo = old.unwrap_or_else(|| tensorfs_core::format::Repo::new(&repo_id));
                if let Some(version) = &revision {
                    repo.set_release(version, tree.clone());
                } else if !repo.checkpoints.contains(&tree) {
                    repo.checkpoints.push(tree.clone());
                }
                Some(repo)
            })?;
            Ok(tree.id())
        }).or_raise(py)
    }

    #[pyo3(signature = (ck, dest, variants = None))]
    fn snapshot(&self, py: Python<'_>, ck: &PyCheckpoint, dest: PathBuf, variants: Option<Vec<String>>) -> PyResult<PathBuf> {
        let access = ck.access(py)?;
        let stored = &access.checkpoint;
        py.detach(|| -> Result<PathBuf, Failure> {
            let tree = match ck.named.as_ref().and_then(|n| n.tree.clone()) {
                Some(tree) => tree,
                None => {
                    let source = stored.path()?;
                    self.store.admit_folder(&source)?
                }
            };
            if let Some(remote) = &access.remote {
                let parsed = self.store.tree(&tree)?;
                let mut objects: Vec<ObjectRef> = parsed.repo_files().map(|(_, o)| o.clone()).collect();
                let headers: Vec<ObjectRef> = parsed.variants().filter(|(v, _)| variants.as_ref().is_none_or(|want| want.iter().any(|w| w == v))).map(|(_, h)| h.clone()).collect();
                remote.fetch(&headers, 0)?;
                for header in headers {
                    objects.extend(Header::from_cbor(&self.store.read(&header)?)?.objects().cloned());
                }
                remote.fetch(&objects, 0)?;
            }
            self.store.snapshot(&tree, &dest, variants.as_deref())?;
            Ok(dest)
        }).or_raise(py)
    }

    #[pyo3(signature = (repo_id, *, revision = None, variant = None, components = None, local_dir = None, local_files_only = false, endpoint = "https://tensorhub.com", token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn snapshot_download(&self, py: Python<'_>, repo_id: &str, revision: Option<String>, variant: Option<String>, components: Option<Vec<String>>, local_dir: Option<PathBuf>, local_files_only: bool, endpoint: &str, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<PathBuf> {
        let client = crate::credentials::client(endpoint, ca_der.as_deref()).or_raise(py)?;
        let endpoint = Arc::new(Endpoint::new(&client, endpoint, crate::credentials::provider(token)?).or_raise(py)?);
        let remote = Remote { store: self.store.clone(), client, endpoint, offline: local_files_only || repo_id.starts_with("local/") };
        py.detach(|| -> Result<PathBuf, Failure> {
            let (root, version) = remote.root(repo_id, revision.as_deref())?;
            let _root_hold = self.store.cache_hold(std::slice::from_ref(&root))?;
            remote.fetch(std::slice::from_ref(&root), 0)?;
            let tree = self.store.tree(&root)?;
            let chosen: Vec<_> = match variant.as_deref() {
                Some(v) => vec![tree.variant(Some(v))?],
                None => tree.variants().collect(),
            };
            let headers: Vec<_> = chosen.iter().map(|(_, h)| (*h).clone()).collect();
            let _header_hold = self.store.cache_hold(&headers)?;
            remote.fetch(&headers, 0)?;
            let mut objects: Vec<_> = tree.repo_files().map(|(_, r)| r.clone()).collect();
            for header in headers {
                let header = Header::from_cbor(&self.store.read(&header)?)?;
                objects.extend(tensorfs_core::store2::wanted(&header, components.as_deref())?);
            }
            let _object_hold = self.store.cache_hold(&objects)?;
            remote.fetch(&objects, 0)?;
            if let Some(version) = version {
                self.store.update_ref(repo_id, |old| Some(fetch::with_tree(old, repo_id, &root, Some(&version))))?;
                self.store.cache_ref(repo_id)?;
            }
            let internal = local_dir.is_none();
            let dest = local_dir.unwrap_or_else(|| self.store.root().join("snapshots").join(&root.sha256));
            let variants: Option<Vec<_>> = variant.map(|v| vec![v]);
            self.store.snapshot(&root, &dest, variants.as_deref())?;
            if internal { self.store.cache_snapshot(&dest)?; }
            self.store.cache_touch(&root)?;
            let _ = self.store.cache_maintain(0)?;
            Ok(dest)
        }).or_raise(py)
    }

    #[pyo3(signature = (repo_id, filename, *, revision = None, local_dir = None, local_files_only = false, endpoint = "https://tensorhub.com", token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn hub_download(&self, py: Python<'_>, repo_id: &str, filename: &str, revision: Option<String>, local_dir: Option<PathBuf>, local_files_only: bool, endpoint: &str, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<PathBuf> {
        let client = crate::credentials::client(endpoint, ca_der.as_deref()).or_raise(py)?;
        let endpoint = Arc::new(Endpoint::new(&client, endpoint, crate::credentials::provider(token)?).or_raise(py)?);
        let remote = Remote { store: self.store.clone(), client, endpoint, offline: local_files_only || repo_id.starts_with("local/") };
        py.detach(|| -> Result<PathBuf, Failure> {
            let (root, version) = remote.root(repo_id, revision.as_deref())?;
            let _root_hold = self.store.cache_hold(std::slice::from_ref(&root))?;
            remote.fetch(std::slice::from_ref(&root), 0)?;
            let tree = self.store.tree(&root)?;
            let object = tree.files.get(filename).ok_or_else(|| Failure { kind: Kind::NotFound, detail: format!("no file {filename}") })?;
            remote.fetch(std::slice::from_ref(object), 0)?;
            if let Some(version) = version {
                self.store.update_ref(repo_id, |old| Some(fetch::with_tree(old, repo_id, &root, Some(&version))))?;
                self.store.cache_ref(repo_id)?;
            }
            let internal = local_dir.is_none();
            let folder = local_dir.unwrap_or_else(|| self.store.root().join("snapshots").join(&root.sha256));
            let dest = folder.join(filename);
            self.store.materialize(object, &dest)?;
            if internal { self.store.cache_snapshot(&folder)?; }
            self.store.cache_touch(&root)?;
            let _ = self.store.cache_maintain(0)?;
            Ok(dest)
        }).or_raise(py)
    }

    /// Imports `src` (a file, directory, `hf://`, `civitai://` or URL; or several) and names it
    /// `name` (`local/<its name>` by default), release 1, variant its main dtype.
    #[pyo3(signature = (src, name = None, converter = None, variant = None))]
    fn import_(
        &self,
        py: Python<'_>,
        src: &Bound<'_, PyAny>,
        name: Option<String>,
        converter: Option<String>,
        variant: Option<String>,
    ) -> PyResult<PyCheckpoint> {
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
        opened(py, &self.store, named)
    }

    #[pyo3(signature = (source, *, converter = None, variant = None, token = None, progress = None))]
    fn convert(&self, py: Python<'_>, source: &Bound<'_, PyAny>, converter: Option<String>, variant: Option<String>, token: Option<&Bound<'_, PyAny>>, progress: Option<Py<PyAny>>) -> PyResult<PyCheckpoint> {
        let sources: Vec<String> = source.extract::<String>().map(|s| vec![s]).or_else(|_| source.extract())?;
        let mut access = client::access();
        access.token = crate::credentials::provider(token)?;
        let callback_error = Mutex::new(None);
        let notify = |stage: &str, done, total| {
            if let Some(callback) = &progress {
                if let Err(error) = Python::attach(|py| callback.call1(py, (stage, done, total))) {
                    *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                }
            }
        };
        let stop = || callback_error.lock().unwrap_or_else(|e| e.into_inner()).is_some();
        let result = py.detach(|| -> Result<Named, Failure> {
            if let [source] = sources.as_slice() {
                let path = PathBuf::from(source);
                if path.is_dir() && std::fs::read_dir(&path).is_ok_and(|entries| entries.flatten().any(|e| tensorfs_core::format::variant_of(&e.file_name().to_string_lossy()).is_some())) {
                    let tree = self.store.admit_folder(&path)?;
                    let parsed = self.store.tree(&tree)?;
                    let (variant, header) = parsed.variant(variant.as_deref())?;
                    return Ok(Named { header: header.clone(), tree: Some(tree), name: None, release: None, variant: Some(variant.to_owned()) });
                }
            }
            let req = tensorfs_core::import::Request { sources: &sources, converter: converter.as_deref(), variant: variant.as_deref(), access: &access, tmp: &self.store.root().join("tmp"), progress: &notify, stop: Some(&stop) };
            let (header, files) = self.store.import(&req, None)?;
            let mut tree = Tree { files };
            tree.set_variant(variant.as_deref().unwrap_or(""), header.clone());
            let tree = self.store.put_tree(&tree)?;
            Ok(Named { header, tree: Some(tree), name: None, release: None, variant })
        });
        if let Some(error) = callback_error.into_inner().unwrap_or_else(|e| e.into_inner()) { return Err(error); }
        opened(py, &self.store, result.or_raise(py)?)
    }

    #[pyo3(signature = (source, token = None))]
    fn resolve_source(&self, py: Python<'_>, source: String, token: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
        let mut access = client::access();
        access.token = crate::credentials::provider(token)?;
        py.detach(|| -> Result<String, Failure> {
            let source = tensorfs_core::import::remote::parse(&source)?;
            Ok(tensorfs_core::import::remote::list(&source, &access)?.pinned)
        }).or_raise(py)
    }

    /// Pulls `ref` (`org/name[@release]`) and its `variant` from Tensorhub (`hub`, else
    /// tensorhub.com; `token` reads private trees), or from a static `origin`'s URL or directory.
    #[pyo3(signature = (r#ref, variant = None, origin = None, hub = None, token = None))]
    fn pull(
        &self,
        py: Python<'_>,
        r#ref: String,
        variant: Option<String>,
        origin: Option<String>,
        hub: Option<String>,
        token: Option<String>,
    ) -> PyResult<PyCheckpoint> {
        let from = match (&origin, &hub, &token) {
            (Some(origin), None, None) => PullFrom::Origin(origin),
            (None, hub, token) => PullFrom::Hub {
                base: hub.as_deref(),
                token: token.as_deref(),
            },
            _ => return Err(invalid(py, "a static origin takes no hub or token")),
        };
        let pulled = py
            .detach(|| self.store.pull(from, &r#ref, variant.as_deref()))
            .or_raise(py)?;
        opened(py, &self.store, pulled.named)
    }

    /// The cozytensors `ref` names: `org/name[@release]` (its `variant`, else the release's
    /// only one), or `sha256:<hex>` of a tree or a header.
    #[pyo3(signature = (r#ref, variant = None))]
    fn checkpoint(&self, py: Python<'_>, r#ref: &Bound<'_, PyAny>, variant: Option<String>) -> PyResult<PyCheckpoint> {
        let named = named(&self.store, r#ref, variant.as_deref())?;
        opened(py, &self.store, named)
    }

    /// Names `ck` (a stored checkpoint) as `variant` (by default its main dtype) of `repo`'s
    /// release `version`, here: the release's tree gains it.
    #[pyo3(signature = (repo, version, ck, variant = None))]
    fn name(&self, py: Python<'_>, repo: String, version: String, ck: &PyCheckpoint, variant: Option<String>) -> PyResult<PyCheckpoint> {
        let header = ck.stored(py)?.id().clone();
        let named = py
            .detach(|| self.store.set_variant(&repo, &version, variant.as_deref(), &header, &Default::default()))
            .or_raise(py)?;
        opened(py, &self.store, named)
    }

    /// A writer of a new cozytensors, carrying `base`'s components, tensors and assets; with
    /// `key`, it resumes what an interrupted writer of that key wrote.
    #[pyo3(signature = (base = None, key = None))]
    fn writer(&self, py: Python<'_>, base: Option<&Bound<'_, PyAny>>, key: Option<String>) -> PyResult<PyWriter> {
        let base = match base {
            Some(b) => {
                let named = named(&self.store, b, None)?;
                Some(py.detach(|| self.store.checkpoint(&named.header)).or_raise(py)?)
            }
            None => None,
        };
        let w = py
            .detach(|| self.store.writer(base.as_ref(), key.as_deref()))
            .or_raise(py)?;
        Ok(PyWriter {
            store: self.store.clone(),
            w: Mutex::new(Some(w)),
            dest: None,
            variant: None,
            repo_id: None,
        })
    }

    /// Writes `ref` into `dir` as a Diffusers/Transformers folder (encoded tensors keep their
    /// parts unless `dequantize` names F32, BF16 or F16), or with `origin=True` as the static
    /// origin layout instead. Returns the paths written.
    #[pyo3(signature = (r#ref, dir, safetensors = true, dequantize = None, *, origin = false, variant = None))]
    #[allow(clippy::too_many_arguments)]
    fn export(
        &self,
        py: Python<'_>,
        r#ref: &Bound<'_, PyAny>,
        dir: PathBuf,
        safetensors: bool,
        dequantize: Option<String>,
        origin: bool,
        variant: Option<String>,
    ) -> PyResult<Vec<String>> {
        let named = named(&self.store, r#ref, variant.as_deref())?;
        let checkpoint = r#ref.cast::<PyCheckpoint>().ok().map(|c| c.get().stored(py)).transpose()?;
        if origin && dequantize.is_some() {
            return Err(invalid(py, "an origin keeps objects as they are: no dequantize"));
        }
        if !origin && !safetensors {
            return Err(invalid(py, "export writes safetensors, or with origin=True an origin"));
        }
        let dtype = dequantize.map(|d| Dtype::parse(&d.to_ascii_uppercase()));
        let store = &self.store;
        py.detach(|| -> Result<Vec<String>, Failure> {
            if origin {
                return client::export_origin(store, &named, &dir);
            }
            let mut repo = std::collections::BTreeMap::new();
            if let Some(tree) = &named.tree {
                for (path, object) in store.tree(tree)?.repo_files() {
                    repo.insert(path.to_owned(), store.path(object)?);
                }
            }
            let checkpoint = checkpoint.map(Ok).unwrap_or_else(|| store.checkpoint(&named.header))?;
            export::safetensors(&checkpoint, &repo, &dir, dtype.as_ref())
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

/// One cozytensors: its pipeline, components and metadata, read from a store; or only its
/// header (`Checkpoint.parse`), with no chunks or assets to read.
#[pyclass(name = "Checkpoint", module = "tensorfs", frozen)]
pub struct PyCheckpoint {
    header: Header,
    id: Option<ObjectRef>,
    named: Option<Named>,
    state: Mutex<CheckpointState>,
}

struct CheckpointState {
    access: Option<Arc<CheckpointAccess>>,
    components: Option<Py<PyDict>>,
    closed: bool,
}

pub(crate) struct CheckpointAccess {
    pub(crate) checkpoint: Checkpoint,
    remote: Option<Remote>,
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
    /// The config's file name when it isn't `config.json`.
    #[pyo3(get)]
    config_file: String,
    /// `key -> Info`, in construction order; reads go through the engine.
    #[pyo3(get)]
    tensors: Py<PyTensors>,
    /// `name -> (sha256, length)`: run data in the component's folder.
    #[pyo3(get)]
    assets: Py<PyDict>,
    ck: Option<Arc<CheckpointAccess>>,
}

#[pymethods]
impl PyComponent {
    #[getter]
    fn path(&self, py: Python<'_>) -> PyResult<PathBuf> {
        let checkpoint = self.ck.as_ref().ok_or_else(|| invalid(py, "a parsed header has no asset path"))?;
        checkpoint.fetch_assets(py, &self.name, None)?;
        let ck = &checkpoint.checkpoint;
        py.detach(|| ck.component_path(&self.name)).or_raise(py)
    }
    /// Asset `name`'s bytes; a parsed header has none.
    fn asset<'py>(&self, py: Python<'py>, name: String) -> PyResult<Bound<'py, PyBytes>> {
        let Some(ck) = &self.ck else {
            return Err(raise(py, Failure { kind: Kind::Unsupported, detail: "a parsed header has no assets here".into() }));
        };
        let checkpoint = ck;
        checkpoint.fetch_assets(py, &self.name, Some(&name))?;
        let ck = &checkpoint.checkpoint;
        let bytes = py.detach(|| ck.asset(&self.name, &name)).or_raise(py)?;
        Ok(PyBytes::new(py, &bytes))
    }
    fn __repr__(&self) -> String {
        format!("Component({}, {})", self.name, self.cls)
    }
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

fn info(py: Python<'_>, t: &Tensor) -> PyResult<PyInfo> {
    let parts = PyDict::new(py);
    let part = |dtype: &Dtype, shape: &[u64], at: (u64, u64, u64)| -> PyResult<Py<PyPart>> {
        let p = PyPart {
            dtype: dtype.name().into(),
            shape: PyTuple::new(py, shape)?.unbind(),
            chunk: at.0,
            offset: at.1,
            length: at.2,
        };
        Py::new(py, p)
    };
    let encoding = match &t.body {
        Body::Data(at) => {
            parts.set_item("", part(&t.dtype, &t.shape, (at.chunk, at.offset, at.length))?)?;
            None
        }
        Body::Encoded { encoding, parts: stored } => {
            for p in stored {
                parts.set_item(&p.role, part(&p.dtype, &p.shape, (p.chunk, p.offset, p.length))?)?;
            }
            let params = metadata(py, &encoding.params)?.unbind();
            Some(Py::new(py, PyEncoding { scheme: encoding.scheme.clone(), params })?)
        }
    };
    Ok(PyInfo {
        dtype: t.dtype.name().into(),
        shape: PyTuple::new(py, &t.shape)?.unbind(),
        encoding,
        parts: parts.unbind(),
    })
}

pub(crate) struct FetchTensor<'a> {
    pub key: &'a str,
    pub role: Option<&'a str>,
    pub range: Option<Range<u64>>,
}

impl PyCheckpoint {
    fn new(header: Header, stored: Option<Checkpoint>, named: Option<Named>) -> PyCheckpoint {
        let id = stored.as_ref().map(|ck| ck.id().clone());
        let access = stored.map(|checkpoint| Arc::new(CheckpointAccess { checkpoint, remote: None }));
        PyCheckpoint { header, id, named, state: Mutex::new(CheckpointState { access, components: None, closed: false }) }
    }

    fn set_remote(&mut self, remote: Remote) {
        let state = self.state.get_mut().unwrap_or_else(|e| e.into_inner());
        let access = Arc::get_mut(state.access.as_mut().expect("a newly opened checkpoint")).expect("not exposed to Python yet");
        access.remote = Some(remote);
    }

    pub(crate) fn access(&self, py: Python<'_>) -> PyResult<Arc<CheckpointAccess>> {
        let (closed, access) = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            (state.closed, state.access.clone())
        };
        if closed { return Err(invalid(py, "the checkpoint is closed")); }
        access.ok_or_else(|| raise(py, Failure { kind: Kind::Unsupported,
            detail: "a parsed header has no store: its chunks and assets are not here".into() }))
    }

    fn ensure_open(&self, py: Python<'_>) -> PyResult<()> {
        let closed = self.state.lock().unwrap_or_else(|e| e.into_inner()).closed;
        if closed { Err(invalid(py, "the checkpoint is closed")) } else { Ok(()) }
    }

    pub(crate) fn native_view(&self, py: Python<'_>) -> PyResult<tensorfs_core::io::view::View> {
        let (closed, access) = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            (state.closed, state.access.clone())
        };
        if closed { return Err(invalid(py, "the checkpoint is closed")); }
        Ok(match access {
            Some(access) => tensorfs_core::io::view::View::from(&access.checkpoint).with_fetch(move |objects| {
                if let Some(remote) = &access.remote { remote.fetch(objects, 0) } else { Ok(()) }
            }),
            None => tensorfs_core::io::view::View::parsed(self.header.clone()),
        })
    }

    pub(crate) fn stored(&self, py: Python<'_>) -> PyResult<Checkpoint> {
        Ok(self.access(py)?.checkpoint.clone())
    }

    fn tensor_dict(&self, py: Python<'_>, c: &tensorfs_core::format::Component) -> PyResult<Py<PyDict>> {
        let tensors = PyDict::new(py);
        for t in &c.tensors { tensors.set_item(&t.key, Py::new(py, info(py, t)?)?)?; }
        Ok(tensors.unbind())
    }
}

impl CheckpointAccess {
    pub(crate) fn fetch_components(&self, py: Python<'_>, components: Option<&[String]>, priority: u64) -> PyResult<()> {
        if let Some(remote) = &self.remote {
            let objects = tensorfs_core::store2::wanted(self.checkpoint.header(), components).or_raise(py)?;
            py.detach(|| remote.fetch(&objects, priority)).or_raise(py)?;
        }
        Ok(())
    }

    pub(crate) fn fetch_tensors(&self, py: Python<'_>, component: &str, reads: &[FetchTensor]) -> PyResult<()> {
        let Some(remote) = &self.remote else { return Ok(()) };
        let ck = &self.checkpoint;
        let mut objects = BTreeMap::new();
        for read in reads {
            let tensor = ck.tensor(component, read.key).or_raise(py)?;
            let mut found = read.role.is_none();
            for (role, at) in tensor.locations() {
                if read.role.is_some_and(|wanted| wanted != role) { continue; }
                found = true;
                let range = read.range.clone().unwrap_or(0..at.length);
                if range.start > range.end || range.end > at.length {
                    return Err(invalid(py, format!("{component}/{}: range {range:?} exceeds {}", read.key, at.length)));
                }
                let mut offset = 0;
                for span in self.checkpoint.header().spans(&at) {
                    if offset < range.end && offset + span.length > range.start {
                        let object = &self.checkpoint.header().chunks[span.chunk];
                        objects.insert(object.sha256.clone(), object.clone());
                    }
                    offset += span.length;
                }
            }
            if !found { return Err(invalid(py, format!("{component}/{} has no part {:?}", read.key, read.role))); }
        }
        if !objects.is_empty() {
            let objects: Vec<ObjectRef> = objects.into_values().collect();
            py.detach(|| remote.fetch(&objects, 0)).or_raise(py)?;
        }
        Ok(())
    }

    fn fetch_assets(&self, py: Python<'_>, component: &str, name: Option<&str>) -> PyResult<()> {
        let component = self.checkpoint.header().component(component).ok_or_else(|| invalid(py, "no such component"))?;
        let objects: Vec<_> = match name {
            Some(name) => vec![component.assets.get(name).ok_or_else(|| invalid(py, "no such asset"))?.clone()],
            None => component.assets.values().cloned().collect(),
        };
        if let Some(remote) = &self.remote {
            py.detach(|| remote.fetch(&objects, 0)).or_raise(py)?;
        }
        Ok(())
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

    fn __exit__(&self, exc_type: &Bound<'_, PyAny>, exc_value: &Bound<'_, PyAny>, traceback: &Bound<'_, PyAny>) {
        let _ = (exc_type, exc_value, traceback);
        self.close();
    }

    fn verify(&self, py: Python<'_>) -> PyResult<Vec<(String, u64)>> {
        let checkpoint = self.stored(py)?;
        let bad = py.detach(|| checkpoint.verify()).or_raise(py)?;
        Ok(bad.into_iter().map(|o| (o.sha256, o.length)).collect())
    }

    fn open_chunk(&self, py: Python<'_>, index: usize) -> PyResult<i32> {
        let access = self.access(py)?;
        let object = access.checkpoint.header().chunks.get(index).ok_or_else(|| invalid(py, format!("no chunk {index}")))?;
        if let Some(remote) = &access.remote { py.detach(|| remote.fetch(std::slice::from_ref(object), 0)).or_raise(py)?; }
        let file = py.detach(|| access.checkpoint.chunk(index)).or_raise(py)?;
        #[cfg(unix)]
        { Ok(std::os::fd::IntoRawFd::into_raw_fd(file)) }
        #[cfg(windows)]
        {
            use std::os::windows::io::{AsRawHandle, IntoRawHandle};
            let fd = py.import("msvcrt")?.call_method1("open_osfhandle", (file.as_raw_handle() as usize, 0x8000))?.extract()?;
            let _ = file.into_raw_handle();
            Ok(fd)
        }
    }

    /// A `cozytensors/1` header's pipeline, components and metadata, with no store behind them.
    #[staticmethod]
    fn parse(py: Python<'_>, header: &[u8]) -> PyResult<PyCheckpoint> {
        let header = Header::from_cbor(header).or_raise(py)?;
        Ok(PyCheckpoint::new(header, None, None))
    }

    /// `sha256:<hex>` of the header: the cozytensors' id. None for a parsed header.
    #[getter]
    fn id(&self) -> Option<String> {
        self.id.as_ref().map(ObjectRef::id)
    }

    #[getter]
    fn header_ref(&self) -> Option<(String, u64)> {
        self.id.as_ref().map(|id| (id.sha256.clone(), id.length))
    }
    #[getter]
    fn tree_ref(&self) -> Option<(String, u64)> {
        self.named.as_ref()?.tree.as_ref().map(|tree| (tree.sha256.clone(), tree.length))
    }

    /// The tree, repo, release and variant naming it, when one does.
    #[getter]
    fn tree(&self) -> Option<String> {
        self.named.as_ref()?.tree.as_ref().map(|t| t.id())
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.named.as_ref()?.name.as_deref()
    }
    #[getter]
    fn release(&self) -> Option<&str> {
        self.named.as_ref()?.release.as_deref()
    }
    #[getter]
    fn variant(&self) -> Option<&str> {
        self.named.as_ref()?.variant.as_deref()
    }

    #[getter]
    fn repo_id(&self) -> Option<&str> { self.name() }
    #[getter]
    fn revision(&self) -> Option<&str> { self.release() }
    #[getter]
    fn path(&self, py: Python<'_>) -> PyResult<PathBuf> {
        let access = self.access(py)?;
        access.fetch_components(py, None, 0)?;
        let ck = &access.checkpoint;
        py.detach(|| ck.path_for(self.variant().unwrap_or(""))).or_raise(py)
    }

    #[getter]
    fn pipeline(&self, py: Python<'_>) -> PyResult<Option<PyPipeline>> {
        (self.header.pipeline())
            .map(|p| Ok(PyPipeline { cls: p.class.clone(), config: config_dict(py, &p.config)?.unbind() }))
            .transpose()
    }

    /// `components[name] -> Component`, in construction order.
    #[getter]
    fn components(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let access = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if state.closed { drop(state); return Err(invalid(py, "the checkpoint is closed")); }
            if let Some(components) = &state.components { return Ok(components.clone_ref(py)); }
            state.access.clone()
        };
        let out = PyDict::new(py);
        for c in self.header.components() {
            let assets = PyDict::new(py);
            for (name, o) in &c.assets { assets.set_item(name, (o.sha256.clone(), o.length))?; }
            let component = PyComponent {
                name: c.name.clone(), cls: c.class.clone(), config: config_dict(py, &c.config)?.unbind(),
                config_file: c.config_file.clone().unwrap_or_else(|| "config.json".into()),
                tensors: Py::new(py, PyTensors::new(&c.name, self.tensor_dict(py, c)?, access.clone()))?,
                assets: assets.unbind(), ck: access.clone(),
            };
            out.set_item(&c.name, Py::new(py, component)?)?;
        }
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed { drop(state); return Err(invalid(py, "the checkpoint is closed")); }
        let components = state.components.get_or_insert_with(|| out.unbind());
        Ok(components.clone_ref(py))
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

    /// The chunks and assets of `components` (all when None) not here: `(sha256, length)`.
    #[pyo3(signature = (components = None))]
    fn missing(&self, py: Python<'_>, components: Option<Vec<String>>) -> PyResult<Vec<(String, u64)>> {
        let ck = self.stored(py)?;
        let missing = py.detach(|| ck.missing(components.as_deref())).or_raise(py)?;
        Ok(missing.into_iter().map(|o| (o.sha256, o.length)).collect())
    }

    #[pyo3(signature = (components = None, *, priority = 0, progress = None))]
    fn fetch(&self, py: Python<'_>, components: Option<Vec<String>>, priority: u64, progress: Option<Py<PyAny>>) -> PyResult<()> {
        let access = self.access(py)?;
        let objects = tensorfs_core::store2::wanted(access.checkpoint.header(), components.as_deref()).or_raise(py)?;
        let callback_error = Mutex::new(None);
        let report = |held, total| -> Result<(), Failure> {
            if let Some(callback) = &progress {
                if let Err(error) = Python::attach(|py| callback.call1(py, ("fetch", held, total))) {
                    *callback_error.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    return Err(Failure { kind: Kind::Invalid, detail: "the download progress callback failed".into() });
                }
            }
            Ok(())
        };
        let result = py.detach(|| -> Result<(), Failure> {
            if let Some(remote) = &access.remote {
                return remote.fetch_progress(&objects, priority, &report);
            }
            let total = objects.iter().map(|object| object.length).sum();
            report(0, total)?;
            let mut held = 0;
            for object in &objects {
                if !access.checkpoint.holds(object)? {
                    return Err(Failure { kind: Kind::NotFound, detail: format!("{} is absent and has no download endpoint", object.id()) });
                }
                held += object.length;
                report(held, total)?;
            }
            Ok(())
        });
        if let Some(error) = callback_error.into_inner().unwrap_or_else(|e| e.into_inner()) { return Err(error); }
        result.or_raise(py)
    }

    /// The header's loading metadata (`source`, `filled`, `safetensors_metadata`).
    #[getter]
    fn metadata<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        metadata(py, &self.header.metadata)
    }


}

/// Groups `ck`'s tensors into `<component>/<key>` regions for a weight plane: tensors sharing a
/// chunk stay together, and neighbours merge while their chunks total at most `max_bytes`,
/// never across a `<component>/<key prefix>` boundary (a block) unless a shared chunk does.
/// `only` limits the work and the result to the tensors a load wants.
#[pyfunction]
#[pyo3(signature = (ck, max_bytes = 64 << 20, boundaries = None, only = None))]
pub(crate) fn regions(ck: &PyCheckpoint, max_bytes: u64, boundaries: Option<Vec<String>>, only: Option<Vec<String>>) -> Vec<Vec<String>> {
    client::regions(&ck.header, max_bytes, &boundaries.unwrap_or_default(), &only.unwrap_or_default())
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
pub(crate) struct DeviceArray(Option<tensorfs_core::io::DeviceBuffer>);

impl DeviceArray {
    pub(crate) fn new(buffer: tensorfs_core::io::DeviceBuffer) -> Arc<Self> {
        Arc::new(Self(Some(buffer)))
    }
}

impl std::ops::Deref for DeviceArray {
    type Target = tensorfs_core::io::DeviceBuffer;
    fn deref(&self) -> &Self::Target { self.0.as_ref().expect("live device allocation") }
}

impl Drop for DeviceArray {
    fn drop(&mut self) {
        // cuMemFree may synchronize pending work. Move its actual final owner to the
        // native cleanup worker, also when a foreign DLPack consumer calls the deleter.
        crate::cleanup::defer(self.0.take());
    }
}

#[derive(Clone)]
pub(crate) enum ArrayBuffer {
    Host(Arc<Aligned>),
    Device(Arc<DeviceArray>),
}

impl ArrayBuffer {
    pub(crate) fn len(&self) -> usize {
        match self { Self::Host(b) => b.len, Self::Device(b) => b.len() }
    }
    pub(crate) fn host(&self) -> Option<&[u8]> {
        match self { Self::Host(b) => Some(b.bytes()), Self::Device(_) => None }
    }
    fn ptr(&self) -> *mut c_void {
        match self { Self::Host(b) => b.ptr() as _, Self::Device(b) => b.ptr() as usize as _ }
    }
    fn device(&self) -> (i32, i32) {
        match self { Self::Host(_) => (KDL_CPU, 0), Self::Device(b) => (KDL_CUDA, b.device()) }
    }
    fn duplicate(&self) -> Result<Self, Failure> {
        use tensorfs_core::io::{self, Copy, Dst, ReadOpts, Src};
        Ok(match self {
            Self::Host(b) => {
                let mut copy = Aligned::new(b.len);
                copy.bytes_mut().copy_from_slice(b.bytes());
                Self::Host(Arc::new(copy))
            }
            Self::Device(b) => {
                let copy = io::DeviceBuffer::new(b.device(), b.len())?;
                let request = Copy { src: Src::Device { device: b.device(), ptr: b.ptr(), stream: 0 }, dst: Dst::Device { device: b.device(), ptr: copy.ptr(), len: copy.len(), stream: 0 } };
                // SAFETY: both allocations remain held until the copy completes.
                unsafe { io::read(&[request], &ReadOpts::default()) }.map_err(|e| Failure::from(e.source))?;
                Self::Device(DeviceArray::new(copy))
            }
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
        if let Some(bytes) = self.buf.host() { return Ok(PyBytes::new(py, bytes)); }
        let ArrayBuffer::Device(buffer) = &self.buf else { unreachable!() };
        let bytes = py.detach(|| -> Result<Vec<u8>, Failure> {
            use tensorfs_core::io::{self, Copy, Dst, ReadOpts, Src};
            let mut bytes = vec![0; buffer.len()];
            let request = Copy { src: Src::Device { device: buffer.device(), ptr: buffer.ptr(), stream: 0 }, dst: Dst::host(&mut bytes) };
            unsafe { io::read(&[request], &ReadOpts::default()) }.map_err(|e| Failure::from(e.source))?;
            Ok(bytes)
        }).or_raise(py)?;
        Ok(PyBytes::new(py, &bytes))
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
        if let Some(stream) = stream {
            let value = stream.extract::<i64>()?;
            if device.0 == KDL_CPU || value == 0 || value < -1 {
                return Err(invalid(py, "DLPack stream must be a CUDA stream, 1/2 for default streams, or -1"));
            }
            if device.0 == KDL_CUDA && copy == Some(true) {
                return Err(pyo3::exceptions::PyBufferError::new_err("CUDA DLPack copy=True with an explicit consumer stream is unsupported"));
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
        let shape = self.shape.iter().map(|&dimension| i64::try_from(dimension).map_err(|_| invalid(py, "DLPack dimension exceeds i64"))).collect::<PyResult<Vec<_>>>()?;
        if self.shape.len() > i32::MAX as usize { return Err(invalid(py, "DLPack rank exceeds i32")); }
        let buf = if copy == Some(true) {
            py.detach(|| self.buf.duplicate()).or_raise(py)?
        } else {
            self.buf.clone()
        };
        let data = buf.ptr();
        let held = Box::into_raw(Box::new(Held {
            shape,
            _buf: buf,
        }));
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

/// The shared contiguous DLPack view used by both the Writer and destination reads.
/// Callers move/drop this owner in a detached closure: the stream drains before the capsule
/// releases its producer allocation, including when an operation refuses the view.
pub(crate) struct Dlpack {
    pub(crate) stream: Option<tensorfs_core::io::cuda::Stream>,
    pub(crate) capsule: Py<PyAny>,
    pub(crate) device: (i32, i32),
    pub(crate) dtype: Dtype,
    pub(crate) shape: Vec<u64>,
    pub(crate) len: usize,
    pub(crate) address: usize,
}

impl Dlpack {
    pub(crate) fn of(py: Python<'_>, data: &Bound<'_, PyAny>, dtype: Option<String>, shape: Option<Vec<u64>>) -> PyResult<Self> {
        let device: (i32, i32) = data.call_method0("__dlpack_device__")?.extract()?;
        let stream = match device {
            (KDL_CPU, _) => None,
            (KDL_CUDA, device) if device >= 0 => Some(py.detach(|| tensorfs_core::io::cuda::Stream::new(device)).or_raise(py)?),
            _ => return Err(raise(py, Failure { kind: Kind::Unsupported, detail: "expected a CPU or CUDA DLPack array".into() })),
        };
        let exported = (|| {
            if let Some(stream) = &stream {
                let kwargs = PyDict::new(py);
                kwargs.set_item("stream", stream.raw())?;
                data.call_method("__dlpack__", (), Some(&kwargs))
            } else { data.call_method0("__dlpack__") }
        })();
        let capsule = match exported {
            Ok(capsule) => capsule.unbind(),
            Err(error) => { py.detach(move || drop(stream)); return Err(error); }
        };
        let parsed = (|| {
            let capsule = capsule.bind(py);
            if unsafe { ffi::PyCapsule_IsValid(capsule.as_ptr(), DLTENSOR.as_ptr()) } != 1 {
                return Err(invalid(py, "__dlpack__ did not return a dltensor capsule"));
            }
            let managed = unsafe { ffi::PyCapsule_GetPointer(capsule.as_ptr(), DLTENSOR.as_ptr()) as *const DLManagedTensor };
            let tensor = unsafe { &(*managed).dl_tensor };
            if tensor.device.device_type != device.0 || (device.0 == KDL_CUDA && tensor.device.device_id != device.1)
                || !(0..=32).contains(&tensor.ndim) || tensor.dtype.lanes != 1 {
                return Err(invalid(py, "DLPack device or scalar-lane tensor metadata disagrees"));
            }
            if tensor.ndim != 0 && tensor.shape.is_null() {
                return Err(invalid(py, "DLPack tensor has no shape"));
            }
            let dims = if tensor.ndim == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) } };
            let inferred_shape: Vec<u64> = dims.iter().map(|d| u64::try_from(*d).map_err(|_| invalid(py, "DLPack shape is negative"))).collect::<PyResult<_>>()?;
            let mut stride = 1i64;
            for i in (0..dims.len()).rev() {
                if !tensor.strides.is_null() && dims[i] > 1 && unsafe { *tensor.strides.add(i) } != stride {
                    return Err(invalid(py, "expected a contiguous DLPack array"));
                }
                stride = stride.checked_mul(dims[i]).ok_or_else(|| invalid(py, "DLPack shape overflows"))?;
            }
            let inferred_dtype = Dtype::known().find(|d| dl_type(d) == Some((tensor.dtype.code, tensor.dtype.bits)))
                .ok_or_else(|| invalid(py, "DLPack dtype is unsupported"))?;
            let dtype = dtype.map(|d| Dtype::parse(&d)).unwrap_or_else(|| inferred_dtype.clone());
            let shape = shape.unwrap_or_else(|| inferred_shape.clone());
            if dtype != inferred_dtype || shape != inferred_shape {
                return Err(invalid(py, "explicit dtype or shape disagrees with the DLPack array"));
            }
            let len = dtype.byte_len(&shape).and_then(|n| usize::try_from(n).ok()).ok_or_else(|| invalid(py, "DLPack byte size overflows"))?;
            let offset = usize::try_from(tensor.byte_offset).map_err(|_| invalid(py, "DLPack byte offset overflows"))?;
            let address = (tensor.data as usize).checked_add(offset).ok_or_else(|| invalid(py, "DLPack address overflows"))?;
            if len != 0 && address == 0 { return Err(invalid(py, "DLPack tensor has no data")); }
            if address.checked_add(len).is_none() { return Err(invalid(py, "DLPack address range overflows")); }
            Ok((dtype, shape, len, address))
        })();
        let (dtype, shape, len, address) = match parsed {
            Ok(parsed) => parsed,
            Err(error) => { py.detach(move || { drop(stream); drop(capsule); }); return Err(error); }
        };
        Ok(Self { stream, capsule, device, dtype, shape, len, address })
    }

    pub(crate) fn destination(&self) -> tensorfs_core::io::Dst {
        match &self.stream {
            Some(stream) => tensorfs_core::io::Dst::Device { device: self.device.1, ptr: self.address as u64, len: self.len, stream: stream.raw() },
            None => tensorfs_core::io::Dst::Host { ptr: self.address as *mut u8, len: self.len },
        }
    }
}

// ---------------------------------------------------------------- the writer

/// One cozytensors out, component by component (`component`), sealed by `commit`.
#[pyclass(name = "Writer", module = "tensorfs", frozen)]
pub struct PyWriter {
    store: Store,
    w: Mutex<Option<Writer>>,
    dest: Option<PathBuf>,
    variant: Option<String>,
    repo_id: Option<String>,
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
    #[new]
    #[pyo3(signature = (dest = None, *, variant = None, base = None, resume = None, cache_dir = None, repo_id = None, endpoint = None, token = None, ca_der = None))]
    #[allow(clippy::too_many_arguments)]
    fn new(py: Python<'_>, dest: Option<PathBuf>, variant: Option<String>, base: Option<&PyCheckpoint>, resume: Option<String>, cache_dir: Option<PathBuf>, repo_id: Option<String>, endpoint: Option<String>, token: Option<&Bound<'_, PyAny>>, ca_der: Option<Vec<u8>>) -> PyResult<Self> {
        let base = base.map(|b| b.stored(py)).transpose()?;
        let root = tensorfs_core::home::resolve(cache_dir.as_deref()).or_raise(py)?;
        if dest.is_some() && repo_id.is_some() { return Err(invalid(py, "a Writer names a folder or an endpoint repo")); }
        let endpoint = if repo_id.is_some() {
            let url = endpoint.unwrap_or_else(fetch::token::default_endpoint);
            let client = crate::credentials::client(&url, ca_der.as_deref()).or_raise(py)?;
            Some(Arc::new(Endpoint::new(&client, &url, crate::credentials::provider(token)?).or_raise(py)?))
        } else { None };
        let (store, w) = py.detach(|| -> Result<_, Failure> {
            let store = Store::open(&root)?.with_cache();
            let w = match (&dest, &repo_id, endpoint) {
                (Some(dest), _, _) => Writer::folder(dest, variant.as_deref().unwrap_or(""), base.as_ref(), resume.is_some())?,
                (_, Some(name), Some(endpoint)) => {
                    let runtime = fetch::runtime()?;
                    let stream = Arc::new(fetch::Stream::new(endpoint, runtime.handle().clone()));
                    store.streaming_writer(stream, name, variant.as_deref().unwrap_or(""), base.as_ref())?
                }
                _ => store.writer(base.as_ref(), resume.as_deref())?,
            };
            Ok((store, w))
        }).or_raise(py)?;
        Ok(Self { store, w: Mutex::new(Some(w)), dest, variant, repo_id })
    }
    /// What `model_index.json` held: `cls` (`diffusers.<Class>`) and its `config`.
    fn pipeline(&self, py: Python<'_>, cls: String, config: &Bound<'_, PyDict>) -> PyResult<()> {
        let config = config_map(config)?;
        self.with(py, |w| {
            w.pipeline(&cls, config);
            Ok(())
        })
    }

    /// Declares component `name` (again: its class and config change, its tensors stay) and
    /// hands back its writer.
    #[pyo3(signature = (name, cls = "", config = None, config_file = None))]
    fn component(
        slf: &Bound<'_, Self>,
        name: String,
        cls: &str,
        config: Option<&Bound<'_, PyDict>>,
        config_file: Option<String>,
    ) -> PyResult<PyComponentWriter> {
        let py = slf.py();
        let config = config.map(config_map).transpose()?.unwrap_or_default();
        slf.get().with(py, |w| {
            w.component(&name, cls, config, config_file.as_deref())?;
            Ok(())
        })?;
        Ok(PyComponentWriter { w: slf.clone().unbind(), name })
    }

    /// A repo file (README, LICENSE): pinned with the header, never inside it.
    fn repo_file(&self, py: Python<'_>, path: String, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        self.with(py, |w| Ok(w.repo_file(&path, bytes.slice())?))
    }

    /// A header metadata field; it overrides the base's.
    fn set_metadata(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = from_py(value)?;
        self.with(py, |w| {
            w.set_metadata(&key, value);
            Ok(())
        })
    }

    /// Written already (or by an interrupted writer of this key); base tensors don't count.
    fn has(&self, py: Python<'_>, component: &str, key: &str) -> bool {
        py.detach(|| {
            let guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
            guard.as_ref().is_some_and(|w| w.has(component, key))
        })
    }

    /// Lays out the chunks and seals the header; it is pinned under the writer's key, or its
    /// digest. Returns the new checkpoint.
    fn commit(&self, py: Python<'_>) -> PyResult<PyCheckpoint> {
        let (result, retained, published, files) = py.detach(|| {
            let mut guard = self.w.lock().unwrap_or_else(|e| e.into_inner());
            let Some(writer) = guard.as_mut() else {
                return (Err(Failure { kind: Kind::Invalid, detail: "the writer has committed or been abandoned".into() }), None, None, Default::default());
            };
            let result = writer.commit().map_err(Failure::from);
            let retained = writer.committed().cloned();
            let published = writer.published().cloned();
            let files = writer.repo_files().clone();
            if result.is_ok() { guard.take(); }
            (result, retained, published, files)
        });
        let header = match result {
            Ok(header) => header,
            Err(failure) => {
                let err = raise(py, failure);
                if let Some(header) = retained {
                    let ck = opened(py, &self.store, Named { header, tree: None, name: None, release: None, variant: self.variant.clone() })?;
                    err.value(py).setattr("checkpoint", Py::new(py, ck)?)?;
                }
                return Err(err);
            }
        };
        let tree = if self.repo_id.is_some() {
            let mut tree = Tree { files };
            tree.set_variant(self.variant.as_deref().unwrap_or(""), header.clone());
            let local = py.detach(|| self.store.put_tree(&tree)).or_raise(py)?;
            if published.as_ref() != Some(&local) { return Err(invalid(py, "the uploaded tree differs from the local tree")); }
            published
        } else { None };
        if let Some(dest) = &self.dest {
            let ck = py.detach(|| Checkpoint::folder(dest, Some(self.variant.as_deref().unwrap_or("")), Some(&self.store))).or_raise(py)?;
            return Ok(PyCheckpoint::new(ck.header().clone(), Some(ck), None));
        }
        let named = Named { header, tree, name: self.repo_id.clone(), release: None, variant: self.variant.clone() };
        opened(py, &self.store, named)
    }

    /// Ends the writer without a checkpoint.
    fn abandon(&self, py: Python<'_>) {
        py.detach(|| {
            let writer = self.w.lock().unwrap_or_else(|e| e.into_inner()).take();
            if let Some(writer) = writer { writer.abandon(); }
        })
    }
}

/// One component of a `Writer`.
#[pyclass(name = "ComponentWriter", module = "tensorfs", frozen)]
pub struct PyComponentWriter {
    w: Py<PyWriter>,
    #[pyo3(get)]
    name: String,
}

#[pymethods]
impl PyComponentWriter {
    #[pyo3(signature = (key, source, as_key = None))]
    fn copy_component(&self, py: Python<'_>, key: String, source: &PyComponent, as_key: Option<String>) -> PyResult<()> {
        let access = source.ck.as_ref().ok_or_else(|| invalid(py, "a parsed component has no bytes"))?;
        access.fetch_tensors(py, &source.name, &[FetchTensor { key: &key, role: None, range: None }])?;
        let into = as_key.unwrap_or_else(|| key.clone());
        self.w.get().with(py, |w| Ok(w.get(&self.name).copy_as(&into, &access.checkpoint, &source.name, &key)?))
    }
    /// A CPU/CUDA DLPack array, or a buffer with an explicit dtype and shape.
    #[pyo3(signature = (key, data, *, dtype = None, shape = None))]
    fn put_array(&self, py: Python<'_>, key: String, data: &Bound<'_, PyAny>, dtype: Option<String>, shape: Option<Vec<u64>>) -> PyResult<()> {
        if !data.hasattr("__dlpack__")? {
            let dtype = dtype.ok_or_else(|| invalid(py, "a buffer needs dtype"))?;
            let shape = shape.ok_or_else(|| invalid(py, "a buffer needs shape"))?;
            return self.put(py, key, dtype, shape, data);
        }
        let view = Dlpack::of(py, data, dtype, shape)?;
        self.w.get().with(py, move |w| {
            let Dlpack { stream, capsule, device, dtype, shape, len, address } = view;
            if let Some(stream) = stream {
                let mut source = crate::dlpack_source::DeviceSource::new(capsule, stream, device.1, address as u64, len)?;
                return Ok(w.get(&self.name).put(&key, &dtype, &shape, Data::Plain(Source::Read(&mut source)))?);
            }
            let _capsule = capsule;
            let mut bytes = if len == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(address as *const u8, len) } };
            Ok(w.get(&self.name).put(&key, &dtype, &shape, Data::Plain(Source::Read(&mut bytes)))?)
        })
    }
    /// A plain tensor: `data` is any C-contiguous buffer of exactly its bytes.
    fn put(&self, py: Python<'_>, key: String, dtype: String, shape: Vec<u64>, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        let dtype = Dtype::parse(&dtype);
        self.w.get().with(py, |w| {
            let mut src = bytes.slice();
            Ok(w.get(&self.name).put(&key, &dtype, &shape, Data::Plain(Source::Read(&mut src)))?)
        })
    }

    /// An encoded tensor: `parts` maps each role to `(dtype, shape, data)`.
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
            Some(p) => match from_py(p.as_any())? {
                Value::Map(map) => map,
                _ => unreachable!("a dict converts to a map"),
            },
            None => Metadata::new(),
        };
        let mut held = Vec::new();
        for (role, part) in parts.iter() {
            let (dtype, shape, data): (String, Vec<u64>, Bound<'_, PyAny>) = part.extract()?;
            held.push((role.extract::<String>()?, Dtype::parse(&dtype), shape, Bytes::of(&data)?));
        }
        let dtype = Dtype::parse(&dtype);
        self.w.get().with(py, |w| {
            let mut slices: Vec<&[u8]> = held.iter().map(|h| h.3.slice()).collect();
            let parts = (held.iter().zip(slices.iter_mut()))
                .map(|((role, dtype, shape, _), data)| PartData {
                    role: role.clone(),
                    dtype: dtype.clone(),
                    shape: shape.clone(),
                    data: Source::Read(data),
                })
                .collect();
            let encoding = Encoding { scheme, params };
            Ok(w.get(&self.name).put(&key, &dtype, &shape, Data::Encoded { encoding, parts })?)
        })
    }

    /// Takes `from_component/from_key` (by default this component and `key`) from `source`, a
    /// stored checkpoint, as `key`; large parts keep their objects.
    #[pyo3(signature = (key, source, from_component = None, from_key = None))]
    fn copy(
        &self,
        py: Python<'_>,
        key: String,
        source: &PyCheckpoint,
        from_component: Option<String>,
        from_key: Option<String>,
    ) -> PyResult<()> {
        let from = source.stored(py)?;
        let fc = from_component.unwrap_or_else(|| self.name.clone());
        let fk = from_key.unwrap_or_else(|| key.clone());
        self.w.get().with(py, |w| Ok(w.get(&self.name).copy_as(&key, &from, &fc, &fk)?))
    }

    /// Run data in the component's folder (`tokenizer.json`), by its path there.
    fn asset(&self, py: Python<'_>, name: String, data: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = Bytes::of(data)?;
        self.w.get().with(py, |w| Ok(w.get(&self.name).asset(&name, bytes.slice())?))
    }

    /// The tensor order when the calls are not it: these keys first, then the rest as written.
    fn order(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<()> {
        self.w.get().with(py, |w| {
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            w.get(&self.name).set_order(&keys);
            Ok(())
        })
    }

    /// Written already (or by an interrupted writer of this key).
    fn has(&self, py: Python<'_>, key: &str) -> bool {
        self.w.get().has(py, &self.name, key)
    }
}

// ---------------------------------------------------------------- the module

#[pyfunction]
fn _kernel_binary<'py>(py: Python<'py>, family: u32, core: &str, dtype: &str) -> PyResult<(Bound<'py, PyBytes>, &'static str, u32, u32)> {
    let binary = tensorfs_core::kernels::BINARIES.iter()
        .find(|b| b.family == family && b.kind == core && b.dtype == dtype)
        .ok_or_else(|| raise(py, Failure { kind: Kind::Unsupported, detail: format!("no prebuilt sm{family}/{core}/{dtype} kernel") }))?;
    Ok((PyBytes::new(py, binary.image), binary.symbol, binary.shared, binary.warps))
}

pub fn install(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStore>()?;
    m.add_class::<PyCheckpoint>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyComponent>()?;
    m.add_class::<PyComponentWriter>()?;
    m.add_class::<PyInfo>()?;
    m.add_class::<PyEncoding>()?;
    m.add_class::<PyPart>()?;
    m.add_class::<PyArray>()?;
    m.add_class::<PyTensors>()?;
    m.add_class::<PyEncoded>()?;
    m.add_class::<PyWriter>()?;
    m.add_function(wrap_pyfunction!(_kernel_binary, m)?)?;
    Ok(())
}
