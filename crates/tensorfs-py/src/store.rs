//! The store, the lease, the read plan and the slot ring — marshalled, never reimplemented.
//!
//! Two properties this module exists to preserve across the boundary:
//!
//! 1. **The caller owns the memory.** Every read fills a buffer Python already owns, through
//!    the stable-ABI `Py_buffer`. TensorFS allocates no destination, and the runtime's one
//!    budgeted pinned pool stays the runtime's. Nothing is copied to hand bytes to Python.
//! 2. **The GIL is not held while bytes move.** Every read runs inside `Python::detach`, so a
//!    reader pool of N genuinely reads N-wide from a Python caller.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::catalog::{Catalog, WriterGuard};
use tensorfs_core::checkpoint;
use tensorfs_core::derived::{
    self, ComponentDeclaration, ConfigDeclaration, Declaration, PartDeclaration, Source,
    TensorDeclaration,
};
use tensorfs_core::err::{refuse, Code, Refusal, Result as TResult};
use tensorfs_core::header::Header;
use tensorfs_core::ids::{prefixed, Doc, ObjectRef};
use tensorfs_core::ingest::source::{
    self as model_source, ModelSourceProfile, PrepareModelSource, VerifiedModelSourceFile,
};
use tensorfs_core::manifest::{Entry, Manifest};
use tensorfs_core::meta::{Hold, Meta};
use tensorfs_core::project;
use tensorfs_core::read::{self, ObjectRange, ReadLease as CoreLease, Source as ReadSource};
use tensorfs_core::receipt::Receipts;
use tensorfs_core::repo_cache::{CacheKind, RepoObjectCache as CoreRepoObjectCache};
use tensorfs_core::repository::{Mutation, ReleaseLane, Repository, RepositoryName};
use tensorfs_core::storage;
use tensorfs_core::store::{Fault, Store as CoreStore, Verdict};

use crate::docs::marshal;
use crate::errors::{to_py, IntoPy};

// ---------------------------------------------------------------- buffers

/// The caller's writable buffer. The one `unsafe` in this crate, and it is the point of the
/// crate: a `readinto` that copied would defeat the whole design.
///
/// Soundness rests on the slot discipline, not on hope: the exporter is kept alive by this
/// `PyBuffer` for the whole call, the GIL is released only after the slice exists, and no
/// Python code can legally touch a slot while it is outstanding — a slot returns to the ring
/// through `release()` and through nothing else.
struct Writable {
    buf: PyBuffer<u8>,
}

impl Writable {
    fn new(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Writable> {
        let buf = PyBuffer::<u8>::get(obj)?;
        if buf.readonly() {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::BUFFER_SIZE,
                    detail: "the destination is read-only — TensorFS fills a buffer the CALLER \
                             owns (a bytearray, a memoryview of one, a pinned host buffer), \
                             never a `bytes`"
                        .into(),
                },
            ));
        }
        if !buf.is_c_contiguous() {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::BUFFER_SIZE,
                    detail: "the destination is not C-contiguous".into(),
                },
            ));
        }
        Ok(Writable { buf })
    }

    fn slice(&mut self) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(self.buf.buf_ptr() as *mut u8, self.buf.len_bytes())
        }
    }
}

/// The caller's byte SOURCE — the mirror of `Writable`, and the reason a worker can pull a
/// multi-gigabyte artifact out of an HTTP response without a second full copy on disk or in
/// RAM.
///
/// The transport adapter (boundaries.md §9: `HTTP/R2 fetch with attempt authority` is
/// cozy-runtime's) hands over anything with `.read(n) -> bytes`. TensorFS never learns the
/// URL, the token or the protocol; it hashes what arrives and admits it at the digest it
/// computed, which is the same door `tfs put` and `tfs fetch admit` cross.
///
/// The GIL is held ONLY for the duration of each `.read()` call: hashing and writing the
/// chunk happen with it released, so one pulling thread does not stall the rest of the
/// process for the length of a download.
struct Readable {
    obj: Py<PyAny>,
    /// The Python exception, kept verbatim. A dropped connection surfaces as the transport's
    /// own error rather than as a flattened `IO_FAILED` string.
    err: Option<PyErr>,
}

impl std::io::Read for Readable {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let want = buf.len();
        Python::attach(|py| {
            let chunk = match self.obj.bind(py).call_method1("read", (want,)) {
                Ok(c) => c,
                Err(e) => {
                    let msg = e.to_string();
                    self.err = Some(e);
                    return Err(std::io::Error::other(msg));
                }
            };
            let bytes: Vec<u8> = if let Ok(b) = chunk.cast::<PyBytes>() {
                b.as_bytes().to_vec()
            } else {
                match chunk.extract::<Vec<u8>>() {
                    Ok(v) => v,
                    Err(e) => {
                        let msg =
                            format!(
                            "the source's read({want}) returned {}, not bytes — a reader is an \
                             object with .read(n) -> bytes, and a str would be a re-encoding: {e}",
                            chunk.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                        );
                        self.err = Some(e);
                        return Err(std::io::Error::other(msg));
                    }
                }
            };
            if bytes.len() > want {
                // Never silently truncate: over-reading is the caller's bug and the bytes
                // beyond the buffer would vanish from the digest without a word.
                return Err(std::io::Error::other(format!(
                    "the source's read({want}) returned {} bytes — more than asked for; the \
                     surplus cannot be hashed and will not be dropped in silence",
                    bytes.len()
                )));
            }
            buf[..bytes.len()].copy_from_slice(&bytes);
            Ok(bytes.len())
        })
    }
}

fn receipts_to_py<'p>(py: Python<'p>, rx: &Receipts) -> PyResult<Bound<'p, PyList>> {
    let l = PyList::empty(py);
    for r in &rx.0 {
        l.append(marshal(py, &r.to_value())?)?;
    }
    Ok(l)
}

fn obj_ref(py: Python<'_>, hex: &str, length: u64) -> PyResult<ObjectRef> {
    let r = ObjectRef {
        sha256: hex.trim_start_matches("sha256:").to_string(),
        length,
    };
    tensorfs_core::ids::hex64("ObjectRef", &r.sha256).or_refuse(py)?;
    Ok(r)
}

fn missing(py: Python<'_>, what: &str, field: &str) -> PyErr {
    to_py(
        py,
        &Refusal {
            code: Code::MISSING_FIELD,
            detail: format!("{what}: missing {field:?}"),
        },
    )
}

fn walk_rows<'p>(py: Python<'p>, walk: &checkpoint::Walk) -> PyResult<Bound<'p, PyList>> {
    let rows = PyList::empty(py);
    for reached in &walk.objects {
        let row = PyDict::new(py);
        row.set_item("id", reached.obj.id())?;
        row.set_item("length", reached.obj.length)?;
        row.set_item("kind", reached.kind)?;
        rows.append(row)?;
    }
    Ok(rows)
}

fn required<'p>(
    py: Python<'p>,
    dict: &Bound<'p, PyDict>,
    what: &str,
    field: &str,
) -> PyResult<Bound<'p, PyAny>> {
    dict.get_item(field)?
        .ok_or_else(|| missing(py, what, field))
}

fn only_keys(
    py: Python<'_>,
    dict: &Bound<'_, PyDict>,
    what: &str,
    allowed: &[&str],
) -> PyResult<()> {
    for (key, _) in dict.iter() {
        let key: String = key.extract()?;
        if !allowed.contains(&key.as_str()) {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::UNKNOWN_FIELD,
                    detail: format!("{what}: unknown field {key:?}"),
                },
            ));
        }
    }
    Ok(())
}

fn parse_declaration(
    py: Python<'_>,
    sources: &Bound<'_, PyDict>,
    targets: &Bound<'_, PyDict>,
    configs: &Bound<'_, PyDict>,
    order: Vec<(String, String)>,
    max_new_bytes: u64,
) -> PyResult<Declaration> {
    let mut source_rows = Vec::with_capacity(sources.len());
    for (alias, value) in sources.iter() {
        let alias: String = alias.extract()?;
        let (manifest, length): (String, u64) = value.extract()?;
        source_rows.push(Source {
            alias,
            manifest: obj_ref(py, &manifest, length)?,
        });
    }

    let mut components = Vec::with_capacity(targets.len());
    for (target, value) in targets.iter() {
        let target: String = target.extract()?;
        let value = value.cast::<PyDict>()?;
        only_keys(
            py,
            value,
            &format!("target {target:?}"),
            &["source", "source_component", "drop", "add"],
        )?;
        let source = value
            .get_item("source")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        let source_component = value
            .get_item("source_component")?
            .map(|value| value.extract::<String>())
            .transpose()?;
        let drop: Vec<String> = required(py, value, "target", "drop")?.extract()?;
        let additions = required(py, value, "target", "add")?.cast_into::<PyDict>()?;
        let mut add = Vec::with_capacity(additions.len());
        for (tensor_key, tensor_value) in additions.iter() {
            let tensor_key: String = tensor_key.extract()?;
            let tensor = tensor_value.cast::<PyDict>()?;
            only_keys(
                py,
                tensor,
                &format!("addition {target}/{tensor_key}"),
                &["logical_dtype", "shape", "encoding", "parts"],
            )?;
            let dtype = tensorfs_core::dtype::Dtype::parse(
                &required(py, tensor, "addition", "logical_dtype")?.extract::<String>()?,
            )
            .or_refuse(py)?;
            let shape: Vec<u64> = required(py, tensor, "addition", "shape")?.extract()?;
            let encoding: String = required(py, tensor, "addition", "encoding")?.extract()?;
            let part_values = required(py, tensor, "addition", "parts")?.cast_into::<PyDict>()?;
            let mut parts = Vec::with_capacity(part_values.len());
            for (role, part_value) in part_values.iter() {
                let role: String = role.extract()?;
                let part = part_value.cast::<PyDict>()?;
                only_keys(
                    py,
                    part,
                    &format!("addition {target}/{tensor_key}#{role}"),
                    &["dtype", "shape"],
                )?;
                let dtype = tensorfs_core::dtype::Dtype::parse(
                    &required(py, part, "part", "dtype")?.extract::<String>()?,
                )
                .or_refuse(py)?;
                let shape: Vec<u64> = required(py, part, "part", "shape")?.extract()?;
                parts.push(PartDeclaration { role, dtype, shape });
            }
            add.push(TensorDeclaration {
                key: tensor_key,
                dtype,
                shape,
                encoding,
                parts,
            });
        }
        components.push(ComponentDeclaration {
            target,
            source,
            source_component,
            drop,
            add,
        });
    }

    let mut config_rows = Vec::with_capacity(configs.len());
    for (target, value) in configs.iter() {
        let target: String = target.extract()?;
        let value = value.cast::<PyDict>()?;
        only_keys(
            py,
            value,
            &format!("config {target:?}"),
            &["kind", "source", "source_config"],
        )?;
        let kind: String = required(py, value, "config", "kind")?.extract()?;
        let source = value.get_item("source")?;
        let source_config = value.get_item("source_config")?;
        match (kind.as_str(), source, source_config) {
            ("copy", Some(source), Some(source_config)) => {
                config_rows.push(ConfigDeclaration::Copy {
                    target,
                    source: source.extract()?,
                    source_config: source_config.extract()?,
                });
            }
            ("add", None, None) => {
                config_rows.push(ConfigDeclaration::Add { target });
            }
            ("derive", Some(source), Some(source_config)) => {
                config_rows.push(ConfigDeclaration::Derive {
                    target,
                    source: source.extract()?,
                    source_config: source_config.extract()?,
                });
            }
            _ => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::MISSING_FIELD,
                        detail: format!("config {target:?} has an invalid kind/source combination"),
                    },
                ));
            }
        }
    }
    let mut declaration = Declaration {
        work_fingerprint: None,
        sources: source_rows,
        components,
        configs: config_rows,
        order,
        max_new_bytes,
    };
    declaration.normalize_and_validate().or_refuse(py)?;
    Ok(declaration)
}

fn source_facts<'p>(py: Python<'p>, views: &[derived::SourceView]) -> PyResult<Bound<'p, PyList>> {
    let rows = PyList::empty(py);
    for view in views {
        let fact = &view.fact;
        let row = PyDict::new(py);
        row.set_item("alias", &fact.alias)?;
        row.set_item("manifest_id", fact.manifest.id())?;
        row.set_item("manifest_length", fact.manifest.length)?;
        row.set_item("header_id", fact.header.id())?;
        row.set_item("header_length", fact.header.length)?;
        row.set_item("components", &fact.components)?;
        rows.append(row)?;
    }
    Ok(rows)
}

fn source_component<'p>(
    py: Python<'p>,
    view: &derived::SourceView,
    component: &str,
) -> PyResult<Bound<'p, PyList>> {
    let tensors = view
        .header
        .components
        .iter()
        .find(|(name, _)| name == component)
        .map(|(_, tensors)| tensors)
        .ok_or_else(|| missing(py, "source component", component))?;
    let rows = PyList::empty(py);
    for (key, tensor) in tensors {
        let row = PyDict::new(py);
        row.set_item("key", key)?;
        row.set_item("logical_dtype", tensor.dtype.name())?;
        row.set_item("shape", &tensor.shape)?;
        row.set_item("encoding", &tensor.encoding)?;
        let parts = PyList::empty(py);
        for (role, part) in &tensor.parts {
            let part_row = PyDict::new(py);
            part_row.set_item("role", role)?;
            part_row.set_item("dtype", part.dtype.name())?;
            part_row.set_item("shape", &part.shape)?;
            parts.append(part_row)?;
        }
        row.set_item("parts", parts)?;
        rows.append(row)?;
    }
    Ok(rows)
}

fn receipt<'p>(py: Python<'p>, receipt: &derived::ReceiptFacts) -> PyResult<Bound<'p, PyAny>> {
    marshal(py, &receipt.to_value())
}

// ---------------------------------------------------------------- store

#[pyclass(name = "RepoObjectCache")]
pub struct PyRepoObjectCache {
    cache: CoreRepoObjectCache,
}

#[pymethods]
impl PyRepoObjectCache {
    #[new]
    fn new(root: PathBuf) -> Self {
        Self {
            cache: CoreRepoObjectCache::new(root),
        }
    }

    #[getter]
    fn root(&self) -> String {
        self.cache.root().display().to_string()
    }

    #[pyo3(signature = (store, object_id, length, *, manifest=false))]
    fn admit(
        &self,
        py: Python<'_>,
        store: &PyStore,
        object_id: &str,
        length: u64,
        manifest: bool,
    ) -> PyResult<&'static str> {
        let object = obj_ref(py, object_id, length)?;
        let kind = if manifest {
            CacheKind::Manifest
        } else {
            CacheKind::Blob
        };
        py.detach(|| self.cache.admit(&store.store, kind, &object))
            .or_refuse(py)
            .map(|status| status.as_str())
    }

    #[pyo3(signature = (store, object_id, length, *, manifest=false))]
    fn backfill(
        &self,
        py: Python<'_>,
        store: &PyStore,
        object_id: &str,
        length: u64,
        manifest: bool,
    ) -> PyResult<&'static str> {
        let object = obj_ref(py, object_id, length)?;
        let kind = if manifest {
            CacheKind::Manifest
        } else {
            CacheKind::Blob
        };
        py.detach(|| self.cache.backfill_store(&store.store, kind, &object))
            .or_refuse(py)
            .map(|status| status.as_str())
    }
}

#[pyclass(name = "Store")]
pub struct PyStore {
    pub store: Arc<CoreStore>,
    pub meta: Arc<Meta>,
}

fn open_store(
    py: Python<'_>,
    root: PathBuf,
    open: fn(&Path) -> tensorfs_core::err::Result<CoreStore>,
) -> PyResult<PyStore> {
    let s = open(&root).or_refuse(py)?;
    let m = Meta::open(&s).or_refuse(py)?;
    Ok(PyStore {
        store: Arc::new(s),
        meta: Arc::new(m),
    })
}

#[pymethods]
impl PyStore {
    #[staticmethod]
    #[pyo3(signature = (root = None))]
    fn ensure(py: Python<'_>, root: Option<PathBuf>) -> PyResult<PyStore> {
        let root = tensorfs_core::home::store_root(root.as_deref()).or_refuse(py)?;
        open_store(py, root, CoreStore::ensure)
    }

    #[staticmethod]
    #[pyo3(signature = (root = None))]
    fn open(py: Python<'_>, root: Option<PathBuf>) -> PyResult<PyStore> {
        let root = tensorfs_core::home::store_root(root.as_deref()).or_refuse(py)?;
        open_store(py, root, CoreStore::open)
    }

    #[staticmethod]
    #[pyo3(signature = (root = None))]
    fn init(py: Python<'_>, root: Option<PathBuf>) -> PyResult<PyStore> {
        let root = tensorfs_core::home::store_root(root.as_deref()).or_refuse(py)?;
        open_store(py, root, CoreStore::init)
    }

    #[getter]
    fn root(&self) -> String {
        self.store.root().display().to_string()
    }

    /// The mounted immutable object cache this Store is bound to, or `None`.
    ///
    /// READ-ONLY on purpose, and there is no way to bind one from here. The binding is a
    /// deployment fact recorded once by `tfs store ensure --repo-cache`; a Python caller
    /// preparing a model source neither knows nor needs a mount point, and giving it one to
    /// pass would put a filesystem path back into a protocol that has no use for it.
    #[getter]
    fn repo_cache(&self) -> Option<String> {
        self.store
            .repo_cache()
            .map(|cache| cache.root().display().to_string())
    }

    fn prepare_readers(&self, py: Python<'_>) -> PyResult<()> {
        self.store.prepare_readers().or_refuse(py)
    }

    fn complete_cozytensors_manifests(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.store.complete_cozytensors_manifests().or_refuse(py)
    }

    fn contains(&self, hex: &str) -> bool {
        self.store.contains(hex.trim_start_matches("sha256:"))
    }

    fn objects(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.store.objects().or_refuse(py)
    }

    fn repo_get<'p>(&self, py: Python<'p>, org: &str, name: &str) -> PyResult<Bound<'p, PyBytes>> {
        let repo = RepositoryName::new(org, name).or_refuse(py)?;
        let path = self.store.repository_path(&repo);
        let bytes = std::fs::read(&path).map_err(|error| {
            to_py(
                py,
                &Refusal {
                    code: if error.kind() == std::io::ErrorKind::NotFound {
                        Code::REPOSITORY_ABSENT
                    } else {
                        Code::IO_FAILED
                    },
                    detail: format!("read {}: {error}", path.display()),
                },
            )
        })?;
        tensorfs_core::repository::Repository::parse(&bytes).or_refuse(py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Resolve one exact local repository release without exporting Repository bytes for a
    /// foreign parser. Package/model selection stays in the caller; TensorFS alone interprets
    /// its stored repository and returns the immutable Manifest reference it names.
    fn resolve_release<'p>(
        &self,
        py: Python<'p>,
        org: &str,
        name: &str,
        version: &str,
        lane: &str,
    ) -> PyResult<Bound<'p, PyDict>> {
        let repo = RepositoryName::new(org, name).or_refuse(py)?;
        let path = self.store.repository_path(&repo);
        let bytes = std::fs::read(&path).map_err(|error| {
            to_py(
                py,
                &Refusal {
                    code: if error.kind() == std::io::ErrorKind::NotFound {
                        Code::REPOSITORY_ABSENT
                    } else {
                        Code::IO_FAILED
                    },
                    detail: format!("read {}: {error}", path.display()),
                },
            )
        })?;
        let repository = Repository::parse(&bytes).or_refuse(py)?;
        let release = repository
            .releases
            .iter()
            .find(|release| release.version == version && !release.yanked)
            .ok_or_else(|| {
                to_py(
                    py,
                    &Refusal {
                        code: Code::RELEASE_ABSENT,
                        detail: format!(
                            "repository {org}/{name} has no release ({version},{lane})"
                        ),
                    },
                )
            })?;
        let selected = release
            .lanes
            .iter()
            .find(|candidate| candidate.lane == lane)
            .ok_or_else(|| {
                to_py(
                    py,
                    &Refusal {
                        code: Code::RELEASE_ABSENT,
                        detail: format!(
                            "repository {org}/{name} release {version:?} has no lane {lane:?}"
                        ),
                    },
                )
            })?;
        let out = PyDict::new(py);
        out.set_item("org", org)?;
        out.set_item("name", name)?;
        out.set_item("version", &release.version)?;
        out.set_item("revision", release.revision)?;
        out.set_item("lane", &selected.lane)?;
        out.set_item("manifest_digest", selected.manifest.id())?;
        out.set_item("manifest_length", selected.manifest.length)?;
        Ok(out)
    }

    /// Resolve Creator's one-row device-local alias. The reserved org/lane and hidden
    /// source-selection version never become a remote catalog reference.
    fn resolve_local<'p>(&self, py: Python<'p>, name: &str) -> PyResult<Bound<'p, PyDict>> {
        let repo = RepositoryName::new("local", name).or_refuse(py)?;
        let path = self.store.repository_path(&repo);
        let bytes = std::fs::read(&path).map_err(|error| {
            to_py(
                py,
                &Refusal {
                    code: if error.kind() == std::io::ErrorKind::NotFound {
                        Code::REPOSITORY_ABSENT
                    } else {
                        Code::IO_FAILED
                    },
                    detail: format!("read {}: {error}", path.display()),
                },
            )
        })?;
        let repository = Repository::parse(&bytes).or_refuse(py)?;
        let release = repository.local_checkpoint().or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("name", name)?;
        out.set_item(
            "source_selection",
            format!(
                "sha256:{}",
                release
                    .source_selection
                    .as_ref()
                    .expect("validated local checkpoint")
            ),
        )?;
        out.set_item("manifest_digest", release.manifest.id())?;
        out.set_item("manifest_length", release.manifest.length)?;
        Ok(out)
    }

    /// Atomically replace the complete local alias after its exact Manifest is resident and
    /// verified. `current` is the exact observed Repository bytes or None for creation. The
    /// optional contract is all-or-none and is checked from resident TensorFS documents.
    #[pyo3(signature = (
        current,
        name,
        source_selection,
        manifest_id,
        manifest_length,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn replace_local<'p>(
        &self,
        py: Python<'p>,
        current: Option<&[u8]>,
        name: &str,
        source_selection: &str,
        manifest_id: &str,
        manifest_length: u64,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let selection = prefixed("local source selection", source_selection).or_refuse(py)?;
        let mutation = Mutation::ReplaceLocal {
            repo: RepositoryName::new("local", name).or_refuse(py)?,
            version: selection.trim_start_matches("sha256:").to_string(),
            manifest: obj_ref(py, manifest_id, manifest_length)?,
        };
        let repository = py
            .detach(|| {
                self.store
                    .apply_repository(current, &mutation, &Fault::default())
            })
            .or_refuse(py)?
            .expect("replace_local retains the repository");
        Ok(PyBytes::new(py, &repository.canonical_bytes()))
    }

    fn remove_local<'p>(
        &self,
        py: Python<'p>,
        current: &[u8],
        name: &str,
    ) -> PyResult<Bound<'p, PyDict>> {
        let mutation = Mutation::DeleteRepository {
            repo: RepositoryName::new("local", name).or_refuse(py)?,
        };
        py.detach(|| {
            self.store
                .apply_repository(Some(current), &mutation, &Fault::default())
        })
        .or_refuse(py)?;
        Ok(PyDict::new(py))
    }

    fn begin_operation(
        &self,
        py: Python<'_>,
        operation_id: &str,
        org: &str,
        name: &str,
    ) -> PyResult<PyOperation> {
        let repo = RepositoryName::new(org, name).or_refuse(py)?;
        let catalog = Catalog::open(self.store.root()).or_refuse(py)?;
        let guard = catalog
            .resume_operation(operation_id, &repo.org, &repo.name)
            .or_refuse(py)?;
        Ok(PyOperation {
            store: Arc::clone(&self.store),
            catalog,
            operation_id: operation_id.to_string(),
            repo,
            guard: Some(guard),
        })
    }

    /// Advance the exact source plan over verified physical carriers. Checkpoint heads are
    /// local progress until the RecordOwner acknowledges their complete upload.
    #[pyo3(signature = (operation_id, source_selection_digest, profiles, roster, landed, *, checkpoints=Vec::new(), adopt_from_operation_id=None))]
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    fn prepare_model_source<'p>(
        &self,
        py: Python<'p>,
        operation_id: &str,
        source_selection_digest: &str,
        profiles: Vec<(String, String)>,
        roster: Vec<(String, String, u64, PathBuf, Vec<u8>)>,
        landed: Vec<String>,
        checkpoints: Vec<(String, String, u64)>,
        adopt_from_operation_id: Option<String>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let checkpoints = checkpoints
            .into_iter()
            .map(|(slot, id, length)| {
                prefixed("source checkpoint", &id).map(|id| {
                    (
                        slot,
                        ObjectRef {
                            sha256: id.trim_start_matches("sha256:").into(),
                            length,
                        },
                    )
                })
            })
            .collect::<TResult<Vec<_>>>()
            .or_refuse(py)?;
        let request = PrepareModelSource {
            operation_id: operation_id.to_string(),
            source_selection_digest: source_selection_digest.to_string(),
            profiles: profiles
                .into_iter()
                .map(|(slot, profile)| ModelSourceProfile { slot, profile })
                .collect(),
            roster: roster
                .into_iter()
                .map(
                    |(member, object_id, length, path, header)| VerifiedModelSourceFile {
                        member,
                        object_id,
                        length,
                        path,
                        header,
                    },
                )
                .collect(),
            landed,
            checkpoints,
            adopt_from_operation_id,
        };
        let prepared = py
            .detach(|| model_source::prepare_model_source(&self.store, &request))
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("replayed", prepared.replayed)?;
        out.set_item("complete", prepared.complete)?;
        out.set_item("spent", prepared.spent)?;
        out.set_item("converted_roles", prepared.converted_roles)?;
        out.set_item("resumed_roles", prepared.resumed_roles)?;
        out.set_item("deferred_ops", prepared.deferred_ops)?;
        out.set_item("converted_bytes", prepared.converted_bytes)?;
        out.set_item("largest_op_bytes", prepared.largest_op_bytes)?;
        let rows = PyList::empty(py);
        for source in prepared.sources {
            let row = PyDict::new(py);
            row.set_item("slot", source.slot)?;
            row.set_item("profile", source.profile)?;
            row.set_item("manifest_digest", source.manifest_digest)?;
            row.set_item("manifest_length", source.manifest_length)?;
            rows.append(row)?;
        }
        out.set_item("sources", rows)?;
        let checkpoints = PyList::empty(py);
        for checkpoint in prepared.checkpoints {
            let Some(head) = checkpoint.head.head else {
                continue;
            };
            let row = PyDict::new(py);
            row.set_item("slot", checkpoint.slot)?;
            row.set_item("plan_digest", checkpoint.plan_digest)?;
            row.set_item("head", head.id())?;
            row.set_item("head_length", head.length)?;
            row.set_item("index", checkpoint.head.links - 1)?;
            row.set_item("bytes", checkpoint.head.bytes)?;
            checkpoints.append(row)?;
        }
        out.set_item("checkpoints", checkpoints)?;
        Ok(out)
    }

    /// One bounded link window, usable as soon as that link alone is resident.
    #[pyo3(signature = (head_id, head_length, *, operation_id, slot, plan_digest, offset=0, limit=128))]
    #[allow(clippy::too_many_arguments)]
    fn checkpoint_page<'p>(
        &self,
        py: Python<'p>,
        head_id: &str,
        head_length: u64,
        operation_id: &str,
        slot: &str,
        plan_digest: &str,
        offset: usize,
        limit: usize,
    ) -> PyResult<Bound<'p, PyDict>> {
        let id = prefixed("source checkpoint", head_id).or_refuse(py)?;
        let object = ObjectRef {
            sha256: id.trim_start_matches("sha256:").into(),
            length: head_length,
        };
        let tensorfs_core::durability::Window {
            link,
            objects,
            next,
        } = py
            .detach(|| tensorfs_core::durability::local_window(&self.store, &object, offset, limit))
            .or_refuse(py)?;
        if link.operation != format!("{operation_id}/{slot}") || link.plan != plan_digest {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::CROSS_SUBJECT_REPLAY,
                    detail: "checkpoint link differs from its operation, slot or plan".into(),
                },
            ));
        }
        let out = PyDict::new(py);
        out.set_item("operation", link.operation)?;
        out.set_item("plan_digest", link.plan)?;
        out.set_item("index", link.index)?;
        out.set_item("bytes", link.bytes)?;
        out.set_item("previous", link.prev.as_ref().map(|r| (r.id(), r.length)))?;
        out.set_item(
            "progress",
            link.progress.as_ref().map(|r| (r.id(), r.length)),
        )?;
        out.set_item("next_offset", next)?;
        let rows = PyList::empty(py);
        for (kind, object) in objects {
            let row = PyDict::new(py);
            row.set_item(
                "kind",
                match kind {
                    tensorfs_core::repo_cache::CacheKind::Blob => "blob",
                    tensorfs_core::repo_cache::CacheKind::Manifest => "manifest",
                },
            )?;
            row.set_item("object_id", object.id())?;
            row.set_item("length", object.length)?;
            rows.append(row)?;
        }
        out.set_item("objects", rows)?;
        Ok(out)
    }

    /// Spend one scoped PUT through the existing TensorFS transport. The caller's expected
    /// length is checked before a socket opens, and the grant never leaves process memory.
    #[pyo3(signature = (object_id, length, manifest, grant, *, allow_local=false))]
    #[allow(clippy::too_many_arguments)]
    fn checkpoint_push<'p>(
        &self,
        py: Python<'p>,
        object_id: &str,
        length: u64,
        manifest: bool,
        grant: &str,
        allow_local: bool,
    ) -> PyResult<Bound<'p, PyDict>> {
        use tensorfs_core::{fetch, transport};
        let id = prefixed("checkpoint object", object_id).or_refuse(py)?;
        let object = ObjectRef {
            sha256: id.trim_start_matches("sha256:").into(),
            length,
        };
        let grant = transport::UploadGrant::parse(grant).or_refuse(py)?;
        let policy = transport::SourcePolicy {
            allowed_hosts: vec![transport::base_host(&grant.url).or_refuse(py)?],
            allow_local,
            max_redirects: 0,
        };
        let pushed = py
            .detach(|| {
                let (plan, _) = if manifest {
                    fetch::FetchPlan::of(
                        &self.store,
                        "checkpoint-upload",
                        &object,
                        std::slice::from_ref(&object),
                    )?
                } else {
                    fetch::FetchPlan::of_objects(
                        &self.store,
                        "checkpoint-upload",
                        std::slice::from_ref(&object),
                    )?
                };
                if !plan.wanted.is_empty() {
                    return refuse(
                        Code::OBJECT_ABSENT,
                        "checkpoint upload object is not verified locally",
                    );
                }
                transport::push_object(
                    &self.store,
                    &object.sha256,
                    manifest,
                    &grant,
                    &policy,
                    &transport::Anonymous,
                    transport::Deadline::after_seconds(None),
                    &transport::Ledger::new(),
                    "application/octet-stream",
                )
            })
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("http_status", pushed.http_status)?;
        out.set_item("transferred", pushed.bytes_sent)?;
        out.set_item("held", pushed.landed_precondition)?;
        Ok(out)
    }

    /// Fetch one exact recovery object through ordinary digest/length admission.
    #[pyo3(signature = (object_id, length, manifest, url, *, allow_local=false, derived_restore=None))]
    #[allow(clippy::too_many_arguments)]
    fn checkpoint_fetch<'p>(
        &self,
        py: Python<'p>,
        object_id: &str,
        length: u64,
        manifest: bool,
        url: &str,
        allow_local: bool,
        derived_restore: Option<(String, u64)>,
    ) -> PyResult<Bound<'p, PyDict>> {
        use tensorfs_core::{fetch, transport};
        let id = prefixed("checkpoint object", object_id).or_refuse(py)?;
        let object = ObjectRef {
            sha256: id.trim_start_matches("sha256:").into(),
            length,
        };
        let policy = transport::SourcePolicy {
            allowed_hosts: vec![transport::base_host(url).or_refuse(py)?],
            allow_local,
            max_redirects: 0,
        };
        let (transferred, held) = py
            .detach(|| {
                let fetch_object = || {
                    let (plan, _) = if manifest {
                        fetch::FetchPlan::of(
                            &self.store,
                            "checkpoint-download",
                            &object,
                            std::slice::from_ref(&object),
                        )?
                    } else {
                        fetch::FetchPlan::of_objects(
                            &self.store,
                            "checkpoint-download",
                            std::slice::from_ref(&object),
                        )?
                    };
                    if plan.wanted.is_empty() {
                        return Ok((0, true));
                    }
                    let grant = fetch::DeliveryGrant::mint(&plan, &object)?;
                    let fetched = transport::fetch_ranged(
                        &self.store,
                        &grant,
                        url,
                        &policy,
                        &transport::Anonymous,
                        transport::Deadline::after_seconds(None),
                        &transport::Ledger::new(),
                        transport::Ranged::default(),
                        None,
                    )?;
                    Ok((fetched.transferred, false))
                };
                if let Some((transaction, epoch)) = &derived_restore {
                    if manifest {
                        return refuse(
                            Code::KEY_GRAMMAR,
                            "derived progress restores blob objects, not manifests",
                        );
                    }
                    tensorfs_core::durability::restore_derived(
                        &self.store,
                        &self.meta,
                        transaction,
                        *epoch,
                        &object,
                        fetch_object,
                    )
                } else {
                    fetch_object()
                }
            })
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("transferred", transferred)?;
        out.set_item("held", held)?;
        Ok(out)
    }

    /// Source-operation cleanup roster; the caller quiesces the owned Store lifecycle.
    /// Does not scan roots or recover coordination lost with a deleted catalog.
    fn model_source_operations(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| {
            tensorfs_core::catalog::Catalog::open(self.store.root())?.model_source_operations()
        })
        .or_refuse(py)
    }

    /// Drop this operation's exact CAS holds after its prepared refs were adopted or rejected.
    fn release_model_source(&self, py: Python<'_>, operation_id: &str) -> PyResult<()> {
        py.detach(|| model_source::release_model_source(&self.store, operation_id))
            .or_refuse(py)
    }

    #[allow(clippy::too_many_arguments)]
    fn update_release<'p>(
        &self,
        py: Python<'p>,
        current: &[u8],
        org: &str,
        name: &str,
        version: &str,
        expected_revision: u64,
        set: Vec<(String, String, u64)>,
        remove: Vec<String>,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let set = set
            .into_iter()
            .map(|(lane, digest, length)| {
                Ok(ReleaseLane {
                    lane,
                    manifest: obj_ref(py, &digest, length)?,
                })
            })
            .collect::<PyResult<Vec<_>>>()?;
        let mutation = Mutation::UpdateRelease {
            expected_revision,
            repo: RepositoryName::new(org, name).or_refuse(py)?,
            remove,
            set,
            version: version.to_string(),
        };
        let repository = py
            .detach(|| {
                self.store
                    .apply_repository(Some(current), &mutation, &Fault::default())
            })
            .or_refuse(py)?
            .expect("release update retains the repository");
        Ok(PyBytes::new(py, &repository.canonical_bytes()))
    }

    fn yank_release<'p>(
        &self,
        py: Python<'p>,
        current: &[u8],
        org: &str,
        name: &str,
        version: &str,
        expected_revision: u64,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let mutation = Mutation::YankRelease {
            expected_revision,
            repo: RepositoryName::new(org, name).or_refuse(py)?,
            version: version.to_string(),
        };
        let repository = py
            .detach(|| {
                self.store
                    .apply_repository(Some(current), &mutation, &Fault::default())
            })
            .or_refuse(py)?
            .expect("release yank retains the repository");
        Ok(PyBytes::new(py, &repository.canonical_bytes()))
    }

    /// Stream a file into the CAS. `expect` is a full ObjectRef (id AND length) or nothing;
    /// a digest without a length is not an admission ticket.
    #[pyo3(signature = (path, expect=None, expect_length=None))]
    fn put_file<'p>(
        &self,
        py: Python<'p>,
        path: PathBuf,
        expect: Option<&str>,
        expect_length: Option<u64>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let want = match (expect, expect_length) {
            (Some(h), Some(n)) => Some(obj_ref(py, h, n)?),
            (None, None) => None,
            _ => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: "expect requires expect_length — an ObjectRef is length-bearing"
                            .into(),
                    },
                ))
            }
        };
        let p = py
            .detach(|| self.store.put_file(&path, want.as_ref(), &Fault::default()))
            .or_refuse(py)?;
        let d = PyDict::new(py);
        d.set_item("id", p.obj.id())?;
        d.set_item("length", p.obj.length)?;
        d.set_item("admitted", p.admitted)?;
        Ok(d)
    }

    /// Stream a READER into the CAS through the one verifying door (tfs-003 row 220).
    ///
    /// `reader` is anything with `.read(n) -> bytes` — an `http.client.HTTPResponse`, a
    /// socket file, a pipe. `expect` is REQUIRED here, unlike `put_file`: a local file can
    /// be re-read and re-hashed, but a network body arrives once, so admitting it without an
    /// expectation would mean trusting whatever the remote sent. That is exactly the
    /// property this method exists to hold, so there is no flag to turn it off.
    ///
    /// A digest or length disagreement refuses typed and commits NOTHING: the partial temp
    /// file is removed on the way out of `put_stream`.
    fn put_reader<'p>(
        &self,
        py: Python<'p>,
        reader: &Bound<'p, PyAny>,
        expect: &str,
        expect_length: u64,
    ) -> PyResult<Bound<'p, PyDict>> {
        let want = obj_ref(py, expect, expect_length)?;
        let mut src = Readable {
            obj: reader.clone().unbind(),
            err: None,
        };
        let put = py.detach(|| {
            self.store
                .put_stream(&mut src, Some(&want), &Fault::default())
        });
        let put = match put {
            Ok(p) => p,
            // The transport's own exception outranks the border's IO_FAILED restatement of
            // it: a dropped connection is the transport's fault to report, with its own
            // traceback, and nothing was committed either way.
            Err(e) => return Err(src.err.take().unwrap_or_else(|| to_py(py, &e))),
        };
        let d = PyDict::new(py);
        d.set_item("id", put.obj.id())?;
        d.set_item("length", put.obj.length)?;
        d.set_item("admitted", put.admitted)?;
        Ok(d)
    }

    /// Strictly parse and admit canonical Manifest bytes into the typed manifests namespace.
    #[pyo3(signature = (data, expect=None, expect_length=None))]
    fn put_manifest<'p>(
        &self,
        py: Python<'p>,
        data: &[u8],
        expect: Option<&str>,
        expect_length: Option<u64>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let manifest = Manifest::parse(data).or_refuse(py)?;
        let observed = ObjectRef::of(data);
        match (expect, expect_length) {
            (Some(identity), Some(length)) => {
                let wanted = obj_ref(py, identity, length)?;
                if wanted != observed {
                    return Err(to_py(
                        py,
                        &Refusal {
                            code: Code::OBJECT_ID_MISMATCH,
                            detail: format!(
                                "manifest bytes are {}, expected {}",
                                observed.id(),
                                wanted.id()
                            ),
                        },
                    ));
                }
            }
            (None, None) => {}
            _ => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::LENGTH_MISMATCH,
                        detail: "expect requires expect_length".into(),
                    },
                ))
            }
        }
        let put = py
            .detach(|| self.store.put_manifest(&manifest))
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("id", put.obj.id())?;
        out.set_item("length", put.obj.length)?;
        out.set_item("admitted", put.admitted)?;
        Ok(out)
    }

    fn verify<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyDict>> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let v = py.detach(|| self.store.verify(&hex)).or_refuse(py)?;
        let d = PyDict::new(py);
        match v {
            Verdict::Verified { rehashed } => {
                d.set_item("verdict", "verified")?;
                d.set_item("rehashed", rehashed)?;
            }
            Verdict::Invalidated { why } => {
                d.set_item("verdict", "invalidated")?;
                d.set_item("why", why)?;
            }
            Verdict::CorruptRemoved { why } => {
                d.set_item("verdict", "corrupt_removed")?;
                d.set_item("why", why)?;
            }
        }
        Ok(d)
    }

    /// A whole small object — a document. Bounded on purpose: this door is not how weights
    /// are read, and the cap says so before anything is allocated.
    #[pyo3(signature = (hex, max_bytes=1<<26))]
    fn document<'p>(
        &self,
        py: Python<'p>,
        hex: &str,
        max_bytes: u64,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let size = std::fs::metadata(self.store.object_path(&hex))
            .map(|m| m.len())
            .unwrap_or(0);
        if size > max_bytes {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::SIZE_CAP,
                    detail: format!("sha256:{hex} is {size} B, over the {max_bytes} B cap"),
                },
            ));
        }
        let v = py
            .detach(|| self.store.read_range(&hex, 0, size))
            .or_refuse(py)?;
        Ok(PyBytes::new(py, &v))
    }

    /// Canonical manifest bytes and the CozyTensors header selected by its typed entry.
    fn manifest<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyDict>> {
        let (m, mb, hb) = self.load(py, hex)?;
        let d = PyDict::new(py);
        d.set_item("manifest_id", m.manifest_id())?;
        d.set_item("manifest", PyBytes::new(py, &mb))?;
        d.set_item("header", PyBytes::new(py, &hb))?;
        Ok(d)
    }

    /// The transitive walk: every blob the manifest commits, with what it was reached AS.
    fn walk<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyList>> {
        let (m, _, _) = self.load(py, hex)?;
        let w = py
            .detach(|| checkpoint::walk(&self.store, &m))
            .or_refuse(py)?;
        walk_rows(py, &w)
    }

    /// The selected CozyTensors runtime closure, excluding ordinary snapshot siblings.
    fn walk_cozytensors<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyList>> {
        let (m, _, _) = self.load(py, hex)?;
        let w = py
            .detach(|| checkpoint::walk_cozytensors(&self.store, &m))
            .or_refuse(py)?;
        walk_rows(py, &w)
    }

    /// `ReadLease = store.acquire(manifest, object_set)`. Verifies every object first and
    /// registers the hold BEFORE the first byte moves.
    #[pyo3(signature = (manifest, objects))]
    fn acquire(
        &self,
        py: Python<'_>,
        manifest: &str,
        objects: Vec<(String, u64)>,
    ) -> PyResult<PyLease> {
        let mut refs = Vec::with_capacity(objects.len());
        for (h, n) in &objects {
            refs.push(obj_ref(py, h, *n)?);
        }
        let snap = manifest.trim_start_matches("sha256:").to_string();
        let (lease, rx) = py
            .detach(|| read::acquire(&self.store, &self.meta, &snap, refs))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
            receipts: rx,
        })
    }

    /// Acquire over everything one manifest commits — the serving case.
    fn acquire_manifest(&self, py: Python<'_>, hex: &str) -> PyResult<PyLease> {
        let (m, _, _) = self.load(py, hex)?;
        let (lease, rx) = py
            .detach(|| read::acquire_manifest(&self.store, &self.meta, &m))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
            receipts: rx,
        })
    }

    /// Acquire the selected CozyTensors runtime closure without optional snapshot siblings.
    fn acquire_cozytensors(&self, py: Python<'_>, hex: &str) -> PyResult<PyLease> {
        let (m, _, _) = self.load(py, hex)?;
        let (lease, rx) = py
            .detach(|| read::acquire_cozytensors(&self.store, &self.meta, &m))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
            receipts: rx,
        })
    }

    /// Tier 2 of the projection ladder: config assets and canonical documents as ordinary
    /// files. A weight path cannot appear, and the refusal says why.
    #[pyo3(signature = (hex, dest, symlink=true))]
    fn checkout<'p>(
        &self,
        py: Python<'p>,
        hex: &str,
        dest: PathBuf,
        symlink: bool,
    ) -> PyResult<Bound<'p, PyDict>> {
        let (m, _, _) = self.load(py, hex)?;
        let p = project::checkout(&self.store, &m, &dest, !symlink).or_refuse(py)?;
        let d = PyDict::new(py);
        d.set_item("dirs", p.dirs)?;
        d.set_item("links", p.links)?;
        d.set_item("copies", p.copies)?;
        d.set_item("bytes", p.bytes)?;
        Ok(d)
    }

    /// Tier 3: a private streamed copy of an ALREADY-RETAINED carrier named by its path in
    /// the manifest. Never a synthesis — an unretained carrier refuses.
    fn materialize(&self, py: Python<'_>, hex: &str, path: &str, dest: PathBuf) -> PyResult<u64> {
        let (m, _, _) = self.load(py, hex)?;
        let entry = m.entries().iter().find(|(p, _)| p == path);
        let blob = match entry {
            Some((_, Entry::File(blob))) => blob.clone(),
            Some((_, Entry::CozyTensors(_))) => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::BLOB_NOT_RETAINED,
                        detail: format!("{path:?} is a CozyTensors header, not an ordinary file"),
                    },
                ))
            }
            None => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::NOT_CONTAINED,
                        detail: format!("{path:?} is not a path in manifest {hex}"),
                    },
                ))
            }
        };
        py.detach(|| project::materialize(&self.store, &blob, &dest))
            .or_refuse(py)
    }

    fn reap_holds(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.meta.reap_holds()).or_refuse(py)
    }

    /// Exact pre-transaction preview for a granted source manifest. Selected tensor
    /// declarations contain no ObjectRefs; selected config bytes cross one verified lease.
    fn inspect_derived_source<'p>(
        &self,
        py: Python<'p>,
        manifest: &str,
        manifest_length: u64,
        components: Vec<String>,
        configs: Vec<String>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let reference = obj_ref(py, manifest, manifest_length)?;
        let inspection = py
            .detach(|| {
                derived::inspect_source(&self.store, &self.meta, reference, components, configs)
            })
            .or_refuse(py)?;
        let out = PyDict::new(py);
        let identity = source_facts(py, std::slice::from_ref(&inspection.view))?;
        out.set_item("source", identity.get_item(0)?)?;
        let component_rows = PyList::empty(py);
        for component in &inspection.view.fact.components {
            let row = PyDict::new(py);
            row.set_item("name", component)?;
            row.set_item(
                "tensors",
                source_component(py, &inspection.view, component)?,
            )?;
            component_rows.append(row)?;
        }
        out.set_item("components", component_rows)?;
        let config_rows = PyList::empty(py);
        for config in &inspection.configs {
            let row = PyDict::new(py);
            row.set_item("name", &config.name)?;
            row.set_item("data", PyBytes::new(py, &config.bytes))?;
            config_rows.append(row)?;
        }
        out.set_item("configs", config_rows)?;
        Ok(out)
    }

    /// Canonical TensorFS declaration bytes for the current producer intent.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (sources, targets, configs, order, max_new_bytes, *, work_fingerprint))]
    fn derived_declaration<'p>(
        &self,
        py: Python<'p>,
        sources: &Bound<'_, PyDict>,
        targets: &Bound<'_, PyDict>,
        configs: &Bound<'_, PyDict>,
        order: Vec<(String, String)>,
        max_new_bytes: u64,
        work_fingerprint: &str,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let mut declaration =
            parse_declaration(py, sources, targets, configs, order, max_new_bytes)?;
        declaration.work_fingerprint = Some(work_fingerprint.to_string());
        let bytes = declaration.canonical_work_bytes().or_refuse(py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transaction_id, declaration, head_id, head_length, *, operation_id, slot, plan_digest))]
    fn validate_derived_checkpoint(
        &self,
        py: Python<'_>,
        transaction_id: &str,
        declaration: &[u8],
        head_id: &str,
        head_length: u64,
        operation_id: &str,
        slot: &str,
        plan_digest: &str,
    ) -> PyResult<()> {
        let head = obj_ref(py, head_id, head_length)?;
        py.detach(|| {
            derived::validate_checkpoint(
                &self.store,
                &self.meta,
                transaction_id,
                declaration,
                &head,
                operation_id,
                slot,
                plan_digest,
            )
        })
        .or_refuse(py)
    }

    /// Open one exact tfs-019 transaction. Python supplies typed plain values; canonical
    /// declaration bytes are constructed and locked only by the compiled core.
    // PyO3 exposes these as six named boundary fields. Packing them into an opaque dict
    // would only move the arity into hand parsing and weaken the typed facade.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transaction_id, writer_session_id, sources, targets, configs, order, max_new_bytes, *, work_fingerprint, checkpoint=None))]
    fn begin_derived(
        &self,
        py: Python<'_>,
        transaction_id: &str,
        writer_session_id: u64,
        sources: &Bound<'_, PyDict>,
        targets: &Bound<'_, PyDict>,
        configs: &Bound<'_, PyDict>,
        order: Vec<(String, String)>,
        max_new_bytes: u64,
        work_fingerprint: &str,
        checkpoint: Option<(String, u64)>,
    ) -> PyResult<PyDerivedWriter> {
        let mut declaration =
            parse_declaration(py, sources, targets, configs, order, max_new_bytes)?;
        declaration.work_fingerprint = Some(work_fingerprint.to_string());
        let checkpoint = checkpoint
            .map(|(id, length)| obj_ref(py, &id, length))
            .transpose()?;
        let begin = py
            .detach(|| {
                derived::begin(
                    &self.store,
                    &self.meta,
                    transaction_id,
                    writer_session_id,
                    declaration,
                    checkpoint.as_ref(),
                )
            })
            .or_refuse(py)?;
        Ok(PyDerivedWriter {
            store: self.store.clone(),
            meta: self.meta.clone(),
            transaction_id: transaction_id.to_string(),
            writer_session_id,
            source_views: begin.source_views,
            guards: Mutex::new(Some(DerivedGuards {
                writer_hold: begin.writer_hold,
                source_leases: begin.source_leases,
            })),
        })
    }

    /// Durable recovery after the TensorFS-commit/Runtime-journal crash gap.
    fn derived_lookup<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
    ) -> PyResult<Bound<'p, PyDict>> {
        let result = py
            .detach(|| derived::lookup(&self.store, &self.meta, transaction_id))
            .or_refuse(py)?;
        let out = PyDict::new(py);
        match result {
            derived::Lookup::Absent => out.set_item("state", "absent")?,
            derived::Lookup::Open { writer_session } => {
                out.set_item("state", "open")?;
                if let Some(writer_session) = writer_session {
                    out.set_item("writer_session_id", writer_session)?;
                }
            }
            derived::Lookup::Committed(value) => {
                out.set_item("state", "committed")?;
                out.set_item("receipt", receipt(py, value.receipt())?)?;
                out.set_item("disposition", marshal(py, &value.disposition_value())?)?;
            }
            derived::Lookup::Abandoned => out.set_item("state", "abandoned")?,
        }
        Ok(out)
    }

    /// Fence one executor session. Its live process holds remain until it explicitly
    /// closes or dies; therefore GC cannot race a stale stream already in progress.
    fn derived_fence(
        &self,
        py: Python<'_>,
        transaction_id: &str,
        writer_session_id: u64,
    ) -> PyResult<bool> {
        py.detach(|| derived::fence(&self.meta, transaction_id, writer_session_id))
            .or_refuse(py)
    }

    /// Receipt-free abandonment. The same metadata lock arbitrates it against commit.
    fn derived_abandon<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
    ) -> PyResult<Bound<'p, PyDict>> {
        let result = py
            .detach(|| derived::abandon(&self.store, &self.meta, transaction_id))
            .or_refuse(py)?;
        let out = PyDict::new(py);
        match result {
            None => out.set_item("state", "abandoned")?,
            Some(value) => {
                out.set_item("state", "committed")?;
                out.set_item("receipt", receipt(py, &value)?)?;
            }
        }
        Ok(out)
    }

    /// Name a private retained root without creating a model repository or changing receipt bytes.
    fn derived_adopt<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
        scratch_root_id: &str,
    ) -> PyResult<Bound<'p, PyAny>> {
        let value = py
            .detach(|| derived::adopt(&self.store, &self.meta, transaction_id, scratch_root_id))
            .or_refuse(py)?;
        let out = marshal(py, &value.to_value())?;
        out.cast::<PyDict>()?
            .set_item("receipt", receipt(py, value.receipt())?)?;
        Ok(out)
    }

    /// Release the committed transaction after its manifest has been published or rejected.
    fn derived_dispose<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
    ) -> PyResult<Bound<'p, PyAny>> {
        let value = py
            .detach(|| derived::dispose(&self.store, &self.meta, transaction_id))
            .or_refuse(py)?;
        let out = marshal(py, &value.to_value())?;
        let out_dict = out.cast::<PyDict>()?;
        out_dict.set_item("receipt", receipt(py, value.receipt())?)?;
        Ok(out)
    }

    fn __repr__(&self) -> String {
        format!("<tensorfs.Store {}>", self.store.root().display())
    }
}

// ---------------------------------------------------------------- exact acquisition operation

#[pyclass(name = "Operation", unsendable)]
pub struct PyOperation {
    store: Arc<CoreStore>,
    catalog: Catalog,
    operation_id: String,
    repo: RepositoryName,
    guard: Option<WriterGuard>,
}

#[pymethods]
impl PyOperation {
    fn hold_blob(&self, py: Python<'_>, identity: &str, length: u64) -> PyResult<()> {
        let digest = identity.trim_start_matches("sha256:");
        let key = storage::blob_key(digest).or_refuse(py)?;
        self.catalog
            .hold(&self.operation_id, &key, length)
            .or_refuse(py)
    }

    fn hold_manifest(&self, py: Python<'_>, identity: &str, length: u64) -> PyResult<()> {
        let digest = identity.trim_start_matches("sha256:");
        let key = storage::manifest_key(digest).or_refuse(py)?;
        self.catalog
            .hold(&self.operation_id, &key, length)
            .or_refuse(py)
    }

    #[pyo3(signature = (current, version, lane, manifest_id, manifest_length))]
    #[allow(clippy::too_many_arguments)]
    fn commit_release<'p>(
        &mut self,
        py: Python<'p>,
        current: Option<&[u8]>,
        version: &str,
        lane: &str,
        manifest_id: &str,
        manifest_length: u64,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let manifest = obj_ref(py, manifest_id, manifest_length)?;
        let checkpoint = Mutation::PutCheckpoint {
            repo: self.repo.clone(),
            manifest: manifest.clone(),
        };
        let repository = py
            .detach(|| {
                let checkpointed = self
                    .store
                    .apply_repository(current, &checkpoint, &Fault::default())?
                    .expect("put_checkpoint retains the repository");
                let expected_revision = checkpointed
                    .releases
                    .iter()
                    .find(|release| release.version == version)
                    .map_or(0, |release| release.revision);
                let mutation = Mutation::UpdateRelease {
                    expected_revision,
                    repo: self.repo.clone(),
                    remove: Vec::new(),
                    set: vec![ReleaseLane {
                        lane: lane.to_string(),
                        manifest,
                    }],
                    version: version.to_string(),
                };
                self.store.apply_repository(
                    Some(&checkpointed.canonical_bytes()),
                    &mutation,
                    &Fault::default(),
                )
            })
            .or_refuse(py)?
            .expect("update_release retains the repository");
        self.catalog
            .complete_operation(&self.operation_id)
            .or_refuse(py)?;
        self.guard = None;
        Ok(PyBytes::new(py, &repository.canonical_bytes()))
    }

    fn __repr__(&self) -> String {
        format!(
            "<tensorfs.Operation {} repo={}/{}>",
            self.operation_id, self.repo.org, self.repo.name
        )
    }
}

// ---------------------------------------------------------------- exact derived manifests

#[pyclass(name = "DerivedWriter")]
pub struct PyDerivedWriter {
    store: Arc<CoreStore>,
    meta: Arc<Meta>,
    transaction_id: String,
    writer_session_id: u64,
    source_views: Vec<derived::SourceView>,
    guards: Mutex<Option<DerivedGuards>>,
}

struct DerivedGuards {
    writer_hold: Hold,
    source_leases: Vec<(String, CoreLease)>,
}

impl PyDerivedWriter {
    fn release_holds(&self) {
        let Some(guards) = self
            .guards
            .lock()
            .expect("writer hold mutex poisoned")
            .take()
        else {
            return;
        };
        for (_, lease) in guards.source_leases {
            if let Err(error) = lease.release(&self.meta) {
                eprintln!(
                    "DEGRADED derived-source-lease-release transaction={} — {error}",
                    self.transaction_id
                );
            }
        }
        if let Err(error) = guards.writer_hold.release(&self.meta) {
            eprintln!(
                "DEGRADED derived-writer-hold-release transaction={} — {error}",
                self.transaction_id
            );
        }
    }
}

#[pymethods]
impl PyDerivedWriter {
    /// Effective exact sources after TensorFS has checked installed identity and closure.
    fn sources<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        source_facts(py, &self.source_views)
    }

    /// Accepted part keys only; object identities stay behind the Runtime capability.
    fn completed_parts(&self, py: Python<'_>) -> PyResult<Vec<(String, String, String)>> {
        py.detach(|| {
            derived::completed(&self.meta, &self.transaction_id, self.writer_session_id)
                .map(|(parts, _)| parts)
        })
        .or_refuse(py)
    }

    fn completed_configs(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| {
            derived::completed(&self.meta, &self.transaction_id, self.writer_session_id)
                .map(|(_, configs)| configs)
        })
        .or_refuse(py)
    }

    #[pyo3(signature = (operation_id, slot, *, previous=None))]
    fn checkpoint<'p>(
        &self,
        py: Python<'p>,
        operation_id: &str,
        slot: &str,
        previous: Option<(String, u64)>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let previous = previous
            .map(|(id, length)| obj_ref(py, &id, length))
            .transpose()?;
        let (plan, head) = py
            .detach(|| {
                derived::checkpoint_progress(
                    &self.store,
                    &self.meta,
                    &self.transaction_id,
                    self.writer_session_id,
                    operation_id,
                    slot,
                    previous.as_ref(),
                )
            })
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("plan_digest", plan)?;
        out.set_item("head", head.head.as_ref().map(ObjectRef::id))?;
        out.set_item("head_length", head.head.as_ref().map_or(0, |r| r.length))?;
        out.set_item("index", head.links.saturating_sub(1))?;
        out.set_item("bytes", head.bytes)?;
        Ok(out)
    }

    /// Read one semantic source-part range through its pinned verified descriptor set.
    /// ObjectRefs never cross this author door: TensorFS resolves the exact locked header,
    /// including inline parts and segmented ranges.
    #[allow(clippy::too_many_arguments)]
    fn source_read_into(
        &self,
        py: Python<'_>,
        source: &str,
        component: &str,
        key: &str,
        role: &str,
        off: u64,
        into: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut buffer = Writable::new(py, into)?;
        let destination = buffer.slice();
        let guards = self.guards.lock().expect("writer hold mutex poisoned");
        let guards = guards.as_ref().ok_or_else(|| released(py))?;
        let lease = guards
            .source_leases
            .iter()
            .find(|(alias, _)| alias == source)
            .map(|(_, lease)| lease)
            .ok_or_else(|| {
                to_py(
                    py,
                    &Refusal {
                        code: Code::ROOT_ABSENT,
                        detail: format!("no source alias {source:?} on this derived writer"),
                    },
                )
            })?;
        let view = self
            .source_views
            .iter()
            .find(|view| view.fact.alias == source)
            .ok_or_else(|| missing(py, "derived source", source))?;
        if !view
            .fact
            .components
            .iter()
            .any(|allowed| allowed == component)
        {
            return Err(to_py(
                py,
                &Refusal {
                    code: Code::MISSING_COMPONENT,
                    detail: format!(
                        "source alias {source:?} component {component:?} is outside the locked target declaration"
                    ),
                },
            ));
        }
        let part = view
            .header
            .components
            .iter()
            .find(|(name, _)| name == component)
            .and_then(|(_, tensors)| tensors.iter().find(|(name, _)| name == key))
            .and_then(|(_, tensor)| tensor.parts.iter().find(|(name, _)| name == role))
            .map(|(_, part)| part)
            .ok_or_else(|| {
                to_py(
                    py,
                    &Refusal {
                        code: Code::MISSING_TENSOR,
                        detail: format!(
                            "source alias {source:?} has no exact part {component}/{key}#{role}"
                        ),
                    },
                )
            })?;
        let what = format!("{component}/{key}#{role}");
        py.detach(|| {
            derived::check_writer(&self.meta, &self.transaction_id, self.writer_session_id)?;
            read::read_part_into(
                lease,
                &what,
                part,
                off,
                destination.len() as u64,
                destination,
            )?;
            derived::check_writer(&self.meta, &self.transaction_id, self.writer_session_id)
        })
        .or_refuse(py)
    }

    /// Stream one declared physical role. A pathname is intentionally not an accepted
    /// shape; Runtime hands in a capability-backed reader.
    fn add_part<'p>(
        &self,
        py: Python<'p>,
        component: &str,
        key: &str,
        role: &str,
        reader: &Bound<'p, PyAny>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let mut source = Readable {
            obj: reader.clone().unbind(),
            err: None,
        };
        let result = py.detach(|| {
            derived::add_part(
                &self.store,
                &self.meta,
                &self.transaction_id,
                self.writer_session_id,
                (component, key, role),
                &mut source,
            )
        });
        let result = match result {
            Ok(value) => value,
            Err(error) => return Err(source.err.take().unwrap_or_else(|| to_py(py, &error))),
        };
        let out = PyDict::new(py);
        out.set_item("bytes", result.bytes)?;
        out.set_item("deduped_bytes", result.deduped)?;
        out.set_item("whole_sha256", result.whole)?;
        let objects = PyList::empty(py);
        for object in result.part.segments() {
            let row = PyDict::new(py);
            row.set_item("sha256", &object.sha256)?;
            row.set_item("length", object.length)?;
            objects.append(row)?;
        }
        out.set_item("objects", objects)?;
        Ok(out)
    }

    fn add_config<'p>(
        &self,
        py: Python<'p>,
        name: &str,
        reader: &Bound<'p, PyAny>,
    ) -> PyResult<()> {
        let mut source = Readable {
            obj: reader.clone().unbind(),
            err: None,
        };
        let result = py.detach(|| {
            derived::add_config(
                &self.meta,
                &self.transaction_id,
                self.writer_session_id,
                name,
                &mut source,
            )
        });
        match result {
            Ok(()) => Ok(()),
            Err(error) => Err(source.err.take().unwrap_or_else(|| to_py(py, &error))),
        }
    }

    fn commit<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let value = py
            .detach(|| {
                let guards = self.guards.lock().expect("writer hold mutex poisoned");
                let guards = guards.as_ref().ok_or_else(|| Refusal {
                    code: Code::LEASE_REVOKED,
                    detail: "derived writer holds were released".into(),
                })?;
                guards.writer_hold.still_registered(&self.meta)?;
                derived::commit(
                    &self.store,
                    &self.meta,
                    &self.transaction_id,
                    self.writer_session_id,
                )
            })
            .or_refuse(py)?;
        self.release_holds();
        receipt(py, &value)
    }

    fn fence(&self, py: Python<'_>) -> PyResult<bool> {
        let fenced = py
            .detach(|| derived::fence(&self.meta, &self.transaction_id, self.writer_session_id))
            .or_refuse(py)?;
        self.release_holds();
        Ok(fenced)
    }

    fn __repr__(&self) -> String {
        format!(
            "<tensorfs.DerivedWriter transaction={} writer_session_id={}>",
            self.transaction_id, self.writer_session_id
        )
    }
}

impl PyStore {
    fn load(&self, py: Python<'_>, hex: &str) -> PyResult<(Manifest, Vec<u8>, Vec<u8>)> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let len = std::fs::metadata(self.store.manifest_path(&hex))
            .map(|m| m.len())
            .unwrap_or(0);
        let r = ObjectRef {
            sha256: hex,
            length: len,
        };
        let m = checkpoint::load_manifest(&self.store, &r).or_refuse(py)?;
        let hr = match m.header() {
            Some(h) => h.clone(),
            None => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::MISSING_FIELD,
                        detail: format!("manifest {} attaches no header", m.manifest_id()),
                    },
                ))
            }
        };
        let h = checkpoint::load_header(&self.store, &hr).or_refuse(py)?;
        Ok((
            m.clone(),
            m.canonical_bytes(),
            h.canonical_bytes().or_refuse(py)?,
        ))
    }
}

// ---------------------------------------------------------------- the plan

#[pyclass(name = "ReadPlan")]
pub struct PyPlan {
    pub plan: read::ReadPlan,
}

#[pymethods]
impl PyPlan {
    /// Select exact existing part `what` identifiers, retaining all their native windows.
    fn select(&self, py: Python<'_>, whats: Vec<String>) -> PyResult<PyPlan> {
        Ok(PyPlan {
            plan: self.plan.select(&whats).or_refuse(py)?,
        })
    }

    #[getter]
    fn bytes(&self) -> u64 {
        self.plan.bytes
    }
    /// The DISK leg — `bytes` minus what the header carries inline. Throughput belongs to
    /// this one; inline bytes never touch a device.
    #[getter]
    fn io_bytes(&self) -> u64 {
        self.plan.io_bytes
    }
    #[getter]
    fn inline_parts(&self) -> usize {
        self.plan.inline_parts
    }
    #[getter]
    fn order(&self) -> &'static str {
        self.plan.order.as_str()
    }
    fn __len__(&self) -> usize {
        self.plan.items.len()
    }

    /// The plan rows, in destination order: what, where it lands, how long, and where the
    /// bytes come from.
    fn items<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let l = PyList::empty(py);
        for it in &self.plan.items {
            let d = PyDict::new(py);
            d.set_item("what", &it.what)?;
            d.set_item("dest_off", it.dest_off)?;
            d.set_item("len", it.len)?;
            match &it.source {
                ReadSource::Object(r) => {
                    d.set_item("source", "object")?;
                    d.set_item("object", r.obj.id())?;
                    d.set_item("object_length", r.obj.length)?;
                    d.set_item("off", r.off)?;
                }
                ReadSource::Inline(_) => {
                    d.set_item("source", "inline")?;
                }
            }
            l.append(d)?;
        }
        Ok(l)
    }

    /// How the plan tiles into slots of `slot_bytes` — the batch count the ring will run.
    fn batches(&self, slot_bytes: u64) -> usize {
        self.plan.batches(slot_bytes).len()
    }
}

/// Build a read plan from canonical header bytes and the CALLER's traversal. Order is the
/// caller's tensor requirements.
///
/// `components` DECLARES the scope completeness is checked against (#570b): the traversal
/// must name every tensor the header carries for those components and none outside them.
/// Absent, it defaults to every component the header carries — the whole-artifact reading,
/// which is what a caller with no tensor requirements to narrow by means.
#[pyfunction]
#[pyo3(signature = (header, traversal, components=None, window=1<<24))]
pub fn plan(
    py: Python<'_>,
    header: &[u8],
    traversal: Vec<(String, String)>,
    components: Option<Vec<String>>,
    window: u64,
) -> PyResult<PyPlan> {
    let h = Header::parse(header).or_refuse(py)?;
    let scope = components.unwrap_or_else(|| h.components.iter().map(|(c, _)| c.clone()).collect());
    let p = read::plan_for_traversal(&h, &traversal, &scope, window).or_refuse(py)?;
    Ok(PyPlan { plan: p })
}

// ---------------------------------------------------------------- the lease

#[pyclass(name = "ReadLease")]
pub struct PyLease {
    // Every attached caller must detach before waiting: a reader holding this lock
    // may need the GIL again to finish I/O or invoke a stream callback.
    lease: Mutex<Option<CoreLease>>,
    meta: Arc<Meta>,
    receipts: Receipts,
}

fn released(py: Python<'_>) -> PyErr {
    to_py(
        py,
        &Refusal {
            code: Code::LEASE_REVOKED,
            detail: "this lease was already released — a released lease never serves bytes".into(),
        },
    )
}

#[pymethods]
impl PyLease {
    #[getter]
    fn hold_id(&self, py: Python<'_>) -> PyResult<String> {
        let g = self.lease.lock_py_attached(py).unwrap();
        Ok(g.as_ref()
            .ok_or_else(|| released(py))?
            .hold_id()
            .to_string())
    }

    #[getter]
    fn manifest(&self, py: Python<'_>) -> PyResult<String> {
        let g = self.lease.lock_py_attached(py).unwrap();
        Ok(g.as_ref().ok_or_else(|| released(py))?.manifest.clone())
    }

    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<u64> {
        let g = self.lease.lock_py_attached(py).unwrap();
        Ok(g.as_ref().ok_or_else(|| released(py))?.bytes())
    }

    #[getter]
    fn objects(&self, py: Python<'_>) -> PyResult<Vec<(String, u64)>> {
        let g = self.lease.lock_py_attached(py).unwrap();
        Ok(g.as_ref()
            .ok_or_else(|| released(py))?
            .objects()
            .iter()
            .map(|o| (o.id(), o.length))
            .collect())
    }

    /// The degraded-behaviour receipts `acquire` emitted (a rehash on an invalidated record).
    #[getter]
    fn receipts<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        receipts_to_py(py, &self.receipts)
    }

    /// Re-check against the authority: one document read, and the arm that makes a revoked
    /// lease refuse instead of serving stale bytes.
    fn recheck(&self, py: Python<'_>) -> PyResult<()> {
        let g = self.lease.lock_py_attached(py).unwrap();
        g.as_ref()
            .ok_or_else(|| released(py))?
            .recheck(&self.meta)
            .or_refuse(py)
    }

    /// THE explicit end. Nothing else releases the lease — dropping it keeps the row for
    /// the reaper and announces itself.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        let l = self.lease.lock_py_attached(py).unwrap().take();
        match l {
            Some(l) => py.detach(|| l.release(&self.meta)).or_refuse(py),
            None => Err(released(py)),
        }
    }

    #[getter]
    fn live(&self, py: Python<'_>) -> bool {
        self.lease.lock_py_attached(py).unwrap().is_some()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.live(py) {
            self.release(py)?;
        }
        Ok(false)
    }

    /// Fill a CALLER-OWNED buffer from one verified range. Exact fit or refusal; the GIL is
    /// released for the read.
    fn read_into(
        &self,
        py: Python<'_>,
        object: &str,
        object_length: u64,
        off: u64,
        length: u64,
        into: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let mut buf = Writable::new(py, into)?;
        let dest = buf.slice();
        let g = self.lease.lock_py_attached(py).unwrap();
        let lease = g.as_ref().ok_or_else(|| released(py))?;
        let r = ObjectRange {
            obj: obj_ref(py, object, object_length)?,
            off,
            len: length,
        };
        py.detach(|| read::read_into(lease, &r, dest)).or_refuse(py)
    }

    /// Read one self-contained model asset through the lease's pinned descriptors.
    #[pyo3(signature = (header, name, max_bytes=67_108_864))]
    fn read_asset<'p>(
        &self,
        py: Python<'p>,
        header: &[u8],
        name: &str,
        max_bytes: u64,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let header_ref = ObjectRef::of(header);
        let header = Header::parse(header).or_refuse(py)?;
        let asset = header
            .assets
            .iter()
            .find(|(candidate, _)| candidate == name)
            .map(|(_, asset)| asset)
            .ok_or_else(|| missing(py, "model asset", name))?;
        let guard = self.lease.lock_py_attached(py).unwrap();
        let lease = guard.as_ref().ok_or_else(|| released(py))?;
        lease
            .covers(&ObjectRange {
                obj: header_ref,
                off: 0,
                len: 0,
            })
            .or_refuse(py)?;
        let bytes = py
            .detach(|| read::read_asset(lease, name, asset, max_bytes))
            .or_refuse(py)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// **The streaming shape.** One plan, a ring of CALLER-owned slots, and a reader pool
    /// that never stops at a batch boundary.
    ///
    /// `on_batch(batch)` is called once per filled slot, in plan order, with the GIL held
    /// and the readers still running. `batch.slot` indexes the caller's own `slots` list, so
    /// the bytes were never copied. `batch.release()` returns the slot to the ring — after
    /// the H2D completion event, not before, exactly like the lease itself.
    #[pyo3(signature = (plan, slots, on_batch, readers=4))]
    fn stream<'p>(
        &self,
        py: Python<'p>,
        plan: &PyPlan,
        slots: Vec<Bound<'p, PyAny>>,
        on_batch: Bound<'p, PyAny>,
        readers: usize,
    ) -> PyResult<Bound<'p, PyDict>> {
        let mut bufs: Vec<Writable> = slots
            .iter()
            .map(|o| Writable::new(py, o))
            .collect::<PyResult<_>>()?;
        let nslots = bufs.len();
        let views: Vec<&mut [u8]> = bufs.iter_mut().map(|w| w.slice()).collect();
        let g = self.lease.lock_py_attached(py).unwrap();
        let lease = g.as_ref().ok_or_else(|| released(py))?;

        let raised: Mutex<Option<PyErr>> = Mutex::new(None);
        let cb = on_batch.unbind();
        let started = std::time::Instant::now();
        let out = py.detach(|| {
            read::pump(
                lease,
                &plan.plan,
                views,
                readers,
                |filled, slot| -> TResult<()> {
                    Python::attach(|py| {
                        let b = PyBatch {
                            index: filled.index,
                            slot: slot.index(),
                            nbytes: filled.batch.bytes,
                            io_bytes: filled.batch.io_bytes,
                            read_ms: filled.read_ms as u64,
                            items: filled
                                .batch
                                .items
                                .iter()
                                .map(|i| (i.what.clone(), i.dest_off, i.len, i.is_io()))
                                .collect(),
                            handle: slot.clone(),
                        };
                        match cb.bind(py).call1((b,)) {
                            Ok(_) => Ok(()),
                            Err(e) => {
                                *raised.lock().unwrap() = Some(e);
                                refuse(
                                    Code::IO_FAILED,
                                    "the consumer's on_batch raised — the ring stopped and the \
                                     lease is untouched",
                                )
                            }
                        }
                    })
                },
            )
        });
        if let Some(e) = raised.lock().unwrap().take() {
            return Err(e);
        }
        let st = out.or_refuse(py)?;
        let ms = started.elapsed().as_millis() as u64;
        let d = PyDict::new(py);
        d.set_item("items", st.items)?;
        d.set_item("bytes", st.bytes)?;
        d.set_item("io_bytes", st.io_bytes)?;
        d.set_item("readers", st.workers)?;
        d.set_item("slots", nslots)?;
        d.set_item("elapsed_ms", ms)?;
        Ok(d)
    }
}

/// One filled slot, described in the caller's own terms.
#[pyclass(name = "Batch")]
pub struct PyBatch {
    #[pyo3(get)]
    index: usize,
    /// Index into the caller's `slots` list. The bytes are in the caller's own buffer.
    #[pyo3(get)]
    slot: usize,
    #[pyo3(get)]
    nbytes: u64,
    #[pyo3(get)]
    io_bytes: u64,
    #[pyo3(get)]
    read_ms: u64,
    items: Vec<(String, u64, u64, bool)>,
    handle: Arc<read::Slot>,
}

#[pymethods]
impl PyBatch {
    /// (what, offset within the slot, length, came-from-disk) for every span in this slot.
    fn items<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyList>> {
        let l = PyList::empty(py);
        for (what, off, len, io) in &self.items {
            let d = PyDict::new(py);
            d.set_item("what", what)?;
            d.set_item("off", off)?;
            d.set_item("len", len)?;
            d.set_item("io", io)?;
            l.append(d)?;
        }
        Ok(l)
    }

    /// Return this slot to the ring. Explicit, and late on purpose: the runtime calls it
    /// when the H2D completion event fires, not when the callback returns.
    fn release(&self) -> bool {
        self.handle.release()
    }

    #[getter]
    fn released(&self) -> bool {
        self.handle.is_released()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, PyAny>) -> bool {
        self.handle.release();
        false
    }

    fn __repr__(&self) -> String {
        format!(
            "<tensorfs.Batch {} slot={} {} B{}>",
            self.index,
            self.slot,
            self.nbytes,
            if self.handle.is_released() {
                " released"
            } else {
                ""
            }
        )
    }
}

// ---------------------------------------------------------------- reclamation

/// One local reclamation pass (tensorfs_core::gc): remove every blob and manifest no
/// repository or open ingest session names, from the filesystem census alone, under the
/// store's exclusive lock. `dry_run` plans without deleting.
#[pyfunction]
#[pyo3(signature = (root, dry_run = false))]
pub fn gc<'p>(py: Python<'p>, root: PathBuf, dry_run: bool) -> PyResult<Bound<'p, PyDict>> {
    let report = py
        .detach(|| tensorfs_core::gc::collect(&root, dry_run))
        .or_refuse(py)?;
    let result = PyDict::new(py);
    result.set_item("dry_run", report.dry_run)?;
    result.set_item("reclaimed_bytes", report.reclaimed_bytes)?;
    result.set_item("reclaimed_blobs", report.reclaimed_blobs)?;
    result.set_item("reclaimed_manifests", report.reclaimed_manifests)?;
    result.set_item("kept_bytes", report.kept_bytes)?;
    result.set_item("kept_objects", report.kept_objects)?;
    result.set_item("sessions", report.sessions)?;
    result.set_item("scratch_reaped", report.scratch_reaped)?;
    Ok(result)
}

// ---------------------------------------------------------------- store transfer

/// Move one exact object between TensorFS stores without exposing either store's layout.
/// The source is a verified pinned descriptor; the destination uses the ordinary hashing,
/// length check, no-clobber publication and verification-record path.
#[pyfunction]
pub fn transfer_object<'p>(
    py: Python<'p>,
    source: &PyStore,
    destination: &PyStore,
    object_id: &str,
    expected_length: u64,
) -> PyResult<Bound<'p, PyDict>> {
    let want = obj_ref(py, object_id, expected_length)?;
    let put = py
        .detach(|| source.store.transfer_object(&destination.store, &want))
        .or_refuse(py)?;
    let result = PyDict::new(py);
    result.set_item("id", put.obj.id())?;
    result.set_item("length", put.obj.length)?;
    result.set_item("admitted", put.admitted)?;
    Ok(result)
}

#[pyfunction]
pub fn transfer_manifest<'p>(
    py: Python<'p>,
    source: &PyStore,
    destination: &PyStore,
    manifest_id: &str,
    expected_length: u64,
) -> PyResult<Bound<'p, PyDict>> {
    let wanted = obj_ref(py, manifest_id, expected_length)?;
    let put = py
        .detach(|| source.store.transfer_manifest(&destination.store, &wanted))
        .or_refuse(py)?;
    let result = PyDict::new(py);
    result.set_item("id", put.obj.id())?;
    result.set_item("length", put.obj.length)?;
    result.set_item("admitted", put.admitted)?;
    Ok(result)
}

/// Split one typed Manifest plus its declared blob closure into held and wanted keys.
/// `fetch_admit` routes the root to `manifests/` and every other grant to `blobs/`.
#[pyfunction]
#[pyo3(signature = (store, session, manifest, manifest_length, declared))]
pub fn fetch_plan<'p>(
    py: Python<'p>,
    store: &PyStore,
    session: &str,
    manifest: &str,
    manifest_length: u64,
    declared: Vec<(String, u64)>,
) -> PyResult<Bound<'p, PyDict>> {
    let manifest = obj_ref(py, manifest, manifest_length)?;
    let refs = declared
        .iter()
        .map(|(identity, length)| obj_ref(py, identity, *length))
        .collect::<PyResult<Vec<_>>>()?;
    let (plan, why) = tensorfs_core::fetch::FetchPlan::of(&store.store, session, &manifest, &refs)
        .or_refuse(py)?;
    let objects = |values: &[ObjectRef]| -> PyResult<Bound<'p, PyList>> {
        let result = PyList::empty(py);
        for object in values {
            result.append((object.id(), object.length))?;
        }
        Ok(result)
    };
    let presence = PyDict::new(py);
    for (object, reason) in why {
        presence.set_item(object.id(), reason.as_str())?;
    }
    let result = PyDict::new(py);
    result.set_item("session", &plan.session)?;
    result.set_item(
        "manifest",
        plan.manifest
            .as_ref()
            .expect("fetch_plan is built with a manifest")
            .id(),
    )?;
    result.set_item("held", objects(&plan.held)?)?;
    result.set_item("wanted", objects(&plan.wanted)?)?;
    result.set_item("wanted_bytes", plan.wanted_bytes())?;
    result.set_item("held_bytes", plan.held_bytes())?;
    result.set_item("presence", presence)?;
    result.set_item("document", PyBytes::new(py, &plan.canonical_bytes()))?;
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (store, plan, object_id, reader))]
pub fn fetch_admit<'p>(
    py: Python<'p>,
    store: &PyStore,
    plan: &[u8],
    object_id: &str,
    reader: &Bound<'p, PyAny>,
) -> PyResult<Bound<'p, PyDict>> {
    let plan = tensorfs_core::fetch::FetchPlan::parse(plan).or_refuse(py)?;
    let digest = object_id.trim_start_matches("sha256:");
    let length = plan
        .wanted
        .iter()
        .chain(&plan.held)
        .find(|object| object.sha256 == digest)
        .map(|object| object.length)
        .unwrap_or(0);
    let object = obj_ref(py, digest, length)?;
    let grant = tensorfs_core::fetch::DeliveryGrant::mint(&plan, &object).or_refuse(py)?;
    let mut source = Readable {
        obj: reader.clone().unbind(),
        err: None,
    };
    let admitted = match py.detach(|| grant.admit(&store.store, &mut source)) {
        Ok(admitted) => admitted,
        Err(error) => return Err(source.err.take().unwrap_or_else(|| to_py(py, &error))),
    };
    let result = PyDict::new(py);
    result.set_item("id", admitted.object.id())?;
    result.set_item("length", admitted.object.length)?;
    result.set_item("admitted", admitted.first_writer)?;
    result.set_item("key", grant.key())?;
    Ok(result)
}

#[pyfunction]
pub fn fetch_complete(py: Python<'_>, store: &PyStore, plan: &[u8]) -> PyResult<u64> {
    let plan = tensorfs_core::fetch::FetchPlan::parse(plan).or_refuse(py)?;
    py.detach(|| plan.complete(&store.store)).or_refuse(py)?;
    Ok(plan.declared_bytes())
}

/// Prove the unchanged FetchPlan against Manifest + selected CozyTensors runtime objects only.
#[pyfunction]
pub fn fetch_complete_cozytensors(py: Python<'_>, store: &PyStore, plan: &[u8]) -> PyResult<u64> {
    let plan = tensorfs_core::fetch::FetchPlan::parse(plan).or_refuse(py)?;
    py.detach(|| plan.complete_cozytensors(&store.store))
        .or_refuse(py)?;
    Ok(plan.declared_bytes())
}

/// The ONE transport, from Python (tfs-050): ask for a repo, get a store that has it.
/// cozy-runtime sheds its own downloader (cr-090); this call IS the mover — closure,
/// plan against the real store, the 16-way walk presigning `wanted` as it reaches it, and
/// the completion proof, all in the compiled core.
///
/// `credential` is the one spelling (`bearer <token>`, or empty for anonymous — which is
/// what a rented pod uses, the hub's closure and presign routes taking no credential),
/// scoped to the hub's host. `timeout` is the caller's deadline or None for none —
/// nothing below invents a number of seconds. `sample_seconds` is the liveness ledger's
/// RESOLUTION (configuration, never a verdict); leave it defaulted outside tests.
#[pyfunction]
#[pyo3(signature = (store, hub, refspec, *, lane=String::new(), credential=String::new(),
       timeout=None, session=String::new(), streams=tensorfs_core::transport::STREAMS,
       allowed_hosts=Vec::new(), allow_local=false, sample_seconds=None))]
#[allow(clippy::too_many_arguments)]
pub fn pull<'p>(
    py: Python<'p>,
    store: &PyStore,
    hub: &str,
    refspec: &str,
    lane: String,
    credential: String,
    timeout: Option<f64>,
    session: String,
    streams: usize,
    allowed_hosts: Vec<String>,
    allow_local: bool,
    sample_seconds: Option<f64>,
) -> PyResult<Bound<'p, PyDict>> {
    use tensorfs_core::transport;
    let hub_host = transport::base_host(hub).or_refuse(py)?;
    let scoped = transport::credential_from_spec(&credential, vec![hub_host]).or_refuse(py)?;
    let policy = transport::SourcePolicy {
        allowed_hosts,
        allow_local,
        max_redirects: 0,
    };
    let report = py
        .detach(|| {
            let mut request =
                transport::PullRequest::new(&store.store, hub, refspec, &scoped, &policy);
            request.lane = &lane;
            request.session = &session;
            request.streams = streams;
            request.deadline = transport::Deadline::after_seconds(timeout);
            if let Some(sample) = sample_seconds {
                request.sample_seconds = sample;
            }
            transport::pull(&request)
        })
        .or_refuse(py)?;
    let result = PyDict::new(py);
    result.set_item("ref", &report.refspec)?;
    result.set_item("model", &report.model)?;
    result.set_item("release", &report.release)?;
    result.set_item("lane", &report.lane)?;
    result.set_item("manifest", &report.manifest)?;
    result.set_item("session", &report.session)?;
    result.set_item("store", store.store.root().display().to_string())?;
    result.set_item("streams", report.streams)?;
    result.set_item("declared", report.declared)?;
    result.set_item("held", report.held)?;
    result.set_item("fetched", report.fetched)?;
    result.set_item("bytes_moved", report.bytes_moved)?;
    result.set_item("bytes_held", report.bytes_held)?;
    result.set_item("bytes_total", report.bytes_total)?;
    let presence = PyDict::new(py);
    for (id, why) in &report.presence {
        presence.set_item(id, why)?;
    }
    result.set_item("presence", presence)?;
    let stamps = PyDict::new(py);
    for (key, values) in &report.stamps {
        stamps.set_item(key, values.clone())?;
    }
    result.set_item("stamps", stamps)?;
    Ok(result)
}
