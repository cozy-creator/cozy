//! The store, the lease and the derived writer — marshalled, never reimplemented.
//!
//! Two properties this module exists to preserve across the boundary:
//!
//! 1. **The caller owns the memory.** Every read fills a buffer Python already owns, through
//!    the stable-ABI `Py_buffer`. TensorFS allocates no destination, and the runtime's one
//!    budgeted pinned pool stays the runtime's. Nothing is copied to hand bytes to Python.
//! 2. **The GIL is not held while bytes move.** Every read runs inside `Python::detach`.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::sync::MutexExt;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::catalog::{Catalog, WriterGuard};
use tensorfs_core::checkpoint;
use tensorfs_core::derived::{
    self, ComponentDeclaration, ConfigDeclaration, Declaration, PartDeclaration, PartSource,
    Source, TensorDeclaration,
};
use tensorfs_core::err::{Code, Refusal};
use tensorfs_core::header::Header;
use tensorfs_core::ids::{prefixed, Doc, ObjectRef};
use tensorfs_core::manifest::Manifest;
use tensorfs_core::meta::{Hold, Meta};
use tensorfs_core::project;
use tensorfs_core::read::{self, ObjectRange, ReadLease as CoreLease};
use tensorfs_core::repo_cache::{CacheKind, RepoObjectCache as CoreRepoObjectCache};
use tensorfs_core::repository::{Mutation, ReleaseLane, RepositoryName};
use tensorfs_core::storage;
use tensorfs_core::store::{Fault, Store as CoreStore};

use crate::docs::marshal;
use crate::errors::{to_py, IntoPy};

// ---------------------------------------------------------------- buffers

/// The caller's writable buffer. The one `unsafe` in this crate, and it is the point of the
/// crate: a `readinto` that copied would defeat the whole design.
///
/// Soundness: the exporter is kept alive by this `PyBuffer` for the whole call, and the GIL is
/// released only after the slice exists.
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
    files: Option<&Bound<'_, PyDict>>,
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
                    &["dtype", "shape", "source"],
                )?;
                let dtype = tensorfs_core::dtype::Dtype::parse(
                    &required(py, part, "part", "dtype")?.extract::<String>()?,
                )
                .or_refuse(py)?;
                let shape: Vec<u64> = required(py, part, "part", "shape")?.extract()?;
                let source = part
                    .get_item("source")?
                    .map(|value| -> PyResult<PartSource> {
                        let selected = value.cast::<PyDict>()?;
                        only_keys(
                            py,
                            selected,
                            "part source",
                            &["source", "component", "tensor", "role"],
                        )?;
                        Ok(PartSource {
                            source: required(py, selected, "part source", "source")?.extract()?,
                            component: required(py, selected, "part source", "component")?
                                .extract()?,
                            tensor: required(py, selected, "part source", "tensor")?.extract()?,
                            role: required(py, selected, "part source", "role")?.extract()?,
                        })
                    })
                    .transpose()?;
                parts.push(PartDeclaration {
                    role,
                    dtype,
                    shape,
                    source,
                });
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
    // A companion is base64 bytes, or `(object_id, length)` naming a verified Store object.
    let (mut companion_files, mut companion_objects) = (Vec::new(), Vec::new());
    if let Some(files) = files {
        for (path, value) in files.iter() {
            let path: String = path.extract()?;
            if let Ok((object, length)) = value.extract::<(String, u64)>() {
                companion_objects.push((path, obj_ref(py, &object, length)?));
                continue;
            }
            let encoded: String = value.extract()?;
            companion_files.push((path, derived::decode_file(&encoded).or_refuse(py)?));
        }
        if companion_files.len() > derived::MAX_COMPANION_FILES {
            return tensorfs_core::err::refuse(
                tensorfs_core::err::Code::COUNT_CAP,
                "at most 16 companion files may be declared",
            )
            .or_refuse(py);
        }
    }
    let mut declaration = Declaration {
        files: companion_files,
        objects: companion_objects,
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
        row.set_item("identity", tensor.encoded_identity().or_refuse(py)?)?;
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

fn tree_root_dict<'p>(
    py: Python<'p>,
    root: &tensorfs_core::source_artifact::TreeRoot,
) -> PyResult<Bound<'p, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("owner", &root.owner)?;
    result.set_item("producer", &root.producer)?;
    if root.complete {
        let receipt = root.receipt().or_refuse(py)?;
        result.set_item("receipt_digest", tensorfs_core::ids::object_id(&receipt))?;
        result.set_item("receipt", PyBytes::new(py, &receipt))?;
    }
    result.set_item("manifest_digest", root.manifest.id())?;
    result.set_item("manifest_length", root.manifest.length)?;
    result.set_item("complete", root.complete)?;
    result.set_item("released", root.released)?;
    Ok(result)
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

    /// Import exact manifest members, retaining each member before byte admission.
    fn import_tree<'p>(
        &self,
        py: Python<'p>,
        owner: &str,
        manifest_bytes: &[u8],
        files: Vec<(String, PathBuf)>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let tree = Manifest::parse(manifest_bytes).or_refuse(py)?;
        let root = py
            .detach(|| {
                tensorfs_core::source_artifact::import_tree(
                    &self.store,
                    owner,
                    &tree,
                    &files,
                    &Fault::default(),
                )
            })
            .or_refuse(py)?;
        tree_root_dict(py, &root)
    }

    fn contains(&self, hex: &str) -> bool {
        self.store.contains(hex.trim_start_matches("sha256:"))
    }

    #[getter]
    fn repo_cache(&self) -> Option<String> {
        self.store
            .repo_cache()
            .map(|cache| cache.root().display().to_string())
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

    /// Canonical manifest bytes and an optional CozyTensors header selected by its typed entry.
    fn manifest<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyDict>> {
        let (m, mb, hb) = self.load(py, hex)?;
        let d = PyDict::new(py);
        d.set_item("manifest_id", m.manifest_id())?;
        d.set_item("manifest", PyBytes::new(py, &mb))?;
        d.set_item("header", hb.as_deref().map(|bytes| PyBytes::new(py, bytes)))?;
        Ok(d)
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
        let (lease, _) = py
            .detach(|| read::acquire(&self.store, &self.meta, &snap, refs))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
        })
    }

    /// Acquire over everything one manifest commits — the serving case.
    fn acquire_manifest(&self, py: Python<'_>, hex: &str) -> PyResult<PyLease> {
        let (m, _, _) = self.load(py, hex)?;
        let (lease, _) = py
            .detach(|| read::acquire_manifest(&self.store, &self.meta, &m))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
        })
    }

    /// Acquire the selected CozyTensors runtime closure without optional snapshot siblings.
    fn acquire_cozytensors(&self, py: Python<'_>, hex: &str) -> PyResult<PyLease> {
        let (m, _, _) = self.load(py, hex)?;
        let (lease, _) = py
            .detach(|| read::acquire_cozytensors(&self.store, &self.meta, &m))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            meta: self.meta.clone(),
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
        // The selected fact is a normalized set. Author inspection must retain
        // the header's construction traversal, including nonlexical components.
        for (component, _) in &inspection.view.header.components {
            if !inspection.view.fact.components.contains(component) {
                continue;
            }
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
    #[pyo3(signature = (sources, targets, configs, order, max_new_bytes, files=None, *, work_fingerprint))]
    fn derived_declaration<'p>(
        &self,
        py: Python<'p>,
        sources: &Bound<'_, PyDict>,
        targets: &Bound<'_, PyDict>,
        configs: &Bound<'_, PyDict>,
        order: Vec<(String, String)>,
        max_new_bytes: u64,
        files: Option<&Bound<'_, PyDict>>,
        work_fingerprint: &str,
    ) -> PyResult<Bound<'p, PyBytes>> {
        let mut declaration =
            parse_declaration(py, sources, targets, configs, order, max_new_bytes, files)?;
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

    /// Adopt exact checkpointed parts into independently owned recipient progress.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transaction_id, source_transaction_id, declaration, head_id, head_length, *, operation_id, slot))]
    fn adopt_derived_checkpoint<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
        source_transaction_id: &str,
        declaration: &[u8],
        head_id: &str,
        head_length: u64,
        operation_id: &str,
        slot: &str,
    ) -> PyResult<Bound<'p, PyDict>> {
        let head = obj_ref(py, head_id, head_length)?;
        let (plan, head) = py
            .detach(|| {
                derived::adopt_checkpoint(
                    &self.store,
                    &self.meta,
                    transaction_id,
                    source_transaction_id,
                    declaration,
                    &head,
                    operation_id,
                    slot,
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

    /// Open one exact tfs-019 transaction. Python supplies typed plain values; canonical
    /// declaration bytes are constructed and locked only by the compiled core.
    // PyO3 exposes these as six named boundary fields. Packing them into an opaque dict
    // would only move the arity into hand parsing and weaken the typed facade.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transaction_id, writer_session_id, sources, targets, configs, order, max_new_bytes, files=None, *, work_fingerprint, checkpoint=None))]
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
        files: Option<&Bound<'_, PyDict>>,
        work_fingerprint: &str,
        checkpoint: Option<(String, u64)>,
    ) -> PyResult<PyDerivedWriter> {
        let mut declaration =
            parse_declaration(py, sources, targets, configs, order, max_new_bytes, files)?;
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
        let store = py
            .detach(|| derived::writer_store(&self.store))
            .or_refuse(py)?;
        Ok(PyDerivedWriter {
            store: Arc::new(store),
            meta: self.meta.clone(),
            transaction_id: transaction_id.to_string(),
            writer_session_id,
            source_views: begin.source_views,
            additions: begin.additions,
            cursor: Mutex::new(None),
            guards: Mutex::new(Some(DerivedGuards {
                writer_hold: begin.writer_hold,
                source_leases: begin.source_leases,
            })),
        })
    }

    /// Remaining native additions before opening a writer or reserving capacity.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (transaction_id, writer_session_id, sources, targets, configs, order, max_new_bytes, files=None, *, work_fingerprint, checkpoint=None))]
    fn derived_write_estimate<'p>(
        &self,
        py: Python<'p>,
        transaction_id: &str,
        writer_session_id: u64,
        sources: &Bound<'_, PyDict>,
        targets: &Bound<'_, PyDict>,
        configs: &Bound<'_, PyDict>,
        order: Vec<(String, String)>,
        max_new_bytes: u64,
        files: Option<&Bound<'_, PyDict>>,
        work_fingerprint: &str,
        checkpoint: Option<(String, u64)>,
    ) -> PyResult<Bound<'p, PyDict>> {
        let mut declaration =
            parse_declaration(py, sources, targets, configs, order, max_new_bytes, files)?;
        declaration.work_fingerprint = Some(work_fingerprint.to_string());
        let checkpoint = checkpoint
            .map(|(id, length)| obj_ref(py, &id, length))
            .transpose()?;
        let estimate = py
            .detach(|| {
                derived::estimate_write(
                    &self.store,
                    &self.meta,
                    transaction_id,
                    writer_session_id,
                    declaration,
                    checkpoint.as_ref(),
                )
            })
            .or_refuse(py)?;
        let out = PyDict::new(py);
        out.set_item("payload_bytes", estimate.payload_bytes)?;
        out.set_item("largest_part_bytes", estimate.largest_part_bytes)?;
        out.set_item("parts", estimate.parts)?;
        Ok(out)
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
                        extra: Default::default(),
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
    additions: derived::Additions,
    cursor: Mutex<Option<derived::CheckpointCursor>>,
    guards: Mutex<Option<DerivedGuards>>,
}

struct DerivedGuards {
    writer_hold: Hold,
    source_leases: Vec<(String, CoreLease)>,
}

impl PyDerivedWriter {
    fn release_holds(&self, py: Python<'_>) {
        let Some(guards) = self
            .guards
            .lock_py_attached(py)
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
    fn part_write_bound(
        &self,
        py: Python<'_>,
        component: &str,
        key: &str,
        role: &str,
    ) -> PyResult<u64> {
        py.detach(|| {
            derived::part_write_bound(
                &self.meta,
                &self.transaction_id,
                self.writer_session_id,
                &self.additions,
                (component, key, role),
            )
        })
        .or_refuse(py)
    }

    fn config_write_bound(&self, py: Python<'_>, name: &str) -> PyResult<u64> {
        py.detach(|| {
            derived::config_write_bound(
                &self.meta,
                &self.transaction_id,
                self.writer_session_id,
                name,
            )
        })
        .or_refuse(py)
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
                derived::checkpoint_indexed(
                    &self.store,
                    &self.meta,
                    &self.transaction_id,
                    self.writer_session_id,
                    operation_id,
                    slot,
                    previous.as_ref(),
                    &mut self.cursor.lock().unwrap(),
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
        let guards = self
            .guards
            .lock_py_attached(py)
            .expect("writer hold mutex poisoned");
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
            derived::add_indexed_part(
                &self.store,
                &self.meta,
                &self.transaction_id,
                self.writer_session_id,
                &self.additions,
                self.cursor.lock().unwrap().as_mut(),
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
            // A config is not a part: the next checkpoint takes the full path.
            *self.cursor.lock().unwrap() = None;
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
                let facts = derived::commit(
                    &self.store,
                    &self.meta,
                    &self.transaction_id,
                    self.writer_session_id,
                )?;
                derived::finish_writer(&self.store)?;
                Ok(facts)
            })
            .or_refuse(py)?;
        self.release_holds(py);
        receipt(py, &value)
    }

    fn fence(&self, py: Python<'_>) -> PyResult<bool> {
        let fenced = py
            .detach(|| derived::fence(&self.meta, &self.transaction_id, self.writer_session_id))
            .or_refuse(py)?;
        self.release_holds(py);
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
    fn load(&self, py: Python<'_>, hex: &str) -> PyResult<(Manifest, Vec<u8>, Option<Vec<u8>>)> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let len = std::fs::metadata(self.store.manifest_path(&hex))
            .map(|m| m.len())
            .unwrap_or(0);
        let r = ObjectRef {
            sha256: hex,
            length: len,
        };
        let m = checkpoint::load_manifest(&self.store, &r).or_refuse(py)?;
        let header = m
            .header()
            .map(|reference| checkpoint::load_header(&self.store, reference)?.canonical_bytes())
            .transpose()
            .or_refuse(py)?;
        Ok((m.clone(), m.canonical_bytes(), header))
    }
}

// ---------------------------------------------------------------- the lease

#[pyclass(name = "ReadLease")]
pub struct PyLease {
    // Every attached caller must detach before waiting: a reader holding this lock
    // may need the GIL again to finish I/O or invoke a stream callback.
    lease: Mutex<Option<CoreLease>>,
    meta: Arc<Meta>,
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

impl PyLease {
    /// Hand the core lease to the weight plane, which ends it; `live` becomes False.
    pub(crate) fn take_for_plane(&self, py: Python<'_>) -> PyResult<CoreLease> {
        let taken = self.lease.lock_py_attached(py).unwrap().take();
        taken.ok_or_else(|| released(py))
    }
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

    /// Read a bounded semantic part from a header already covered by this read lease.
    /// This is the same native range reader used by derived writers, without a writer.
    #[allow(clippy::too_many_arguments)]
    fn read_part_into(
        &self,
        py: Python<'_>,
        header: &[u8],
        component: &str,
        key: &str,
        role: &str,
        off: u64,
        into: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let header_ref = ObjectRef::of(header);
        let header = Header::parse(header).or_refuse(py)?;
        let name = format!("{component}/{key}/{role}");
        let part = header
            .components
            .iter()
            .find(|(candidate, _)| candidate == component)
            .and_then(|(_, tensors)| tensors.iter().find(|(candidate, _)| candidate == key))
            .and_then(|(_, tensor)| tensor.parts.iter().find(|(candidate, _)| candidate == role))
            .map(|(_, part)| part)
            .ok_or_else(|| missing(py, "model tensor part", &name))?;
        let mut buffer = Writable::new(py, into)?;
        let destination = buffer.slice();
        let guard = self.lease.lock_py_attached(py).unwrap();
        let lease = guard.as_ref().ok_or_else(|| released(py))?;
        lease
            .covers(&ObjectRange {
                obj: header_ref,
                off: 0,
                len: 0,
            })
            .or_refuse(py)?;
        py.detach(|| {
            lease.recheck(&self.meta)?;
            read::read_part_into(
                lease,
                &name,
                part,
                off,
                destination.len() as u64,
                destination,
            )?;
            lease.recheck(&self.meta)
        })
        .or_refuse(py)
    }

}

// ---------------------------------------------------------------- reclamation

/// One local reclamation pass (tensorfs_core::gc): remove every blob and manifest no
/// repository or open ingest session names, from the filesystem census alone, under the
/// store's exclusive lock.
#[pyfunction]
pub fn gc<'p>(py: Python<'p>, root: PathBuf) -> PyResult<Bound<'p, PyDict>> {
    let report = py
        .detach(|| tensorfs_core::gc::collect(&root, false))
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
