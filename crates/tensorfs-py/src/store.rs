//! The store, the lease, the read plan and the slot ring — marshalled, never reimplemented.
//!
//! Two properties this module exists to preserve across the boundary:
//!
//! 1. **The caller owns the memory.** Every read fills a buffer Python already owns, through
//!    the stable-ABI `Py_buffer`. TensorFS allocates no destination, and the runtime's one
//!    budgeted pinned pool stays the runtime's. Nothing is copied to hand bytes to Python.
//! 2. **The GIL is not held while bytes move.** Every read runs inside `allow_threads`, so a
//!    reader pool of N genuinely reads N-wide from a Python caller.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use pyo3::buffer::PyBuffer;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use tensorfs_core::checkpoint;
use tensorfs_core::err::{refuse, Code, Refusal, Result as TResult};
use tensorfs_core::header::Header;
use tensorfs_core::ids::{Doc, ObjectRef};
use tensorfs_core::manifest::{BlobDescriptor, Entry, SnapshotManifest};
use tensorfs_core::meta::Meta;
use tensorfs_core::project;
use tensorfs_core::read::{self, ObjectRange, ReadLease as CoreLease, Source};
use tensorfs_core::receipt::Receipts;
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

// ---------------------------------------------------------------- store

#[pyclass(name = "Store")]
pub struct PyStore {
    pub store: Arc<CoreStore>,
    pub meta: Arc<Meta>,
}

fn open_store(py: Python<'_>, root: PathBuf, init: bool) -> PyResult<PyStore> {
    let s = if init {
        CoreStore::init(&root).or_refuse(py)?
    } else {
        CoreStore::open(&root).or_refuse(py)?
    };
    let m = Meta::open(&s).or_refuse(py)?;
    Ok(PyStore {
        store: Arc::new(s),
        meta: Arc::new(m),
    })
}

#[pymethods]
impl PyStore {
    #[staticmethod]
    fn open(py: Python<'_>, root: PathBuf) -> PyResult<PyStore> {
        open_store(py, root, false)
    }

    #[staticmethod]
    fn init(py: Python<'_>, root: PathBuf) -> PyResult<PyStore> {
        open_store(py, root, true)
    }

    #[getter]
    fn root(&self) -> String {
        self.store.root().display().to_string()
    }

    #[getter]
    fn generation(&self) -> u64 {
        self.store.generation()
    }

    #[getter]
    fn era(&self) -> &'static str {
        tensorfs_core::store::ERA
    }

    fn contains(&self, hex: &str) -> bool {
        self.store.contains(hex.trim_start_matches("sha256:"))
    }

    fn objects(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        self.store.objects().or_refuse(py)
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
            .allow_threads(|| self.store.put_file(&path, want.as_ref(), &Fault::default()))
            .or_refuse(py)?;
        let d = PyDict::new(py);
        d.set_item("id", p.obj.id())?;
        d.set_item("length", p.obj.length)?;
        d.set_item("admitted", p.admitted)?;
        Ok(d)
    }

    fn verify<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyDict>> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let v = py.allow_threads(|| self.store.verify(&hex)).or_refuse(py)?;
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
            Verdict::Quarantined { to, why } => {
                d.set_item("verdict", "quarantined")?;
                d.set_item("to", to.display().to_string())?;
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
            .allow_threads(|| self.store.read_range(&hex, 0, size))
            .or_refuse(py)?;
        Ok(PyBytes::new(py, &v))
    }

    /// Canonical bytes of a snapshot's manifest and of the header it attaches.
    fn snapshot<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyDict>> {
        let (m, mb, hb) = self.load(py, hex)?;
        let d = PyDict::new(py);
        d.set_item("snapshot_id", m.snapshot_id())?;
        d.set_item("checkpoint_id", m.checkpoint_id().or_refuse(py)?)?;
        d.set_item("manifest", PyBytes::new(py, &mb))?;
        d.set_item("header", PyBytes::new(py, &hb))?;
        Ok(d)
    }

    /// The transitive walk: every object the snapshot commits, with what it was reached AS.
    fn walk<'p>(&self, py: Python<'p>, hex: &str) -> PyResult<Bound<'p, PyList>> {
        let (m, _, _) = self.load(py, hex)?;
        let w = py
            .allow_threads(|| checkpoint::walk(&self.store, &m))
            .or_refuse(py)?;
        let l = PyList::empty(py);
        for r in &w.objects {
            let d = PyDict::new(py);
            d.set_item("id", r.obj.id())?;
            d.set_item("length", r.obj.length)?;
            d.set_item("kind", r.kind)?;
            l.append(d)?;
        }
        Ok(l)
    }

    /// `ReadLease = store.acquire(snapshot, object_set)`. Verifies every object first and
    /// registers the hold BEFORE the first byte moves.
    #[pyo3(signature = (snapshot, objects))]
    fn acquire(
        &self,
        py: Python<'_>,
        snapshot: &str,
        objects: Vec<(String, u64)>,
    ) -> PyResult<PyLease> {
        let mut refs = Vec::with_capacity(objects.len());
        for (h, n) in &objects {
            refs.push(obj_ref(py, h, *n)?);
        }
        let snap = snapshot.trim_start_matches("sha256:").to_string();
        let (lease, rx) = py
            .allow_threads(|| read::acquire(&self.store, &self.meta, &snap, refs))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            store: self.store.clone(),
            meta: self.meta.clone(),
            receipts: rx,
        })
    }

    /// Acquire over everything one snapshot commits — the serving case.
    fn acquire_snapshot(&self, py: Python<'_>, hex: &str) -> PyResult<PyLease> {
        let (m, _, _) = self.load(py, hex)?;
        let (lease, rx) = py
            .allow_threads(|| read::acquire_snapshot(&self.store, &self.meta, &m))
            .or_refuse(py)?;
        Ok(PyLease {
            lease: Mutex::new(Some(lease)),
            store: self.store.clone(),
            meta: self.meta.clone(),
            receipts: rx,
        })
    }

    /// Tier 2 of the projection ladder: config assets and canonical documents as ordinary
    /// files. A weight path cannot appear, and the refusal says why.
    #[pyo3(signature = (hex, dest, symlink=true))]
    fn project<'p>(
        &self,
        py: Python<'p>,
        hex: &str,
        dest: PathBuf,
        symlink: bool,
    ) -> PyResult<Bound<'p, PyDict>> {
        let (m, _, _) = self.load(py, hex)?;
        let mut rx = Receipts::default();
        let p =
            project::project_snapshot(&self.store, &m, &dest, !symlink, &mut rx).or_refuse(py)?;
        let d = PyDict::new(py);
        d.set_item("dirs", p.dirs)?;
        d.set_item("links", p.links)?;
        d.set_item("copies", p.copies)?;
        d.set_item("bytes", p.bytes)?;
        d.set_item("receipts", receipts_to_py(py, &rx)?)?;
        Ok(d)
    }

    /// Tier 3: a private streamed copy of an ALREADY-RETAINED carrier named by its path in
    /// the snapshot. Never a synthesis — an unretained carrier refuses.
    fn materialize(&self, py: Python<'_>, hex: &str, path: &str, dest: PathBuf) -> PyResult<u64> {
        let (m, _, _) = self.load(py, hex)?;
        let entry = m.entries().iter().find(|(p, _)| p == path);
        let b: BlobDescriptor = match entry {
            Some((_, Entry::Blob(b))) => b.clone(),
            Some((_, _)) => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::BLOB_NOT_RETAINED,
                        detail: format!("{path:?} is not a retained blob in this snapshot"),
                    },
                ))
            }
            None => {
                return Err(to_py(
                    py,
                    &Refusal {
                        code: Code::NOT_CONTAINED,
                        detail: format!("{path:?} is not a path in snapshot {hex}"),
                    },
                ))
            }
        };
        py.allow_threads(|| project::materialize_blob(&self.store, path, &b, &dest))
            .or_refuse(py)
    }

    /// The local root/hold authority, as plain data. Read-only here: roots and sweeps are
    /// the store owner's verbs, not a reader's.
    fn state<'p>(&self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let s = self.meta.read().or_refuse(py)?;
        marshal(py, &s.to_value())
    }

    fn reap_holds(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.allow_threads(|| self.meta.reap_holds()).or_refuse(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<tensorfs.Store {} era={} generation={}>",
            self.store.root().display(),
            tensorfs_core::store::ERA,
            self.store.generation()
        )
    }
}

impl PyStore {
    fn load(&self, py: Python<'_>, hex: &str) -> PyResult<(SnapshotManifest, Vec<u8>, Vec<u8>)> {
        let hex = hex.trim_start_matches("sha256:").to_string();
        let len = std::fs::metadata(self.store.object_path(&hex))
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
                        detail: format!("snapshot {} attaches no header", m.snapshot_id()),
                    },
                ))
            }
        };
        let h = checkpoint::load_header(&self.store, &hr).or_refuse(py)?;
        Ok((m.clone(), m.canonical_bytes(), h.canonical_bytes()))
    }
}

// ---------------------------------------------------------------- the plan

#[pyclass(name = "ReadPlan")]
pub struct PyPlan {
    pub plan: read::ReadPlan,
}

#[pymethods]
impl PyPlan {
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
                Source::Object(r) => {
                    d.set_item("source", "object")?;
                    d.set_item("object", r.obj.id())?;
                    d.set_item("object_length", r.obj.length)?;
                    d.set_item("off", r.off)?;
                }
                Source::Inline(_) => {
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
/// caller's code topology; a traversal that does not name every tensor refuses.
#[pyfunction]
#[pyo3(signature = (header, traversal, window=1<<24))]
pub fn plan(
    py: Python<'_>,
    header: &[u8],
    traversal: Vec<(String, String)>,
    window: u64,
) -> PyResult<PyPlan> {
    let h = Header::parse(header).or_refuse(py)?;
    let p = read::plan_for_traversal(&h, &traversal, window).or_refuse(py)?;
    Ok(PyPlan { plan: p })
}

/// The header's own key order. BENCHMARK-ONLY, and it says so: it exists so the
/// destination-order contract can be measured rather than quoted.
#[pyfunction]
#[pyo3(signature = (header, window=1<<24))]
pub fn plan_in_key_order(py: Python<'_>, header: &[u8], window: u64) -> PyResult<PyPlan> {
    let h = Header::parse(header).or_refuse(py)?;
    let p = read::plan_in_key_order(&h, window).or_refuse(py)?;
    Ok(PyPlan { plan: p })
}

// ---------------------------------------------------------------- the lease

#[pyclass(name = "ReadLease")]
pub struct PyLease {
    lease: Mutex<Option<CoreLease>>,
    store: Arc<CoreStore>,
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
        let g = self.lease.lock().unwrap();
        Ok(g.as_ref()
            .ok_or_else(|| released(py))?
            .hold_id()
            .to_string())
    }

    #[getter]
    fn snapshot(&self, py: Python<'_>) -> PyResult<String> {
        let g = self.lease.lock().unwrap();
        Ok(g.as_ref().ok_or_else(|| released(py))?.snapshot.clone())
    }

    #[getter]
    fn bytes(&self, py: Python<'_>) -> PyResult<u64> {
        let g = self.lease.lock().unwrap();
        Ok(g.as_ref().ok_or_else(|| released(py))?.bytes())
    }

    #[getter]
    fn objects(&self, py: Python<'_>) -> PyResult<Vec<(String, u64)>> {
        let g = self.lease.lock().unwrap();
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
        let g = self.lease.lock().unwrap();
        g.as_ref()
            .ok_or_else(|| released(py))?
            .recheck(&self.meta)
            .or_refuse(py)
    }

    /// THE explicit end. Nothing else releases the lease — dropping it keeps the row for
    /// the reaper and announces itself.
    fn release(&self, py: Python<'_>) -> PyResult<()> {
        let l = self.lease.lock().unwrap().take();
        match l {
            Some(l) => py.allow_threads(|| l.release(&self.meta)).or_refuse(py),
            None => Err(released(py)),
        }
    }

    #[getter]
    fn live(&self) -> bool {
        self.lease.lock().unwrap().is_some()
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, py: Python<'_>, _args: &Bound<'_, PyAny>) -> PyResult<bool> {
        if self.live() {
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
        let g = self.lease.lock().unwrap();
        let lease = g.as_ref().ok_or_else(|| released(py))?;
        let r = ObjectRange {
            obj: obj_ref(py, object, object_length)?,
            off,
            len: length,
        };
        py.allow_threads(|| read::read_into(&self.store, lease, &r, dest))
            .or_refuse(py)
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
        let g = self.lease.lock().unwrap();
        let lease = g.as_ref().ok_or_else(|| released(py))?;

        let raised: Mutex<Option<PyErr>> = Mutex::new(None);
        let cb = on_batch.unbind();
        let started = std::time::Instant::now();
        let out = py.allow_threads(|| {
            read::pump(
                &self.store,
                lease,
                &plan.plan,
                views,
                readers,
                |filled, slot| -> TResult<()> {
                    Python::with_gil(|py| {
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

// ---------------------------------------------------------------- ingest transaction

/// The ingest border's TRANSACTION seam. Carrier decode, classification and conversion stay
/// inside the border (they are the border), but the quarantine root that bounds them is a
/// caller-visible fact: who opened it, what it expires at, and whether it installed.
#[pyfunction]
#[pyo3(signature = (store, session, tenant, ttl_secs))]
pub fn ingest_open_root<'p>(
    py: Python<'p>,
    store: &PyStore,
    session: &str,
    tenant: &str,
    ttl_secs: u64,
) -> PyResult<Bound<'p, PyAny>> {
    let r = tensorfs_core::ingest::transaction::open_root(
        store.store.root(),
        session,
        tenant,
        ttl_secs,
    )
    .or_refuse(py)?;
    tensorfs_core::ingest::transaction::write_root(store.store.root(), &r).or_refuse(py)?;
    marshal(py, &r.to_value())
}

#[pyfunction]
pub fn ingest_read_root<'p>(
    py: Python<'p>,
    store: &PyStore,
    session: &str,
) -> PyResult<Bound<'p, PyAny>> {
    let r =
        tensorfs_core::ingest::transaction::read_root(store.store.root(), session).or_refuse(py)?;
    marshal(py, &r.to_value())
}

#[pyfunction]
pub fn ingest_reap<'p>(py: Python<'p>, store: &PyStore, now: u64) -> PyResult<Bound<'p, PyDict>> {
    let (expired, kept) =
        tensorfs_core::ingest::transaction::reap(store.store.root(), now).or_refuse(py)?;
    let d = PyDict::new(py);
    d.set_item("expired", expired)?;
    d.set_item("kept", kept)?;
    Ok(d)
}

#[pyfunction]
pub fn ingest_installed(store: &PyStore) -> Vec<String> {
    tensorfs_core::ingest::transaction::installed(&store.store)
}
