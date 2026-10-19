//! `tensorfs.varena.Ledger`: the per-GPU ledger (`varena::ledger`). Marshal only; every call
//! that can block or probe runs detached from the GIL.

use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;

create_exception!(tensorfs.varena, LedgerError, PyException, "A ledger error; `.code` names its class.");

#[cfg(unix)]
mod imp {
    use std::os::fd::{AsRawFd, BorrowedFd};
    use std::sync::Arc;

    use pyo3::prelude::*;
    use varena::ledger::{Admit, Facts, Ledger, Row, RowFacts, State, Victim};

    use super::LedgerError;

    fn err(py: Python<'_>, e: varena::Error) -> PyErr {
        let x = LedgerError::new_err(e.to_string());
        let _ = x.value(py).setattr("code", e.code());
        x
    }

    trait OrRaise<T> {
        fn or_raise(self, py: Python<'_>) -> PyResult<T>;
    }

    impl<T> OrRaise<T> for varena::Result<T> {
        fn or_raise(self, py: Python<'_>) -> PyResult<T> {
            self.map_err(|e| err(py, e))
        }
    }

    #[pyclass(name = "Ledger", module = "tensorfs.varena", frozen)]
    pub struct PyLedger(Arc<Ledger>);

    #[pymethods]
    impl PyLedger {
        #[staticmethod]
        fn create(py: Python<'_>, gpu_uuid: &str, capacity: u64) -> PyResult<Self> {
            Ok(Self(Arc::new(Ledger::create(gpu_uuid, capacity).or_raise(py)?.0)))
        }

        /// The table behind `fd`, and the eventfd beside it (`freed` signals it).
        #[staticmethod]
        #[pyo3(signature = (fd, events_fd = None))]
        fn adopt(py: Python<'_>, fd: i32, events_fd: Option<i32>) -> PyResult<Self> {
            // SAFETY: the caller holds `fd` and `events_fd` open for the call.
            let (fd, events) = unsafe { (BorrowedFd::borrow_raw(fd), events_fd.map(|e| BorrowedFd::borrow_raw(e))) };
            Ok(Self(Arc::new(Ledger::adopt(fd, events).or_raise(py)?)))
        }

        /// This process's description of the table: what to pass to a child.
        #[getter]
        fn fd(&self) -> i32 {
            self.0.fd().as_raw_fd()
        }

        /// The eventfd beside the table, when this process holds it.
        #[getter]
        fn events_fd(&self) -> Option<i32> {
            self.0.events().map(|e| e.as_raw_fd())
        }

        /// Bytes freed here: the free generation moves and the eventfd is signalled.
        fn freed(&self) {
            self.0.freed();
        }

        fn freed_gen(&self) -> u64 {
            self.0.freed_gen()
        }

        /// Blocks until the free generation is other than `seen`; returns it.
        fn wait_freed(&self, py: Python<'_>, seen: u64) -> u64 {
            let l = self.0.clone();
            py.detach(move || l.wait_freed(seen))
        }

        /// The measured peak recorded under `key` (`<package>|<shape>`).
        fn peak(&self, key: &str) -> Option<u64> {
            self.0.peak(key)
        }

        fn set_peak(&self, py: Python<'_>, key: &str, nbytes: u64) -> PyResult<()> {
            self.0.set_peak(key, nbytes).or_raise(py)
        }

        /// The key TensorD dispatched `request_id` under.
        fn dispatched(&self, request_id: &str) -> Option<String> {
            self.0.dispatched(request_id)
        }

        fn dispatch(&self, py: Python<'_>, request_id: &str, key: &str) -> PyResult<()> {
            self.0.dispatch(request_id, key).or_raise(py)
        }

        fn undispatch(&self, request_id: &str) {
            self.0.undispatch(request_id);
        }

        #[pyo3(signature = (package, shape = None, spawn = false))]
        fn fits(&self, py: Python<'_>, package: &str, shape: Option<&str>, spawn: bool) -> PyResult<bool> {
            let (l, package, shape) = (self.0.clone(), package.to_string(), shape.map(str::to_string));
            py.detach(move || l.fits(&package, shape.as_deref(), spawn)).or_raise(py)
        }

        #[getter]
        fn gpu_uuid(&self) -> String {
            self.0.gpu_uuid()
        }

        #[getter]
        fn capacity(&self) -> u64 {
            self.0.capacity()
        }

        /// A row for this process.
        fn claim(&self, py: Python<'_>) -> PyResult<PyRow> {
            let l = self.0.clone();
            let row = py.detach(move || l.claim(std::process::id())).or_raise(py)?;
            Ok(PyRow(Arc::new(row)))
        }

        #[pyo3(signature = (need, asker = None))]
        fn ask(&self, py: Python<'_>, need: u64, asker: Option<u32>) -> PyResult<Vec<PyVictim>> {
            let l = self.0.clone();
            Ok(py.detach(move || l.ask(asker, need)).or_raise(py)?.into_iter().map(PyVictim).collect())
        }

        fn answered(&self, py: Python<'_>, victim: &PyVictim) -> PyResult<bool> {
            self.0.answered(&victim.0).or_raise(py)
        }

        fn wait(&self, py: Python<'_>, victim: &PyVictim) -> PyResult<()> {
            let (l, v) = (self.0.clone(), victim.0.clone());
            py.detach(move || l.wait(&v)).or_raise(py)
        }

        fn admit(&self, py: Python<'_>, need: u64) -> PyResult<PyAdmit> {
            let l = self.0.clone();
            Ok(match py.detach(move || l.admit(need)).or_raise(py)? {
                Admit::Now => PyAdmit { now: true, victims: vec![], short: 0 },
                Admit::After { victims, short } => PyAdmit { now: false, victims, short },
            })
        }

        fn reap(&self, py: Python<'_>, pid: u32) -> PyResult<u64> {
            let l = self.0.clone();
            py.detach(move || l.reap(pid)).or_raise(py)
        }

        fn returned(&self, py: Python<'_>, pid: u32) -> PyResult<()> {
            self.0.returned(pid).or_raise(py)
        }

        fn stalled(&self, py: Python<'_>) -> PyResult<Vec<u32>> {
            let l = self.0.clone();
            py.detach(move || l.stalled()).or_raise(py)
        }

        fn snapshot(&self, py: Python<'_>) -> PyResult<Vec<PyRowFacts>> {
            let l = self.0.clone();
            Ok(py.detach(move || l.snapshot()).or_raise(py)?.into_iter().map(PyRowFacts::from).collect())
        }
    }

    #[pyclass(name = "Row", module = "tensorfs.varena", frozen)]
    pub struct PyRow(pub(crate) Arc<Row>);

    #[pymethods]
    impl PyRow {
        #[getter]
        fn id(&self) -> u32 {
            self.0.id()
        }

        #[getter]
        fn pid(&self) -> u32 {
            self.0.pid()
        }

        #[pyo3(signature = (*, context = 0, resident = 0, cold = 0, kept = 0, pinned = 0, torch_reserved = 0,
            torch_allocated = 0, cache = 0, idle = 0, active_peak = 0, last_use = 0, waiting = 0))]
        #[allow(clippy::too_many_arguments)]
        fn publish(
            &self,
            context: u64,
            resident: u64,
            cold: u64,
            kept: u64,
            pinned: u64,
            torch_reserved: u64,
            torch_allocated: u64,
            cache: u64,
            idle: u64,
            active_peak: u64,
            last_use: u64,
            waiting: u64,
        ) {
            self.0.publish(&Facts {
                context,
                resident,
                cold,
                kept,
                pinned,
                torch_reserved,
                torch_allocated,
                cache,
                idle,
                active_peak,
                last_use,
                waiting,
            });
        }

        fn want(&self) -> u64 {
            self.0.want()
        }

        fn wait_want(&self, py: Python<'_>) -> u64 {
            let r = self.0.clone();
            py.detach(move || r.wait_want())
        }

        fn ack(&self, freed: u64) {
            self.0.ack(freed);
        }

        /// Its requests running now: (request id, remaining measured peak; -1 never measured).
        fn running(&self, py: Python<'_>, requests: Vec<(String, i64)>) -> PyResult<()> {
            self.0.running(&requests).or_raise(py)
        }
    }

    #[pyclass(name = "Victim", module = "tensorfs.varena", frozen)]
    pub struct PyVictim(Victim);

    #[pymethods]
    impl PyVictim {
        #[getter]
        fn slot(&self) -> u32 {
            self.0.slot
        }

        #[getter]
        fn pid(&self) -> u32 {
            self.0.pid
        }

        #[getter]
        fn bytes(&self) -> u64 {
            self.0.bytes
        }
    }

    #[pyclass(name = "Admit", module = "tensorfs.varena", frozen)]
    pub struct PyAdmit {
        #[pyo3(get)]
        now: bool,
        victims: Vec<Victim>,
        #[pyo3(get)]
        short: u64,
    }

    #[pymethods]
    impl PyAdmit {
        #[getter]
        fn victims(&self) -> Vec<PyVictim> {
            self.victims.iter().cloned().map(PyVictim).collect()
        }
    }

    #[pyclass(name = "RowFacts", module = "tensorfs.varena", frozen, get_all)]
    pub struct PyRowFacts {
        slot: u32,
        gen: u64,
        pid: u32,
        state: &'static str,
        context: u64,
        resident: u64,
        cold: u64,
        kept: u64,
        pinned: u64,
        torch_reserved: u64,
        torch_allocated: u64,
        cache: u64,
        idle: u64,
        active_peak: u64,
        last_use: u64,
        waiting: u64,
        want: u64,
        acked: u64,
        progress: u64,
        running: Vec<(String, i64)>,
        held: u64,
        unevictable: u64,
        reclaimable: u64,
    }

    impl From<RowFacts> for PyRowFacts {
        fn from(r: RowFacts) -> Self {
            let f = r.facts;
            PyRowFacts {
                slot: r.slot,
                gen: r.gen,
                pid: r.pid,
                state: match r.state {
                    State::Live => "live",
                    State::Dead => "dead",
                    State::Reaping => "reaping",
                },
                context: f.context,
                resident: f.resident,
                cold: f.cold,
                kept: f.kept,
                pinned: f.pinned,
                torch_reserved: f.torch_reserved,
                torch_allocated: f.torch_allocated,
                cache: f.cache,
                idle: f.idle,
                active_peak: f.active_peak,
                last_use: f.last_use,
                waiting: f.waiting,
                want: r.want,
                acked: r.acked,
                progress: r.progress,
                running: r.running.clone(),
                held: r.held(),
                unevictable: r.unevictable(),
                reclaimable: r.reclaimable(),
            }
        }
    }
}

/// Off Unix the ledger does not exist: the names import, and creating or adopting one raises.
#[cfg(not(unix))]
mod imp {
    use pyo3::prelude::*;

    use super::LedgerError;

    fn unsupported(py: Python<'_>) -> PyErr {
        let e = LedgerError::new_err("the ledger needs a Linux memfd");
        let _ = e.value(py).setattr("code", "UNSUPPORTED");
        e
    }

    #[pyclass(name = "Ledger", module = "tensorfs.varena", frozen)]
    pub struct PyLedger;

    #[pymethods]
    impl PyLedger {
        #[staticmethod]
        fn create(py: Python<'_>, gpu_uuid: &str, capacity: u64) -> PyResult<Self> {
            let _ = (gpu_uuid, capacity);
            Err(unsupported(py))
        }

        #[staticmethod]
        #[pyo3(signature = (fd, events_fd = None))]
        fn adopt(py: Python<'_>, fd: i32, events_fd: Option<i32>) -> PyResult<Self> {
            let _ = (fd, events_fd);
            Err(unsupported(py))
        }
    }

    #[pyclass(name = "Row", module = "tensorfs.varena", frozen)]
    pub struct PyRow;
    #[pyclass(name = "Victim", module = "tensorfs.varena", frozen)]
    pub struct PyVictim;
    #[pyclass(name = "Admit", module = "tensorfs.varena", frozen)]
    pub struct PyAdmit;
    #[pyclass(name = "RowFacts", module = "tensorfs.varena", frozen)]
    pub struct PyRowFacts;
}

pub(crate) use imp::PyRow;

/// Adds the ledger to `tensorfs._ext.varena`.
pub fn install(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = parent.getattr("varena")?.cast_into::<PyModule>()?;
    m.add_class::<imp::PyLedger>()?;
    m.add_class::<imp::PyRow>()?;
    m.add_class::<imp::PyVictim>()?;
    m.add_class::<imp::PyAdmit>()?;
    m.add_class::<imp::PyRowFacts>()?;
    m.add("LedgerError", parent.py().get_type::<LedgerError>())
}
