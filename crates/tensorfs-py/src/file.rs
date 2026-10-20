//! The native half of `tensorfs.CheckpointFile` (`api.py`), an `io.RawIOBase` over the Rust
//! `CheckpointFile`. Each call runs detached from the GIL and locks only while detached.

use std::io::{Read, Seek, SeekFrom};
use std::sync::Mutex;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyType;
use tensorfs::{Checkpoint, CheckpointFile, Failure};

use crate::store::{invalid, OrRaise};

#[pyclass(name = "_CheckpointFile", module = "tensorfs._ext", frozen)]
pub struct PyCheckpointFile(Mutex<Option<CheckpointFile>>);

/// `checkpoint.open_file()`: a `tensorfs.CheckpointFile` holding what it may read until closed.
pub(crate) fn open<'py>(py: Python<'py>, checkpoint: Checkpoint) -> PyResult<Bound<'py, PyAny>> {
    static CLASS: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    let file = checkpoint.open_file().or_raise(py)?;
    let native = Py::new(py, PyCheckpointFile(Mutex::new(Some(file))))?;
    CLASS
        .import(py, "tensorfs.api", "CheckpointFile")?
        .call1((native,))
}

impl PyCheckpointFile {
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut CheckpointFile) -> std::io::Result<T> + Send,
    ) -> PyResult<T> {
        let done = py.detach(|| {
            let mut file = self.0.lock().unwrap_or_else(|e| e.into_inner());
            file.as_mut().map(f)
        });
        match done {
            None => Err(PyValueError::new_err("I/O operation on closed file")),
            Some(result) => result.map_err(Failure::from).or_raise(py),
        }
    }
}

#[pymethods]
impl PyCheckpointFile {
    fn readinto(&self, py: Python<'_>, buffer: &Bound<'_, PyAny>) -> PyResult<usize> {
        let view = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((buffer,))?;
        if view.getattr("readonly")?.extract::<bool>()?
            || !view.getattr("c_contiguous")?.extract::<bool>()?
        {
            return Err(invalid(py, "readinto needs a writable C-contiguous buffer"));
        }
        let buf: PyBuffer<u8> = PyBuffer::get(&view.call_method1("cast", ("B",))?)?;
        let (ptr, len) = (buf.buf_ptr() as usize, buf.len_bytes());
        self.with(py, |file| {
            // SAFETY: `buf` holds the writable buffer until this call returns.
            #[allow(unsafe_code)]
            let out = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, len) };
            file.read(out)
        })
    }

    #[pyo3(signature = (offset, whence = 0))]
    fn seek(&self, py: Python<'_>, offset: i64, whence: i32) -> PyResult<u64> {
        let to =
            match whence {
                0 => SeekFrom::Start(u64::try_from(offset).map_err(|_| {
                    PyValueError::new_err(format!("negative seek position {offset}"))
                })?),
                1 => SeekFrom::Current(offset),
                2 => SeekFrom::End(offset),
                _ => return Err(PyValueError::new_err(format!("invalid whence ({whence})"))),
            };
        self.with(py, |file| file.seek(to))
    }

    fn tell(&self, py: Python<'_>) -> PyResult<u64> {
        self.with(py, |file| file.stream_position())
    }

    /// Releases what it holds; later calls raise `ValueError`.
    fn close(&self, py: Python<'_>) {
        py.detach(|| drop(self.0.lock().unwrap_or_else(|e| e.into_inner()).take()));
    }
}
