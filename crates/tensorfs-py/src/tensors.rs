//! `Component.tensors`: `key -> Info` in construction order, read through TensorFS's read engine
//! (`tensorfs_core::io`) with the GIL released.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator};
use tensorfs_core::failure::{Failure, Kind};
use tensorfs_core::format::{decode, Body, Dtype, Encoding, PartBytes};
use tensorfs_core::io::{Dst, ReadOpts, Target};
use tensorfs_core::store2::{Checkpoint, Error};

use crate::store::{invalid, metadata, raise, Aligned, FetchTensor, OrRaise, PyArray, PyCheckpoint};

/// A component's tensors: a mapping of key to `Info`, with the reads.
#[pyclass(name = "Tensors", module = "tensorfs", frozen, mapping)]
pub struct PyTensors {
    component: String,
    infos: Py<PyDict>,
    ck: Option<Py<PyCheckpoint>>,
}

/// An encoded tensor as stored (`decode=False`): its scheme and params, and each part.
#[pyclass(name = "Encoded", module = "tensorfs", frozen, get_all)]
pub struct PyEncoded {
    scheme: String,
    params: Py<PyDict>,
    parts: Py<PyDict>,
}

/// One stored range a read fills: a plain tensor's data (role `""`) or one part.
struct Want {
    key: usize,
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    buf: Aligned,
}

impl PyTensors {
    pub(crate) fn new(
        component: &str,
        infos: Py<PyDict>,
        ck: Option<Py<PyCheckpoint>>,
    ) -> PyTensors {
        PyTensors {
            component: component.to_owned(),
            infos,
            ck,
        }
    }

    fn checkpoint<'a>(&'a self, py: Python<'_>) -> PyResult<&'a Checkpoint> {
        match &self.ck {
            Some(ck) => ck.get().stored(py),
            None => Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "a parsed header has no chunks here".into(),
                },
            )),
        }
    }

    /// `keys` as Arrays (or `Encoded`), every stored range in one engine call.
    fn load<'py>(
        &self,
        py: Python<'py>,
        keys: &[String],
        decode_: bool,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let ck = self.checkpoint(py)?;
        let reads: Vec<FetchTensor> = keys.iter().map(|key| FetchTensor { key, role: None, range: None }).collect();
        self.ck.as_ref().unwrap().get().fetch_tensors(py, &self.component, &reads)?;
        let mut wants = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let t = ck.tensor(&self.component, key).or_raise(py)?;
            let parts: Vec<(&str, &Dtype, &[u64], u64)> = match &t.body {
                Body::Data(at) => vec![("", &t.dtype, &t.shape, at.length)],
                Body::Encoded { parts, .. } => parts
                    .iter()
                    .map(|p| (p.role.as_str(), &p.dtype, &p.shape[..], p.length))
                    .collect(),
            };
            for (role, dtype, shape, length) in parts {
                let buf = Aligned::new(length as usize);
                wants.push(Want {
                    key: i,
                    role: role.to_owned(),
                    dtype: dtype.clone(),
                    shape: shape.to_vec(),
                    buf,
                });
            }
        }
        let targets: Vec<Target> = (wants.iter_mut())
            .map(|w| Target {
                component: &self.component,
                key: &keys[w.key],
                role: &w.role,
                offset: 0,
                dst: Dst::Host {
                    ptr: w.buf.as_mut_ptr(),
                    len: w.buf.len,
                },
            })
            .collect();
        // SAFETY: each destination is its own buffer, held until the read returns.
        py.detach(|| unsafe { ck.read_into(&targets, &ReadOpts::default()) })
            .or_raise(py)?;
        drop(targets);
        let mut by_key: Vec<Vec<Want>> = keys.iter().map(|_| Vec::new()).collect();
        for w in wants {
            by_key[w.key].push(w);
        }
        let array = |w: Want| PyArray {
            buf: Arc::new(w.buf),
            dtype: w.dtype,
            shape: w.shape,
        };
        let mut out = Vec::with_capacity(keys.len());
        for (key, mut parts) in keys.iter().zip(by_key) {
            let t = ck.tensor(&self.component, key).or_raise(py)?;
            let Body::Encoded { encoding, .. } = &t.body else {
                out.push(Bound::new(py, array(parts.remove(0)))?.into_any());
                continue;
            };
            if decode_ {
                let bytes = py
                    .detach(|| decoded(&t.dtype, &t.shape, encoding, &parts))
                    .or_raise(py)?;
                let mut buf = Aligned::new(bytes.len());
                buf.bytes_mut().copy_from_slice(&bytes);
                let a = PyArray {
                    buf: Arc::new(buf),
                    dtype: t.dtype.clone(),
                    shape: t.shape.clone(),
                };
                out.push(Bound::new(py, a)?.into_any());
                continue;
            }
            let dict = PyDict::new(py);
            for w in parts {
                dict.set_item(w.role.clone(), Bound::new(py, array(w))?)?;
            }
            let e = PyEncoded {
                scheme: encoding.scheme.clone(),
                params: metadata(py, &encoding.params)?.unbind(),
                parts: dict.unbind(),
            };
            out.push(Bound::new(py, e)?.into_any());
        }
        Ok(out)
    }
}

fn decoded(
    dtype: &Dtype,
    shape: &[u64],
    encoding: &Encoding,
    parts: &[Want],
) -> Result<Vec<u8>, Error> {
    let views: BTreeMap<&str, PartBytes> = (parts.iter())
        .map(|w| {
            (
                w.role.as_str(),
                PartBytes {
                    dtype: &w.dtype,
                    shape: &w.shape,
                    bytes: w.buf.bytes(),
                },
            )
        })
        .collect();
    Ok(decode(dtype, shape, encoding, &views)?)
}

#[pymethods]
impl PyTensors {
    fn __len__(&self, py: Python<'_>) -> usize {
        self.infos.bind(py).len()
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        self.infos
            .bind(py)
            .get_item(key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_owned()))
    }
    fn __contains__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        self.infos.bind(py).contains(key)
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.infos.bind(py).try_iter()
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.infos.bind(py).call_method0("keys")
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.infos.bind(py).call_method0("values")
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.infos.bind(py).call_method0("items")
    }
    #[pyo3(signature = (key, default = None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &str,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        Ok(self.infos.bind(py).get_item(key)?.or(default))
    }
    fn __repr__(&self, py: Python<'_>) -> String {
        format!("Tensors({}, {} tensors)", self.component, self.__len__(py))
    }

    /// One tensor on the CPU: an `Array` (decoded), or `Encoded` with `decode=False`.
    #[pyo3(signature = (key, *, decode = true))]
    fn read<'py>(&self, py: Python<'py>, key: String, decode: bool) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.load(py, &[key], decode)?.remove(0))
    }

    /// `keys` (all when None, in construction order) read at once: `{key: Array | Encoded}`.
    #[pyo3(signature = (keys = None, *, device = "cpu", decode = true))]
    fn read_all<'py>(
        &self,
        py: Python<'py>,
        keys: Option<Vec<String>>,
        device: &str,
        decode: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        if device != "cpu" {
            let detail = format!(
                "device {device:?}: CUDA arrays come with the CUDA engine; read on \"cpu\""
            );
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail,
                },
            ));
        }
        let keys = match keys {
            Some(keys) => keys,
            None => self.infos.bind(py).keys().extract()?,
        };
        let out = PyDict::new(py);
        for (key, value) in keys.iter().zip(self.load(py, &keys, decode)?) {
            out.set_item(key, value)?;
        }
        Ok(out)
    }

    /// `len(out)` bytes of `key` from `offset` into the writable buffer `out`: its stored data,
    /// one part (`role`), or, for an encoded tensor with no role, its decoded bytes.
    #[pyo3(signature = (key, out, *, role = None, offset = 0))]
    fn read_into(
        &self,
        py: Python<'_>,
        key: String,
        out: &Bound<'_, PyAny>,
        role: Option<String>,
        offset: u64,
    ) -> PyResult<()> {
        let view = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((out,))?;
        if view.getattr("readonly")?.extract::<bool>()?
            || !view.getattr("c_contiguous")?.extract::<bool>()?
        {
            return Err(invalid(py, "out must be a writable C-contiguous buffer"));
        }
        let buf: PyBuffer<u8> = PyBuffer::get(&view.call_method1("cast", ("B",))?)?;
        let len = buf.len_bytes();
        let ck = self.checkpoint(py)?;
        let t = ck.tensor(&self.component, &key).or_raise(py)?;
        if role.is_none() && matches!(t.body, Body::Encoded { .. }) {
            let length = t.dtype.byte_len(&t.shape).ok_or_else(|| invalid(py, "unknown logical byte length"))?;
            if offset.checked_add(len as u64).is_none_or(|end| end > length) {
                return Err(invalid(py, format!("{key}: {len} bytes from {offset} of {length}")));
            }
            let array = self
                .load(py, std::slice::from_ref(&key), true)?
                .remove(0)
                .cast_into::<PyArray>()?;
            let bytes = array.get().buf.bytes();
            let Some(range) = usize::try_from(offset)
                .ok()
                .and_then(|at| bytes.get(at..))
                .and_then(|b| b.get(..len))
            else {
                return Err(invalid(
                    py,
                    format!("{key}: {len} bytes from {offset} of {}", bytes.len()),
                ));
            };
            // SAFETY: the buffer is held writable for `len` bytes while `buf` lives.
            if len != 0 {
                unsafe { std::slice::from_raw_parts_mut(buf.buf_ptr() as *mut u8, len) }
                    .copy_from_slice(range);
            }
            return Ok(());
        }
        let end = offset.checked_add(len as u64).ok_or_else(|| invalid(py, "read range overflow"))?;
        let read = FetchTensor { key: &key, role: Some(role.as_deref().unwrap_or("")), range: Some(offset..end) };
        self.ck.as_ref().unwrap().get().fetch_tensors(py, &self.component, &[read])?;
        let target = Target {
            component: &self.component,
            key: &key,
            role: role.as_deref().unwrap_or(""),
            offset,
            dst: Dst::Host {
                ptr: buf.buf_ptr() as *mut u8,
                len,
            },
        };
        // SAFETY: the exporter keeps `out` alive and writable while `buf` is held.
        py.detach(|| unsafe { ck.read_into(&[target], &ReadOpts::default()) })
            .or_raise(py)?;
        Ok(())
    }
}
