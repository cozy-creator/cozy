//! `Component.tensors`: `key -> TensorMetadata` in construction order, read through TensorFS's read engine
//! (`tensorfs::io`) with the GIL released.

use std::collections::BTreeMap;
use std::sync::Arc;

use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyIterator, PyString, PyTuple};
use tensorfs::format::{decode, Body, Dtype, Encoding, PartBytes};
use tensorfs::io::{DeviceBuffer, Dst, ReadOpts, Target};
use tensorfs::{Checkpoint, Failure, Kind};

use crate::store::{
    defer_failed_cuda, invalid, metadata, raise, Aligned, ArrayBuffer, CheckpointAccess,
    DeviceArray, FetchTensor, OrRaise, PyArray,
};

/// A component's tensors: a mapping of key to `TensorMetadata`, with the reads.
#[pyclass(name = "Tensors", module = "tensorfs", frozen, mapping)]
pub struct PyTensors {
    component: String,
    infos: Py<PyDict>,
    ck: Option<Arc<CheckpointAccess>>,
}

/// An encoded tensor as stored (`decode=False`): its logical dtype and shape, its scheme and
/// params, and each part.
#[pyclass(name = "Encoded", module = "tensorfs", frozen, get_all)]
pub struct PyEncoded {
    dtype: String,
    shape: Py<PyTuple>,
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
    device: Option<Arc<DeviceArray>>,
}

impl PyTensors {
    pub(crate) fn new(
        component: &str,
        infos: Py<PyDict>,
        ck: Option<Arc<CheckpointAccess>>,
    ) -> PyTensors {
        PyTensors {
            component: component.to_owned(),
            infos,
            ck,
        }
    }

    fn checkpoint(&self, py: Python<'_>) -> PyResult<&Checkpoint> {
        match &self.ck {
            Some(ck) => Ok(ck),
            None => Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "a parsed header has no chunks here".into(),
                },
            )),
        }
    }

    fn fill_target(
        &self,
        py: Python<'_>,
        key: String,
        role: Option<&str>,
        decode_: bool,
        offset: u64,
        dst: Dst,
    ) -> PyResult<()> {
        let len = dst.len();
        let ck = self.checkpoint(py)?;
        let t = ck
            .header()
            .tensor(&self.component, &key)
            .ok_or_else(|| invalid(py, "no such tensor"))?;
        if role.is_some() && decode_ {
            return Err(invalid(
                py,
                "role and decode exclude each other: a role reads one stored part",
            ));
        }
        if let (None, Body::Encoded { encoding, parts }) = (role, &t.body) {
            if !decode_ {
                let roles: Vec<_> = parts.iter().map(|p| p.role.as_str()).collect();
                let roles = roles.join(", ");
                let detail = format!("{key} is encoded: pass decode=True or a role ({roles})");
                return Err(invalid(py, detail));
            }
            let length = t
                .dtype
                .byte_len(&t.shape)
                .ok_or_else(|| invalid(py, "unknown logical byte length"))?;
            let end = offset
                .checked_add(len as u64)
                .filter(|end| *end <= length)
                .ok_or_else(|| {
                    invalid(py, format!("{key}: {len} bytes from {offset} of {length}"))
                })?;
            if len == 0 {
                return Ok(());
            }
            if let Dst::Device { device, .. } = dst {
                let parts = self
                    .stored(py, std::slice::from_ref(&key), true, None)?
                    .remove(0);
                let width = t
                    .dtype
                    .byte_len(&[1])
                    .ok_or_else(|| invalid(py, "unknown logical dtype width"))?;
                let begin = offset / width;
                let count = end.div_ceil(width) - begin;
                let result = py.detach(|| -> Result<(), Failure> {
                    let buffer = tensorfs::io::decode::read(
                        &t.dtype,
                        &t.shape,
                        encoding,
                        &views(&parts),
                        device,
                        begin,
                        count,
                    )?;
                    let copy = tensorfs::io::Copy {
                        src: tensorfs::io::Src::Device {
                            device,
                            ptr: buffer.ptr() + offset % width,
                            stream: 0,
                        },
                        dst,
                    };
                    if let Err(error) = unsafe { tensorfs::io::read(&[copy], &ReadOpts::default()) }
                    {
                        defer_failed_cuda(device, buffer);
                        return Err(error.source.into());
                    }
                    Ok(())
                });
                if result.is_err() {
                    defer_failed_cuda(device, parts);
                }
                return result.or_raise(py);
            }
            let array = self
                .load(py, std::slice::from_ref(&key), true, None)?
                .remove(0)
                .cast_into::<PyArray>()?;
            let bytes = array.get().buf.host().expect("read on CPU");
            let range = &bytes[offset as usize..end as usize];
            let copy = tensorfs::io::Copy {
                src: tensorfs::io::Src::Host(range.as_ptr()),
                dst,
            };
            return py
                .detach(|| {
                    unsafe { tensorfs::io::read(&[copy], &ReadOpts::default()) }
                        .map_err(|e| Failure::from(e.source))
                })
                .or_raise(py)
                .map(|_| ());
        }
        let end = offset
            .checked_add(len as u64)
            .ok_or_else(|| invalid(py, "read range overflow"))?;
        let read = FetchTensor {
            key: &key,
            role: Some(role.unwrap_or("")),
            range: Some(offset..end),
        };
        self.ck
            .as_ref()
            .unwrap()
            .fetch_tensors(py, &self.component, &[read])?;
        let target = Target {
            component: &self.component,
            key: &key,
            role: role.unwrap_or(""),
            offset,
            dst,
        };
        // SAFETY: the exporter keeps `out` alive and writable while `buf` is held.
        py.detach(|| unsafe { tensorfs::raw::read_into(ck, &[target], &ReadOpts::default()) })
            .or_raise(py)?;
        Ok(())
    }

    fn stored(
        &self,
        py: Python<'_>,
        keys: &[String],
        decode_: bool,
        device: Option<i32>,
    ) -> PyResult<Vec<Vec<Want>>> {
        let ck = self.checkpoint(py)?;
        let reads: Vec<FetchTensor> = keys
            .iter()
            .map(|key| FetchTensor {
                key,
                role: None,
                range: None,
            })
            .collect();
        self.ck
            .as_ref()
            .unwrap()
            .fetch_tensors(py, &self.component, &reads)?;
        let mut wants = Vec::new();
        for (i, key) in keys.iter().enumerate() {
            let t = ck
                .header()
                .tensor(&self.component, key)
                .ok_or_else(|| invalid(py, "no such tensor"))?;
            let parts: Vec<(&str, &Dtype, &[u64], u64)> = match &t.body {
                Body::Data(at) => vec![("", &t.dtype, &t.shape, at.length)],
                Body::Encoded { parts, .. } => parts
                    .iter()
                    .map(|p| (p.role.as_str(), &p.dtype, &p.shape[..], p.length))
                    .collect(),
            };
            for (role, dtype, shape, length) in parts {
                let length = usize::try_from(length)
                    .map_err(|_| invalid(py, "stored tensor exceeds address space"))?;
                let direct = device.filter(|_| !decode_ || matches!(t.body, Body::Data(_)));
                let gpu = direct
                    .map(|device| {
                        py.detach(|| DeviceBuffer::new(device, length).map(DeviceArray::new))
                            .or_raise(py)
                    })
                    .transpose()?;
                let buf = Aligned::new(if gpu.is_some() { 0 } else { length });
                wants.push(Want {
                    key: i,
                    role: role.to_owned(),
                    dtype: dtype.clone(),
                    shape: shape.to_vec(),
                    buf,
                    device: gpu,
                });
            }
        }
        let targets: Vec<Target> = (wants.iter_mut())
            .map(|w| Target {
                component: &self.component,
                key: &keys[w.key],
                role: &w.role,
                offset: 0,
                dst: if let Some(gpu) = &w.device {
                    Dst::Device {
                        device: gpu.device(),
                        ptr: gpu.ptr(),
                        len: gpu.len(),
                        stream: 0,
                    }
                } else {
                    Dst::Host {
                        ptr: w.buf.as_mut_ptr(),
                        len: w.buf.len,
                    }
                },
            })
            .collect();
        // SAFETY: each destination is its own buffer, held until the read returns.
        let result =
            py.detach(|| unsafe { tensorfs::raw::read_into(ck, &targets, &ReadOpts::default()) });
        drop(targets);
        if let Err(error) = result {
            if let Some(device) = wants
                .iter()
                .find_map(|want| want.device.as_ref().map(|buffer| buffer.device()))
            {
                defer_failed_cuda(device, wants);
            }
            return Err(raise(py, error));
        }
        let mut by_key: Vec<Vec<Want>> = keys.iter().map(|_| Vec::new()).collect();
        for w in wants {
            by_key[w.key].push(w);
        }
        Ok(by_key)
    }

    /// `keys` as Arrays (or `Encoded`), every stored range in one engine call.
    fn load<'py>(
        &self,
        py: Python<'py>,
        keys: &[String],
        decode_: bool,
        device: Option<i32>,
    ) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let ck = self.checkpoint(py)?;
        let by_key = self.stored(py, keys, decode_, device)?;
        let array = |w: Want| PyArray {
            buf: match w.device {
                Some(device) => ArrayBuffer::Device(device),
                None => ArrayBuffer::Host(Arc::new(w.buf)),
            },
            dtype: w.dtype,
            shape: w.shape,
        };
        let mut out = Vec::with_capacity(keys.len());
        for (key, mut parts) in keys.iter().zip(by_key) {
            let t = ck
                .header()
                .tensor(&self.component, key)
                .ok_or_else(|| invalid(py, "no such tensor"))?;
            let Body::Encoded { encoding, .. } = &t.body else {
                out.push(Bound::new(py, array(parts.remove(0)))?.into_any());
                continue;
            };
            if decode_ {
                if let Some(device) = device {
                    let decoded = py.detach(|| {
                        let views = views(&parts);
                        let count = t
                            .shape
                            .iter()
                            .try_fold(1u64, |n, d| n.checked_mul(*d))
                            .ok_or_else(|| Failure {
                                kind: Kind::Invalid,
                                detail: "logical tensor size overflow".into(),
                            })?;
                        tensorfs::io::decode::read(
                            &t.dtype, &t.shape, encoding, &views, device, 0, count,
                        )
                        .map(DeviceArray::new)
                    });
                    let buffer = match decoded {
                        Ok(buffer) => buffer,
                        Err(error) => {
                            defer_failed_cuda(device, parts);
                            return Err(raise(py, error));
                        }
                    };
                    out.push(
                        Bound::new(
                            py,
                            PyArray {
                                buf: ArrayBuffer::Device(buffer),
                                dtype: t.dtype.clone(),
                                shape: t.shape.clone(),
                            },
                        )?
                        .into_any(),
                    );
                    continue;
                }
                let bytes = py
                    .detach(|| decoded(&t.dtype, &t.shape, encoding, &parts))
                    .or_raise(py)?;
                let mut buf = Aligned::new(bytes.len());
                buf.bytes_mut().copy_from_slice(&bytes);
                let a = PyArray {
                    buf: ArrayBuffer::Host(Arc::new(buf)),
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
                dtype: t.dtype.name().to_owned(),
                shape: PyTuple::new(py, &t.shape)?.unbind(),
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
) -> tensorfs::Result<Vec<u8>> {
    Ok(decode(dtype, shape, encoding, &views(parts))?)
}

fn views(parts: &[Want]) -> BTreeMap<&str, PartBytes<'_>> {
    (parts.iter())
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
        .collect()
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

    /// Read one tensor, a chosen sequence, or all tensors in construction order.
    #[pyo3(signature = (keys = None, *, device = "cpu", decode = false))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        keys: Option<&Bound<'py, PyAny>>,
        device: &str,
        decode: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let device = match device {
            "cpu" => None,
            "cuda" => Some(0),
            text => Some(
                text.strip_prefix("cuda:")
                    .and_then(|n| n.parse::<i32>().ok())
                    .filter(|n| *n >= 0)
                    .ok_or_else(|| {
                        invalid(
                            py,
                            format!("device {text:?}: expected cpu, cuda, or cuda:N"),
                        )
                    })?,
            ),
        };
        let single = keys.is_some_and(|keys| keys.is_instance_of::<PyString>());
        let mut keys = match keys {
            Some(keys) if single => vec![keys.extract::<String>()?],
            Some(keys) => keys.extract::<Vec<String>>()?,
            None => self.infos.bind(py).keys().extract()?,
        };
        let mut seen = std::collections::HashSet::new();
        keys.retain(|key| seen.insert(key.clone()));
        for key in &keys {
            if !self.infos.bind(py).contains(key)? {
                return Err(PyKeyError::new_err(key.clone()));
            }
        }
        if single {
            return Ok(self.load(py, &keys, decode, device)?.remove(0));
        }
        let out = PyDict::new(py);
        if !keys.is_empty() {
            for (key, value) in keys.iter().zip(self.load(py, &keys, decode, device)?) {
                out.set_item(key, value)?;
            }
        }
        Ok(out.into_any())
    }

    /// `len(out)` bytes of `key` from `offset` into the writable buffer `out`: a plain tensor's
    /// data, one stored part (`role`), or with `decode=True` an encoded tensor's logical bytes.
    /// An encoded tensor with neither is an `InvalidError` naming its roles.
    #[pyo3(signature = (key, out, *, role = None, decode = false, offset = 0))]
    fn read_into(
        &self,
        py: Python<'_>,
        key: String,
        out: &Bound<'_, PyAny>,
        role: Option<String>,
        decode: bool,
        offset: u64,
    ) -> PyResult<()> {
        if out.hasattr("__dlpack__")? {
            let mut held = crate::store::Dlpack::of(py, out, None, None)?;
            let result =
                self.fill_target(py, key, role.as_deref(), decode, offset, held.destination());
            if result.is_err() {
                held.record_cuda_error();
            }
            let finished = py.detach(move || held.finish()).or_raise(py);
            return result.and(finished);
        }
        let view = py
            .import("builtins")?
            .getattr("memoryview")?
            .call1((out,))?;
        if view.getattr("readonly")?.extract::<bool>()?
            || !view.getattr("c_contiguous")?.extract::<bool>()?
        {
            return Err(invalid(
                py,
                "out must be a writable C-contiguous buffer or DLPack array",
            ));
        }
        let buf: PyBuffer<u8> = PyBuffer::get(&view.call_method1("cast", ("B",))?)?;
        let dst = Dst::Host {
            ptr: buf.buf_ptr() as *mut u8,
            len: buf.len_bytes(),
        };
        self.fill_target(py, key, role.as_deref(), decode, offset, dst)
    }
}
