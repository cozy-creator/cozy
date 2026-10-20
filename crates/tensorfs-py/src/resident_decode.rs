//! Advanced resident-part decoding. This module only lends validated buffers to the
//! shared Core decoder; it owns no schema table, decoder, copy loop or CUDA driver.

use pyo3::buffer::PyUntypedBuffer;
use pyo3::prelude::*;
use pyo3::types::PyMapping;
use tensorfs::format::decode::PartMetadata;
use tensorfs::format::{Dtype, Encoding, Metadata, Value};
use tensorfs::io::decode::{Output, Part};
use tensorfs::io::{Dst, Src};
use tensorfs::{Failure, Kind};

use crate::store::{from_py, invalid, raise, Dlpack, OrRaise};

struct Description<'py> {
    role: String,
    dtype: Dtype,
    shape: Vec<u64>,
    len: usize,
    data: Bound<'py, PyAny>,
}

enum Storage {
    Dlpack(Dlpack),
    Buffer(PyUntypedBuffer),
}

impl Storage {
    fn open(py: Python<'_>, data: &Bound<'_, PyAny>) -> PyResult<Self> {
        if data.hasattr("__dlpack__")? {
            // Reuse the unique shared consumer. Its actual descriptor describes the
            // carrier, not the encoded format (F4, for example, is packed U8 storage).
            Ok(Self::Dlpack(Dlpack::of(py, data, None, None)?))
        } else {
            let buffer = PyUntypedBuffer::get(data)?;
            if !buffer.is_c_contiguous() {
                return Err(invalid(py, "decode storage must be C-contiguous"));
            }
            let format = buffer.format().to_bytes();
            let format = match format.first() {
                Some(b'@' | b'=' | b'<') => &format[1..],
                _ => format,
            };
            if format.len() != 1 || !b"bB?hHiIlLqQefd".contains(&format[0]) {
                return Err(invalid(
                    py,
                    "decode storage must contain primitive POD bytes",
                ));
            }
            Ok(Self::Buffer(buffer))
        }
    }

    fn address(&self) -> usize {
        match self {
            Self::Dlpack(value) => value.address,
            Self::Buffer(value) => value.buf_ptr() as usize,
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Dlpack(value) => value.len,
            Self::Buffer(value) => value.len_bytes(),
        }
    }

    fn device(&self) -> Option<i32> {
        match self {
            Self::Dlpack(value) if value.device.0 == 2 => Some(value.device.1),
            _ => None,
        }
    }

    fn source(&self) -> Src<'_> {
        match self {
            Self::Dlpack(value) => match value.stream() {
                Some(stream) => Src::Device {
                    device: value.device.1,
                    ptr: value.address as u64,
                    stream: stream.raw(),
                },
                None => Src::Host(value.address as *const u8),
            },
            Self::Buffer(value) => Src::Host(value.buf_ptr().cast()),
        }
    }

    fn destination(&self) -> Dst {
        match self {
            Self::Dlpack(value) => value.destination(),
            Self::Buffer(value) => Dst::Host {
                ptr: value.buf_ptr().cast(),
                len: value.len_bytes(),
            },
        }
    }

    fn output_matches(&self, dtype: &Dtype, shape: &[u64]) -> bool {
        match self {
            Self::Dlpack(value) => value.dtype == *dtype && value.shape == shape,
            Self::Buffer(value) => {
                if value.readonly() {
                    return false;
                }
                let actual_shape = if value.dimensions() == 0 {
                    Vec::new()
                } else {
                    value.shape().iter().map(|n| *n as u64).collect()
                };
                let format = value.format().to_bytes();
                let format = match format.first() {
                    Some(b'@' | b'=' | b'<') => &format[1..],
                    _ => format,
                };
                let element = match dtype {
                    Dtype::F32 => (b"f".as_slice(), 4),
                    Dtype::F16 => (b"e".as_slice(), 2),
                    _ => return false,
                };
                actual_shape == shape && format == element.0 && value.item_size() == element.1
            }
        }
    }

    fn checked_span(&self, py: Python<'_>) -> PyResult<std::ops::Range<usize>> {
        let start = self.address();
        let len = self.len();
        if len != 0 && start == 0 {
            return Err(invalid(py, "decode storage has no data"));
        }
        let end = start
            .checked_add(len)
            .ok_or_else(|| invalid(py, "decode address range overflows"))?;
        Ok(start..end)
    }
}

/// No member is released until the engine AND every producer have proved completion.
/// If one proof fails, preserve the entire actual input/output owner graph.
struct CudaOwners {
    device: i32,
    owners: Option<Vec<Storage>>,
}

/// Armed before entering native code, including unwinding out of an unsuccessful call.
struct Owners {
    device: Option<i32>,
    values: Option<Vec<Storage>>,
}

impl Drop for Owners {
    fn drop(&mut self) {
        if let Some(device) = self.device {
            crate::cleanup::defer(CudaOwners {
                device,
                owners: self.values.take(),
            });
        }
    }
}

impl Drop for CudaOwners {
    fn drop(&mut self) {
        let Some(mut owners) = self.owners.take() else {
            return;
        };
        let mut complete = tensorfs::io::drain(self.device).is_ok();
        if complete {
            for owner in &mut owners {
                if let Storage::Dlpack(owner) = owner {
                    if owner.checked_drain().is_err() {
                        complete = false;
                        break;
                    }
                }
            }
        }
        if !complete {
            for owner in &mut owners {
                if let Storage::Dlpack(owner) = owner {
                    owner.retain();
                }
            }
            std::mem::forget(owners);
        }
        // Successful owners drop here, outside the GIL and all engine/Writer locks.
        // Their shared managed-deleter cleanup attaches Python only after the drains.
    }
}

fn byte_len(py: Python<'_>, dtype: &Dtype, shape: &[u64]) -> PyResult<usize> {
    dtype
        .byte_len(shape)
        .and_then(|n| usize::try_from(n).ok())
        .ok_or_else(|| invalid(py, "decode dtype or byte length is invalid"))
}

fn parameters(py: Python<'_>, params: Option<&Bound<'_, PyAny>>) -> PyResult<Metadata> {
    match params {
        None => Ok(Metadata::new()),
        Some(params) => {
            let dict = py.import("builtins")?.getattr("dict")?.call1((params,))?;
            match from_py(&dict)? {
                Value::Map(value) => Ok(value),
                _ => unreachable!("dict conversion produces a metadata map"),
            }
        }
    }
}

#[pyfunction]
#[pyo3(signature = (scheme, parts, out, *, dtype, shape, params = None))]
fn decode_into(
    py: Python<'_>,
    scheme: String,
    parts: &Bound<'_, PyMapping>,
    out: &Bound<'_, PyAny>,
    dtype: String,
    shape: Vec<u64>,
    params: Option<&Bound<'_, PyAny>>,
) -> PyResult<()> {
    let encoding = Encoding {
        scheme,
        params: parameters(py, params)?,
    };
    let dtype = Dtype::parse(&dtype);
    let length = byte_len(py, &dtype, &shape)?;
    let mut descriptions = Vec::new();
    for item in parts.items()?.iter() {
        let (role, value): (String, Bound<'_, PyAny>) = item.extract()?;
        let (dtype, shape, data): (String, Vec<u64>, Bound<'_, PyAny>) = value.extract()?;
        let dtype = Dtype::parse(&dtype);
        let len = byte_len(py, &dtype, &shape)?;
        descriptions.push(Description {
            role,
            dtype,
            shape,
            len,
            data,
        });
    }
    let metadata: Vec<_> = descriptions
        .iter()
        .map(|part| PartMetadata {
            role: &part.role,
            dtype: &part.dtype,
            shape: &part.shape,
            len: part.len,
        })
        .collect();
    // The same Core metadata rules run before a producer is asked for a capsule.
    tensorfs::io::decode::check(&encoding, &dtype, &shape, &metadata).or_raise(py)?;

    let mut owners = Vec::new();
    for part in &descriptions {
        let owner = Storage::open(py, &part.data)?;
        if owner.len() != part.len {
            return Err(invalid(
                py,
                "decode part byte length disagrees with its descriptor",
            ));
        }
        owner.checked_span(py)?;
        owners.push(owner);
    }
    let destination = Storage::open(py, out)?;
    if !destination.output_matches(&dtype, &shape) || destination.len() != length {
        return Err(invalid(
            py,
            "decode output must exactly match logical dtype and shape",
        ));
    }
    let span = destination.checked_span(py)?;
    let device = destination.device();
    for owner in &owners {
        if owner.device() != device {
            return Err(raise(
                py,
                Failure {
                    kind: Kind::Unsupported,
                    detail: "decode parts and output must have the same placement".into(),
                },
            ));
        }
        let source = owner.checked_span(py)?;
        if !source.is_empty()
            && !span.is_empty()
            && source.start < span.end
            && span.start < source.end
        {
            return Err(invalid(py, "decode input and output storage overlap"));
        }
    }
    owners.push(destination);
    let held = Owners {
        device,
        values: Some(owners),
    };
    let owners = held
        .values
        .as_ref()
        .expect("all actual decode owners are held");
    let sources: Vec<_> = descriptions
        .iter()
        .zip(owners)
        .map(|(part, owner)| Part {
            role: &part.role,
            dtype: &part.dtype,
            shape: &part.shape,
            src: owner.source(),
            len: owner.len(),
        })
        .collect();
    let output = Output {
        dtype: &dtype,
        shape: &shape,
        dst: owners
            .last()
            .expect("output owner was appended")
            .destination(),
    };
    // SAFETY: every actual owner/lease remains in `owners`, all spans and descriptors
    // were checked, output is disjoint, and Core returns only after joined CPU work or
    // checked CUDA completion. An error still transfers all CUDA owners to custody.
    let result = py.detach(|| unsafe {
        tensorfs::raw::decode_into(&encoding, &dtype, &shape, &sources, output)
    });
    drop(sources);
    drop(held);
    result.or_raise(py)
}

pub(crate) fn install(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(decode_into, module)?)
}
