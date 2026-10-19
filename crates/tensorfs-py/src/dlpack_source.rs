//! A contiguous CUDA DLPack tensor as a bounded Writer input. The Python bridge validates
//! the shared DLPack view and orders its producer onto the consumer stream before this
//! reader takes ownership. All transfers go through TensorFS's byte engine.

use std::io::{self, Read};

use pyo3::{Py, PyAny};
use tensorfs::io::{self as engine, cuda::Stream, Copy, Dst, ReadOpts, Src};

const MAX_READ: usize = 8 << 20;

pub(crate) struct DeviceSource {
    // Drop the synchronizing stream before releasing the producer's allocation, including
    // when construction succeeded but the Writer never requested a byte.
    stream: Stream,
    _capsule: Py<PyAny>,
    device: i32,
    ptr: u64,
    nbytes: usize,
    offset: usize,
}

impl DeviceSource {
    /// `capsule` is the unconsumed capsule for the bridge's validated, contiguous CUDA
    /// tensor. `ptr` already includes its byte_offset; `nbytes` is the validated dtype/shape
    /// size. The producer's `__dlpack__(stream=stream.raw())` orders access to these bytes.
    pub(crate) fn new(
        capsule: Py<PyAny>,
        stream: Stream,
        device: i32,
        ptr: u64,
        nbytes: usize,
    ) -> io::Result<Self> {
        if device < 0 || stream.device() != device {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DLPack source and consumer stream use different CUDA devices",
            ));
        }
        let len = u64::try_from(nbytes).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "DLPack byte size overflows")
        })?;
        if ptr.checked_add(len).is_none() || (nbytes != 0 && ptr == 0) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DLPack device address is invalid",
            ));
        }
        Ok(Self {
            stream,
            _capsule: capsule,
            device,
            ptr,
            nbytes,
            offset: 0,
        })
    }
}

impl Read for DeviceSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let len = buf.len().min(self.nbytes - self.offset).min(MAX_READ);
        if len == 0 {
            return Ok(0);
        }
        let copy = Copy {
            src: Src::Device {
                device: self.device,
                ptr: self.ptr + self.offset as u64,
                stream: self.stream.raw(),
            },
            dst: Dst::host(&mut buf[..len]),
        };
        // SAFETY: the bridge's validated capsule retains the source allocation and the
        // stream orders producer access. The constructor checked the complete address
        // range; this copy stays within it. The exclusive buffer borrow is its destination.
        // The engine completes or drains the transfer before returning, including errors.
        unsafe { engine::read(&[copy], &ReadOpts::default()) }.map_err(|error| error.source)?;
        self.offset += len;
        Ok(len)
    }
}
