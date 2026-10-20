//! A contiguous CUDA DLPack tensor as a bounded Writer input. The Python bridge validates
//! the shared DLPack view and orders its producer onto the consumer stream before this
//! reader takes ownership. All transfers go through TensorFS's byte engine.

use std::io::{self, Read};

use tensorfs::io::{self as engine, Copy, Dst, ReadOpts, Src};

use crate::store::Dlpack;

const MAX_READ: usize = 8 << 20;

pub(crate) struct DeviceSource {
    // The shared parser's consumed producer owner and checked stream lifetime.
    view: Dlpack,
    offset: usize,
    failed: bool,
}

impl DeviceSource {
    /// Take the shared parser's validated, consumed CUDA tensor. No second ABI parse.
    pub(crate) fn new(view: Dlpack) -> io::Result<Self> {
        let stream = view.stream().ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidInput, "a CUDA source needs a stream")
        })?;
        if view.device.1 < 0 || stream.device() != view.device.1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "DLPack source and consumer stream use different CUDA devices",
            ));
        }
        Ok(Self {
            view,
            offset: 0,
            failed: false,
        })
    }

    /// Called after the Writer releases its mutex, including on operation errors.
    pub(crate) fn finish(&mut self) -> tensorfs::Result<()> {
        self.view.finish()
    }
}

impl Read for DeviceSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::other("the previous CUDA source read failed"));
        }
        let len = buf.len().min(self.view.len - self.offset).min(MAX_READ);
        if len == 0 {
            return Ok(0);
        }
        let copy = Copy {
            src: Src::Device {
                device: self.view.device.1,
                ptr: self.view.address as u64 + self.offset as u64,
                stream: self.view.stream().expect("live CUDA source").raw(),
            },
            dst: Dst::host(&mut buf[..len]),
        };
        // SAFETY: the shared parser's unique managed owner retains the source allocation and the
        // stream orders producer access. The constructor checked the complete address
        // range; this copy stays within it. The exclusive buffer borrow is its destination.
        // A failed engine read is not a completion proof for its other copy streams.
        if let Err(error) = unsafe { engine::read(&[copy], &ReadOpts::default()) } {
            self.failed = true;
            self.view.record_cuda_error();
            return Err(error.source);
        }
        self.offset += len;
        Ok(len)
    }
}
