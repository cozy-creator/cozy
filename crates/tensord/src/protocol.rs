use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

/// Not a size rule: the most one length prefix may make a reader allocate, on this socket
/// and the executor seam alike (Runtime's `seam.MAX_FRAME`). Load replies grow with their
/// checkpoint and stay far below it, so it refuses only a corrupt or hostile prefix.
pub const MAX_FRAME: usize = 256 << 20;
/// One frame's body, read as it arrives: a prefix is never trusted with an allocation. EOF is
/// clean only between frames.
pub fn read_body(stream: &mut UnixStream) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0; 4];
    if stream.read(&mut header[..1])? == 0 {
        return Ok(None);
    }
    stream.read_exact(&mut header[1..])?;
    let length = u32::from_be_bytes(header) as usize;
    if length == 0 || length > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {length} outside 1..={MAX_FRAME}"),
        ));
    }
    let mut bytes = Vec::with_capacity(length.min(1 << 20));
    Read::take(&mut *stream, length as u64).read_to_end(&mut bytes)?;
    if bytes.len() < length {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(Some(bytes))
}
pub fn write_body(stream: &mut UnixStream, bytes: &[u8]) -> io::Result<()> {
    if bytes.len() > MAX_FRAME {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("a {} B frame is over the {MAX_FRAME} B guard", bytes.len()),
        ));
    }
    stream.write_all(&(bytes.len() as u32).to_be_bytes())?;
    stream.write_all(bytes)
}
