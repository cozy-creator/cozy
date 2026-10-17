use serde::{Deserialize, Serialize};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;

/// Not a size rule: the most one length prefix may make a reader allocate, on this socket
/// and the executor seam alike (Runtime's `seam.MAX_FRAME`). Load replies grow with their
/// checkpoint and stay far below it, so it refuses only a corrupt or hostile prefix.
pub const MAX_FRAME: usize = 256 << 20;
pub const CAPS: &[&str] = &["weights.hosted/1", "objects.put-fd/1", "execution.cpu/1"];

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Object {
    pub sha256: String,
    pub length: u64,
}
#[derive(Debug, Deserialize)]
pub struct Request {
    pub seq: u64,
    #[serde(flatten)]
    pub command: Command,
}
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Command {
    Hello {
        #[serde(default)]
        runtime: String,
        #[serde(default)]
        capabilities: Vec<String>,
    },
    Import {
        object: Object,
    },
    Attach {
        object: Object,
    },
    Release {
        lease: u64,
        incarnation: String,
    },
    Stats,
    Submit {
        key: String,
        generation: String,
        entrypoint: String,
        input: serde_json::Value,
    },
    Execution {
        id: String,
    },
    Executions,
    Cancel {
        id: String,
    },
    ReadResult {
        id: String,
        index: usize,
    },
    Shutdown,
    #[serde(other)]
    Unknown,
}
#[derive(Debug, Serialize)]
pub struct Reply {
    pub seq: u64,
    #[serde(flatten)]
    pub body: Body,
}
#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Body {
    Hello {
        version: &'static str,
        tensorfs: &'static str,
        incarnation: String,
        capabilities: Vec<String>,
    },
    Imported {
        object: Object,
        admitted: bool,
    },
    Attached {
        object: Object,
        lease: u64,
        incarnation: String,
        tier: &'static str,
    },
    Released,
    Stats {
        host_bytes: u64,
        cached_objects: usize,
        active_leases: usize,
        host_budget: u64,
    },
    Shutdown,
    Execution {
        record: Box<crate::journal::Execution>,
    },
    Executions {
        records: Vec<crate::journal::Execution>,
    },
    ResultArtifact {
        id: String,
        artifact: crate::journal::Artifact,
    },
    Error {
        code: &'static str,
        detail: String,
    },
}
pub fn read(stream: &mut UnixStream) -> io::Result<Option<Request>> {
    read_body(stream)?
        .map(|bytes| serde_json::from_slice(&bytes).map_err(io::Error::other))
        .transpose()
}
pub fn write(stream: &mut UnixStream, reply: &Reply) -> io::Result<()> {
    write_body(stream, &serde_json::to_vec(reply)?)
}
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
