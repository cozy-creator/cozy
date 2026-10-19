//! The framing of TensorD's executor channels (fetch, derive, source): a kind byte, a big-endian
//! u32 length, then a JSON object or at most 1 MiB of bytes.
use std::io::{self, Read, Write};

const JSON: u8 = 0;
const BYTES: u8 = 1;
/// The most bytes one data frame carries.
pub const CHUNK: usize = 1 << 20;
/// The largest JSON frame.
pub const JSON_MAX: usize = 64 << 20;

#[derive(Debug, PartialEq)]
pub enum Frame {
    Json(serde_json::Value),
    Bytes(Vec<u8>),
}

pub struct Channel<S>(pub S);

impl<S: Read + Write> Channel<S> {
    fn write(&mut self, kind: u8, body: &[u8]) -> io::Result<()> {
        let length = u32::try_from(body.len()).map_err(|_| invalid("frame too long"))?;
        let mut head = [kind, 0, 0, 0, 0];
        head[1..].copy_from_slice(&length.to_be_bytes());
        self.0.write_all(&head)?;
        self.0.write_all(body)?;
        self.0.flush()
    }

    /// One JSON object.
    pub fn send(&mut self, value: &serde_json::Value) -> io::Result<()> {
        let body = serde_json::to_vec(value)?;
        if !value.is_object() || body.len() > JSON_MAX {
            return Err(invalid("a JSON frame is an object of at most 64 MiB"));
        }
        self.write(JSON, &body)
    }

    /// `bytes` as data frames of at most [`CHUNK`].
    pub fn data(&mut self, bytes: &[u8]) -> io::Result<()> {
        for chunk in bytes.chunks(CHUNK) {
            self.write(BYTES, chunk)?;
        }
        Ok(())
    }

    /// The next frame; `None` when the peer closed between frames.
    pub fn receive(&mut self) -> io::Result<Option<Frame>> {
        let mut head = [0u8; 5];
        match self.0.read_exact(&mut head[..1]) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }
        self.0.read_exact(&mut head[1..])?;
        let length = u32::from_be_bytes(head[1..].try_into().expect("four bytes")) as usize;
        let max = match head[0] {
            JSON => JSON_MAX,
            BYTES => CHUNK,
            _ => return Err(invalid("unknown frame kind")),
        };
        if length > max {
            return Err(invalid("frame too long"));
        }
        let mut body = vec![0; length];
        self.0.read_exact(&mut body)?;
        Ok(Some(match head[0] {
            JSON => match serde_json::from_slice(&body)? {
                value @ serde_json::Value::Object(_) => Frame::Json(value),
                _ => return Err(invalid("a JSON frame is an object")),
            },
            _ => Frame::Bytes(body),
        }))
    }
}

fn invalid(why: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn frames_round_trip_and_split_data() {
        let (a, b) = cozy_platform::ipc::pair().unwrap();
        let (mut tx, mut rx) = (Channel(a), Channel(b));
        let big = vec![7u8; CHUNK + 5];
        std::thread::scope(|s| {
            s.spawn(|| {
                tx.send(&json!({"op": "x"})).unwrap();
                tx.data(&big).unwrap();
                drop(tx);
            });
            assert_eq!(rx.receive().unwrap(), Some(Frame::Json(json!({"op": "x"}))));
            assert_eq!(rx.receive().unwrap(), Some(Frame::Bytes(vec![7; CHUNK])));
            assert_eq!(rx.receive().unwrap(), Some(Frame::Bytes(vec![7; 5])));
            assert_eq!(rx.receive().unwrap(), None);
        });
    }
}
