//! Strict bounded decode of a foreign SAFE carrier. Nothing here executes, imports, or
//! deserializes a publisher object graph: a carrier is a length, a JSON header, and a byte
//! region, and every one of those is checked against arithmetic before a buffer exists.
//!
//! Three properties this file is built to hold:
//!
//! 1. **Refuse before allocation.** The 8-byte length prefix is compared with the cap AND
//!    with the file's real length before `Vec::with_capacity` is ever reached. A header
//!    claiming 2^63 bytes costs one `read_exact` of 8 bytes.
//! 2. **One JSON reader.** The foreign header goes through `canon::parse` — the same
//!    bounded, integer-only, ASCII-only, duplicate-key-refusing reader every canonical
//!    document uses. A second tolerant parser at the border is exactly the drift this repo
//!    is built against. The foreign header is NOT required to be canonically ENCODED (it is
//!    someone else's file, pretty-printed as they please); it is required to be inside the
//!    canonical GRAMMAR.
//! 3. **Geometry is arithmetic, not trust.** Every tensor's declared byte run must equal
//!    `checked_prod(shape) x sizeof(dtype)`, must lie inside the file, and must not overlap
//!    its neighbour. A lying header is refused by multiplication, never by a failed read.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use crate::canon::{self, as_arr, as_str, Fields, Value};
use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::limits;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

// ---------------------------------------------------------------- the closed dialect set

/// What a carrier CALLS its element types, mapped into the closed cozytensors enum. A
/// spelling outside this table refuses `DTYPE_UNKNOWN` — a foreign name is never guessed at.
const FOREIGN_DTYPES: [(&str, Dtype); 14] = [
    ("F64", Dtype::F64),
    ("F32", Dtype::F32),
    ("F16", Dtype::F16),
    ("BF16", Dtype::Bf16),
    ("F8_E4M3", Dtype::F8E4M3FN),
    ("F8_E4M3FN", Dtype::F8E4M3FN),
    ("F8_E5M2", Dtype::F8E5M2),
    ("I64", Dtype::I64),
    ("I32", Dtype::I32),
    ("I16", Dtype::I16),
    ("I8", Dtype::I8),
    ("U8", Dtype::U8),
    ("BOOL", Dtype::Bool),
    ("BYTES", Dtype::U8),
];

fn foreign_dtype(key: &str, name: &str) -> Result<Dtype> {
    match FOREIGN_DTYPES.iter().find(|(n, _)| *n == name) {
        Some((_, d)) => Ok(*d),
        None => refuse(
            Code::DTYPE_UNKNOWN,
            format!("{key}: carrier dtype {name:?} is outside the border's closed table"),
        ),
    }
}

// ---------------------------------------------------------------- magic sniff

/// What the first bytes SAY the file is. Observed, never opened: a pickle refusal that had
/// to unpickle to reach its verdict is not a refusal, it is an exploit with a log line.
///
/// Ordered so the strongest evidence wins. The safe carrier has no magic at all (it opens
/// with a little-endian length), so it is the fallthrough — which is why every hostile shape
/// must be named here explicitly rather than inferred from "not one of ours".
pub fn sniff(head: &[u8]) -> Result<()> {
    if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
        return refuse(
            Code::ARCHIVE_REFUSED,
            "zip archive (a pickled `.pt`/`.bin` object graph): normal ingest opens no archive \
             and imports no publisher class; the remedy is a safe-carrier export of the \
             same weights",
        );
    }
    if head.starts_with(b"GGUF") {
        return refuse(
            Code::GGUF_REFUSED,
            "GGUF: consumer-gated, never admitted by the normal door (tensorfs.md §6)",
        );
    }
    // PROTO opcode + a protocol number is a pickle stream; the classic protocol-0/1 streams
    // open with a MARK, EMPTY_DICT or a memo op. Both are named, neither is fed to anything.
    if head.first() == Some(&0x80) && matches!(head.get(1), Some(2..=5)) {
        return refuse(
            Code::PICKLE_REFUSED,
            "pickle PROTO opcode: refused on the magic, with no opcode interpreted and no \
             publisher class imported",
        );
    }
    if matches!(head.first(), Some(b'(') | Some(b']') | Some(b'}') | Some(b'c')) {
        return refuse(
            Code::PICKLE_REFUSED,
            "protocol-0/1 pickle opening opcode: refused on the magic, nothing interpreted",
        );
    }
    if head.starts_with(b"\x1f\x8b") || head.starts_with(b"BZh") || head.starts_with(b"\xfd7zXZ") {
        return refuse(
            Code::ARCHIVE_REFUSED,
            "compressed stream: the border decompresses nothing (an unbounded expansion is \
             a resource claim the caller never made)",
        );
    }
    Ok(())
}

// ---------------------------------------------------------------- decoded shape

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceTensor {
    pub key: String,
    pub dtype: Dtype,
    pub shape: Vec<u64>,
    /// Offsets are relative to the data region, exactly as the carrier states them.
    pub begin: u64,
    pub end: u64,
}

impl SourceTensor {
    pub fn nbytes(&self) -> u64 {
        self.end - self.begin
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SourceHeader {
    pub tensors: Vec<SourceTensor>,
    /// `__metadata__` — foreign JSON, decoded as flat strings only, never interpreted here.
    pub metadata: Vec<(String, String)>,
    pub header_bytes: u64,
    pub data_start: u64,
    pub file_len: u64,
    /// Declared bytes that no tensor claims. Recorded, not silently absorbed.
    pub gap_bytes: u64,
}

impl SourceHeader {
    pub fn declared_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.nbytes()).sum()
    }
    pub fn get(&self, key: &str) -> Option<&SourceTensor> {
        self.tensors.iter().find(|t| t.key == key)
    }
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------- the decode

/// Decode one carrier's header from an open file. Reads at most
/// `8 + CARRIER_HEADER_MAX_BYTES` bytes and never touches the data region.
pub fn read_header(path: &Path) -> Result<SourceHeader> {
    let mut f = File::open(path).map_err(|e| io(format!("open {}", path.display()), e))?;
    let file_len = f
        .metadata()
        .map_err(|e| io("stat", e))?
        .len();

    let mut head = [0u8; 16];
    let n = read_upto(&mut f, &mut head)?;
    sniff(&head[..n])?;
    if n < 8 {
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!("{n} bytes: too short to carry an 8-byte header length"),
        );
    }
    let declared = u64::from_le_bytes(head[0..8].try_into().unwrap());

    // Refuse BEFORE the allocation, against both the cap and the real file.
    if declared > limits::CARRIER_HEADER_MAX_BYTES as u64 {
        return refuse(
            Code::CARRIER_HEADER_CAP,
            format!(
                "header claims {declared} B, cap is {} B — refused before allocating",
                limits::CARRIER_HEADER_MAX_BYTES
            ),
        );
    }
    let data_start = match declared.checked_add(8) {
        Some(v) => v,
        None => return refuse(Code::ARITH_OVERFLOW, "header length + 8 overflows"),
    };
    if data_start > file_len {
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!("header claims {declared} B but the file holds {file_len} B in total"),
        );
    }

    f.seek(SeekFrom::Start(8)).map_err(|e| io("seek", e))?;
    let mut buf = vec![0u8; declared as usize];
    f.read_exact(&mut buf)
        .map_err(|e| io(format!("read {declared} header bytes"), e))?;

    parse_header(&buf, data_start, Some(file_len))
}

/// The header bytes alone → the decoded, geometry-checked shape. Separated from IO so a
/// committed header (real evidence with no artifact behind it) decodes through the exact
/// same code.
///
/// `file_len` is `None` for that headerless-evidence case: the data region is then taken to
/// be exactly what the runs declare, which is the only honest reading when there is no file
/// to measure. Every other check is identical.
pub fn parse_header(bytes: &[u8], data_start: u64, file_len: Option<u64>) -> Result<SourceHeader> {
    let v = canon::parse(bytes, limits::CARRIER_HEADER_MAX_BYTES)?;
    let entries = match &v {
        Value::Obj(m) => m,
        o => {
            return refuse(
                Code::WRONG_TYPE,
                format!("carrier header: expected object, got {}", o.kind()),
            )
        }
    };
    if entries.len() > limits::MAX_TENSORS {
        return refuse(
            Code::COUNT_CAP,
            format!("carrier header holds {} keys, cap {}", entries.len(), limits::MAX_TENSORS),
        );
    }

    let mut out = SourceHeader {
        header_bytes: data_start - 8,
        data_start,
        file_len: file_len.unwrap_or(data_start),
        ..Default::default()
    };

    for (key, ev) in entries {
        if key == "__metadata__" {
            for (mk, mv) in canon::as_obj("carrier", "__metadata__", ev)? {
                out.metadata.push((
                    mk.clone(),
                    match mv {
                        Value::Str(s) => s.clone(),
                        // Foreign metadata is decoded as flat strings under a strict schema
                        // or discarded. A nested blob is not silently carried forward.
                        o => {
                            return refuse(
                                Code::WRONG_TYPE,
                                format!("__metadata__.{mk}: expected string, got {}", o.kind()),
                            )
                        }
                    },
                ));
            }
            continue;
        }
        if key.is_empty() || key.len() > limits::MAX_KEY_BYTES {
            return refuse(
                Code::KEY_GRAMMAR,
                format!("carrier key length {} out of bounds", key.len()),
            );
        }
        let mut f = Fields::new("carrier.tensor", ev)?;
        let offsets = as_arr("carrier.tensor", "data_offsets", f.req("data_offsets")?)?;
        let dtype = foreign_dtype(key, as_str("carrier.tensor", "dtype", f.req("dtype")?)?)?;
        let shape = as_arr("carrier.tensor", "shape", f.req("shape")?)?
            .iter()
            .map(|x| canon::as_uint("carrier.tensor", "shape", x))
            .collect::<Result<Vec<u64>>>()?;
        f.done()?;

        if offsets.len() != 2 {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{key}: data_offsets holds {} values, not 2", offsets.len()),
            );
        }
        let begin = canon::as_uint("carrier.tensor", "data_offsets", &offsets[0])?;
        let end = canon::as_uint("carrier.tensor", "data_offsets", &offsets[1])?;
        if end < begin {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!("{key}: data_offsets [{begin}, {end}) runs backwards"),
            );
        }
        // The anchor equation, at the BORDER: what the geometry implies must be what the
        // file claims. This is the check that makes a truncated or doctored file arithmetic.
        let want = checked_bytes(key, &shape, dtype)?;
        if end - begin != want {
            return refuse(
                Code::CARRIER_GEOMETRY,
                format!(
                    "{key}: declared run {} B != {:?} x {} = {want} B",
                    end - begin,
                    shape,
                    dtype.name()
                ),
            );
        }
        out.tensors.push(SourceTensor {
            key: key.clone(),
            dtype,
            shape,
            begin,
            end,
        });
    }

    if out.tensors.is_empty() {
        return refuse(Code::MISSING_FIELD, "carrier declares no tensors");
    }

    let declared_end = out.tensors.iter().map(|t| t.end).max().unwrap_or(0);
    let region = match file_len {
        Some(f) => f - data_start,
        None => declared_end,
    };
    if declared_end > region {
        return refuse(
            Code::CARRIER_TRUNCATED,
            format!(
                "a run ends at {declared_end} but the data region holds {region} B \
                 (file {} B, data starts at {data_start})",
                out.file_len
            ),
        );
    }
    out.file_len = data_start + region;

    // Overlap refuses. Two keys sharing bytes is either corruption or an aliasing trick,
    // and neither has a meaning the border is willing to invent.
    let mut order: Vec<usize> = (0..out.tensors.len()).collect();
    order.sort_by_key(|i| out.tensors[*i].begin);
    let mut cursor = 0u64;
    for i in order {
        let t = &out.tensors[i];
        if t.begin < cursor {
            return refuse(
                Code::CARRIER_OVERLAP,
                format!(
                    "{}: run starts at {} inside the preceding run that ends at {cursor}",
                    t.key, t.begin
                ),
            );
        }
        out.gap_bytes += t.begin - cursor;
        cursor = t.end;
    }
    out.gap_bytes += region - cursor;
    // UNCLAIMED BYTES REFUSE. A safe carrier's data region is exactly the concatenation of
    // its declared runs — every real artifact measured here (SDXL's four components, 6.9 GB,
    // and the shipped H3 DiT header) reports zero unclaimed bytes, and the v1 quarry's own
    // strictest reader demanded the same gapless coverage. Bytes inside the file that no
    // key claims are a channel: they are ingested by nobody, verified by nothing, and
    // preserved by any tool that copies the file whole.
    if out.gap_bytes != 0 {
        return refuse(
            Code::CARRIER_GEOMETRY,
            format!(
                "{} B of the {region} B data region are claimed by no tensor — a safe \
                 carrier's region is exactly its declared runs, with no padding and no gaps",
                out.gap_bytes
            ),
        );
    }
    Ok(out)
}

fn read_upto(f: &mut File, buf: &mut [u8]) -> Result<usize> {
    let mut n = 0;
    while n < buf.len() {
        match f.read(&mut buf[n..]).map_err(|e| io("read magic", e))? {
            0 => break,
            k => n += k,
        }
    }
    Ok(n)
}

// ---------------------------------------------------------------- bounded region reader

/// A reader over ONE tensor's declared byte run. Bounded by the declaration, positioned
/// once, and never allowed to read past the run — the writer downstream streams from this
/// and therefore can never be handed more bytes than the geometry promised.
pub struct Region {
    f: File,
    left: u64,
}

impl Region {
    pub fn open(path: &Path, data_start: u64, t: &SourceTensor) -> Result<Region> {
        let mut f = File::open(path).map_err(|e| io(format!("open {}", path.display()), e))?;
        f.seek(SeekFrom::Start(data_start + t.begin))
            .map_err(|e| io("seek to run", e))?;
        Ok(Region {
            f,
            left: t.nbytes(),
        })
    }
}

impl Read for Region {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.left == 0 {
            return Ok(0);
        }
        let cap = (buf.len() as u64).min(self.left) as usize;
        let n = self.f.read(&mut buf[..cap])?;
        self.left -= n as u64;
        Ok(n)
    }
}
