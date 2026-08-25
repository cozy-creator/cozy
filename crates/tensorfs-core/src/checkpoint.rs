//! The checkpoint plane: objectize real tensor bytes through the real CAS, emit the
//! canonical header, and reconstruct from the header alone.
//!
//! The store is shape-agnostic (tfs-002) and stays that way — every shape fact is decided
//! here, before a single body byte moves. The anchor equation
//! `checked_prod(shape) × sizeof(dtype) = Σ segment lengths` is checked BEFORE any fetch,
//! so a lying header is refused by arithmetic rather than by a failed download.

use std::io::{Read, Write};

use crate::b64;
use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Body, Closure, Header, Located, Part};
use crate::ids::{Doc, ObjectRef};
use crate::limits;
use crate::sha256::{self, Sha256};
use crate::spec::EncodingSpec;
use crate::store::{Fault, Store};

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

/// A reader that hashes what passes through it — the whole-part digest costs no second pass.
struct Tee<'a, R: Read> {
    r: R,
    h: &'a mut Sha256,
}

impl<R: Read> Read for Tee<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.r.read(buf)?;
        self.h.update(&buf[..n]);
        Ok(n)
    }
}

struct TeeW<'a, W: Write> {
    w: &'a mut W,
    h: &'a mut Sha256,
}

impl<W: Write> Write for TeeW<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.h.update(buf);
        self.w.write_all(buf)?;
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.w.flush()
    }
}

#[derive(Debug, Clone)]
pub struct Written {
    pub part: Part,
    /// sha256 of the part's whole logical byte run — the end-to-end identity claim.
    pub whole: String,
    pub bytes: u64,
    /// Bytes whose object was already installed: byte-identical part-run dedup, and only that.
    pub deduped: u64,
    pub segments: usize,
}

/// Writer policy, PROVISIONAL until proto-001/tfs-018 ratifies the constants: a role at or
/// under `INLINE_MAX_BYTES` is inline, above it objectized on the part-relative
/// `GRID_BYTES` grid. Never author-selectable — the same bytes have exactly one form.
/// Readers consume the emitted segment records and never re-derive this policy.
pub fn objectize<R: Read>(
    store: &Store,
    what: &str,
    dtype: Dtype,
    shape: Vec<u64>,
    src: &mut R,
) -> Result<Written> {
    let want = checked_bytes(what, &shape, dtype)?;
    let mut h = Sha256::new();

    if Part::is_inline(want) {
        let mut buf = vec![0u8; want as usize];
        src.read_exact(&mut buf)
            .map_err(|e| io(format!("{what}: read {want} inline bytes"), e))?;
        h.update(&buf);
        return Ok(Written {
            part: Part {
                dtype,
                shape,
                body: Body::Inline(b64::encode(&buf)),
            },
            whole: sha256::hex(&h.finish()),
            bytes: want,
            deduped: 0,
            segments: 0,
        });
    }

    let mut segs = Vec::new();
    let mut left = want;
    let mut deduped = 0u64;
    while left > 0 {
        let n = left.min(limits::GRID_BYTES);
        let mut t = Tee {
            r: (&mut *src).take(n),
            h: &mut h,
        };
        let put = store.put_stream(&mut t, None, &Fault::default())?;
        if put.obj.length != n {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{what}: source gave {} bytes for a {n}-byte segment",
                    put.obj.length
                ),
            );
        }
        if !put.admitted {
            deduped += n;
        }
        segs.push(put.obj);
        left -= n;
    }
    let segments = segs.len();
    Ok(Written {
        part: Part {
            dtype,
            shape,
            body: Body::Segments(segs),
        },
        whole: sha256::hex(&h.finish()),
        bytes: want,
        deduped,
        segments,
    })
}

/// Reconstruct a part's whole byte run. The anchor equation is validated first; each
/// segment's delivered length is checked against the record that promised it.
pub fn materialize<W: Write>(store: &Store, what: &str, part: &Part, w: &mut W) -> Result<String> {
    part.check_bytes(what)?;
    let mut h = Sha256::new();
    match &part.body {
        Body::Inline(s) => {
            let b = b64::decode(s)?;
            h.update(&b);
            w.write_all(&b).map_err(|e| io("write inline", e))?;
        }
        Body::Segments(segs) => {
            for (i, s) in segs.iter().enumerate() {
                let mut t = TeeW { w, h: &mut h };
                let n = store.read_into(&s.sha256, &mut t)?;
                if n != s.length {
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        format!(
                            "{what}: segment {i} delivered {n} bytes, its record promised {}",
                            s.length
                        ),
                    );
                }
            }
        }
    }
    Ok(sha256::hex(&h.finish()))
}

/// A bounded slice of a part, fetched through the derived offsets and nothing else.
pub fn read_span(store: &Store, what: &str, part: &Part, off: u64, len: u64) -> Result<Vec<u8>> {
    match part.locate(what, off, len)? {
        Located::Inline { off, len } => {
            let b = match &part.body {
                Body::Inline(s) => b64::decode(s)?,
                Body::Segments(_) => unreachable!(),
            };
            Ok(b[off as usize..(off + len) as usize].to_vec())
        }
        Located::Segments(spans) => {
            let mut out = Vec::with_capacity(len as usize);
            for s in spans {
                out.extend_from_slice(&store.read_range(&s.obj.sha256, s.off, s.len)?);
            }
            Ok(out)
        }
    }
}

// ------------------------------------------------------------------ documents

/// Read one length-bearing reference. The length refuses BEFORE the body is trusted.
pub fn read_object(store: &Store, r: &ObjectRef, max: usize) -> Result<Vec<u8>> {
    if r.length > max as u64 {
        return refuse(
            Code::SIZE_CAP,
            format!(
                "{}: {} bytes over the {max} B document cap",
                r.id(),
                r.length
            ),
        );
    }
    let mut v = Vec::with_capacity(r.length as usize);
    store.read_into(&r.sha256, &mut v)?;
    if v.len() as u64 != r.length {
        return refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{}: delivered {} bytes, reference says {}",
                r.id(),
                v.len(),
                r.length
            ),
        );
    }
    Ok(v)
}

/// Documents admit as their EXACT canonical bytes: `put_expected` proves the id the writer
/// advertises is the id the store installed.
pub fn put_doc<D: Doc>(store: &Store, d: &D) -> Result<ObjectRef> {
    let bytes = d.canonical_bytes();
    let want = ObjectRef::of(&bytes);
    let put = store.put_stream(&mut bytes.as_slice(), Some(&want), &Fault::default())?;
    Ok(put.obj)
}

pub fn load_header(store: &Store, r: &ObjectRef) -> Result<Header> {
    Header::parse(&read_object(store, r, limits::DOC_MAX_BYTES)?)
}

/// The closure the header's `encodings` list names. An absent spec object is state (3) of
/// cozytensors.md §4 — it cannot install as self-describing at all.
pub fn load_closure(store: &Store, h: &Header) -> Result<Closure> {
    let mut c = Closure::default();
    for r in &h.encodings {
        c.insert(EncodingSpec::parse(&read_object(
            store,
            r,
            limits::SPEC_MAX_BYTES,
        )?)?);
    }
    Ok(c)
}
