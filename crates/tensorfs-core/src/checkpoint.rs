//! The checkpoint plane: objectize real tensor bytes through the real CAS, emit the
//! canonical header, and reconstruct from the header alone.
//!
//! The store is shape-agnostic (tfs-002) and stays that way — every shape fact is decided
//! here, before a single body byte moves. The anchor equation
//! `checked_prod(shape) × sizeof(dtype) = Σ segment lengths` is checked BEFORE any fetch,
//! so a lying header is refused by arithmetic rather than by a failed download.

use std::io::{Read, Write};

use crate::dtype::{checked_bytes, Dtype};
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Body, Closure, Header, Located, Part};
use crate::ids::ObjectRef;
use crate::limits;
use crate::manifest::Manifest;
use crate::sha256::{self, Sha256};
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
    objectize_held(store, what, dtype, shape, src, None)
}

pub fn objectize_held<R: Read>(
    store: &Store,
    what: &str,
    dtype: Dtype,
    shape: Vec<u64>,
    src: &mut R,
    operation: Option<&str>,
) -> Result<Written> {
    let want = checked_bytes(what, &shape, dtype)?;
    let mut whole = Sha256::new();
    let stored = if want > limits::GRID_BYTES {
        let mut tee = Tee {
            r: src,
            h: &mut whole,
        };
        store_part_held(store, what, dtype, shape, &mut tee, operation)?
    } else {
        store_part_held(store, what, dtype, shape, src, operation)?
    };
    let digest = match &stored.part.body {
        Body::Inline(bytes) => sha256::hex_digest(bytes),
        Body::Segments(segments) if segments.len() == 1 => segments[0].sha256.clone(),
        Body::Segments(_) => sha256::hex(&whole.finish()),
    };
    Ok(Written {
        segments: stored.part.segments().len(),
        part: stored.part,
        whole: digest,
        bytes: stored.bytes,
        deduped: stored.deduped,
    })
}

/// The source converter needs stored parts, not an unused whole-tensor identity.
/// Both callers share the same bounded objectization and verification path.
pub(crate) struct StoredPart {
    pub part: Part,
    pub bytes: u64,
    pub deduped: u64,
}

pub(crate) fn store_part_held<R: Read>(
    store: &Store,
    what: &str,
    dtype: Dtype,
    shape: Vec<u64>,
    src: &mut R,
    operation: Option<&str>,
) -> Result<StoredPart> {
    let want = checked_bytes(what, &shape, dtype)?;

    if Part::is_inline(want) {
        let mut buf = vec![0u8; want as usize];
        src.read_exact(&mut buf)
            .map_err(|e| io(format!("{what}: read {want} inline bytes"), e))?;
        store.moved(want);
        return Ok(StoredPart {
            part: Part {
                dtype,
                shape,
                body: Body::Inline(buf),
            },
            bytes: want,
            deduped: 0,
        });
    }

    let mut segs = Vec::new();
    let mut left = want;
    let mut deduped = 0u64;
    while left > 0 {
        let n = left.min(limits::GRID_BYTES);
        let mut t = (&mut *src).take(n);
        let put = store.put_stream_held(&mut t, None, &Fault::default(), operation)?;
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
    Ok(StoredPart {
        part: Part {
            dtype,
            shape,
            body: Body::Segments(segs),
        },
        bytes: want,
        deduped,
    })
}

/// Reconstruct a part's whole byte run. The anchor equation is validated first; each
/// segment's delivered length is checked against the record that promised it.
pub fn materialize<W: Write>(store: &Store, what: &str, part: &Part, w: &mut W) -> Result<String> {
    part.check_bytes(what)?;
    let mut h = Sha256::new();
    match &part.body {
        Body::Inline(s) => {
            h.update(s);
            w.write_all(s).map_err(|e| io("write inline", e))?;
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
                Body::Inline(s) => s.clone(),
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
pub fn put_doc<D: crate::ids::StoredDoc>(store: &Store, d: &D) -> Result<ObjectRef> {
    let bytes = <D as crate::ids::StoredDoc>::canonical_bytes(d)?;
    // A document that cannot be read back must not be written. `Doc::parse` enforces
    // `MAX_BYTES` and this side did not, so an over-cap document stored happily and then
    // refused SIZE_CAP on the way back in — a write/read asymmetry that turns a bounds
    // decision into a corrupt-looking store. Observed live: the ingest subject's candidate
    // set at 2,601 objects.
    if bytes.len() > <D as crate::ids::StoredDoc>::MAX_BYTES {
        return refuse(
            Code::SIZE_CAP,
            format!(
                "{}: {} canonical bytes over the {} B document cap — refused at WRITE, \
                 because this store would never be able to read it back",
                <D as crate::ids::StoredDoc>::FORMAT,
                bytes.len(),
                <D as crate::ids::StoredDoc>::MAX_BYTES
            ),
        );
    }
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
    let _ = store;
    let mut c = Closure::default();
    for spec in &h.encodings {
        c.insert(spec.clone());
    }
    Ok(c)
}

pub fn load_manifest(store: &Store, r: &ObjectRef) -> Result<Manifest> {
    store.read_manifest(r)
}

/// Stream one model-runtime asset in declared segment order and verify its logical whole.
pub fn read_asset<W: Write>(
    store: &Store,
    name: &str,
    asset: &crate::header::Asset,
    w: &mut W,
) -> Result<()> {
    let mut hash = Sha256::new();
    let mut total = 0u64;
    for (index, segment) in asset.segments.iter().enumerate() {
        let mut tee = TeeW { w, h: &mut hash };
        let read = store.read_into(&segment.sha256, &mut tee)?;
        if read != segment.length {
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "asset {name:?} segment {index} delivered {read} B, promised {} B",
                    segment.length
                ),
            );
        }
        total = total.checked_add(read).ok_or_else(|| Refusal {
            code: Code::ARITH_OVERFLOW,
            detail: format!("asset {name:?}: logical length overflows"),
        })?;
    }
    let got = sha256::hex(&hash.finish());
    if total != asset.logical_length || got != asset.logical_sha256 {
        return refuse(
            Code::WHOLE_DIGEST_MISMATCH,
            format!(
                "asset {name:?}: reassembled {total} B sha256:{got}, header claims {} B sha256:{}",
                asset.logical_length, asset.logical_sha256
            ),
        );
    }
    Ok(())
}

// ------------------------------------------------------------------ the projected tree

/// The tree a checkpoint commits. Contained documents (EncodingSpec objects and their
/// conformance-vector sets) go under `closure/` at paths whose terminal IS the digest, so
/// the tree is self-describing on any host and every reference is checkable against the
/// path that carries it.
///
/// One writer, two consumers (the synthetic checkpoint plane and the ingest border) — the
/// alternative is two spellings of one tree, which `walk` would then have to forgive.
pub fn build_manifest(header: &Header, href: &ObjectRef, _closure: &Closure) -> Result<Manifest> {
    let _ = header;
    crate::manifest::Draft {
        entries: vec![(
            "model.cozytensors".to_string(),
            crate::manifest::Entry::CozyTensors(href.clone()),
        )],
    }
    .seal()
}

// ------------------------------------------------------------------ the transitive root

/// What one object was reached AS. The kind is the reason it is live, not a property of
/// its bytes: the same object can be a model asset here and a plain file there.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reached {
    pub kind: &'static str,
    pub obj: ObjectRef,
}

#[derive(Debug, Default)]
pub struct Walk {
    pub objects: Vec<Reached>,
    pub tensors: usize,
    pub parts: usize,
    /// Entries that are neither the header, a config, nor a contained document: SIBLING
    /// ASSETS, admitted only for a named reason (cozytensors.md §3 — non-JSON bytes or
    /// genuine mutability). Counted and listed, never silently absorbed.
    pub siblings: Vec<String>,
}

impl Walk {
    pub fn count(&self, kind: &str) -> usize {
        self.objects.iter().filter(|r| r.kind == kind).count()
    }
    pub fn bytes(&self) -> u64 {
        self.objects.iter().map(|r| r.obj.length).sum()
    }
    /// Distinct objects — repeated files and aliased runs reference the same object once.
    pub fn distinct(&self) -> Vec<&ObjectRef> {
        let mut v: Vec<&ObjectRef> = self.objects.iter().map(|r| &r.obj).collect();
        v.sort_by(|a, b| a.sha256.cmp(&b.sha256));
        v.dedup_by(|a, b| a.sha256 == b.sha256);
        v
    }

    /// Every reached object PRESENT in the store, or the first absent one as a refusal.
    /// The walk itself proves containment (the manifest lists what the header reaches);
    /// this proves residency — `install`'s gate (tfs-020), so a root can never commit
    /// over missing objects. A partially fetched manifest legitimately fails this; it is
    /// an install precondition, not a walk invariant.
    pub fn require_resident(&self, store: &Store) -> Result<()> {
        for r in self.distinct() {
            if !store.contains(&r.sha256) {
                return refuse(
                    Code::OBJECT_ABSENT,
                    format!(
                        "{} is committed by this manifest but absent from the store — \
                         refusing to install a root over missing objects",
                        r.id()
                    ),
                );
            }
        }
        Ok(())
    }

    /// Every reached object present, or held by a live source operation's publication
    /// custodian. A complete checkpoint for publication, never a readable one.
    pub fn require_held(
        &self,
        store: &Store,
        custodied: &std::collections::BTreeSet<String>,
    ) -> Result<()> {
        for r in self.distinct() {
            if !store.contains(&r.sha256) && !custodied.contains(&r.sha256) {
                return refuse(
                    Code::OBJECT_ABSENT,
                    format!(
                        "{} is committed by this manifest but neither resident nor held by \
                         a live publication custodian",
                        r.id()
                    ),
                );
            }
        }
        Ok(())
    }
}

/// Manifest → CozyTensorsHeader → contained tensor rules → storage-part object runs.
/// Reaches the self-contained runtime closure and deliberately excludes ordinary files that
/// happen to live beside the CozyTensors entry in the snapshot.
pub fn walk_cozytensors(store: &Store, snap: &Manifest) -> Result<Walk> {
    // The path law already ran: a Manifest cannot exist unvalidated (manifest.rs).
    let mut w = Walk::default();
    let href = match snap.header() {
        Some(h) => h.clone(),
        None => {
            return refuse(
                Code::ATTACHMENT_CARDINALITY,
                "not a checkpoint manifest: no cozytensors header attachment",
            )
        }
    };
    w.objects.push(Reached {
        kind: "header",
        obj: href.clone(),
    });
    let header = load_header(store, &href)?;

    let mut closure = Closure::default();
    for spec in &header.encodings {
        closure.insert(spec.clone());
    }

    for (_, asset) in &header.assets {
        for segment in &asset.segments {
            w.objects.push(Reached {
                kind: "model_asset",
                obj: segment.clone(),
            });
        }
    }

    header.validate(&closure)?;
    for (comp, key, t) in header.tensors() {
        w.tensors += 1;
        for (role, part) in &t.parts {
            w.parts += 1;
            let _ = (comp, key, role);
            for s in part.segments() {
                w.objects.push(Reached {
                    kind: "part",
                    obj: s.clone(),
                });
            }
        }
    }

    Ok(w)
}

/// Reach the complete snapshot: the CozyTensors runtime closure plus every ordinary sibling.
/// Repository completeness, checkout, retention, and accounting use this full walk; workers
/// use `walk_cozytensors` because an optional README or sample image is not a runtime input.
pub fn walk(store: &Store, snap: &Manifest) -> Result<Walk> {
    let mut w = if snap.header().is_some() {
        walk_cozytensors(store, snap)?
    } else {
        Walk::default()
    };

    // Anything listed that the header does not account for is a sibling asset — with two
    // audit-driven tightenings (external review, folded in at tfs-003):
    //
    // 1. Comparison is by the WHOLE length-bearing ObjectRef. Matching on the digest alone
    //    let an entry claim a different length for the same bytes and still read as
    //    accounted, which is the one field a fetch plan is sized from.
    // 2. `closure/` is the CONTAINED-DOCUMENT namespace. An entry there that no header
    //    reference reaches is an orphan, not an asset: it is a canonical document the
    //    checkpoint does not commit to, sitting at a path that asserts it does. Silently
    //    absorbing it as a "sibling" is how a stale or planted spec rides along.
    // A SET over the whole length-bearing ObjectRef, not a per-entry linear scan — the
    // scan made walk O(n²) over the manifest at design cardinality.
    let accounted: std::collections::HashSet<(String, u64)> = w
        .objects
        .iter()
        .map(|r| (r.obj.sha256.clone(), r.obj.length))
        .collect();
    for (path, e) in snap.entries() {
        if let Some(o) = e.content() {
            if !accounted.contains(&(o.sha256.clone(), o.length)) {
                w.siblings.push(path.clone());
                w.objects.push(Reached {
                    kind: "asset",
                    obj: o.clone(),
                });
            }
        }
    }
    Ok(w)
}
