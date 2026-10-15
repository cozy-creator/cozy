//! The verified read path (tfs-005): one byte door, caller-owned buffers, and a lease that
//! only the caller ends.
//!
//! `read_into(range, caller_buffer, lease)` is the whole contract. The CALLER owns buffer
//! allocation, pinning, queue depth, asynchronous H2D and completion lifetime; TensorFS owns
//! range validation, verified identity, short-read detection and the lease. TensorFS never
//! allocates a destination and never owns a pinned buffer — exactly one budgeted pinned pool
//! per executor lives with the caller (the v1 FillSink lesson).
//!
//! **`contains` is not a resume predicate, and the API shape says so.** There is no door
//! here that takes a bare digest and hands back bytes. Every read is a range inside an
//! object the LEASE already verified, and a lease is obtained by `acquire`, which verifies
//! against durable records (or rehashes and says it did). A caller cannot reach these bytes
//! by asking whether the store "has" them.
//!
//! Buffered positional reads only. `io_uring`, `O_DIRECT`, mmap refill windows and GDS are
//! benchmark-gated upgrades behind this unchanged API, and none of them appears in any
//! contract — tfs-018's banked baselines are the bar they would have to miss.

use std::collections::{HashMap, HashSet};

#[cfg(feature = "store")]
use crate::err::Refusal;
use crate::err::{refuse, Code, Result};
use crate::header::{Body, Header, Tensor};
use crate::ids::ObjectRef;

// The lease and the byte door need a store; planning does not (a plane-only build plans and
// adopts layouts its machine filled).
#[cfg(feature = "store")]
pub use lease::*;

/// A byte range inside ONE identified object. The identity is part of the range: there is no
/// such thing here as "an offset in a file".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRange {
    pub obj: ObjectRef,
    pub off: u64,
    pub len: u64,
}

impl ObjectRange {
    #[cfg(feature = "store")]
    fn check(&self) -> Result<()> {
        let end = self.off.checked_add(self.len).ok_or(Refusal {
            code: Code::RANGE_BOUNDS,
            detail: format!("{}: offset {} + length overflows", self.obj.id(), self.off),
        })?;
        if end > self.obj.length {
            return refuse(
                Code::RANGE_BOUNDS,
                format!(
                    "{}: range [{}, {end}) leaves the object's declared {} bytes",
                    self.obj.id(),
                    self.off,
                    self.obj.length
                ),
            );
        }
        Ok(())
    }
}

#[cfg(feature = "store")]
mod lease {
use super::*;
use crate::checkpoint;
use crate::header::{Asset, Located, Part};
use crate::manifest::Manifest;
use crate::meta::{Hold, Meta};
use crate::receipt::{Receipt, Receipts};
use crate::store::{Store, VerifiedFile};

// ---------------------------------------------------------------- the lease

/// `ReadLease = store.acquire(manifest, object_set)` — verified PINNED descriptors.
///
/// The hold is registered BEFORE the first byte is verified or fetched, so GC racing a
/// starting fill is fenced by construction rather than by timing. The lease lives until the
/// caller explicitly releases it: the runtime's FillLease holds this value and every source
/// mapping until the completion fence, and dropping it without release keeps the row for the
/// reaper (over-keeping, the only permitted direction) with a loud degraded line — it never
/// silently frees source bytes that may still be in DMA.
///
/// THE FILE VERIFIED IS THE FILE READ (tfs-021). `acquire` opens one `VerifiedFile` per
/// distinct object and every `read_into` goes through that pinned descriptor: no pathname
/// is resolved after verification, so a replacement at the path — same length or not —
/// cannot reach a reader for the lease's whole life, and a 90 GiB fill pays zero
/// per-item open/close syscalls. The descriptor-count contract is explicit: acquire proves
/// `RLIMIT_NOFILE` headroom for the complete set, raising the soft limit toward the hard
/// limit when it must, and refuses `FD_HEADROOM` typed only past the hard limit — chosen
/// over a bounded fd cache because a cache's reopen is a second pathname resolution, which
/// is exactly the defect this type exists to kill.
pub struct ReadLease {
    hold: Hold,
    pub manifest: String,
    objects: Vec<ObjectRef>,
    files: HashMap<String, VerifiedFile>,
}

impl ReadLease {
    pub fn hold_id(&self) -> &str {
        self.hold.id()
    }
    pub fn objects(&self) -> &[ObjectRef] {
        &self.objects
    }
    pub fn bytes(&self) -> u64 {
        self.objects.iter().map(|o| o.length).sum()
    }

    /// Full-ObjectRef coverage. Matching on the digest alone would let a caller read a range
    /// sized from a length the lease never verified.
    pub fn covers(&self, r: &ObjectRange) -> Result<()> {
        if self.objects.contains(&r.obj) {
            return Ok(());
        }
        refuse(
            Code::LEASE_NOT_COVERED,
            format!(
                "{} (length {}) is outside this lease's {} verified objects",
                r.obj.id(),
                r.obj.length,
                self.objects.len()
            ),
        )
    }

    /// Re-check the lease against the authority. Cheap (one document read) and the arm that
    /// makes a revoked lease refuse instead of serving stale bytes.
    pub fn recheck(&self, meta: &Meta) -> Result<()> {
        self.hold.still_registered(meta)
    }

    /// THE explicit end. Nothing else releases the lease.
    pub fn release(self, meta: &Meta) -> Result<()> {
        self.hold.release(meta)
    }
}

/// The complete-descriptor-set headroom question, answered from `/proc` with std alone.
/// `None` means the platform cannot answer (no `/proc`, an "unlimited" soft limit): the
/// pre-check is skipped and a failing open at acquire still refuses before any read.
/// `None` when the process can hold `need` more descriptors, raising its soft limit up to
/// the hard limit if that is what it takes. `Some((limit, open, want))` only past the hard
/// limit.
fn fd_shortfall(need: usize) -> Option<(u64, u64, u64)> {
    use rustix::process::{getrlimit, setrlimit, Resource, Rlimit};
    /// Fds the pre-check leaves free for stdio, locks and whatever the caller opens next.
    const MARGIN: u64 = 64;
    let open = std::fs::read_dir("/proc/self/fd").ok()?.count() as u64;
    let want = need as u64 + open + MARGIN;
    let limit = getrlimit(Resource::Nofile);
    let soft = limit.current?;
    if want <= soft {
        return None;
    }
    let raised = limit.maximum.unwrap_or(want);
    if raised >= want
        && setrlimit(
            Resource::Nofile,
            Rlimit {
                current: Some(raised),
                maximum: limit.maximum,
            },
        )
        .is_ok()
    {
        return None;
    }
    Some((limit.maximum.unwrap_or(soft).max(soft), open, want))
}

/// Acquire a lease over an explicit object set, verifying every object first and PINNING
/// its descriptor. Verification and reading share one `open` (`open_verified`); nothing
/// after this function resolves an object pathname.
pub fn acquire(
    store: &Store,
    meta: &Meta,
    manifest: &str,
    objects: Vec<ObjectRef>,
) -> Result<(ReadLease, Receipts)> {
    // The hold FIRST: registered before the first fetch (tensorfs.md §2, pin root class 2).
    // The owner's GC waits for an admission in flight and keeps what it pins.
    let owner = crate::meta::owner(store.root());
    let _admitting = owner.as_ref().map(|owner| owner.gate.read().unwrap());
    let hold = meta.read_hold(owner.as_ref(), &objects)?;
    // Admission proves RLIMIT_NOFILE headroom for the COMPLETE descriptor set (H3 serves
    // 3,627 distinct objects — never an assumable count) or refuses typed.
    let distinct: HashSet<&str> = objects.iter().map(|o| o.sha256.as_str()).collect();
    if let Some((soft, open, want)) = fd_shortfall(distinct.len()) {
        let _ = hold.release(meta);
        return refuse(
            Code::FD_HEADROOM,
            format!(
                "this lease pins {} verified descriptors and the process already holds \
                 {open} fds under an RLIMIT_NOFILE hard limit of {soft} ({want} needed \
                 with margin) — raise the hard limit; the reader will not fall back to \
                 re-resolving pathnames",
                distinct.len()
            ),
        );
    }
    let mut rx = Receipts::default();
    let mut files: HashMap<String, VerifiedFile> = HashMap::with_capacity(distinct.len());
    for o in &objects {
        if let Some(vf) = files.get(&o.sha256) {
            if vf.len() != o.length {
                let _ = hold.release(meta);
                return refuse(
                    Code::LENGTH_MISMATCH,
                    format!(
                        "{}: the lease already pinned these bytes at {} B, the reference \
                         claims {}",
                        o.id(),
                        vf.len(),
                        o.length
                    ),
                );
            }
            continue;
        }
        // A missing or disagreeing record makes open_verified pay exactly one rehash —
        // reported, because a read path that silently re-hashes is hiding a fault.
        let (vf, rehashed) = match store.open_verified_reporting(&o.sha256) {
            Ok(x) => x,
            Err(e) => {
                let _ = hold.release(meta);
                return Err(e);
            }
        };
        if vf.len() != o.length {
            let _ = hold.release(meta);
            return refuse(
                Code::LENGTH_MISMATCH,
                format!(
                    "{}: the pinned verified descriptor carries {} bytes, the reference \
                     claims {}",
                    o.id(),
                    vf.len(),
                    o.length
                ),
            );
        }
        if rehashed {
            rx.emit(Receipt::new(
                "acquire",
                "record-invalidated-rehash",
                manifest,
                format!("{}: rehashed on acquire", o.id()),
            ));
        }
        files.insert(o.sha256.clone(), vf);
    }
    Ok((
        ReadLease {
            hold,
            manifest: manifest.to_string(),
            objects,
            files,
        },
        rx,
    ))
}

/// Acquire over everything one Manifest commits, including ordinary snapshot siblings. Use this
/// for full-snapshot operations; model execution uses `acquire_cozytensors` below.
pub fn acquire_manifest(
    store: &Store,
    meta: &Meta,
    snap: &Manifest,
) -> Result<(ReadLease, Receipts)> {
    crate::stats::presence_pass();
    let walk = checkpoint::walk(store, snap)?;
    let objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
    let hex = snap.manifest_id();
    acquire(
        store,
        meta,
        hex.strip_prefix("sha256:").unwrap_or(&hex),
        objects,
    )
}

/// Acquire only the self-contained CozyTensors runtime closure selected by a Manifest.
/// Ordinary snapshot siblings remain retained by the Manifest but do not consume worker file
/// descriptors and need not be resident for model execution.
pub fn acquire_cozytensors(
    store: &Store,
    meta: &Meta,
    snap: &Manifest,
) -> Result<(ReadLease, Receipts)> {
    crate::stats::presence_pass();
    let walk = checkpoint::walk_cozytensors(store, snap)?;
    let objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
    let hex = snap.manifest_id();
    acquire(
        store,
        meta,
        hex.strip_prefix("sha256:").unwrap_or(&hex),
        objects,
    )
}

// ---------------------------------------------------------------- the one byte door

/// Fill a CALLER-OWNED buffer from one verified range, THROUGH THE LEASE'S PINNED
/// DESCRIPTOR. Exact fit or refusal: a buffer that is not exactly the range's length is a
/// caller bug, and a short read is a refusal that never claims the bytes it did move.
///
/// No pathname is resolved here — the fd pinned at acquire is the only door, so a
/// replacement installed at the object's path after acquire never reaches this read.
pub fn read_into(lease: &ReadLease, r: &ObjectRange, buf: &mut [u8]) -> Result<()> {
    lease.covers(r)?;
    r.check()?;
    if buf.len() as u64 != r.len {
        return refuse(
            Code::BUFFER_SIZE,
            format!(
                "caller buffer is {} B for a {} B range — TensorFS never resizes a caller's \
                 buffer and never returns a partial fill",
                buf.len(),
                r.len
            ),
        );
    }
    let vf = match lease.files.get(&r.obj.sha256) {
        Some(vf) => vf,
        // Unreachable past `covers`, but stated: a lease without the descriptor must
        // refuse rather than fall back to a pathname.
        None => {
            return refuse(
                Code::LEASE_NOT_COVERED,
                format!("{}: no pinned descriptor on this lease", r.obj.id()),
            )
        }
    };
    // `fstat` on the PINNED fd against the declared length, before any byte moves: an
    // in-place truncation of this inode is caught by arithmetic rather than by a read
    // that half-succeeds. (A swap at the pathname is not caught — it is INEFFECTIVE.)
    let size = vf.current_len().map_err(|e| crate::store::classify_io(format!("fstat {}", r.obj.id()), e))?;
    if size != r.obj.length {
        return refuse(
            Code::SHORT_READ,
            format!(
                "{} is {size} B on disk, the reference declares {} B — refusing the range \
                 rather than exposing a partial object",
                r.obj.id(),
                r.obj.length
            ),
        );
    }
    match vf.read_exact_at(buf, r.off) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => refuse(
            Code::SHORT_READ,
            format!(
                "{}: range [{}, {}) ended early — no partial success is exposed",
                r.obj.id(),
                r.off,
                r.off + r.len
            ),
        ),
        Err(e) => Err(crate::store::classify_io(format!("pread {}", r.obj.id()), e)),
    }
}

/// Reassemble one header-declared asset through this lease's already verified descriptors.
/// `max_bytes` is the caller's allocation bound; assets above it refuse before allocation.
pub fn read_asset(lease: &ReadLease, name: &str, asset: &Asset, max_bytes: u64) -> Result<Vec<u8>> {
    if asset.logical_length > max_bytes {
        return refuse(
            Code::SIZE_CAP,
            format!(
                "asset {name:?} is {} B, above the caller's {max_bytes} B bound",
                asset.logical_length
            ),
        );
    }
    let length = usize::try_from(asset.logical_length).map_err(|_| Refusal {
        code: Code::SIZE_CAP,
        detail: format!("asset {name:?} length does not fit this platform"),
    })?;
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(length).map_err(|error| Refusal {
        code: Code::SIZE_CAP,
        detail: format!("asset {name:?}: cannot allocate {length} B: {error}"),
    })?;
    bytes.resize(length, 0);

    let mut offset = 0usize;
    for segment in &asset.segments {
        let segment_length = usize::try_from(segment.length).map_err(|_| Refusal {
            code: Code::SIZE_CAP,
            detail: format!("asset {name:?}: segment length does not fit this platform"),
        })?;
        let end = offset.checked_add(segment_length).ok_or_else(|| Refusal {
            code: Code::ARITH_OVERFLOW,
            detail: format!("asset {name:?}: segment offsets overflow"),
        })?;
        read_into(
            lease,
            &ObjectRange {
                obj: segment.clone(),
                off: 0,
                len: segment.length,
            },
            &mut bytes[offset..end],
        )?;
        offset = end;
    }
    let digest = crate::sha256::hex_digest(&bytes);
    if offset as u64 != asset.logical_length || digest != asset.logical_sha256 {
        return refuse(
            Code::WHOLE_DIGEST_MISMATCH,
            format!(
                "asset {name:?}: reassembled {offset} B sha256:{digest}, header claims {} B sha256:{}",
                asset.logical_length, asset.logical_sha256
            ),
        );
    }
    Ok(bytes)
}

/// A sequential helper over one storage-part run: the part's declared geometry locates the
/// segments, the lease covers them, and the caller's buffer receives the exact span.
pub fn read_part_into(
    lease: &ReadLease,
    what: &str,
    part: &Part,
    off: u64,
    len: u64,
    buf: &mut [u8],
) -> Result<()> {
    if buf.len() as u64 != len {
        return refuse(
            Code::BUFFER_SIZE,
            format!("caller buffer is {} B for a {len} B span", buf.len()),
        );
    }
    match part.locate(what, off, len)? {
        Located::Inline { off, len } => {
            let b = match &part.body {
                Body::Inline(s) => s.clone(),
                Body::Segments(_) => unreachable!("locate returned Inline for a segmented part"),
            };
            buf.copy_from_slice(&b[off as usize..(off + len) as usize]);
            Ok(())
        }
        Located::Segments(spans) => {
            let mut cur = 0usize;
            for s in spans {
                let n = s.len as usize;
                read_into(
                    lease,
                    &ObjectRange {
                        obj: s.obj.clone(),
                        off: s.off,
                        len: s.len,
                    },
                    &mut buf[cur..cur + n],
                )?;
                cur += n;
            }
            Ok(())
        }
    }
}

}

// ---------------------------------------------------------------- the read plan

/// Where one plan item's bytes come from. An inline part is a first-class item, not a
/// skipped one: a fill that silently omits declared bytes is a wrong model, not a fast one.
/// Inline bytes cost no I/O — they ride the header the caller already holds — but they still
/// occupy their exact destination span, in traversal order.
#[derive(Debug, Clone)]
pub enum Source {
    Object(ObjectRange),
    Inline(Vec<u8>),
}

#[derive(Debug, Clone)]
pub struct PlanItem {
    pub what: String,
    pub dest_off: u64,
    pub len: u64,
    pub source: Source,
}

impl PlanItem {
    pub fn is_io(&self) -> bool {
        matches!(self.source, Source::Object(_))
    }
}

/// The header's construction order, after exact equality with Runtime's traversal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Destination,
}

impl Order {
    pub fn as_str(self) -> &'static str {
        "construction"
    }
}

#[derive(Debug, Clone)]
pub struct ReadPlan {
    pub items: Vec<PlanItem>,
    /// Every declared byte the traversal covers.
    pub bytes: u64,
    /// The DISK LEG — `bytes` minus what the header carries inline. Throughput is reported
    /// against this one, because inline bytes never touch a device.
    pub io_bytes: u64,
    pub inline_parts: usize,
    pub order: Order,
}

impl ReadPlan {
    /// Select whole physical parts from this already planned scope. Source ranges and
    /// their order/windows are unchanged; destination offsets become contiguous again.
    pub fn select(&self, whats: &[String]) -> Result<ReadPlan> {
        let mut requested = HashSet::new();
        for what in whats {
            if !requested.insert(what.as_str()) {
                return refuse(
                    Code::DUPLICATE_KEY,
                    format!("read selection repeats {what:?}"),
                );
            }
        }
        let mut missing = requested.clone();
        let mut items = Vec::new();
        let mut dest = 0;
        for item in &self.items {
            if requested.contains(item.what.as_str()) {
                missing.remove(item.what.as_str());
                let mut selected = item.clone();
                selected.dest_off = dest;
                dest += selected.len;
                items.push(selected);
            }
        }
        if let Some(what) = missing.into_iter().min() {
            return refuse(
                Code::MISSING_FIELD,
                format!("read plan has no part {what:?}"),
            );
        }
        let (bytes, io_bytes, inline_parts) = tally(&items);
        Ok(ReadPlan {
            items,
            bytes,
            io_bytes,
            inline_parts,
            order: self.order,
        })
    }
}

fn build(ordered: &[(&str, &str, &Tensor)], window: u64) -> Result<ReadPlan> {
    let mut items = Vec::new();
    let mut dest: u64 = 0;
    for (comp, key, tensor) in ordered {
        for (role, part) in &tensor.parts {
            let what = format!("{comp}/{key}#{role}");
            if let Body::Inline(b) = &part.body {
                let n = part.nbytes(&what)?;
                items.push(PlanItem {
                    what: what.clone(),
                    dest_off: dest,
                    len: n,
                    source: Source::Inline(b.clone()),
                });
                dest += n;
                continue;
            }
            for segment in part.segments() {
                let mut off = 0u64;
                while off < segment.length {
                    let n = if window == 0 {
                        segment.length - off
                    } else {
                        window.min(segment.length - off)
                    };
                    items.push(PlanItem {
                        what: what.clone(),
                        dest_off: dest,
                        len: n,
                        source: Source::Object(ObjectRange {
                            obj: segment.clone(),
                            off,
                            len: n,
                        }),
                    });
                    dest += n;
                    off += n;
                }
            }
        }
    }
    let (bytes, io_bytes, inline_parts) = tally(&items);
    debug_assert_eq!(bytes, dest);
    Ok(ReadPlan {
        items,
        bytes,
        io_bytes,
        inline_parts,
        order: Order::Destination,
    })
}

fn tally(items: &[PlanItem]) -> (u64, u64, usize) {
    let bytes = items.iter().map(|i| i.len).sum();
    let io_bytes = items.iter().filter(|i| i.is_io()).map(|i| i.len).sum();
    let inline = items.iter().filter(|i| !i.is_io()).count();
    (bytes, io_bytes, inline)
}

/// The read planner. The destination follows the CALLER's traversal, which is the order it
/// constructs in; the header's tensor order is a storage choice of whoever wrote it. The two
/// are written by independently versioned peers, so a permutation is planned, never refused:
/// only a traversal that names a different set of tensors is.
///
/// COMPLETENESS SCOPES TO THE REQUESTED COMPONENT SET (#570b). `requested` is the caller's
/// explicit declaration of which components this plan is for; the traversal must name every
/// tensor the header carries FOR THOSE COMPONENTS, and may name nothing outside them.
///
/// The check used to be whole-header, and tfs-007's reason for it is unchanged and still
/// enforced: a fill that silently skips a tensor is a wrong model, not a fast one. What was
/// wrong was the SCOPE. An N-ary artifact packages several components under one manifest —
/// the canonical shape for every multi-task family — and a role class constructs only its
/// own graph, so a whole-header rule made the standard packaging unservable: H3's Fl2VA
/// class names 917 of 3445 tensors and was refused for the 2528 belonging to a sibling
/// transformer it must not touch. It also contradicted the per-component residency the rest
/// of the design assumes everywhere, which stages components independently by construction.
///
/// So the hole this closes is stated per component rather than per manifest. A traversal
/// missing one tensor of a component it REQUESTED still refuses, exactly as before. A
/// traversal naming a tensor from a component it did not request refuses too — declaring a
/// narrow scope and then reading outside it is the smuggling case, and it is the reason the
/// scope is DECLARED rather than inferred from the keys the caller happened to name.
pub fn plan_for_traversal(
    header: &Header,
    traversal: &[(String, String)],
    requested: &[String],
    window: u64,
) -> Result<ReadPlan> {
    let mut scope: Vec<&str> = requested.iter().map(|c| c.as_str()).collect();
    if scope.is_empty() {
        return refuse(
            Code::TRAVERSAL_INCOMPLETE,
            "the plan declares no requested component: completeness is scoped to a declared \
             set, and an empty one names nothing to be complete about"
                .to_string(),
        );
    }
    let before = scope.len();
    scope.sort();
    scope.dedup();
    if scope.len() != before {
        return refuse(
            Code::TRAVERSAL_INCOMPLETE,
            "requested components repeat; scope is an exact set",
        );
    }
    for comp in &scope {
        if !header.components.iter().any(|(c, _)| c == comp) {
            let carried: Vec<&str> = header.components.iter().map(|(c, _)| c.as_str()).collect();
            return refuse(
                Code::TRAVERSAL_INCOMPLETE,
                format!(
                    "the plan requests component {comp:?}, which this header does not carry \
                     (it carries: {carried:?})"
                ),
            );
        }
    }

    let mut carried: HashMap<(&str, &str), &Tensor> = header
        .tensors()
        .filter(|(c, _, _)| scope.contains(&c.as_str()))
        .map(|(c, k, t)| ((c.as_str(), k.as_str()), t))
        .collect();
    let mut ordered = Vec::with_capacity(traversal.len());
    for (c, k) in traversal {
        let Some(tensor) = carried.remove(&(c.as_str(), k.as_str())) else {
            return refuse(
                Code::TRAVERSAL_INCOMPLETE,
                format!(
                    "the traversal names {c}/{k}, which requested components {scope:?} do \
                     not carry, or names it twice"
                ),
            );
        };
        ordered.push((c.as_str(), k.as_str(), tensor));
    }
    if !carried.is_empty() {
        let mut missing: Vec<String> = carried.keys().map(|(c, k)| format!("{c}/{k}")).collect();
        missing.sort();
        missing.truncate(4);
        return refuse(
            Code::TRAVERSAL_INCOMPLETE,
            format!(
                "the traversal omits {} tensor(s) requested components {scope:?} carry: \
                 {missing:?}",
                carried.len()
            ),
        );
    }
    build(&ordered, window)
}
