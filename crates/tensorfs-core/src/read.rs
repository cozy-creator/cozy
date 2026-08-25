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

use std::os::unix::fs::FileExt;
use std::sync::Mutex;
use std::time::Instant;

use crate::checkpoint;
use crate::err::{refuse, Code, Refusal, Result};
use crate::header::{Body, Header, Located, Part};
use crate::ids::ObjectRef;
use crate::manifest::SnapshotManifest;
use crate::meta::{Hold, Meta};
use crate::receipt::{Receipt, Receipts};
use crate::store::{Store, Verdict};

/// A byte range inside ONE identified object. The identity is part of the range: there is no
/// such thing here as "an offset in a file".
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectRange {
    pub obj: ObjectRef,
    pub off: u64,
    pub len: u64,
}

impl ObjectRange {
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

// ---------------------------------------------------------------- the lease

/// `ReadLease = store.acquire(snapshot, object_set)` — verified held ranges.
///
/// The hold is registered BEFORE the first byte is verified or fetched, so GC racing a
/// starting fill is fenced by construction rather than by timing. The lease lives until the
/// caller explicitly releases it: the runtime's FillLease holds this value and every source
/// mapping until the completion fence, and dropping it without release keeps the row for the
/// reaper (over-keeping, the only permitted direction) with a loud degraded line — it never
/// silently frees source bytes that may still be in DMA.
pub struct ReadLease {
    hold: Hold,
    pub snapshot: String,
    objects: Vec<ObjectRef>,
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

/// Acquire a lease over an explicit object set, verifying every object first.
pub fn acquire(
    store: &Store,
    meta: &Meta,
    snapshot: &str,
    objects: Vec<ObjectRef>,
) -> Result<(ReadLease, Receipts)> {
    // The hold FIRST: registered before the first fetch (tensorfs.md §2, pin root class 2).
    let hold = meta.acquire_hold("read", snapshot, Vec::new())?;
    let mut rx = Receipts::default();
    for o in &objects {
        match store.record_valid(&o.sha256) {
            Ok(rec) => {
                if rec.length != o.length {
                    let _ = hold.release(meta);
                    return refuse(
                        Code::LENGTH_MISMATCH,
                        format!(
                            "{}: the verification record carries {} bytes, the reference \
                             claims {}",
                            o.id(),
                            rec.length,
                            o.length
                        ),
                    );
                }
            }
            Err(why) => match store.verify(&o.sha256, false) {
                Ok(Verdict::Quarantined { to, why: w }) => {
                    let _ = hold.release(meta);
                    return refuse(
                        Code::OBJECT_CORRUPT,
                        format!("{} quarantined to {}: {w}", o.id(), to.display()),
                    );
                }
                Ok(_) => rx.emit(Receipt::new(
                    "acquire",
                    "record-invalidated-rehash",
                    snapshot,
                    store.generation(),
                    format!("{}: {why} — rehashed on acquire", o.id()),
                )),
                Err(e) => {
                    let _ = hold.release(meta);
                    return Err(e);
                }
            },
        }
    }
    Ok((
        ReadLease {
            hold,
            snapshot: snapshot.to_string(),
            objects,
        },
        rx,
    ))
}

/// Acquire over everything one snapshot commits — the serving case. The transitive walk is
/// the object set, so the lease covers exactly what the checkpoint declares and nothing else.
pub fn acquire_snapshot(
    store: &Store,
    meta: &Meta,
    snap: &SnapshotManifest,
) -> Result<(ReadLease, Receipts)> {
    let walk = checkpoint::walk(store, snap)?;
    let objects: Vec<ObjectRef> = walk.distinct().into_iter().cloned().collect();
    let hex = snap.snapshot_id();
    acquire(
        store,
        meta,
        hex.strip_prefix("sha256:").unwrap_or(&hex),
        objects,
    )
}

// ---------------------------------------------------------------- the one byte door

/// Fill a CALLER-OWNED buffer from one verified range. Exact fit or refusal: a buffer that
/// is not exactly the range's length is a caller bug, and a short read is a refusal that
/// never claims the bytes it did move.
pub fn read_into(store: &Store, lease: &ReadLease, r: &ObjectRange, buf: &mut [u8]) -> Result<()> {
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
    let f = store.open_nofollow(&r.obj.sha256)?;
    // File identity against the DECLARED length, before any byte moves: a truncated object
    // is caught by arithmetic rather than by a read that half-succeeds.
    let size = f
        .metadata()
        .map_err(|e| Refusal {
            code: Code::IO_FAILED,
            detail: format!("stat {}: {e}", r.obj.id()),
        })?
        .len();
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
    match f.read_exact_at(buf, r.off) {
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
        Err(e) => Err(Refusal {
            code: Code::IO_FAILED,
            detail: format!("pread {}: {e}", r.obj.id()),
        }),
    }
}

/// A sequential helper over one storage-part run: the part's declared geometry locates the
/// segments, the lease covers them, and the caller's buffer receives the exact span.
pub fn read_part_into(
    store: &Store,
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
                Body::Inline(s) => crate::b64::decode(s)?,
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
                    store,
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

// ---------------------------------------------------------------- the read plan

/// Where one plan item's bytes come from. An inline part is a first-class item, not a
/// skipped one: a fill that silently omits declared bytes is a wrong model, not a fast one.
/// Inline bytes cost no I/O — they ride the header the caller already holds — but they still
/// occupy their exact destination span, in traversal order.
#[derive(Debug, Clone)]
pub enum Source {
    Object(ObjectRange),
    Inline(String),
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

/// Where the ORDER came from. It is a field because it is a contract: fill iterates in
/// DESTINATION order, and reading in the header's key order is the v1 ~10x cold-load defect
/// (disk-order = memory-order took a ~50 GB load from hundreds of seconds to ~10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// The caller's code-topology / module traversal. The only order fill may use.
    Destination,
    /// The header's key order. BENCHMARK-ONLY — it exists so the lesson can be measured
    /// rather than quoted, and `execute` names it in its output every time.
    HeaderKey,
}

impl Order {
    pub fn as_str(self) -> &'static str {
        match self {
            Order::Destination => "destination (code-topology traversal)",
            Order::HeaderKey => "header key order (BENCHMARK-ONLY)",
        }
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
    /// Cut the plan into sub-plans that each fit a BOUNDED destination, destination offsets
    /// rebased from zero. This is what makes the caller's buffer a ring rather than a
    /// materialization: peak resident bytes become a function of the ring, not of the
    /// artifact (tfs-018 measured 0.61–0.86 GiB bounded-ring RSS against 5.4–5.5 GiB for the
    /// stock loaders on the same 4.78 GiB working set).
    pub fn batches(&self, ring: u64) -> Vec<ReadPlan> {
        let mut out: Vec<ReadPlan> = Vec::new();
        let (mut items, mut bytes, mut base) = (Vec::new(), 0u64, 0u64);
        for it in &self.items {
            if !items.is_empty() && bytes + it.len > ring {
                let (b, io, il) = tally(&items);
                out.push(ReadPlan {
                    items: std::mem::take(&mut items),
                    bytes: b,
                    io_bytes: io,
                    inline_parts: il,
                    order: self.order,
                });
                base += bytes;
                bytes = 0;
            }
            items.push(PlanItem {
                what: it.what.clone(),
                dest_off: it.dest_off - base,
                len: it.len,
                source: it.source.clone(),
            });
            bytes += it.len;
        }
        if !items.is_empty() {
            let (b, io, il) = tally(&items);
            out.push(ReadPlan {
                items,
                bytes: b,
                io_bytes: io,
                inline_parts: il,
                order: self.order,
            });
        }
        out
    }
}

fn build(header: &Header, keys: &[(String, String)], order: Order, window: u64) -> Result<ReadPlan> {
    let mut items = Vec::new();
    let mut dest: u64 = 0;
    for (comp, key) in keys {
        let t = header
            .components
            .iter()
            .find(|(c, _)| c == comp)
            .and_then(|(_, ts)| ts.iter().find(|(k, _)| k == key))
            .map(|(_, t)| t);
        let t = match t {
            Some(t) => t,
            None => {
                return refuse(
                    Code::TRAVERSAL_INCOMPLETE,
                    format!("{comp}/{key}: the traversal names a tensor this header does not carry"),
                )
            }
        };
        for (role, part) in &t.parts {
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
            for s in part.segments() {
                let mut off = 0u64;
                while off < s.length {
                    let n = if window == 0 {
                        s.length - off
                    } else {
                        window.min(s.length - off)
                    };
                    items.push(PlanItem {
                        what: what.clone(),
                        dest_off: dest,
                        len: n,
                        source: Source::Object(ObjectRange {
                            obj: s.clone(),
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
        order,
    })
}

fn tally(items: &[PlanItem]) -> (u64, u64, usize) {
    let bytes = items.iter().map(|i| i.len).sum();
    let io_bytes = items.iter().filter(|i| i.is_io()).map(|i| i.len).sum();
    let inline = items.iter().filter(|i| !i.is_io()).count();
    (bytes, io_bytes, inline)
}

/// The read planner. Order comes from the CALLER's code-topology traversal — never from the
/// header. A traversal that does not name every tensor the header carries refuses: a fill
/// that silently skips a tensor is a wrong model, not a fast one.
pub fn plan_for_traversal(
    header: &Header,
    traversal: &[(String, String)],
    window: u64,
) -> Result<ReadPlan> {
    let mut named: Vec<(&str, &str)> = traversal
        .iter()
        .map(|(c, k)| (c.as_str(), k.as_str()))
        .collect();
    named.sort();
    named.dedup();
    let mut have: Vec<(&str, &str)> = header
        .tensors()
        .map(|(c, k, _)| (c.as_str(), k.as_str()))
        .collect();
    have.sort();
    if named != have {
        let missing: Vec<String> = have
            .iter()
            .filter(|x| !named.contains(x))
            .take(3)
            .map(|(c, k)| format!("{c}/{k}"))
            .collect();
        return refuse(
            Code::TRAVERSAL_INCOMPLETE,
            format!(
                "the traversal names {} distinct tensors, the header carries {} — first \
                 missing: {missing:?}",
                named.len(),
                have.len()
            ),
        );
    }
    build(header, traversal, Order::Destination, window)
}

/// The header's own key order. BENCHMARK-ONLY: it is what the destination-order contract
/// exists to beat, and it is here so the margin can be measured on real bytes.
pub fn plan_in_key_order(header: &Header, window: u64) -> Result<ReadPlan> {
    let keys: Vec<(String, String)> = header
        .tensors()
        .map(|(c, k, _)| (c.clone(), k.clone()))
        .collect();
    build(header, &keys, Order::HeaderKey, window)
}

// ---------------------------------------------------------------- the reader pool

#[derive(Debug, Clone, Default)]
pub struct Stats {
    pub items: usize,
    pub bytes: u64,
    pub io_bytes: u64,
    pub workers: usize,
    pub elapsed_ms: u128,
}

/// Execute a plan into ONE caller-owned destination buffer with `workers` concurrent
/// readers. tfs-018 proved the depth load-bearing: at queue depth 1 cozytensors LOSES cold
/// (1.26 vs 1.51 GB/s) and at depth 4 it wins (3.14 vs 2.42) — the 1679-object layout costs
/// read serially and pays read concurrently, so a reader pool of at least 4 is a claim
/// requirement, not a tuning preference.
///
/// Issue order is plan order: workers pull from one cursor in sequence, so the DESTINATION
/// order is what reaches the disk even though completions are concurrent.
pub fn execute(
    store: &Store,
    lease: &ReadLease,
    plan: &ReadPlan,
    dest: &mut [u8],
    workers: usize,
) -> Result<Stats> {
    if dest.len() as u64 != plan.bytes {
        return refuse(
            Code::BUFFER_SIZE,
            format!(
                "destination is {} B, the plan fills {} B",
                dest.len(),
                plan.bytes
            ),
        );
    }
    let workers = workers.max(1);
    let t0 = Instant::now();

    // Split the caller's buffer into the plan's disjoint destination spans, in order. Any
    // overlap or gap is a planner bug and refuses here rather than corrupting a fill.
    let mut slices: Vec<(usize, &mut [u8])> = Vec::with_capacity(plan.items.len());
    let mut rest: &mut [u8] = dest;
    let mut cur: u64 = 0;
    for (i, it) in plan.items.iter().enumerate() {
        if it.dest_off < cur {
            return refuse(
                Code::RANGE_BOUNDS,
                format!(
                    "plan item {i} ({}) starts at {} inside the span ending at {cur}",
                    it.what, it.dest_off
                ),
            );
        }
        let (_, r) = rest.split_at_mut((it.dest_off - cur) as usize);
        let (mine, r2) = r.split_at_mut(it.len as usize);
        slices.push((i, mine));
        rest = r2;
        cur = it.dest_off + it.len;
    }

    let queue = Mutex::new(slices.into_iter());
    let errs: Mutex<Vec<Refusal>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..workers {
            s.spawn(|| loop {
                let next = queue.lock().unwrap().next();
                let (i, buf) = match next {
                    None => return,
                    Some(x) => x,
                };
                let it = &plan.items[i];
                let r = match &it.source {
                    Source::Object(r) => read_into(store, lease, r, buf),
                    // Inline bytes come out of the header the caller already holds: decoded
                    // into their exact destination span, in order, with no device touched.
                    Source::Inline(b) => crate::b64::decode(b).and_then(|v| {
                        if v.len() != buf.len() {
                            return refuse(
                                Code::BYTE_LENGTH_MISMATCH,
                                format!("inline decodes to {} B for a {} B span", v.len(), buf.len()),
                            );
                        }
                        buf.copy_from_slice(&v);
                        Ok(())
                    }),
                };
                if let Err(e) = r {
                    errs.lock().unwrap().push(Refusal {
                        code: e.code,
                        detail: format!("{}: {}", it.what, e.detail),
                    });
                    return;
                }
            });
        }
    });
    if let Some(e) = errs.into_inner().unwrap().into_iter().next() {
        return Err(e);
    }
    Ok(Stats {
        items: plan.items.len(),
        bytes: plan.bytes,
        io_bytes: plan.io_bytes,
        workers,
        elapsed_ms: t0.elapsed().as_millis(),
    })
}
