//! Resumable ranged fetch of ONE source object (tfs-062).
//!
//! **The prize is resume, not parallelism.** `pull`'s N-way walk parallelises ACROSS
//! objects, which is the whole story for a repo pull: publish already cut the artifact at
//! 64 MiB, so a dropped connection costs one segment. A foreign SOURCE has no such grid — a
//! HuggingFace member is whatever the uploader pushed, and MiniMax-H3's selection is 44
//! shards totalling 210.3 GB, one of them 5.2 GB. There was no byte-range resume inside an
//! object, so a connection that died at 4.9 GB of that shard re-bought the shard. On a
//! 210 GB pull over a link that has been watched drop repeatedly, that is the recurring
//! cost, and it is what this module removes.
//!
//! **The throughput case was demoted by measurement, and is not claimed.** It was briefed
//! as 49.8 MB/s on one stream against 117.6 MB/s at sixteen; those were transport-PROBE
//! numbers. Production says otherwise: H3 run 290 sustained 70-129 MB/s per lane against a
//! ~516 MB/s aggregate that looks bounded by the NIC or the origin, and a real 3.087 GB
//! HuggingFace fetch through this code moved at **18.9 MB/s on one stream and 19.1 MB/s on
//! eight 64 MiB parts** — the link was already saturated by one. On a link like that,
//! cutting an object into parts moves bytes from one socket to another and changes no
//! total. `SOURCE_STREAMS` is therefore 1. The one case parallelism wins — the TAIL of a
//! multi-object pull, where fewer objects remain than lanes and an origin answers some
//! asks application-limited (run 2322) — is the hub walk's `hedge`.
//!
//! So there are three paths, and the default is the cheap one:
//!
//! - **one stream (`sequential`, the default).** Ascending ranges, one at a time. The bytes
//!   arrive in the order the door wants them, so `put_stream` reads straight off the wire
//!   and there is no scratch file, no second write, and nothing buffered anywhere. A
//!   dropped connection re-asks `bytes=<reached>-<part end>`; the door has been hashing
//!   that prefix all along and never learns the socket under it was replaced.
//! - **N streams (`assemble`).** Parts in flight at once, landing out of order into a
//!   scratch file, read back in offset order. Costs a second full write of every byte,
//!   which is why it is not the default.
//! - **no ranges (`whole`).** `pull::fetch_url`, byte for byte the path a member took
//!   before this module. An origin that answers the probe with a 200, or with an unusable
//!   `Content-Range`, or is merely busy, lands here. A source that cannot be ranged must
//!   still work.
//!
//! **The admission door does not move.** Every byte still enters at
//! `DeliveryGrant::admit` → `Store::put_stream`, hashed in one sequential pass over the
//! object's own byte order, and the object is admitted only if it hashes to the id the
//! record owner named. Ranges change WHICH SOCKET a byte arrives on and nothing about how
//! it is judged: the door is always fed bytes in strictly ascending offset order, by
//! construction on the sequential path and by a reader that blocks rather than skip a hole
//! on the parallel one. A range that never completes is a read error at the door, and
//! `put_stream` deletes its temp and refuses — a partially assembled object has no path to
//! admission.
//!
//! **A ranged ask is `pull::open_object` called again, and nothing else.** Every request
//! here goes through the single-stream path's own opener with one extra header, which is
//! the only structure that makes the guarantee cheap to state and impossible to drift
//! from: the fence runs on the URL and on every redirect hop of it, the credential is
//! re-asked per host, a relative `Location` resolves against the hop that sent it, the hop
//! budget is the same budget, and the `Retryable`/`Expired` arms are the same arms. Ranged
//! requests are not a way to reach a host the predicate never saw.
//!
//! Redirect targets are not cached. It is tempting — the probe already walked the
//! chain, and an ask could start at the delivery URL and save a 302 — and that is exactly
//! the trade tfs-060 was filed about: a per-request behaviour quietly demoted to a
//! per-object one, correct on the day it ships and wrong the first time an origin changes
//! its mind mid-transfer. The chain is re-walked per ask, at the cost of one cheap hop per
//! 64 MiB, and a delivery URL that ages out during a 5 GB shard simply never happens
//! because nothing is holding one.
//!
use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{Admitted, DeliveryGrant, Destination};
use crate::store::Store;
use crate::transport::http;
use crate::transport::hub::CredentialProvider;
use crate::transport::ledger::{Deadline, Ledger, FETCH_ATTEMPTS};
use crate::transport::policy::SourcePolicy;
use crate::transport::pull::{self, FetchObserver, Fetched, Opened};
use fs2::FileExt as _;
use std::fs::{File, OpenOptions};
use std::io::Read;
use std::os::unix::fs::{FileExt as _, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex};

/// How many ranged asks are in flight for ONE object, by default: **one**.
///
/// This started at eight and was demoted by measurement. The throughput case for splitting
/// one object across N sockets rests on the aggregate being per-STREAM bound, and every
/// number available says it is not: a real 3.087 GB HuggingFace fetch through this code
/// moved at 18.9 MB/s on one stream and 19.1 MB/s on eight 64 MiB parts — the link was
/// already saturated by one — and H3 run 290 has a single production lane sustaining
/// 70-129 MB/s against an aggregate that looks bounded by the NIC or the origin. On a link
/// like that, cutting an object into parts moves bytes from one socket to another and
/// changes no total, while costing a second full write of every byte.
///
/// So the default buys the thing that is NOT in question — byte-exact resume inside an
/// object — and buys it at zero extra disk, because one stream needs no scratch file.
/// Parallelism stays one parameter away for the case still argued for: the TAIL of a
/// multi-object pull, where fewer objects remain than there are lanes and per-object
/// concurrency is the only thing that can fill them. Run 2322 measured that case on hub
/// pulls, and `hedge` answers it there: a slow tail object is split over idle lanes.
pub const SOURCE_STREAMS: usize = 1;

/// One ranged ask's slice and resume unit. The original close-per-request client measured
/// 0.41 s connect/TLS/first-byte cost against HuggingFace, motivating 64 MiB parts. Exact
/// ranged GETs can now reuse a fully consumed, currently authorized connection, but that
/// does not change this established resume unit or the per-request redirect checks.
pub const PART_BYTES: u64 = 64 << 20;

/// What `PART_BYTES` implies: an object smaller than two parts is one part, and one part is
/// one ask, which is what the object would have been anyway. DERIVED and stated, never a
/// second threshold — the size gate is `Ranged::part` alone, so a caller that cuts smaller
/// parts gets a smaller floor with it rather than a constant here silently overriding the
/// parameter it was handed.
pub const RANGED_MIN_BYTES: u64 = 2 * PART_BYTES;

/// The smallest part a hedge cuts. Granularity for spreading one slow object over idle
/// lanes, and the floor `Ranged::normalized` already holds every part to.
pub const HEDGE_PART: u64 = 1 << 20;

/// How ONE object is asked for. `streams` of 1 — the default — is one resumable stream of
/// ascending ranges; above 1 is that many ranges in flight at once, assembled through a
/// scratch file; 0 is `Ranged::whole`, no ranges at all.
#[derive(Debug, Clone, Copy)]
pub struct Ranged {
    pub streams: usize,
    pub part: u64,
}

impl Default for Ranged {
    fn default() -> Ranged {
        Ranged {
            streams: SOURCE_STREAMS,
            part: PART_BYTES,
        }
    }
}

impl Ranged {
    /// Do not range at all: one whole-object GET, exactly the path a member took before
    /// this module, with no resume inside it. This is what an origin that refuses ranges
    /// gets, and it stays reachable by name because a link where none of this helps is a
    /// real link.
    pub fn whole() -> Ranged {
        Ranged {
            streams: 0,
            part: PART_BYTES,
        }
    }

    fn normalized(self) -> Ranged {
        Ranged {
            // NOT clamped up to 1: zero means "do not range", and a floor here would
            // silently turn the caller's opt-out into the default.
            streams: self.streams.min(64),
            part: self.part.clamp(1 << 20, 1 << 30),
        }
    }
}

/// Move ONE granted object across the door as resumable byte ranges when the origin
/// supports them, and as one whole-object GET when it does not.
///
/// `url` is the caller's; nothing here re-authorizes it, which is why this is the SOURCE
/// plane's entry point and not the hub walk's — a presigned hub URL has a lease and
/// `pull::fetch_authorized` owns that question.
#[allow(clippy::too_many_arguments)]
pub fn fetch_ranged(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    ranged: Ranged,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let ranged = ranged.normalized();
    let total = grant.object.length;
    // A manifest is bytes the door reads whole into memory and compares; it is never a
    // gigabyte and ranging it would be all cost. Small objects are the same argument. A
    // STAGED carrier is the opposite case and ranges like a blob: it is the 5 GiB shard
    // this module exists for.
    if grant.destination == Destination::Manifest
        || ranged.streams == 0
        || total < ranged.part.saturating_mul(2)
    {
        return whole(
            store, grant, url, policy, credential, deadline, ledger, observer,
        );
    }
    let probe = match probe(url, policy, credential, deadline, ledger)? {
        Some(probe) => probe,
        // The origin does not do ranges, or would not say so on this ask. Not an error:
        // this is the same whole-object GET the member took before this module.
        None => {
            return whole(
                store, grant, url, policy, credential, deadline, ledger, observer,
            )
        }
    };
    if probe.total != total {
        return refuse(
            Code::LENGTH_MISMATCH,
            format!(
                "{}: the plan declares {total} B and the origin's Content-Range says \
                 {} B — the two halves of an identity may not disagree",
                grant.object.id(),
                probe.total
            ),
        );
    }
    // One stream is the DEFAULT and is not a degraded case: the parts arrive in order, so
    // the door is fed straight off the wire and nothing is ever buffered anywhere. Resume
    // is the whole point and is unaffected — a dropped connection re-asks from the byte it
    // reached, and the door, which has been hashing that prefix all along, never notices.
    if ranged.streams == 1 {
        return sequential(
            store,
            grant,
            url,
            probe.total,
            policy,
            credential,
            deadline,
            ledger,
            ranged,
            observer,
        );
    }
    assemble(
        store,
        grant,
        &|_| Ok(url.to_string()),
        pull::shared_client(),
        probe.total,
        policy,
        credential,
        deadline,
        ledger,
        ranged,
        None,
        observer,
    )
}

/// A hub object's hedge: the whole object again as ranged parts on fresh connections, one
/// stream per lane `lanes` holds — a count the walk raises as its lanes come free — each
/// ask authorized by the walk's own lease supply so a part that outlives its URL re-mints
/// instead of presenting it dead.
#[allow(clippy::too_many_arguments)]
pub(super) fn hedge(
    store: &Store,
    grant: &DeliveryGrant,
    authorize: &(dyn Fn(bool, &Ledger) -> Result<String> + Sync),
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    lanes: &AtomicUsize,
) -> Result<Fetched> {
    let total = grant.object.length;
    let streams = lanes.load(Ordering::Acquire);
    // Four parts a lane: a part stuck on a cold chunk is split or re-asked by the lanes that
    // finish theirs, and each part is a fresh ranged ask.
    let ranged = Ranged {
        streams,
        part: total.div_ceil(4 * streams as u64).max(HEDGE_PART),
    };
    assemble(
        store,
        grant,
        &|stale| authorize(stale, ledger),
        &http::Client::new(),
        total,
        policy,
        credential,
        deadline,
        ledger,
        ranged,
        Some(lanes),
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn whole(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    pull::fetch_url(
        store, grant, url, policy, credential, deadline, ledger, observer,
    )
}

// ---------------------------------------------------------------- the probe

/// What one `bytes=0-0` GET learned. Deliberately only the two facts a fetch plan needs:
/// the object's length, and that the origin answers ranges at all. The URL the chain
/// resolved to is NOT carried out of here — see the module doc.
struct Probe {
    total: u64,
}

/// Ask the origin for one byte and read the answer's shape. `Ok(None)` means "not
/// rangeable, or not answerable right now" and is a fall-back, never a failure; only a
/// refusal the single-stream path could not improve on is returned as an error.
fn probe(
    url: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<Option<Probe>> {
    let opened = pull::open_object(
        pull::shared_client(),
        url,
        policy,
        credential,
        deadline,
        ledger,
        &[("range".to_string(), "bytes=0-0".to_string())],
        &mut |_checked, _status| {},
    )?;
    match opened {
        Opened::Body(response) if response.status == 206 => {
            match content_range(response.header("content-range")) {
                // The origin must also agree about WHERE it started: a 206 that answers
                // from somewhere other than the byte asked for cannot be assembled by
                // offset, whatever its total says.
                Some((0, 0, total)) if total > 0 => Ok(Some(Probe { total })),
                _ => Ok(None),
            }
        }
        // 200 to a ranged ask is the origin saying it does not do ranges. Everything else
        // — busy, expired signature — is a question the single-stream path is better
        // equipped to ask, because it can re-ask and re-authorize and this cannot.
        Opened::Body(_) | Opened::Retryable(..) | Opened::Expired(_) => Ok(None),
        Opened::Refused(refusal) => Err(refusal),
    }
}

/// `bytes <first>-<last>/<total>` — the three numbers, or nothing. A `*` total is a
/// declaration that the origin will not say, which is not something to assemble against.
fn content_range(header: Option<&str>) -> Option<(u64, u64, u64)> {
    let value = header?.trim();
    let spec = value.strip_prefix("bytes")?.trim_start();
    let (range, total) = spec.split_once('/')?;
    let (first, last) = range.trim().split_once('-')?;
    Some((
        first.trim().parse().ok()?,
        last.trim().parse().ok()?,
        total.trim().parse().ok()?,
    ))
}

// ---------------------------------------------------------------- the assembly

/// What every part reports into and what the door's reader reads out of.
struct Wire {
    state: Mutex<State>,
    /// Woken when a part lands, when a part fails, and when the door is done with the
    /// object — the three facts a waiting reader or worker can act on.
    filled: Condvar,
    /// Redirect hops the parts have walked, queued for the calling thread to announce. The
    /// observer is the CALLER's and is touched by the calling thread alone; a worker that
    /// called it directly would need it to be `Send` for no reason.
    hops: Mutex<Vec<(u32, String, u16)>>,
    /// Bytes actually written into the scratch file, across every part. Monotone by
    /// construction — a re-asked part resumes at its own offset and re-counts nothing — so
    /// it is a sensible running total even though the parts land out of order.
    moved: AtomicU64,
    stop: AtomicBool,
    /// Workers that have not yet returned. The reader waits on `filled` for bytes that only
    /// a worker can produce, so "every worker is gone and the object is not whole" has to be
    /// a recorded FAILURE and not a silence — otherwise a worker that dies in a way this
    /// module did not anticipate is an unkillable hang on a pod, hours into a transfer,
    /// which is precisely the failure mode this whole lane exists to stop having.
    live: AtomicUsize,
}

/// One run of the object some racer is asked to fill. The parts start as `ranged.part`
/// segments; a hedge splits the largest unfinished one when a lane goes idle.
struct Segment {
    /// Inclusive bounds. `end` only shrinks, when the segment is split.
    start: u64,
    end: u64,
    /// Below this mark the segment's bytes are on disk. Every racer starts at it, so what
    /// is below it stays one contiguous run.
    reached: u64,
    racers: usize,
    done: bool,
    /// A racer exhausted its asks here; it is not raced again.
    refused: bool,
}

struct State {
    /// The initial parts in offset order, then splits in the order they were cut.
    segments: Vec<Segment>,
    /// Initial parts nobody has asked for start here.
    next: usize,
    initial: usize,
    /// Bytes below `high` are complete on disk in one run from the first byte, so they
    /// may be handed to the door. Nothing above it may, hole or not.
    high: u64,
    failure: Option<Refusal>,
}

impl State {
    fn coverage(&self) -> u64 {
        let mut order: Vec<&Segment> = self.segments.iter().collect();
        order.sort_by_key(|segment| segment.start);
        let mut high = 0;
        for segment in order {
            if segment.start != high || !segment.done {
                break;
            }
            high = segment.end + 1;
        }
        high
    }
}

impl Wire {
    fn stopped(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    fn hop(&self, attempt: u32, host: &str, status: u16) {
        self.hops
            .lock()
            .unwrap()
            .push((attempt, host.to_string(), status));
    }

    /// The next part nobody has asked for. A hedge then takes the unfinished segment with
    /// the most bytes left: it splits off that segment's upper half when there is a part's
    /// worth left, and races what remains from the segment's mark when there is not. A lane
    /// that finished its own part thereby speeds the slowest one instead of idling.
    fn claim(&self, race: bool) -> Option<(usize, u64)> {
        let mut state = self.state.lock().unwrap();
        if state.failure.is_some() || self.stopped() {
            return None;
        }
        if state.next < state.initial {
            let index = state.next;
            state.next += 1;
            state.segments[index].racers += 1;
            return Some((index, state.segments[index].reached));
        }
        if !race {
            return None;
        }
        let index = (0..state.segments.len())
            .filter(|&index| !state.segments[index].done && !state.segments[index].refused)
            .max_by_key(|&index| {
                let segment = &state.segments[index];
                segment.end + 1 - segment.reached
            })?;
        let segment = &mut state.segments[index];
        let left = segment.end + 1 - segment.reached;
        if segment.racers == 0 || left < HEDGE_PART {
            segment.racers += 1;
            return Some((index, segment.reached));
        }
        let middle = segment.reached + left / 2;
        let end = std::mem::replace(&mut segment.end, middle - 1);
        state.segments.push(Segment {
            start: middle,
            end,
            reached: middle,
            racers: 1,
            done: false,
            refused: false,
        });
        Some((state.segments.len() - 1, middle))
    }

    /// The last byte segment `index` still wants, or nothing once it is complete.
    fn wants(&self, index: usize) -> Option<u64> {
        let state = self.state.lock().unwrap();
        let segment = &state.segments[index];
        (!segment.done).then_some(segment.end)
    }

    /// Bytes `..to` of segment `index` are on disk. Answers its last wanted byte, which a
    /// split may have moved closer, or nothing once another racer finished it.
    fn advance(&self, index: usize, to: u64) -> Option<u64> {
        let mut state = self.state.lock().unwrap();
        let segment = &mut state.segments[index];
        segment.reached = segment.reached.max(to);
        (!segment.done).then_some(segment.end)
    }

    /// A racer left segment `index`. A refusal retires the segment from racing, and fails
    /// the object once nobody else is on it.
    fn leave(&self, index: usize, refusal: Option<Refusal>) {
        let mut state = self.state.lock().unwrap();
        let segment = &mut state.segments[index];
        segment.racers -= 1;
        segment.refused |= refusal.is_some();
        let orphaned = segment.racers == 0 && !segment.done;
        drop(state);
        if let Some(refusal) = refusal.filter(|_| orphaned) {
            self.failed(refusal);
        }
    }

    fn landed(&self, index: usize) {
        let mut state = self.state.lock().unwrap();
        state.segments[index].done = true;
        state.high = state.coverage();
        drop(state);
        self.filled.notify_all();
    }

    fn failed(&self, refusal: Refusal) {
        let mut state = self.state.lock().unwrap();
        if state.failure.is_none() {
            state.failure = Some(refusal);
        }
        drop(state);
        self.filled.notify_all();
    }

    /// One worker has returned, however it returned. The LAST one out with the object still
    /// incomplete and nothing recorded leaves a refusal behind, so the reader wakes and the
    /// door refuses instead of blocking on bytes nobody is fetching any more. Poison-tolerant
    /// because this runs in a `Drop` and panicking there would abort the process.
    fn gone(&self) {
        if self.live.fetch_sub(1, Ordering::AcqRel) > 1 {
            return;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total = state.segments.iter().map(|segment| segment.end + 1).max();
        if state.failure.is_none() && Some(state.high) != total && !self.stopped() {
            state.failure = Some(Refusal {
                code: Code::TRANSFER_FAILED,
                detail: format!(
                    "every stream stopped with {} of {} bytes assembled",
                    state.high,
                    total.unwrap_or(0)
                ),
            });
        }
        drop(state);
        self.filled.notify_all();
    }
}

/// Decrements the live count however the worker leaves — return, refusal, or unwind.
struct Worker<'a>(&'a Wire);

impl Drop for Worker<'_> {
    fn drop(&mut self) {
        self.0.gone();
    }
}

/// Sets `stop` however the door leaves, so a reader that unwinds does not leave eight
/// streams pulling bytes into a scratch file nobody will read.
struct Door<'a>(&'a Wire);

impl Drop for Door<'_> {
    fn drop(&mut self) {
        // Under the state lock, so a waiter that just found `stop` clear is already waiting
        // when the notice comes. Poison-tolerant: this runs in a `Drop`.
        let state = self
            .0
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        self.0.stop.store(true, Ordering::Relaxed);
        drop(state);
        self.0.filled.notify_all();
    }
}

/// The scratch file the parts are written into, at the offsets they belong at.
///
/// It lives in the store's own `tmp/` under the `put-` prefix and holds the same exclusive
/// lock `put_stream`'s temps hold, so the store's existing orphan rule covers it unchanged:
/// a temp whose lock the kernel still holds has a live writer behind it, and one whose lock
/// `reap` is handed is an orphan. A process killed mid-fetch leaves a file the store
/// already knows how to reclaim rather than a new class of litter.
struct Scratch {
    path: PathBuf,
    file: File,
}

impl Scratch {
    fn open(store: &Store, length: u64) -> Result<Scratch> {
        let path = store.root().join("tmp").join(format!(
            "put-range-{}-{}",
            std::process::id(),
            crate::meta::now_nanos_unique()
        ));
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: format!("create scratch {}: {error}", path.display()),
            })?;
        if file.try_lock_exclusive().is_err() {
            let _ = std::fs::remove_file(&path);
            return refuse(Code::IO_FAILED, "could not lock the assembly scratch file");
        }
        file.set_len(length).map_err(|error| Refusal {
            code: Code::IO_FAILED,
            detail: format!("size scratch to {length} B: {error}"),
        })?;
        Ok(Scratch { path, file })
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[allow(clippy::too_many_arguments)]
fn assemble(
    store: &Store,
    grant: &DeliveryGrant,
    ask: &(dyn Fn(bool) -> Result<String> + Sync),
    client: &http::Client,
    total: u64,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    ranged: Ranged,
    // A hedge's lane count. A hedge races and splits unfinished parts with every stream it
    // has, and runs a stream for each lane it is lent while the object is still incomplete.
    lanes: Option<&AtomicUsize>,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let race = lanes.is_some();
    let parts = total.div_ceil(ranged.part) as usize;
    let scratch = Scratch::open(store, total)?;
    let wire = Wire {
        state: Mutex::new(State {
            segments: (0..parts as u64)
                .map(|index| Segment {
                    start: index * ranged.part,
                    end: ((index + 1) * ranged.part).min(total) - 1,
                    reached: index * ranged.part,
                    racers: 0,
                    done: false,
                    refused: false,
                })
                .collect(),
            next: 0,
            initial: parts,
            high: 0,
            failure: None,
        }),
        filled: Condvar::new(),
        hops: Mutex::new(Vec::new()),
        moved: AtomicU64::new(0),
        stop: AtomicBool::new(false),
        live: AtomicUsize::new(0),
    };
    let streams = ranged.streams.min(parts);
    wire.live.store(streams, Ordering::Relaxed);
    let (wire, file) = (&wire, &scratch.file);
    let stream = move || {
        let _worker = Worker(wire);
        while let Some((index, from)) = wire.claim(race) {
            let outcome = part(
                wire, file, index, from, total, ask, client, policy, credential, deadline, ledger,
            );
            match outcome {
                Ok(true) => {
                    wire.landed(index);
                    wire.leave(index, None);
                }
                // Another racer finished the part, or the door is done with the object and
                // nothing is waiting for these bytes.
                Ok(false) => wire.leave(index, None),
                Err(refusal) => wire.leave(index, Some(refusal)),
            }
        }
    };

    let admitted = std::thread::scope(|scope| -> Result<Admitted> {
        for _ in 0..streams {
            scope.spawn(stream);
        }
        if let Some(lanes) = lanes {
            scope.spawn(move || {
                let mut running = streams;
                let mut state = wire.state.lock().unwrap();
                while !wire.stopped() && state.failure.is_none() && state.high < total {
                    while running < lanes.load(Ordering::Acquire) {
                        wire.live.fetch_add(1, Ordering::AcqRel);
                        scope.spawn(stream);
                        running += 1;
                    }
                    state = wire.filled.wait_timeout(state, ledger.sample()).unwrap().0;
                }
            });
        }
        let mut reader = Assembled {
            scratch: &scratch.file,
            at: 0,
            total,
            wire,
            observer,
            reported: 0,
            tick: ledger.sample(),
        };
        // The door has read every byte it is going to read, whichever way it answered —
        // including by unwinding. Anything still on the wire is unwanted after that.
        let outcome = {
            let _door = Door(wire);
            grant.admit(store, &mut reader)
        };
        reader.report();
        match outcome {
            Ok(admitted) => Ok(admitted),
            // A part that could not be moved reaches the door as "read source: ...", which
            // names the symptom and not the cause. The cause is the refusal the part
            // recorded. A door that refused for its OWN reason — the bytes hashed wrong —
            // recorded nothing here and propagates untouched.
            Err(door) => Err(wire.state.lock().unwrap().failure.take().unwrap_or(door)),
        }
    })?;
    // Racing parts write some bytes twice; the ledger counts those, the object once.
    Ok(Fetched {
        admitted,
        transferred: wire.moved.load(Ordering::Relaxed).min(total),
    })
}

/// Move ONE part into the scratch file. `Ok(true)` is the whole part on disk; `Ok(false)`
/// is "the caller stopped wanting it".
///
/// Re-asking is per part and resumes at the byte the part has reached, which is the point
/// of the whole module: a connection that dies at 63 MiB of a 64 MiB part asks for the
/// last MiB, and one that dies at 4.9 GB of a 5.2 GB object asks for one part.
#[allow(clippy::too_many_arguments)]
fn part(
    wire: &Wire,
    scratch: &File,
    index: usize,
    from: u64,
    total: u64,
    ask: &(dyn Fn(bool) -> Result<String> + Sync),
    client: &http::Client,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<bool> {
    let mut at = from;
    let mut last: Option<Refusal> = None;
    let mut attempt = 0;
    let mut stale = false;
    while attempt < FETCH_ATTEMPTS {
        attempt += 1;
        ledger.check_running()?;
        if wire.stopped() {
            return Ok(false);
        }
        let Some(end) = wire.wants(index) else {
            return Ok(false);
        };
        if at > end {
            return Ok(true);
        }
        let asked = format!("bytes={at}-{end}");
        let mut hops: Vec<(String, u16)> = Vec::new();
        // Authorized per ask: a part that outlives its URL is re-minted by whoever owns
        // the lease rather than presenting it dead. The predicate runs on it and on every
        // hop of it, the credential is re-asked per host, and a relative `Location`
        // resolves against the hop that sent it — because this is the single-stream path's
        // own opener, not a second copy of it that could fall behind.
        let url = ask(std::mem::take(&mut stale))?;
        let opened = pull::open_object(
            client,
            &url,
            policy,
            credential,
            deadline,
            ledger,
            &[("range".to_string(), asked.clone())],
            &mut |checked, status| hops.push((checked.host.clone(), status)),
        );
        for (host, status) in &hops {
            wire.hop(attempt, host, *status);
        }
        let last_attempt = attempt == FETCH_ATTEMPTS;
        let opened = match opened {
            Ok(opened) => opened,
            Err(refusal) if refusal.code == Code::TRANSFER_FAILED && !last_attempt => {
                last = Some(refusal);
                continue;
            }
            Err(refusal) => return Err(ranged_detail(refusal, index, &asked)),
        };
        let response = match opened {
            Opened::Body(response) if response.status == 206 => response,
            // 200 where 206 was asked for: the origin answered with the WHOLE object on a
            // socket that is carrying one part of it. Assembling that would put the first
            // bytes of the object at this part's offset, which the digest would catch and
            // the operator would not understand. Refuse it by name instead.
            Opened::Body(_) => {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!(
                        "part {index}: asked for {asked} and the origin answered 200 — it \
                         stopped honouring ranges mid-object, which is a fact about the \
                         origin and not something a re-ask can fix"
                    ),
                )
            }
            Opened::Retryable(status, stated) => {
                let (wait, counted) = ledger.retry_wait(stated);
                if last_attempt && counted {
                    return refuse(
                        Code::TRANSFER_FAILED,
                        format!(
                            "part {index}: the origin answered {status} on every one of \
                             {FETCH_ATTEMPTS} asks for {asked}"
                        ),
                    );
                }
                if !counted {
                    attempt -= 1;
                }
                last = Some(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("part {index}: the origin answered {status}"),
                });
                ledger.pause(wait, deadline)?;
                continue;
            }
            // 401/403 on a source URL is the origin refusing the request, not a lease
            // this side can re-mint — `fetch_authorized` owns that question, and only for
            // presigned hub URLs. Here it is re-asked like any other transient answer,
            // against a chain re-walked from the caller's URL, which is the one thing that
            // could actually change the answer.
            Opened::Expired(status) if !last_attempt => {
                stale = true;
                last = Some(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("part {index}: the origin answered {status}"),
                });
                continue;
            }
            Opened::Expired(status) => {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!(
                        "part {index}: the origin answered {status} on every one of \
                         {FETCH_ATTEMPTS} asks for {asked}"
                    ),
                )
            }
            Opened::Refused(refusal) => return Err(ranged_detail(refusal, index, &asked)),
        };
        // The origin agrees it is answering the range that was asked for, and agrees about
        // the object's size. Without this a 206 that quietly restarts at zero would be
        // assembled at this part's offset and only the digest would notice.
        match content_range(response.header("content-range")) {
            Some((first, _, said)) if first == at && said == total => {}
            other => {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!(
                        "part {index}: asked for {asked} of {total} B and the origin \
                         answered Content-Range {other:?}"
                    ),
                )
            }
        }
        match body(response, wire, scratch, index, &mut at) {
            Ok(()) => {}
            Err(refusal) if !last_attempt => {
                last = Some(refusal);
                continue;
            }
            Err(refusal) => return Err(ranged_detail(refusal, index, &asked)),
        }
        match wire.wants(index) {
            Some(end) if at > end => return Ok(true),
            Some(_) if !wire.stopped() => {}
            _ => return Ok(false),
        }
    }
    Err(ranged_detail(
        last.unwrap_or(Refusal {
            code: Code::TRANSFER_FAILED,
            detail: "the origin did not answer".into(),
        }),
        index,
        &format!("bytes={at}-"),
    ))
}

/// Stream one 206 body into the scratch file at the offsets it belongs at, advancing `at`
/// as bytes land so a re-ask starts from where this one stopped rather than from the part.
fn body(
    response: crate::transport::http::Response<'_>,
    wire: &Wire,
    scratch: &File,
    index: usize,
    at: &mut u64,
) -> Result<()> {
    let mut reader = response.into_body();
    let mut buf = vec![0u8; 256 * 1024];
    let Some(mut end) = wire.wants(index) else {
        return Ok(());
    };
    loop {
        if *at > end || wire.stopped() {
            return Ok(());
        }
        let want = (end - *at + 1).min(buf.len() as u64) as usize;
        // `read_refusing` carries the ledger's stall verdict and the caller's deadline as
        // TYPED refusals, which is what makes a black-holed part re-askable rather than an
        // opaque io error.
        let read = reader.read_refusing(&mut buf[..want])?;
        if read == 0 {
            return refuse(
                Code::TRANSFER_FAILED,
                format!(
                    "the origin closed a ranged answer {} B short",
                    end - *at + 1
                ),
            );
        }
        scratch
            .write_at(&buf[..read], *at)
            .map_err(|error| Refusal {
                code: Code::IO_FAILED,
                detail: format!("write scratch at {at}: {error}"),
            })?;
        *at += read as u64;
        wire.moved.fetch_add(read as u64, Ordering::Relaxed);
        // A racer of the same segment writes the same bytes at the same offsets; the
        // door's digest is the authority if an origin ever answers two asks differently.
        // A split moves this segment's end closer; past it the answer is abandoned.
        match wire.advance(index, *at) {
            Some(wanted) => end = wanted,
            None => return Ok(()),
        }
    }
}

fn ranged_detail(refusal: Refusal, index: usize, asked: &str) -> Refusal {
    Refusal {
        code: refusal.code,
        detail: format!("part {index} ({asked}): {}", refusal.detail),
    }
}

/// The object as the door reads it: strictly ascending offsets, and a wait rather than a
/// skip when the next bytes have not landed.
///
/// This is the whole reason parallelism does not weaken the digest. `put_stream` hashes
/// exactly what it reads in exactly the order it reads it; this reader can only ever hand
/// it byte `n` after byte `n-1`, and can only hand it bytes belonging to a part that has
/// completely landed. Out-of-order arrival is a fact about the sockets and never about the
/// hash.
struct Assembled<'a, 'o> {
    scratch: &'a File,
    at: u64,
    total: u64,
    wire: &'a Wire,
    /// The CALLER's observer, and its own lifetime: it outlives this fetch, while the
    /// scratch file and the wire do not.
    observer: Option<&'o mut (dyn FetchObserver + 'o)>,
    reported: u64,
    tick: std::time::Duration,
}

impl Assembled<'_, '_> {
    /// Hand the caller's observer what the parts have accumulated. Called on every read AND
    /// on every sampling tick of a wait, because the interesting case is precisely the one
    /// where the reader is blocked on part 0 while seven other parts are moving: a
    /// supervisor that only heard from completed reads would see nothing at all then.
    fn report(&mut self) {
        let Some(observer) = self.observer.as_mut() else {
            return;
        };
        let hops = std::mem::take(&mut *self.wire.hops.lock().unwrap());
        for (attempt, host, status) in hops {
            observer.hop(attempt, &host, status);
        }
        let moved = self.wire.moved.load(Ordering::Relaxed);
        if moved != self.reported {
            self.reported = moved;
            observer.moved(moved);
        }
    }
}

impl Read for Assembled<'_, '_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.at >= self.total || out.is_empty() {
            return Ok(0);
        }
        let high = loop {
            self.report();
            let waited = {
                let state = self.wire.state.lock().unwrap();
                if let Some(refusal) = &state.failure {
                    return Err(std::io::Error::other(refusal.to_string()));
                }
                let high = state.high;
                if high > self.at {
                    Some(high)
                } else {
                    let _unused = self.wire.filled.wait_timeout(state, self.tick);
                    None
                }
            };
            if let Some(high) = waited {
                break high;
            }
        };
        self.report();
        let n = out.len().min((high - self.at) as usize);
        self.scratch.read_at(&mut out[..n], self.at)?;
        self.at += n as u64;
        Ok(n)
    }
}

// ---------------------------------------------------------------- one stream, resumable

/// One stream, ranged — the DEFAULT, and the whole point of the module.
///
/// The object is asked for one part at a time in ascending order, so the bytes arrive in the
/// order the door wants them and there is no scratch file, no second write, and nothing
/// buffered anywhere: `put_stream` reads straight off the wire exactly as it does for a
/// whole-object GET. The only thing that changed is what happens when the wire stops.
///
/// **A dropped connection costs the bytes that did not arrive, and nothing else.** The
/// reader remembers the offset it has reached; the next ask is `bytes=<reached>-<part end>`.
/// The door has been hashing that prefix all along and never learns that the socket under it
/// was replaced. Before this, a connection that died at 4.9 GB of a 5.2 GB shard re-bought
/// the shard.
#[allow(clippy::too_many_arguments)]
fn sequential(
    store: &Store,
    grant: &DeliveryGrant,
    url: &str,
    total: u64,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    ranged: Ranged,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let mut reader = Resumed {
        at: 0,
        total,
        part: ranged.part,
        open: None,
        attempt: 0,
        url,
        policy,
        credential,
        deadline,
        ledger,
        observer,
        failure: None,
    };
    let admitted = match grant.admit(store, &mut reader) {
        Ok(admitted) => admitted,
        // Same rule the parallel path follows: a range that could not be moved reaches the
        // door as "read source: ...", which names the symptom. The cause is the refusal the
        // reader recorded. A door that refused for its OWN reason — the bytes hashed wrong —
        // recorded nothing here and propagates untouched.
        Err(door) => return Err(reader.failure.take().unwrap_or(door)),
    };
    Ok(Fetched {
        admitted,
        transferred: reader.at,
    })
}

/// The object as one resumable stream: ascending ranges, re-asked from the byte reached.
struct Resumed<'a, 'l, 'o> {
    at: u64,
    total: u64,
    part: u64,
    /// The body currently being read and the byte its range runs to.
    open: Option<(http::BodyReader<'l>, u64)>,
    /// Asks spent since the last byte that landed. Reset by PROGRESS, not by a part
    /// boundary: a link that delivers 60 MiB and then dies has not spent an attempt on the
    /// next 60, and counting it as if it had would turn a long slow transfer into a failure.
    attempt: u32,
    url: &'a str,
    policy: &'a SourcePolicy,
    credential: &'a dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'l Ledger,
    observer: Option<&'o mut (dyn FetchObserver + 'o)>,
    /// The typed refusal behind whatever `io::Error` the door will see.
    failure: Option<Refusal>,
}

impl<'l> Resumed<'_, 'l, '_> {
    /// Ask for `bytes=<at>-<end of the part containing at>`, through the single-stream
    /// path's own opener so the fence, the credential and the redirect rules are the same
    /// ones — not a copy of them.
    /// `Ok(None)`: the origin stated a wait and it was waited out; ask again without
    /// spending an attempt.
    fn ask(&mut self) -> Result<Option<(http::BodyReader<'l>, u64)>> {
        let end = (self.at / self.part + 1)
            .saturating_mul(self.part)
            .min(self.total)
            - 1;
        let asked = format!("bytes={}-{end}", self.at);
        let mut hops: Vec<(String, u16)> = Vec::new();
        let opened = pull::open_object(
            pull::shared_client(),
            self.url,
            self.policy,
            self.credential,
            self.deadline,
            self.ledger,
            &[("range".to_string(), asked.clone())],
            &mut |checked, status| hops.push((checked.host.clone(), status)),
        );
        if let Some(observer) = self.observer.as_mut() {
            for (host, status) in &hops {
                observer.hop(self.attempt + 1, host, *status);
            }
        }
        let response = match opened? {
            Opened::Body(response) if response.status == 206 => response,
            // 200 where 206 was asked for. At offset 0 that is simply an origin that does
            // not range, and the whole body IS the object, so it is read as one — the same
            // answer the probe's fall-back gives, arrived at later. Past offset 0 it would
            // put the object's first bytes at this offset, and only the digest would notice.
            Opened::Body(response) if self.at == 0 && end + 1 == self.total => response,
            Opened::Body(_) => {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!(
                        "asked for {asked} and the origin answered 200 — it stopped \
                         honouring ranges mid-object, which no re-ask can fix"
                    ),
                )
            }
            Opened::Retryable(status, stated) => {
                let (wait, counted) = self.ledger.retry_wait(stated);
                if !counted || self.attempt + 1 < FETCH_ATTEMPTS {
                    self.ledger.pause(wait, self.deadline)?;
                }
                if !counted {
                    return Ok(None);
                }
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!("the origin answered {status} to {asked}"),
                );
            }
            Opened::Expired(status) => {
                return refuse(
                    Code::TRANSFER_FAILED,
                    format!("the origin answered {status} to {asked}"),
                )
            }
            Opened::Refused(refusal) => return Err(refusal),
        };
        if response.status == 206 {
            match content_range(response.header("content-range")) {
                Some((first, _, said)) if first == self.at && said == self.total => {}
                other => {
                    return refuse(
                        Code::TRANSFER_FAILED,
                        format!(
                            "asked for {asked} of {} B and the origin answered \
                             Content-Range {other:?}",
                            self.total
                        ),
                    )
                }
            }
        }
        Ok(Some((response.into_body(), end)))
    }

    /// Keep the typed refusal and hand the door the `io::Error` a `Read` can carry.
    fn record(&mut self, refusal: Refusal) -> std::io::Error {
        let text = refusal.to_string();
        self.failure = Some(refusal);
        std::io::Error::other(text)
    }
}

impl Read for Resumed<'_, '_, '_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.at >= self.total || out.is_empty() {
            return Ok(0);
        }
        loop {
            // A body is held only while it still owes bytes, so `end` is never behind `at`
            // and the next ask is always the next thing to do.
            let Some((mut body, end)) = self.open.take() else {
                match self.ask() {
                    Ok(Some(open)) => {
                        self.open = Some(open);
                        continue;
                    }
                    Ok(None) => continue,
                    Err(refusal) => {
                        self.attempt += 1;
                        if self.attempt >= FETCH_ATTEMPTS {
                            return Err(self.record(refusal));
                        }
                        continue;
                    }
                }
            };
            let want = out.len().min((end - self.at + 1) as usize);
            match body.read_refusing(&mut out[..want]) {
                Ok(0) => {
                    // The origin closed inside the range it promised. The bytes already
                    // handed to the door STAY handed to the door — `at` is where the next
                    // ask begins, and that is the whole feature.
                    self.attempt += 1;
                    if self.attempt >= FETCH_ATTEMPTS {
                        return Err(self.record(Refusal {
                            code: Code::TRANSFER_FAILED,
                            detail: format!(
                                "the origin closed {} B short of the range it promised, on \
                                 every one of {FETCH_ATTEMPTS} asks",
                                end - self.at + 1
                            ),
                        }));
                    }
                }
                Ok(read) => {
                    self.at += read as u64;
                    // Progress, so the budget is whole again: an object that keeps moving
                    // never spends an attempt, however many sockets it took to move it.
                    self.attempt = 0;
                    if self.at <= end {
                        self.open = Some((body, end));
                    }
                    if let Some(observer) = self.observer.as_mut() {
                        observer.moved(self.at);
                    }
                    return Ok(read);
                }
                Err(refusal) => {
                    self.attempt += 1;
                    if self.attempt >= FETCH_ATTEMPTS {
                        return Err(self.record(refusal));
                    }
                }
            }
        }
    }
}
