//! Byte-range parallel fetch of ONE source object (tfs-061).
//!
//! **The recovery unit is an object only where the object grid is fine.** `pull`'s N-way
//! walk parallelises ACROSS objects, which is the whole story for a repo pull: publish
//! already cut the artifact at 64 MiB, so sixteen streams have sixteen things to carry and
//! a dropped connection costs one segment. A foreign SOURCE has no such grid. A
//! HuggingFace member is whatever the uploader pushed — MiniMax-H3's selection is 44
//! safetensors shards totalling 210.3 GB, one of them 5.2 GB — and the walk that moves it
//! has exactly ONE object to walk, so it runs on ONE stream and a connection that dies at
//! 4.9 GB of 5.2 GB re-buys the whole shard.
//!
//! This module cuts the grid at fetch time instead of publish time. An object is split
//! into PARTS, the parts are fetched in parallel with `Range`, and the object is assembled
//! in offset order. Measured against HuggingFace's CDN from a developer box: 14.2 MB/s on
//! one stream, 18.9 MB/s at eight — the same shape #566b measured pod-side at a different
//! ceiling (49.8 MB/s serial, 117.6 MB/s at sixteen), and the reason the number here is
//! smaller is that this box's own link saturates near 19 MB/s.
//!
//! **The admission door does not move.** Every byte still enters at
//! `DeliveryGrant::admit` → `Store::put_stream`, hashed in one sequential pass over the
//! object's own byte order, and the object is admitted only if it hashes to the id the
//! record owner named. Parallelism changes WHICH SOCKET a byte arrives on and nothing
//! about how it is judged: the parts land in a scratch file, and the door is fed a reader
//! that hands out bytes strictly in offset order and blocks rather than skip a hole. A
//! part that never completes is a read error at the door, and `put_stream` deletes its
//! temp and refuses — a partially assembled object has no path to admission.
//!
//! **A part is `pull::open_object` called again, and nothing else.** Every ranged request
//! goes through the single-stream path's own opener with one extra header, which is the
//! only structure that makes the guarantee cheap to state and impossible to drift from:
//! the fence runs on the part's URL and on every redirect hop of it, the credential is
//! re-asked per host, a relative `Location` resolves against the hop that sent it, the hop
//! budget is the same budget, and the `Retryable`/`Expired` status arms are the same arms.
//! Parallel ranges are not a way to reach a host the predicate never saw.
//!
//! Nothing is resolved once and reused. It is tempting — the probe already walked the
//! chain, and a part could start at the delivery URL and save a 302 — and that is exactly
//! the trade tfs-060 was filed about: a per-request behaviour quietly demoted to a
//! per-object one, correct on the day it ships and wrong the first time an origin changes
//! its mind mid-transfer. The redirect is re-walked per part, at the cost of one cheap hop
//! per 64 MiB, and a delivery URL that ages out during a 5 GB shard simply never happens
//! because no part is holding one.
//!
//! **Resume is per part, and within a part it is per byte.** A part whose connection dies
//! at byte k re-asks for `bytes=(start+k)-end` and moves nothing it already holds. The
//! object is re-asked for only when the caller re-runs, and then `FetchPlan` already
//! answers that it is not held.
//!
//! **A source that cannot be ranged still works.** The probe is one `bytes=0-0` GET: an
//! origin that answers 200, or answers 206 without a usable `Content-Range`, or is merely
//! busy, drops this object onto `pull::fetch_url`'s single-stream path with its full
//! re-ask and re-authorize machinery. Ranged fetch is an accelerator, never a requirement.

use crate::err::{refuse, Code, Refusal, Result};
use crate::fetch::{Admitted, DeliveryGrant};
use crate::store::Store;
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

/// How many ranged asks are in flight for ONE object. Eight rather than `STREAMS`'s
/// sixteen because these all belong to a single object: the walk's sixteen are sixteen
/// different objects and cost sixteen sockets across a whole closure, while these sixteen
/// would be sixteen sockets aimed at one file, which the measurement did not reward
/// (14.2 MB/s at one stream, 18.9 MB/s at eight, 19.1 MB/s at sixteen on the same link).
pub const SOURCE_STREAMS: usize = 8;

/// One ranged ask's slice, and the resume unit. MEASURED, not chosen: this crate's client
/// sends `Connection: close`, so every part pays a fresh TCP connect plus TLS handshake
/// plus time-to-first-byte, which against HuggingFace's CDN measured 0.41 s. A part must
/// therefore be large enough that 0.41 s is noise against its own transfer — at the
/// ~14 MB/s a single stream showed, 64 MiB transfers for 4.6 s and spends 8% of a stream's
/// time handshaking. Smaller parts resume faster and waste more link; this is the trade,
/// and it is a parameter because the right number is a property of the LINK.
pub const PART_BYTES: u64 = 64 << 20;

/// What `PART_BYTES` implies: an object smaller than two parts is one part, and one part
/// is one stream. DERIVED and stated, never a second threshold — the size gate is
/// `Ranged::part` alone, so a caller that cuts smaller parts gets a smaller floor with it
/// rather than a constant here silently overriding the parameter it was handed.
pub const RANGED_MIN_BYTES: u64 = 2 * PART_BYTES;

/// How ONE object is cut. `streams` of 1 is the single-stream path and it stays reachable,
/// for the same reason `fetch_wanted` keeps a serial walk: a link where concurrency does
/// not help is a real link.
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
    /// Do not cut: the object crosses on one stream, exactly as it did before this module.
    pub fn single() -> Ranged {
        Ranged {
            streams: 1,
            part: PART_BYTES,
        }
    }

    fn normalized(self) -> Ranged {
        Ranged {
            streams: self.streams.clamp(1, 64),
            part: self.part.clamp(1 << 20, 1 << 30),
        }
    }
}

/// Move ONE granted object across the door, in parallel byte ranges when the origin
/// supports them and on one stream when it does not.
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
    // gigabyte and cutting it would be all cost. Small objects are the same argument.
    if grant.manifest || ranged.streams < 2 || total < ranged.part.saturating_mul(2) {
        return single(
            store, grant, url, policy, credential, deadline, ledger, observer,
        );
    }
    let probe = match probe(url, policy, credential, deadline, ledger)? {
        Some(probe) => probe,
        // The origin does not do ranges, or would not say so on this ask. Not an error:
        // the single-stream path is the same path this object took before this module.
        None => {
            return single(
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
    assemble(
        store, grant, url, &probe, policy, credential, deadline, ledger, ranged, observer,
    )
}

#[allow(clippy::too_many_arguments)]
fn single(
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
        Opened::Body(_) | Opened::Retryable(_) | Opened::Expired(_) => Ok(None),
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
    next: AtomicUsize,
    stop: AtomicBool,
}

struct State {
    done: Vec<bool>,
    /// Parts `0..contiguous` are complete on disk, so bytes below `contiguous * part` may
    /// be handed to the door. Nothing above it may, hole or not.
    contiguous: usize,
    failure: Option<Refusal>,
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

    fn landed(&self, index: usize) {
        let mut state = self.state.lock().unwrap();
        state.done[index] = true;
        while state.contiguous < state.done.len() && state.done[state.contiguous] {
            state.contiguous += 1;
        }
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
    url: &str,
    probe: &Probe,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    ranged: Ranged,
    observer: Option<&mut dyn FetchObserver>,
) -> Result<Fetched> {
    let total = probe.total;
    let parts = total.div_ceil(ranged.part) as usize;
    let scratch = Scratch::open(store, total)?;
    let wire = Wire {
        state: Mutex::new(State {
            done: vec![false; parts],
            contiguous: 0,
            failure: None,
        }),
        filled: Condvar::new(),
        hops: Mutex::new(Vec::new()),
        moved: AtomicU64::new(0),
        next: AtomicUsize::new(0),
        stop: AtomicBool::new(false),
    };
    let streams = ranged.streams.min(parts);

    let admitted = std::thread::scope(|scope| -> Result<Admitted> {
        for _ in 0..streams {
            scope.spawn(|| {
                loop {
                    if wire.stopped() || wire.state.lock().unwrap().failure.is_some() {
                        return;
                    }
                    let index = wire.next.fetch_add(1, Ordering::Relaxed);
                    if index >= parts {
                        return;
                    }
                    match part(
                        &wire,
                        &scratch.file,
                        index,
                        total,
                        ranged.part,
                        url,
                        policy,
                        credential,
                        deadline,
                        ledger,
                    ) {
                        Ok(true) => wire.landed(index),
                        // Stopped mid-part: the door is finished with this object and
                        // nothing is waiting for these bytes.
                        Ok(false) => return,
                        Err(refusal) => {
                            wire.failed(refusal);
                            return;
                        }
                    }
                }
            });
        }
        let mut reader = Assembled {
            scratch: &scratch.file,
            at: 0,
            total,
            part: ranged.part,
            wire: &wire,
            observer,
            reported: 0,
            tick: ledger.sample(),
        };
        let outcome = grant.admit(store, &mut reader);
        // The door has read every byte it is going to read, whichever way it answered.
        // Anything still on the wire is unwanted, and a worker blocked on the window must
        // not hold the scope open waiting for a reader that has gone.
        wire.stop.store(true, Ordering::Relaxed);
        wire.filled.notify_all();
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
    Ok(Fetched {
        admitted,
        transferred: wire.moved.load(Ordering::Relaxed),
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
    total: u64,
    size: u64,
    original: &str,
    policy: &SourcePolicy,
    credential: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
) -> Result<bool> {
    let start = index as u64 * size;
    let end = (start + size).min(total) - 1;
    let mut at = start;
    let mut last: Option<Refusal> = None;
    for attempt in 1..=FETCH_ATTEMPTS {
        if wire.stopped() {
            return Ok(false);
        }
        if at > end {
            return Ok(true);
        }
        let asked = format!("bytes={at}-{end}");
        let mut hops: Vec<(String, u16)> = Vec::new();
        // From the URL the CALLER named, every time. The predicate runs on it and on every
        // hop of it, the credential is re-asked per host, and a relative `Location`
        // resolves against the hop that sent it — because this is the single-stream path's
        // own opener, not a second copy of it that could fall behind.
        let opened = pull::open_object(
            pull::shared_client(),
            original,
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
            Opened::Retryable(status) if !last_attempt => {
                last = Some(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("part {index}: the origin answered {status}"),
                });
                continue;
            }
            // 401/403 on a source URL is the origin refusing the request, not a lease
            // this side can re-mint — `fetch_authorized` owns that question, and only for
            // presigned hub URLs. Here it is re-asked like any other transient answer,
            // against a chain re-walked from the caller's URL, which is the one thing that
            // could actually change the answer.
            Opened::Expired(status) if !last_attempt => {
                last = Some(Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("part {index}: the origin answered {status}"),
                });
                continue;
            }
            Opened::Retryable(status) | Opened::Expired(status) => {
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
        match body(response, wire, scratch, &mut at, end) {
            Ok(()) => {}
            Err(refusal) if !last_attempt => {
                last = Some(refusal);
                continue;
            }
            Err(refusal) => return Err(ranged_detail(refusal, index, &asked)),
        }
        if at > end {
            return Ok(true);
        }
        if wire.stopped() {
            return Ok(false);
        }
    }
    Err(ranged_detail(
        last.unwrap_or(Refusal {
            code: Code::TRANSFER_FAILED,
            detail: "the origin did not answer".into(),
        }),
        index,
        &format!("bytes={at}-{end}"),
    ))
}

/// Stream one 206 body into the scratch file at the offsets it belongs at, advancing `at`
/// as bytes land so a re-ask starts from where this one stopped rather than from the part.
fn body(
    response: crate::transport::http::Response<'_>,
    wire: &Wire,
    scratch: &File,
    at: &mut u64,
    end: u64,
) -> Result<()> {
    let mut reader = response.into_body();
    let mut buf = vec![0u8; 256 * 1024];
    loop {
        if *at > end {
            return Ok(());
        }
        if wire.stopped() {
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
    part: u64,
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
                let high = (state.contiguous as u64 * self.part).min(self.total);
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
