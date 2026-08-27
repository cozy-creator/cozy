//! Roots, leases and local GC (tfs-006), made EXCLUSIVE AND EXPLICIT (tfs-020, decisions
//! #613) over the metadata authority.
//!
//! Two processes never hold conflicting authority over one byte, so:
//!
//! * **GC never INFERS what is alive.** Every known non-tombstoned snapshot is live by
//!   default — pinned, warm, unpublished, in-flight alike. A snapshot's bytes become
//!   collectable ONLY via the explicit tombstone (`tfs rm` / the worker-protocol purge
//!   path); liveness is never computed from pin structure.
//! * **GC never runs beside a writer.** The sweep takes store-wide exclusivity at start:
//!   any open ingest transaction root or any held lease refuses it (`STORE_BUSY`, or wait
//!   under `--wait`), and because opening a root and registering a hold both pass through
//!   the metadata lock the sweep holds for its whole cycle, a border run arriving
//!   mid-sweep blocks the border, never the bytes. No incremental refresh and no
//!   census-vs-writer race is representable.
//!
//! One rule shapes the failure direction: **over-keeping the index is the only permitted
//! failure direction.** A sweep that cannot read its liveness census refuses to delete; a
//! row dies WITH its object and after it; recovery may keep bytes it did not need, never
//! invent a root and never delete from an incomplete census.
//!
//! The whole cycle — census, mark, claim, delete — runs inside ONE metadata write
//! transaction, so the scrub cannot race a seal or a release: a live root can never be
//! judged garbage because no root can change while the sweep holds the lock. That is what
//! makes the scrub-vs-delete proof deterministic rather than intermittently green (v1's
//! `a_scrub_racing_a_deletion` is still flaky, and an intermittently-green test is not a
//! proof).
//!
//! Removal takes the NAME first: an object is renamed out of `objects/` into
//! `gc/<generation>/` before it is unlinked, so exactly one remover wins and a failed
//! unlink leaves unreachable scratch for the next reap rather than a removal failure.
//!
//! ## The state machines, and where each one linearizes
//!
//! A database transaction cannot atomically contain an fsync, a link, a rename or an unlink,
//! so every transition names its own crash point and its own recovery direction.
//!
//! ```text
//! object      writing -> verified-temp -> installed          (store.rs)
//!             linearizes at link(); a crash leaves a temp, never a partial object
//! snapshot    unknown -> known -> rooted | cached -> reclaimed
//!             linearizes at the DB root commit; `known` is knowledge, `rooted` is liveness
//! hold        acquired -> released | reaped(owner-dead)
//!             linearizes at commit; the OWNER LOCK, not a clock, decides `reaped`
//! GC object   live -> deleting(claim_generation) -> absent
//!             linearizes at the claim checkpoint; the rename makes `absent` irreversible
//! projection  absent -> building -> ready -> disposable         (project.rs)
//!             pins nothing at any point, so it has no recovery obligation at all
//! ```
//!
//! Recovery direction is one rule in all five: over-keep. A crash between claim and unlink
//! leaves rows for objects that still exist (a later sweep re-decides them); a crash between
//! unlink and row removal leaves rows for objects that are gone (the next sweep drops them).
//! Neither invents a root, and neither deletes from an incomplete census.
//!
//! ## The device-copy read hold
//!
//! `read`-kind holds are the lease-until-CUDA-event of tfs-005 and are first-class here.
//! Their interaction with quarantine and reap is fully determined by two facts already
//! stated: the mark runs inside the write transaction, so an object under such a hold can
//! never enter a claim set; and the hold is reaped only on PROVEN owner death, so a copy
//! still in flight cannot have its source collected — the process owning the DMA is by
//! definition alive and by definition still holding its lock. A caller that drops the lease
//! without releasing it keeps the row until the reaper proves it dead, which is over-keeping
//! and therefore safe.

use std::fs;
use std::path::PathBuf;

use crate::checkpoint::{self, Reached};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::ObjectRef;
use crate::ingest::transaction as ingest_txn;
use crate::meta::{now_secs, Meta, State};
use crate::store::Store;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

/// The ONE hash algorithm the address namespace admits (tensorfs.md §7). Every algorithm in
/// the namespace must be in the sweep set, fence-tested both directions: an unswept prefix
/// would be unreachable AND uncollectable — billed forever. A migration is an era hard-cut,
/// never coexistence, which is why this is a list of exactly one.
pub const SWEEP_SET: &[&str] = &["sha256"];

// ---------------------------------------------------------------- removal

/// A removal RESULT, never a bool. A failed open is not evidence about content, so the
/// three answers are kept apart: we took it, it was already gone, or someone else is on it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Removal {
    Taken { bytes: u64 },
    Absent,
    Deferred { why: String },
}

/// Name-first removal: rename out of the object namespace, then unlink. After the rename the
/// object can never satisfy another read, and the unlink is pure cleanup that a later reap
/// can finish.
pub fn remove_object(store: &Store, hex: &str, generation: u64) -> Result<Removal> {
    let from = store.object_path(hex);
    let md = match fs::symlink_metadata(&from) {
        Ok(md) => md,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Removal::Absent),
        Err(e) => return Ok(Removal::Deferred { why: e.to_string() }),
    };
    let dir = store.root().join("gc").join(generation.to_string());
    fs::create_dir_all(&dir).map_err(|e| io("mkdir gc", e))?;
    let to = dir.join(hex);
    match fs::rename(&from, &to) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Removal::Absent),
        Err(e) => return Ok(Removal::Deferred { why: e.to_string() }),
    }
    // The name is ours now. Anything that fails past this point leaves scratch, not a live
    // object, and the next reap finishes it.
    let _ = fs::remove_file(&to);
    let _ = fs::remove_file(
        store
            .root()
            .join("verified")
            .join(&hex[0..2])
            .join(&hex[2..4])
            .join(hex),
    );
    Ok(Removal::Taken { bytes: md.len() })
}

/// Finish removals a crash interrupted. Everything under `gc/` was already renamed out of
/// the object namespace, so it is unreachable by construction — deleting it can never lose
/// a live object, which is why recovery is allowed to be this blunt.
pub fn reap_removed(store: &Store) -> Result<(usize, u64)> {
    let base = store.root().join("gc");
    let (mut n, mut b) = (0usize, 0u64);
    let rd = match fs::read_dir(&base) {
        Ok(rd) => rd,
        Err(_) => return Ok((0, 0)),
    };
    for gen in rd.flatten() {
        if let Ok(inner) = fs::read_dir(gen.path()) {
            for e in inner.flatten() {
                b += e.metadata().map(|m| m.len()).unwrap_or(0);
                if fs::remove_file(e.path()).is_ok() {
                    n += 1;
                }
            }
        }
        let _ = fs::remove_dir(gen.path());
    }
    Ok((n, b))
}

// ---------------------------------------------------------------- the census

/// Why one object is live. The reason is the point: usage accounting and every refusal
/// message need to name what is holding the bytes, not merely that something is.
#[derive(Debug, Clone)]
pub struct Live {
    pub object: String,
    pub length: u64,
    pub kind: &'static str,
    pub via: String,
}

#[derive(Debug, Default)]
pub struct Census {
    pub live: Vec<Live>,
    pub roots: usize,
    pub holds: usize,
    pub ingest_roots: usize,
    /// snapshot -> its distinct object set, for usage accounting. Covers every readable
    /// snapshot, tombstoned or not: "how many bytes does removing this reclaim" has to be
    /// answerable before the explicit `tfs rm`.
    pub per_snapshot: Vec<(String, Vec<ObjectRef>)>,
    /// TOMBSTONED snapshots this store can no longer read — collection already underway.
    /// A LIVE snapshot that does not read refuses the census instead.
    pub unreadable: Vec<String>,
}

impl Census {
    pub fn is_live(&self, hex: &str) -> bool {
        self.live.iter().any(|l| l.object == hex)
    }
    /// The whole live set as one lookup structure — the sweep asks per swept object, and a
    /// linear `is_live` there is O(n²) at store cardinality.
    pub fn live_set(&self) -> std::collections::HashSet<&str> {
        self.live.iter().map(|l| l.object.as_str()).collect()
    }
    pub fn distinct(&self) -> usize {
        let mut v: Vec<&str> = self.live.iter().map(|l| l.object.as_str()).collect();
        v.sort();
        v.dedup();
        v.len()
    }
    pub fn bytes(&self) -> u64 {
        let mut v: Vec<(&str, u64)> = self
            .live
            .iter()
            .map(|l| (l.object.as_str(), l.length))
            .collect();
        v.sort();
        v.dedup_by(|a, b| a.0 == b.0);
        v.iter().map(|(_, n)| n).sum()
    }
}

/// The MARK phase. EVERY KNOWN NON-TOMBSTONED SNAPSHOT IS LIVE (tfs-020): pinned or not,
/// published or not, its whole transitive walk is marked. Tombstoned snapshots are walked
/// only for accounting; their unshared objects are what the sweep collects.
///
/// A live snapshot that cannot be loaded or walked fails the WHOLE census — pinned or
/// merely known: a partial mark is exactly the state from which deleting is wrong, so it
/// refuses instead. Only a TOMBSTONED snapshot that no longer reads is skipped (recorded
/// in `unreadable`): its bytes are collectable, so an unreadable manifest there is the
/// collection already underway, not a hole in the census.
pub fn mark(store: &Store, state: &State) -> Result<Census> {
    let mut c = Census {
        roots: state.roots.len(),
        holds: state.holds.len(),
        ..Default::default()
    };

    let pin_sites = state.pinned_snapshots();
    let mut subjects: Vec<(String, Option<String>)> = pin_sites
        .iter()
        .map(|(s, via)| (s.clone(), Some(via.clone())))
        .collect();
    for row in &state.snapshots {
        if !subjects.iter().any(|(s, _)| *s == row.snapshot) {
            subjects.push((
                row.snapshot.clone(),
                if row.tombstoned {
                    None
                } else {
                    Some("known snapshot (live by default, not tombstoned)".to_string())
                },
            ));
        }
    }

    for (snap_hex, via) in subjects {
        let len = fs::metadata(store.object_path(&snap_hex))
            .map(|m| m.len())
            .unwrap_or(0);
        let r = ObjectRef {
            sha256: snap_hex.clone(),
            length: len,
        };
        let walked = checkpoint::load_manifest(store, &r).and_then(|m| {
            let w = checkpoint::walk(store, &m)?;
            Ok(w)
        });
        let w = match (walked, &via) {
            (Ok(w), _) => w,
            // A LIVE snapshot this store cannot read — a pin, or any known non-tombstoned
            // row — fails the whole census: a partial mark is exactly the state from which
            // deleting is wrong.
            (Err(e), Some(via)) => {
                return refuse(
                    Code::CENSUS_INCOMPLETE,
                    format!(
                        "{via} keeps sha256:{snap_hex} live, and this store cannot read it \
                         as a snapshot ({e}) — refusing to delete from an incomplete \
                         census; `tfs rm` the snapshot if it is truly unwanted"
                    ),
                )
            }
            // A TOMBSTONED row that no longer reads: collection already underway.
            (Err(_), None) => {
                c.unreadable.push(snap_hex.clone());
                continue;
            }
        };
        if let Some(via) = &via {
            c.live.push(Live {
                object: snap_hex.clone(),
                length: len,
                kind: "snapshot",
                via: via.clone(),
            });
            for Reached { kind, obj } in &w.objects {
                c.live.push(Live {
                    object: obj.sha256.clone(),
                    length: obj.length,
                    kind,
                    via: via.clone(),
                });
            }
        }
        let mut objs: Vec<ObjectRef> = w.distinct().into_iter().cloned().collect();
        objs.push(r);
        if !c.per_snapshot.iter().any(|(s, _)| *s == snap_hex) {
            c.per_snapshot.push((snap_hex, objs));
        }
    }

    // A snapshot's EVIDENCE: the ingest receipt/subject/profile chain that attests how it
    // came to exist. Nothing in the snapshot references them — they bind TO it — so they
    // are marked from the SNAPSHOT row (tfs-020): provenance lives while the snapshot
    // lives and is collected with it, never merely because a root was released.
    //
    // Evidence joins the snapshot's accounting set as well as the live set: it is part of
    // what removing the snapshot reclaims, so leaving it out would make `exclusive`
    // understate the reclaim by exactly the evidence bytes.
    for row in &state.snapshots {
        for e in &row.evidence {
            let length = fs::metadata(store.object_path(e))
                .map(|m| m.len())
                .unwrap_or(0);
            if !row.tombstoned {
                c.live.push(Live {
                    object: e.clone(),
                    length,
                    kind: "evidence",
                    via: format!("snapshot sha256:{}", row.snapshot),
                });
            }
            if let Some((_, objs)) = c.per_snapshot.iter_mut().find(|(s, _)| *s == row.snapshot) {
                if !objs.iter().any(|o| o.sha256 == *e) {
                    objs.push(ObjectRef {
                        sha256: e.clone(),
                        length,
                    });
                }
            }
        }
    }

    // Holds that name explicit objects rather than a snapshot (a partial fill).
    for h in &state.holds {
        for o in &h.objects {
            let len = fs::metadata(store.object_path(o))
                .map(|m| m.len())
                .unwrap_or(0);
            c.live.push(Live {
                object: o.clone(),
                length: len,
                kind: "hold-object",
                via: format!("hold {}", h.id),
            });
        }
    }

    // Expiring ingest transactions are pin root class (4). They live in their own plane
    // (tfs-003) and are marked from it, which is what makes "still mark from roots" true
    // even against an empty authority document.
    let now = now_secs();
    if let Ok(rd) = fs::read_dir(store.root().join("ingest")) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            let root = match ingest_txn::read_root(store.root(), &name) {
                Ok(r) => r,
                Err(_) => continue,
            };
            if root.expires_at <= now {
                continue;
            }
            c.ingest_roots += 1;
            for cand in &root.candidates {
                c.live.push(Live {
                    object: cand.sha256.clone(),
                    length: cand.length,
                    kind: "ingest-candidate",
                    via: format!("ingest root {name}"),
                });
            }
        }
    }
    Ok(c)
}

// ---------------------------------------------------------------- the sweep

#[derive(Debug, Default, Clone)]
pub struct Sweep {
    pub scanned: usize,
    pub live: usize,
    pub collected: usize,
    pub bytes: u64,
    pub deferred: usize,
    pub absent: usize,
    pub claim_generation: u64,
}

/// The whole cycle in one transaction. `limit` bounds the sweep (0 = unbounded), which
/// bounds the claim set and therefore the size of the authority document.
///
/// `allow_empty` is the SECOND half of "correct against an empty DB". The first half is that
/// an empty authority over a non-empty store REFUSES; the flag is how a genuinely empty
/// store says so out loud, once, rather than every sweep silently accepting the state that
/// would delete everything.
/// Dev-only kill points inside the cycle. `claimed` stops with the quarantine rows durable
/// and not one byte removed; `deleting` stops after the first object is taken. Both wait to
/// be killed from OUTSIDE — a real SIGKILL to a real process, never a simulation.
pub const SWEEP_FAULTS: &[&str] = &["claimed", "deleting"];

fn fault(stage: &str, at: Option<&str>) {
    if at != Some(stage) {
        return;
    }
    eprintln!("FAULT-READY {stage} pid={}", std::process::id());
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
}

pub fn sweep(
    store: &Store,
    meta: &Meta,
    limit: usize,
    dry_run: bool,
    allow_empty: bool,
) -> Result<(Sweep, Census)> {
    sweep_at(store, meta, limit, dry_run, allow_empty, false, None)
}

/// Ingest transaction roots that are OPEN — unexpired. An expired root is reapable debris,
/// pins nothing, and does not block the sweep.
fn open_ingest_roots(store: &Store) -> Vec<String> {
    let now = now_secs();
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(store.root().join("ingest")) {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if let Ok(r) = ingest_txn::read_root(store.root(), &name) {
                if r.expires_at > now {
                    out.push(name);
                }
            }
        }
    }
    out.sort();
    out
}

pub fn sweep_at(
    store: &Store,
    meta: &Meta,
    limit: usize,
    dry_run: bool,
    allow_empty: bool,
    wait: bool,
    at: Option<&str>,
) -> Result<(Sweep, Census)> {
    // The sweep set fence, in the direction that can actually be violated: an algorithm
    // directory the namespace admits but this sweep does not know is refused loudly. An
    // unswept prefix would be both unreachable and uncollectable.
    for e in fs::read_dir(store.root().join("objects"))
        .map_err(|e| io("read objects/", e))?
        .flatten()
    {
        let n = e.file_name().to_string_lossy().to_string();
        if !SWEEP_SET.contains(&n.as_str()) {
            return refuse(
                Code::SWEEP_SET_INCOMPLETE,
                format!(
                    "objects/{n}/ is an address prefix outside the sweep set {SWEEP_SET:?} — \
                     it would be unreachable AND uncollectable; a hash migration is an era \
                     hard-cut, never coexistence"
                ),
            );
        }
    }

    // STORE-WIDE EXCLUSIVITY, checked at sweep start under the one lock scope (tfs-020):
    // an open ingest root or a held lease refuses the sweep (or waits, by flag). Opening a
    // root and registering a hold both pass through this same lock, and the sweep keeps it
    // for the whole cycle — so a border run arriving mid-sweep BLOCKS THE BORDER, not the
    // bytes, and no writer can appear between this check and the deletions.
    let mut t = loop {
        let t = meta.txn()?;
        let holds: Vec<String> = t.state.holds.iter().map(|h| h.id.clone()).collect();
        let roots = open_ingest_roots(store);
        if holds.is_empty() && roots.is_empty() {
            break t;
        }
        if !wait {
            return refuse(
                Code::STORE_BUSY,
                format!(
                    "the sweep is EXCLUSIVE and the store has writers: {} open ingest \
                     root(s) {roots:?}, {} held lease(s) {holds:?} — GC never runs beside \
                     a writer; retry after they finish, or pass --wait",
                    roots.len(),
                    holds.len()
                ),
            );
        }
        drop(t);
        std::thread::sleep(std::time::Duration::from_millis(250));
    };
    let all = store.objects()?;

    if meta.is_empty() && !all.is_empty() && !allow_empty {
        // Both claims, in one refusal: we do NOT delete, and we still say what the roots
        // that DO exist reach, so an operator can see the census was not merely skipped.
        let c = mark(store, &t.state)?;
        return refuse(
            Code::CENSUS_INCOMPLETE,
            format!(
                "the metadata authority has never been written and this store holds {} \
                 objects — refusing to delete. Marking from the roots that do exist reached \
                 {} objects via {} ingest roots; pass --allow-empty only for a store that \
                 genuinely has no roots",
                all.len(),
                c.distinct(),
                c.ingest_roots
            ),
        );
    }

    let census = mark(store, &t.state)?;
    let claim_generation = t.state.generation + 1;
    let mut s = Sweep {
        claim_generation,
        ..Default::default()
    };

    let live = census.live_set();
    let mut claimed: Vec<String> = Vec::new();
    for hex in &all {
        s.scanned += 1;
        if live.contains(hex.as_str()) {
            s.live += 1;
            continue;
        }
        if limit > 0 && claimed.len() >= limit {
            break;
        }
        claimed.push(hex.clone());
    }

    if dry_run {
        s.collected = claimed.len();
        s.bytes = claimed
            .iter()
            .filter_map(|h| fs::metadata(store.object_path(h)).ok().map(|m| m.len()))
            .sum();
        return Ok((s, census));
    }

    // CLAIM, durably, WITHOUT dropping the lock: the quarantine rows record what this
    // generation intends to delete, so a crash between claim and unlink leaves evidence
    // rather than a guess — and because the lock is still ours, no root or hold can be
    // registered between the census and the deletion. That is the whole scrub-in-the-same-
    // transaction property, and it is why the race proof is deterministic.
    for hex in &claimed {
        t.state.quarantine.push(crate::meta::QRow {
            object: hex.clone(),
            claim_generation,
        });
    }
    t.state.quarantine.sort_by(|a, b| a.object.cmp(&b.object));
    t.state.quarantine.dedup_by(|a, b| a.object == b.object);
    let generation = t.checkpoint()?;
    fault("claimed", at);

    let mut done: Vec<String> = Vec::new();
    for hex in &claimed {
        fault("deleting", at.filter(|_| !done.is_empty()));
        match remove_object(store, hex, generation)? {
            Removal::Taken { bytes } => {
                s.collected += 1;
                s.bytes += bytes;
                done.push(hex.clone());
            }
            Removal::Absent => {
                s.absent += 1;
                done.push(hex.clone());
            }
            Removal::Deferred { .. } => s.deferred += 1,
        }
    }

    // The row dies WITH its object and AFTER it. A crash before this leaves a quarantine row
    // for an object already gone, which over-keeps the index — the permitted direction, and
    // the reason the retain also drops rows whose object is simply no longer there: without
    // that, every interrupted sweep leaks one row forever and "over-keep" stops being
    // bounded. A row for an absent object is a row whose object already died.
    t.state
        .quarantine
        .retain(|q| !done.contains(&q.object) && store.contains(&q.object));
    // THE TOMBSTONED STATE IS COLLECTED TOO: a tombstoned row whose manifest object is
    // gone has nothing left to account for and dies here — after its objects, never
    // before. A deferred removal keeps the row for the next sweep (over-keeping).
    t.state
        .snapshots
        .retain(|row| !row.tombstoned || store.contains(&row.snapshot));
    t.commit()?;
    Ok((s, census))
}

// ---------------------------------------------------------------- usage accounting

#[derive(Debug, Clone)]
pub struct Usage {
    pub snapshot: String,
    pub logical: u64,
    pub resident: u64,
    pub exclusive: u64,
}

/// logical / resident / exclusive, as a by-product of the mark walk.
///
/// `exclusive` carries a numeric proof obligation: deleting this snapshot's last root must
/// reclaim exactly these bytes. A live workspace head makes nothing exclusive — `exclusive`
/// is the freed-if-deleted number or it is nothing.
pub fn usage(store: &Store, census: &Census) -> Vec<Usage> {
    use std::collections::{HashMap, HashSet};
    // Precomputed once: how many snapshots reach each object, and which objects a hold
    // names. The per-object inner scans made this O(S²·N²) at design cardinality.
    let mut owners: HashMap<&str, usize> = HashMap::new();
    for (_, objs) in &census.per_snapshot {
        let mut seen: HashSet<&str> = HashSet::with_capacity(objs.len());
        for o in objs {
            if seen.insert(o.sha256.as_str()) {
                *owners.entry(o.sha256.as_str()).or_insert(0) += 1;
            }
        }
    }
    let held: HashSet<&str> = census
        .live
        .iter()
        .filter(|l| l.kind == "hold-object")
        .map(|l| l.object.as_str())
        .collect();

    let mut out = Vec::new();
    for (snap, objs) in &census.per_snapshot {
        let mut u = Usage {
            snapshot: snap.clone(),
            logical: 0,
            resident: 0,
            exclusive: 0,
        };
        for o in objs {
            u.logical += o.length;
            let resident = store.contains(&o.sha256);
            if resident {
                u.resident += o.length;
            }
            // Exclusive = reached by THIS snapshot and by no other pinned snapshot. Holds
            // that name a snapshot are already among `per_snapshot`, so a leased object is
            // correctly not exclusive to anybody.
            let shared = owners.get(o.sha256.as_str()).copied().unwrap_or(0) > 1;
            if !shared && !held.contains(o.sha256.as_str()) && resident {
                u.exclusive += o.length;
            }
        }
        out.push(u);
    }
    out
}

// ---------------------------------------------------------------- reconciliation

/// The supervisor holds one `roots` row per live binding. Binding churn moves ROOTS, never
/// bytes: a removed binding drops its root and the bytes stay until a later mark, and a
/// restart re-reconciles the authority against the coordinator's binding set rather than
/// trusting whatever survived.
pub fn reconcile_bindings(
    meta: &Meta,
    bindings: &[(String, String)],
    kind: &str,
) -> Result<(Vec<String>, Vec<String>)> {
    let mut t = meta.txn()?;
    let (mut added, mut removed) = (Vec::new(), Vec::new());
    let names: Vec<&str> = bindings.iter().map(|(n, _)| n.as_str()).collect();
    let stale: Vec<String> = t
        .state
        .roots
        .iter()
        .filter(|r| r.kind == kind && !names.contains(&r.name.as_str()))
        .map(|r| r.name.clone())
        .collect();
    for n in stale {
        t.release_root(&n)?;
        removed.push(n);
    }
    for (name, snapshot) in bindings {
        let current = t.state.root(name).map(|r| r.snapshot.clone());
        if current.as_deref() != Some(snapshot.as_str()) {
            t.register_root(name, snapshot, kind, None)?;
            added.push(name.clone());
        }
    }
    t.commit()?;
    Ok((added, removed))
}

/// Debug-only rendering of the roots as files. This directory is CACHE with zero authority:
/// nothing reads it back, GC never consults it, and a projection that disagrees with the
/// authority has no effect on what is collected or served.
pub fn render_refs(store: &Store, state: &State) -> Result<PathBuf> {
    let dir = store.root().join("refs");
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).map_err(|e| io("mkdir refs", e))?;
    for r in &state.roots {
        fs::write(dir.join(&r.name), format!("sha256:{}\n", r.snapshot))
            .map_err(|e| io("write ref", e))?;
    }
    Ok(dir)
}
