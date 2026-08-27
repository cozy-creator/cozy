//! The local metadata plane — the **SOLE local root authority** (tensorfs.md §2, tfs-006).
//!
//! ```text
//! <root>/meta/LOCK              the exclusive advisory lock: one writer, serialized transitions
//! <root>/meta/state.json        the authority document — snapshots / roots / holds / quarantine
//! <root>/meta/holds/<id>.lock   one owner-liveness lock per hold
//! ```
//!
//! **Storage divergence, recorded** (tensorfs.md §2 and this issue's opening both name a
//! local Turso primary at `metadata.turso`): the ENGINE here is one canonical document
//! committed by fsync-rename under an OS advisory lock, not libSQL. The invariant the audit
//! actually bought — ONE authority no second writer can contradict, one transaction per
//! transition — is what is preserved, and it is preserved exactly:
//!
//! * one transaction = one `rename(2)` of one document that carries all four record kinds,
//!   which is strictly more atomic than a multi-row commit needs to be;
//! * serialized writes are the launch baseline the design asks for, and an exclusive
//!   `flock` IS that baseline;
//! * `tensorfs-core` takes no external crate and no `unsafe`, so a database would have to
//!   live OUTSIDE core — which would put the root authority in a different crate from the
//!   GC that depends on it, a worse layering than the engine buys back.
//!
//! Hold liveness is the reason the lock is not merely convenient. A local hold is reapable
//! **exactly when its owner is provably dead**, with no wall-clock grace: the owner holds an
//! exclusive lock on its own hold file for the hold's lifetime, and the kernel drops that
//! lock when the process dies — including under `SIGKILL`. A reaper that can take the lock
//! has PROOF, not a timeout. (v1's durable leases had no owner-liveness contract, so a
//! SIGKILL could strand a permanent GC root.)
//!
//! Door: libSQL lands as a `tensorfs-meta` crate above core when the local plane needs a
//! QUERY surface, replication, or a row count that outgrows one rewritable document — none
//! of which local GC requires. Vanilla SQLite never (it is already in the dependency fence).

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::canon::{as_arr, Fields, Value};
use crate::err::{refuse, Code, Refusal, Result};
use crate::ids::{ascii_name, check_format, hex64, Doc};
use crate::limits;
use crate::store::Store;

fn io(what: impl AsRef<str>, e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("{}: {e}", what.as_ref()),
    }
}

/// A partial hold names its objects; a whole-snapshot hold names its snapshot and lets the
/// mark walk reach the transitive graph. The cap keeps one document one document.
pub const MAX_HOLD_OBJECTS: usize = 4096;

pub fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The owner's boot epoch. Two processes with the same pid across a reboot are different
/// owners, and this is the fact that says so. Absent (non-Linux) reads as 0 — the advisory
/// lock is the PROOF of liveness; the epoch is recorded evidence beside it.
pub fn boot_epoch() -> u64 {
    let s = match fs::read_to_string("/proc/stat") {
        Ok(s) => s,
        Err(_) => return 0,
    };
    s.lines()
        .find_map(|l| l.strip_prefix("btime "))
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------- the four record kinds

/// A KNOWN immutable manifest. Under the tfs-020 rule a known row IS liveness unless it is
/// tombstoned: GC never infers what is alive, so a snapshot's bytes become collectable ONLY
/// through the explicit tombstone (`tfs rm` / the worker-protocol purge path). `source_url`
/// is PROVENANCE and nothing more — it authorizes nothing and evicts nothing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnapshotRow {
    pub snapshot: String,
    pub known_at: u64,
    /// The explicit act. True means the sweep may collect this snapshot's unshared objects
    /// and then drop the row; false means LIVE, whatever pins do or do not exist.
    pub tombstoned: bool,
    pub source_url: String,
    pub last_used: u64,
    /// The objects this snapshot keeps alive BESIDES its transitive graph: the ingest
    /// receipt/stamp/subject/profile/recipe chain attesting how it came to exist. A
    /// per-SNAPSHOT fact (tfs-020): it lives while the snapshot lives and is collected
    /// with it — releasing a root must not collect the provenance of a warm artifact.
    pub evidence: Vec<String>,
}

/// A durable named pin: one per live binding, per workspace/pull. Provenance evidence
/// lives on the SNAPSHOT row, not here — a root is a pin, and releasing a pin must not
/// collect the provenance of an artifact that stays warm (tfs-020).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RootRow {
    pub name: String,
    pub snapshot: String,
    pub kind: String,
    pub created: u64,
}

/// Process/request/fill/ingest liveness, including the lease-until-CUDA-event read hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HoldRow {
    pub id: String,
    pub kind: String,
    pub snapshot: String,
    pub objects: Vec<String>,
    pub owner_pid: u64,
    pub boot_epoch: u64,
    pub acquired: u64,
}

/// An object unrooted at a named mark generation, claimed for deletion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QRow {
    pub object: String,
    pub claim_generation: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct State {
    /// Bumped on every committed transaction. It is the expected-generation token of a root
    /// swap AND the `claim_generation` a GC deletion is fenced by — one counter, two jobs.
    pub generation: u64,
    pub snapshots: Vec<SnapshotRow>,
    pub roots: Vec<RootRow>,
    pub holds: Vec<HoldRow>,
    pub quarantine: Vec<QRow>,
}

fn sorted<T, K: Ord>(v: &[T], key: impl Fn(&T) -> K) -> bool {
    v.windows(2).all(|w| key(&w[0]) < key(&w[1]))
}

impl Doc for State {
    const FORMAT: &'static str = "tensorfs.meta/1";
    const MAX_BYTES: usize = limits::DOC_MAX_BYTES;

    fn from_value(v: &Value) -> Result<Self> {
        let mut f = Fields::new("State", v)?;
        check_format("State", f.req_str("format")?, Self::FORMAT)?;
        let generation = f.req_uint("generation")?;

        let mut holds = Vec::new();
        for h in as_arr("State", "holds", f.req("holds")?)? {
            let mut hf = Fields::new("State.hold", h)?;
            let acquired = hf.req_uint("acquired")?;
            let boot_epoch = hf.req_uint("boot_epoch")?;
            let id = hf.req_str("id")?.to_string();
            let kind = hf.req_str("kind")?.to_string();
            let mut objects = Vec::new();
            for o in as_arr("State.hold", "objects", hf.req("objects")?)? {
                objects.push(hex64(
                    "State.hold.object",
                    crate::canon::as_str("State.hold", "object", o)?,
                )?);
            }
            let owner_pid = hf.req_uint("owner_pid")?;
            let snapshot = hf.req_str("snapshot")?.to_string();
            hf.done()?;
            ascii_name("State.hold.id", &id, limits::MAX_NAME_BYTES)?;
            ascii_name("State.hold.kind", &kind, limits::MAX_NAME_BYTES)?;
            if objects.len() > MAX_HOLD_OBJECTS {
                return refuse(
                    Code::COUNT_CAP,
                    format!("hold {id:?} names {} objects, over the cap", objects.len()),
                );
            }
            if !snapshot.is_empty() {
                hex64("State.hold.snapshot", &snapshot)?;
            }
            holds.push(HoldRow {
                id,
                kind,
                snapshot,
                objects,
                owner_pid,
                boot_epoch,
                acquired,
            });
        }

        let mut quarantine = Vec::new();
        for q in as_arr("State", "quarantine", f.req("quarantine")?)? {
            let mut qf = Fields::new("State.quarantine", q)?;
            let claim_generation = qf.req_uint("claim_generation")?;
            let object = hex64("State.quarantine.object", qf.req_str("object")?)?;
            qf.done()?;
            quarantine.push(QRow {
                object,
                claim_generation,
            });
        }

        let mut roots = Vec::new();
        for r in as_arr("State", "roots", f.req("roots")?)? {
            let mut rf = Fields::new("State.root", r)?;
            let created = rf.req_uint("created")?;
            let kind = rf.req_str("kind")?.to_string();
            let name = rf.req_str("name")?.to_string();
            let snapshot = hex64("State.root.snapshot", rf.req_str("snapshot")?)?;
            rf.done()?;
            ascii_name("State.root.name", &name, limits::MAX_NAME_BYTES)?;
            ascii_name("State.root.kind", &kind, limits::MAX_NAME_BYTES)?;
            roots.push(RootRow {
                name,
                snapshot,
                kind,
                created,
            });
        }

        let mut snapshots = Vec::new();
        for s in as_arr("State", "snapshots", f.req("snapshots")?)? {
            let mut sf = Fields::new("State.snapshot", s)?;
            let mut evidence = Vec::new();
            for e in as_arr("State.snapshot", "evidence", sf.req("evidence")?)? {
                evidence.push(hex64(
                    "State.snapshot.evidence",
                    crate::canon::as_str("State.snapshot", "evidence", e)?,
                )?);
            }
            let known_at = sf.req_uint("known_at")?;
            let last_used = sf.req_uint("last_used")?;
            let snapshot = hex64("State.snapshot.snapshot", sf.req_str("snapshot")?)?;
            let source_url = sf.req_str("source_url")?.to_string();
            let tombstoned = matches!(sf.req("tombstoned")?, Value::Bool(true));
            sf.done()?;
            if source_url.len() > limits::MAX_KEY_BYTES {
                return refuse(Code::SIZE_CAP, "State.snapshot.source_url over cap");
            }
            if evidence.len() > MAX_HOLD_OBJECTS {
                return refuse(
                    Code::COUNT_CAP,
                    format!("snapshot {snapshot} carries too much evidence"),
                );
            }
            snapshots.push(SnapshotRow {
                snapshot,
                known_at,
                tombstoned,
                source_url,
                last_used,
                evidence,
            });
        }
        f.done()?;

        // Deterministic order is part of the document, not a convention: an unsorted twin
        // would round-trip to different bytes and is refused before it can be believed.
        if !sorted(&roots, |r| r.name.clone())
            || !sorted(&holds, |h| h.id.clone())
            || !sorted(&quarantine, |q| q.object.clone())
            || !sorted(&snapshots, |s| s.snapshot.clone())
        {
            return refuse(
                Code::SORT_ORDER,
                "State records must be strictly ascending and unique within each kind",
            );
        }
        Ok(State {
            generation,
            snapshots,
            roots,
            holds,
            quarantine,
        })
    }

    fn to_value(&self) -> Value {
        Value::obj(vec![
            ("format", Value::str(Self::FORMAT)),
            ("generation", Value::uint(self.generation)),
            (
                "holds",
                Value::arr(
                    self.holds
                        .iter()
                        .map(|h| {
                            Value::obj(vec![
                                ("acquired", Value::uint(h.acquired)),
                                ("boot_epoch", Value::uint(h.boot_epoch)),
                                ("id", Value::str(h.id.clone())),
                                ("kind", Value::str(h.kind.clone())),
                                (
                                    "objects",
                                    Value::arr(
                                        h.objects.iter().map(|o| Value::str(o.clone())).collect(),
                                    ),
                                ),
                                ("owner_pid", Value::uint(h.owner_pid)),
                                ("snapshot", Value::str(h.snapshot.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "quarantine",
                Value::arr(
                    self.quarantine
                        .iter()
                        .map(|q| {
                            Value::obj(vec![
                                ("claim_generation", Value::uint(q.claim_generation)),
                                ("object", Value::str(q.object.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "roots",
                Value::arr(
                    self.roots
                        .iter()
                        .map(|r| {
                            Value::obj(vec![
                                ("created", Value::uint(r.created)),
                                ("kind", Value::str(r.kind.clone())),
                                ("name", Value::str(r.name.clone())),
                                ("snapshot", Value::str(r.snapshot.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            (
                "snapshots",
                Value::arr(
                    self.snapshots
                        .iter()
                        .map(|s| {
                            Value::obj(vec![
                                (
                                    "evidence",
                                    Value::arr(
                                        s.evidence.iter().map(|e| Value::str(e.clone())).collect(),
                                    ),
                                ),
                                ("known_at", Value::uint(s.known_at)),
                                ("last_used", Value::uint(s.last_used)),
                                ("snapshot", Value::str(s.snapshot.clone())),
                                ("source_url", Value::str(s.source_url.clone())),
                                ("tombstoned", Value::Bool(s.tombstoned)),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

impl State {
    fn normalize(&mut self) {
        self.roots.sort_by(|a, b| a.name.cmp(&b.name));
        self.holds.sort_by(|a, b| a.id.cmp(&b.id));
        self.quarantine.sort_by(|a, b| a.object.cmp(&b.object));
        self.snapshots.sort_by(|a, b| a.snapshot.cmp(&b.snapshot));
    }
    pub fn root(&self, name: &str) -> Option<&RootRow> {
        self.roots.iter().find(|r| r.name == name)
    }
    pub fn snapshot(&self, id: &str) -> Option<&SnapshotRow> {
        self.snapshots.iter().find(|s| s.snapshot == id)
    }
    /// Every snapshot a live root or hold names — the mark walk's entry set.
    pub fn pinned_snapshots(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .roots
            .iter()
            .map(|r| (r.snapshot.clone(), format!("root {}", r.name)))
            .chain(
                self.holds
                    .iter()
                    .filter(|h| !h.snapshot.is_empty())
                    .map(|h| (h.snapshot.clone(), format!("hold {}", h.id))),
            )
            .collect();
        v.sort();
        v.dedup_by(|a, b| a.0 == b.0);
        v
    }
}

// ---------------------------------------------------------------- the plane

#[derive(Debug, Clone)]
pub struct Meta {
    dir: PathBuf,
}

impl Meta {
    pub fn open(store: &Store) -> Result<Meta> {
        let dir = store.root().join("meta");
        fs::create_dir_all(dir.join("holds")).map_err(|e| io("mkdir meta", e))?;
        Ok(Meta { dir })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
    fn state_path(&self) -> PathBuf {
        self.dir.join("state.json")
    }
    fn hold_lock_path(&self, id: &str) -> PathBuf {
        self.dir.join("holds").join(format!("{id}.lock"))
    }

    /// Read the census. No lock: a commit is one `rename`, so a reader never sees a torn
    /// document. An UNPARSABLE document is a refusal, never an empty state — a sweep that
    /// cannot read its liveness census must not delete (tensorfs.md §2).
    pub fn read(&self) -> Result<State> {
        match fs::read(self.state_path()) {
            Ok(b) => State::parse(&b),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(State::default()),
            Err(e) => Err(io("read metadata state", e)),
        }
    }

    /// True when the authority document has never been written — an EMPTY census, which is
    /// a different fact from an unreadable one and is treated differently by the sweep.
    pub fn is_empty(&self) -> bool {
        !self.state_path().is_file()
    }

    /// Begin the one write transaction. Blocks until the exclusive lock is ours, then reads
    /// the state under it — so read-modify-write is serialized against every other writer.
    pub fn txn(&self) -> Result<Txn<'_>> {
        let p = self.dir.join("LOCK");
        let lock = File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&p)
            .map_err(|e| io(format!("open {}", p.display()), e))?;
        lock.lock().map_err(|e| io("acquire metadata lock", e))?;
        let state = self.read()?;
        Ok(Txn {
            meta: self,
            _lock: lock,
            state,
        })
    }
}

/// A transaction. Dropping without `commit` writes nothing — rollback is the default.
pub struct Txn<'a> {
    meta: &'a Meta,
    _lock: File,
    pub state: State,
}

impl Txn<'_> {
    /// Commit: bump the generation, then fsync-rename the whole document. This rename is
    /// the visibility linearization point for every record kind at once.
    pub fn commit(mut self) -> Result<u64> {
        self.write()
    }

    /// Write the document and KEEP the lock. This is what lets a GC claim become durable
    /// before any byte is unlinked while still excluding every other writer: no root can be
    /// registered between the claim and the deletion, so a new complete root can never
    /// reference an object this sweep is taking.
    pub fn checkpoint(&mut self) -> Result<u64> {
        self.write()
    }

    fn write(&mut self) -> Result<u64> {
        self.state.generation += 1;
        self.state.normalize();
        let bytes = self.state.canonical_bytes();
        let p = self.meta.state_path();
        let tmp = self
            .meta
            .dir
            .join(format!(".state.{}.{}", std::process::id(), now_secs()));
        let mut f = File::create(&tmp).map_err(|e| io("create state temp", e))?;
        f.write_all(&bytes).map_err(|e| io("write state", e))?;
        f.sync_all().map_err(|e| io("fsync state", e))?;
        drop(f);
        fs::rename(&tmp, &p).map_err(|e| io("rename state", e))?;
        File::open(&self.meta.dir)
            .and_then(|d| d.sync_all())
            .map_err(|e| io("fsync meta dir", e))?;
        Ok(self.state.generation)
    }

    /// Root swap with an EXPECTED generation. Passing `None` accepts any; passing the
    /// generation you read refuses if anything committed in between.
    pub fn register_root(
        &mut self,
        name: &str,
        snapshot: &str,
        kind: &str,
        expected: Option<u64>,
    ) -> Result<()> {
        ascii_name("root name", name, limits::MAX_NAME_BYTES)?;
        ascii_name("root kind", kind, limits::MAX_NAME_BYTES)?;
        hex64("root snapshot", snapshot)?;
        if let Some(g) = expected {
            if g != self.state.generation {
                return refuse(
                    Code::ROOT_GENERATION,
                    format!(
                        "root {name:?} expected metadata generation {g}, the authority is at {}",
                        self.state.generation
                    ),
                );
            }
        }
        let now = now_secs();
        match self.state.roots.iter_mut().find(|r| r.name == name) {
            Some(r) => {
                r.snapshot = snapshot.to_string();
                r.kind = kind.to_string();
                r.created = now;
            }
            None => self.state.roots.push(RootRow {
                name: name.to_string(),
                snapshot: snapshot.to_string(),
                kind: kind.to_string(),
                created: now,
            }),
        }
        match self
            .state
            .snapshots
            .iter_mut()
            .find(|s| s.snapshot == snapshot)
        {
            // Registering a root over a tombstoned row is the explicit act that
            // un-tombstones it — an operator re-pinning wins over a pending collection.
            Some(s) => s.tombstoned = false,
            None => self.state.snapshots.push(SnapshotRow {
                snapshot: snapshot.to_string(),
                known_at: now,
                tombstoned: false,
                source_url: String::new(),
                last_used: now,
                evidence: Vec::new(),
            }),
        }
        Ok(())
    }

    /// Attach provenance evidence to a KNOWN snapshot — the ingest receipt chain. Sorted,
    /// deduped, and additive: two installs of one snapshot union their evidence.
    pub fn set_evidence(&mut self, snapshot: &str, evidence: &[String]) -> Result<()> {
        hex64("snapshot", snapshot)?;
        for e in evidence {
            hex64("snapshot evidence", e)?;
        }
        match self
            .state
            .snapshots
            .iter_mut()
            .find(|s| s.snapshot == snapshot)
        {
            Some(s) => {
                s.evidence.extend(evidence.iter().cloned());
                s.evidence.sort();
                s.evidence.dedup();
                if s.evidence.len() > MAX_HOLD_OBJECTS {
                    return refuse(
                        Code::COUNT_CAP,
                        format!("snapshot {snapshot} carries too much evidence"),
                    );
                }
                Ok(())
            }
            None => refuse(
                Code::ROOT_ABSENT,
                format!("sha256:{snapshot} is not a known snapshot — note it first"),
            ),
        }
    }

    /// THE EXPLICIT TOMBSTONE (tfs-020) — the ONLY act that makes a snapshot's bytes
    /// collectable. Refuses while any root or hold names the snapshot: a pinned artifact
    /// is released first, so deletion is always two explicit acts away from a live one.
    pub fn tombstone_snapshot(&mut self, snapshot: &str) -> Result<()> {
        hex64("snapshot", snapshot)?;
        if let Some(r) = self.state.roots.iter().find(|r| r.snapshot == snapshot) {
            return refuse(
                Code::SNAPSHOT_PINNED,
                format!(
                    "sha256:{snapshot} is pinned by root {:?} — release the root before \
                     tombstoning",
                    r.name
                ),
            );
        }
        if let Some(h) = self.state.holds.iter().find(|h| h.snapshot == snapshot) {
            return refuse(
                Code::SNAPSHOT_PINNED,
                format!("sha256:{snapshot} is under live hold {:?}", h.id),
            );
        }
        match self
            .state
            .snapshots
            .iter_mut()
            .find(|s| s.snapshot == snapshot)
        {
            Some(s) => {
                s.tombstoned = true;
                Ok(())
            }
            None => refuse(
                Code::ROOT_ABSENT,
                format!("sha256:{snapshot} is not a known snapshot — nothing to tombstone"),
            ),
        }
    }

    pub fn release_root(&mut self, name: &str) -> Result<String> {
        match self.state.roots.iter().position(|r| r.name == name) {
            Some(i) => Ok(self.state.roots.remove(i).snapshot),
            None => refuse(
                Code::ROOT_ABSENT,
                format!("no root {name:?} in the metadata authority"),
            ),
        }
    }

    pub fn note_snapshot(&mut self, snapshot: &str, source_url: &str) -> Result<()> {
        hex64("snapshot", snapshot)?;
        let now = now_secs();
        match self
            .state
            .snapshots
            .iter_mut()
            .find(|s| s.snapshot == snapshot)
        {
            Some(s) => {
                if !source_url.is_empty() {
                    s.source_url = source_url.to_string();
                }
                s.last_used = now;
            }
            None => self.state.snapshots.push(SnapshotRow {
                snapshot: snapshot.to_string(),
                known_at: now,
                tombstoned: false,
                source_url: source_url.to_string(),
                last_used: now,
                evidence: Vec::new(),
            }),
        }
        Ok(())
    }
}

// ---------------------------------------------------------------- holds

/// A registered liveness hold. The struct OWNS the owner-liveness lock file: while this
/// value is alive the kernel says the owner is alive, and when the process dies — by exit
/// or by SIGKILL — the kernel says it is dead. There is no third answer and no timeout.
#[derive(Debug)]
pub struct Hold {
    pub row: HoldRow,
    lock: Option<File>,
    released: bool,
}

impl Hold {
    pub fn id(&self) -> &str {
        &self.row.id
    }

    /// Explicit release — the ONLY silent, complete teardown. Removes the row inside one
    /// transaction, then drops the lock and the lock file.
    pub fn release(mut self, meta: &Meta) -> Result<()> {
        let mut t = meta.txn()?;
        t.state.holds.retain(|h| h.id != self.row.id);
        t.commit()?;
        self.lock = None;
        let _ = fs::remove_file(meta.hold_lock_path(&self.row.id));
        self.released = true;
        Ok(())
    }

    /// Is this hold still the authority's? A hold that vanished from the state while its
    /// owner still lives is evidence of an out-of-band act (the `steal` red arm), and every
    /// read against it must refuse rather than trust a lease the authority no longer knows.
    pub fn still_registered(&self, meta: &Meta) -> Result<()> {
        let s = meta.read()?;
        if s.holds.iter().any(|h| h.id == self.row.id) {
            return Ok(());
        }
        refuse(
            Code::LEASE_REVOKED,
            format!(
                "hold {:?} is no longer in the metadata authority — the lease it protects is \
                 not live and its bytes are not guaranteed",
                self.row.id
            ),
        )
    }
}

impl Drop for Hold {
    /// Dropping an unreleased hold does NOT quietly release it. The row stays (over-keeping
    /// is the only permitted failure direction) and the drop is announced. What the kernel
    /// does release is the lock — which is correct: this process is no longer holding
    /// anything, so a reaper may now prove the hold dead and reclaim it. Bytes are kept
    /// until then; nothing is lost, and the loud line is how a caller learns it skipped the
    /// completion fence.
    fn drop(&mut self) {
        if !self.released {
            eprintln!(
                "DEGRADED lease-dropped-unreleased hold={} kind={} — the row is KEPT for the \
                 reaper; a caller that owns a device copy must release after its fence",
                self.row.id, self.row.kind
            );
        }
    }
}

impl Meta {
    /// Register a hold BEFORE the first fetch (tensorfs.md §2, pin root class 2): GC racing
    /// a starting fill is fenced by construction, not by timing.
    pub fn acquire_hold(&self, kind: &str, snapshot: &str, objects: Vec<String>) -> Result<Hold> {
        ascii_name("hold kind", kind, limits::MAX_NAME_BYTES)?;
        if !snapshot.is_empty() {
            hex64("hold snapshot", snapshot)?;
        }
        if objects.len() > MAX_HOLD_OBJECTS {
            return refuse(
                Code::COUNT_CAP,
                format!(
                    "a hold names at most {MAX_HOLD_OBJECTS} objects; a whole-snapshot hold \
                     names its snapshot instead ({} given)",
                    objects.len()
                ),
            );
        }
        let mut objects = objects;
        for o in &objects {
            hex64("hold object", o)?;
        }
        objects.sort();
        objects.dedup();

        let id = format!(
            "{kind}-{}-{}",
            std::process::id(),
            now_secs() % 1_000_000 * 1000 + unique()
        );
        let lp = self.hold_lock_path(&id);
        let lock = File::options()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lp)
            .map_err(|e| io(format!("open hold lock {}", lp.display()), e))?;
        // Non-blocking on purpose: a fresh id must never be contended, and if it is, the id
        // is not fresh and the hold must not exist twice.
        lock.try_lock().map_err(|_| Refusal {
            code: Code::LOCK_CONTENDED,
            detail: format!("hold lock {id:?} is already held — refusing to double-register"),
        })?;

        let row = HoldRow {
            id: id.clone(),
            kind: kind.to_string(),
            snapshot: snapshot.to_string(),
            objects,
            owner_pid: std::process::id() as u64,
            boot_epoch: boot_epoch(),
            acquired: now_secs(),
        };
        let mut t = self.txn()?;
        t.state.holds.push(row.clone());
        t.commit()?;
        Ok(Hold {
            row,
            lock: Some(lock),
            released: false,
        })
    }

    /// Reap every hold whose owner is PROVABLY dead. Proof is taking the owner's lock: a
    /// live owner still holds it, and no elapsed time can substitute for that.
    pub fn reap_holds(&self) -> Result<Vec<String>> {
        let mut reaped = Vec::new();
        let mut t = self.txn()?;
        let candidates: Vec<HoldRow> = t.state.holds.clone();
        for h in candidates {
            let p = self.hold_lock_path(&h.id);
            let f = match File::options().write(true).open(&p) {
                Ok(f) => f,
                // No lock file at all: nothing can be holding it.
                Err(_) => {
                    reaped.push(h.id.clone());
                    continue;
                }
            };
            // owner alive (Err) — the hold stands, whatever its age
            if f.try_lock().is_ok() {
                let _ = f.unlock();
                drop(f);
                let _ = fs::remove_file(&p);
                reaped.push(h.id.clone());
            }
        }
        if reaped.is_empty() {
            return Ok(reaped);
        }
        t.state.holds.retain(|h| !reaped.contains(&h.id));
        t.commit()?;
        reaped.sort();
        Ok(reaped)
    }

    /// DEV-ONLY red arm: remove a hold row WITHOUT the owner-liveness proof. Nothing in the
    /// normal path does this; it exists so the "a stolen lease must refuse, not serve" claim
    /// can be fired live instead of asserted.
    pub fn steal_hold(&self, id: &str) -> Result<()> {
        let mut t = self.txn()?;
        let before = t.state.holds.len();
        t.state.holds.retain(|h| h.id != id);
        if t.state.holds.len() == before {
            return refuse(Code::ROOT_ABSENT, format!("no hold {id:?}"));
        }
        t.commit()?;
        Ok(())
    }

    /// `open_root(name)` resolves ONCE to an immutable snapshot id and takes a hold. The
    /// hold — not the name's continued existence — protects the reader from then on.
    pub fn open_root(&self, name: &str) -> Result<(String, Hold)> {
        let s = self.read()?;
        let snapshot = match s.root(name) {
            Some(r) => r.snapshot.clone(),
            None => {
                return refuse(
                    Code::ROOT_ABSENT,
                    format!("no root {name:?} in the metadata authority"),
                )
            }
        };
        let hold = self.acquire_hold("open-root", &snapshot, Vec::new())?;
        Ok((snapshot, hold))
    }
}

fn unique() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    N.fetch_add(1, Ordering::Relaxed) % 1000
}
