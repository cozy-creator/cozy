//! Durable, concurrent source downloads. Called only while the selected-source writer holds
//! its native owner fence; each object's directory lock fences every other writer of it.
//!
//! An object downloads into one sparse `bytes.part` as RUNS: contiguous verified byte ranges,
//! each with its own durable extent and SHA-256 state file. The run at offset 0 is the
//! prefix, and the object digest is only ever computed in offset order: the prefix absorbs
//! each finished run behind it by re-reading those bytes. A run is asked for with ONE ranged
//! GET from its reached byte to its end and checkpointed every `checkpoint_bytes` mid-body.
//!
//! Up to `streams` workers share every wanted object of the call. When no whole run is left
//! to start, an idle worker splits the largest range in flight in two — never below one
//! checkpoint per half — so a single large member, or the tail of a many-member source,
//! still fills every stream. A delivery URL an origin redirected to is reused for later
//! asks of the same object until the origin refuses it.
use super::{
    pull::{self, Opened},
    CredentialProvider, Deadline, Ledger, PullCancellation, SourcePolicy, FETCH_ATTEMPTS, STREAMS,
};
use crate::{
    canon::{Fields, Value},
    err::{refuse, Code, Refusal, Result},
    fetch::DeliveryGrant,
    ids::{prefixed, ObjectRef, Plain},
    sha256::{hex, Sha256},
    store::{Fault, Store, Verified},
};
use fs2::FileExt;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Read,
    os::unix::fs::{FileExt as _, MetadataExt},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex,
    },
};

#[derive(Clone, Debug)]
pub struct SourceDownload {
    /// Durable progress cadence per run, and the smallest half a split may leave.
    pub checkpoint_bytes: u64,
    /// Concurrent ranged GETs across the whole call.
    pub streams: usize,
    pub fault: Fault,
    pub cancellation: Option<PullCancellation>,
}
impl Default for SourceDownload {
    fn default() -> Self {
        Self {
            checkpoint_bytes: super::PART_BYTES,
            streams: STREAMS,
            fault: Fault::default(),
            cancellation: None,
        }
    }
}

/// Bytes of the call's distinct objects present locally (held before it, or durably
/// written by it) and their total. The present figure drops only if retained bytes are
/// discarded (an origin that stopped honouring ranges, or bytes that hashed wrong).
pub type SourceProgress<'a> = &'a (dyn Fn(u64, u64) + Sync);

/// One wanted object: its grant and the source URL it is fetched from.
pub(crate) struct SourceJob {
    pub grant: DeliveryGrant,
    pub url: String,
}

#[derive(Clone)]
struct State {
    owner: String,
    object: ObjectRef,
    start: u64,
    extent: u64,
    prefix_sha256: String,
}
impl Plain for State {
    const MAX_BYTES: usize = 4096;
    fn from_value(value: &Value) -> Result<Self> {
        let mut f = Fields::new("SourcePrefix", value)?;
        let owner = prefixed("source prefix owner", f.req_str("owner")?)?;
        let object = ObjectRef::from_value("source prefix object", f.req("object")?)?;
        let start = match f.opt("start") {
            Some(v) => crate::canon::as_uint("SourcePrefix", "start", v)?,
            None => 0,
        };
        let extent = f.req_uint("extent")?;
        let prefix_sha256 = crate::ids::hex64("source prefix digest", f.req_str("prefix_sha256")?)?;
        if f.req_str("representation")? != "identity" {
            return refuse(Code::WRONG_TYPE, "source prefix has another representation");
        }
        f.done()?;
        if start
            .checked_add(extent)
            .is_none_or(|end| end > object.length)
        {
            return refuse(Code::LENGTH_MISMATCH, "prefix exceeds object");
        }
        Ok(Self {
            owner,
            object,
            start,
            extent,
            prefix_sha256,
        })
    }
    fn to_value(&self) -> Value {
        let mut fields = vec![
            ("owner", Value::str(&self.owner)),
            ("object", self.object.to_value()),
            ("extent", Value::uint(self.extent)),
            ("prefix_sha256", Value::str(&self.prefix_sha256)),
            ("representation", Value::str("identity")),
        ];
        if self.start > 0 {
            fields.push(("start", Value::uint(self.start)));
        }
        Value::obj(fields)
    }
}
fn io(e: std::io::Error) -> Refusal {
    Refusal {
        code: Code::IO_FAILED,
        detail: format!("source prefix: {e}"),
    }
}
fn area(store: &Store) -> PathBuf {
    store.root().join("roots/source-downloads")
}
fn owner_dir(store: &Store, owner: &str) -> Result<PathBuf> {
    prefixed("prefix owner", owner)?;
    Ok(area(store).join(&owner[7..]))
}
/// The prefix keeps the original `state.json`; a run starting later is `run-<start>.json`.
fn state_name(start: u64) -> String {
    if start == 0 {
        "state".into()
    } else {
        format!("run-{start}")
    }
}
fn read_state(path: &Path) -> Result<Option<State>> {
    match File::open(path) {
        Ok(file) => {
            let mut bytes = vec![];
            file.take(State::MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)
                .map_err(io)?;
            State::parse(&bytes).map(Some)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(io(e)),
    }
}
fn hash_range(file: &File, start: u64, length: u64, hash: &mut Sha256) -> std::io::Result<bool> {
    let mut buffer = vec![0u8; (1 << 20).min(length.max(1)) as usize];
    let mut at = start;
    while at < start + length {
        let n = buffer.len().min((start + length - at) as usize);
        match file.read_at(&mut buffer[..n], at)? {
            0 => return Ok(false),
            n => {
                hash.update(&buffer[..n]);
                at += n as u64;
            }
        }
    }
    Ok(true)
}

/// Bytes `[start, start + extent)` of the object are in `bytes.part` and hash to `hash`;
/// `durable` of them are covered by the run's last checkpoint.
struct Run {
    start: u64,
    extent: u64,
    hash: Sha256,
    durable: u64,
}
impl Run {
    fn new(start: u64) -> Self {
        Self {
            start,
            extent: 0,
            hash: Sha256::new(),
            durable: 0,
        }
    }
    fn reached(&self) -> u64 {
        self.start + self.extent
    }
}

/// One object's download area: sparse bytes, one state file per run, and the lock.
struct Part {
    dir: PathBuf,
    owner: String,
    object: ObjectRef,
    file: File,
    _lock: File,
}
impl Part {
    /// Lock and verify. A writable open discards every run whose bytes no longer hash to its
    /// state and then refuses once, so the retry starts that range again; a read-only open
    /// (a donor) verifies only the prefix and refuses on corruption.
    fn open(
        store: &Store,
        owner: &str,
        object: &ObjectRef,
        writable: bool,
    ) -> Result<(Self, Vec<Run>)> {
        crate::ids::hex64("prefix object", &object.sha256)?;
        let dir = owner_dir(store, owner)?.join(&object.sha256);
        fs::create_dir_all(&dir).map_err(io)?;
        let lock = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(dir.join("lock"))
            .map_err(io)?;
        lock.try_lock_exclusive().map_err(|_| Refusal {
            code: Code::STORE_BUSY,
            detail: "source prefix has a live writer".into(),
        })?;
        let file = OpenOptions::new()
            .read(true)
            .write(writable)
            .create(writable)
            .truncate(false)
            .open(dir.join("bytes.part"))
            .map_err(io)?;
        let part = Self {
            dir,
            owner: owner.into(),
            object: object.clone(),
            file,
            _lock: lock,
        };
        let mut states = vec![];
        for entry in fs::read_dir(&part.dir).map_err(io)? {
            let name = entry.map_err(io)?.file_name();
            let Some(stem) = name.to_str().and_then(|name| name.strip_suffix(".json")) else {
                continue;
            };
            if stem != "state" && (!writable || !stem.starts_with("run-")) {
                continue;
            }
            let Some(state) = read_state(&part.dir.join(&name))? else {
                continue;
            };
            if state.owner != owner || state.object != *object || state_name(state.start) != stem {
                return refuse(
                    Code::TRANSACTION_CONFLICT,
                    "source prefix has another immutable identity",
                );
            }
            states.push(state);
        }
        states.sort_by_key(|state| state.start);
        if states.first().is_none_or(|state| state.start != 0) {
            states.insert(
                0,
                State {
                    owner: owner.into(),
                    object: object.clone(),
                    start: 0,
                    extent: 0,
                    prefix_sha256: hex(&Sha256::new().finish()),
                },
            );
        }
        let (mut runs, mut corrupt) = (Vec::<Run>::new(), false);
        for state in states {
            // A run the prefix already absorbed (stopped before its file was removed).
            if runs.last().is_some_and(|run| run.reached() > state.start) {
                part.drop_state(state.start)?;
                continue;
            }
            let mut hash = Sha256::new();
            let whole = hash_range(&part.file, state.start, state.extent, &mut hash).map_err(io)?;
            let mut run = Run {
                start: state.start,
                extent: state.extent,
                hash,
                durable: state.extent,
            };
            if !whole || hex(&run.hash.clone().finish()) != state.prefix_sha256 {
                if !writable {
                    return refuse(
                        Code::OBJECT_CORRUPT,
                        "source donor prefix failed integrity validation",
                    );
                }
                // Untrusted local bytes are never an input to a resumed completed object.
                corrupt = true;
                if run.start > 0 {
                    part.drop_state(run.start)?;
                    continue;
                }
                run = Run::new(0);
            }
            runs.push(run);
        }
        if writable {
            let end = runs.last().map_or(0, Run::reached);
            if part.file.metadata().map_err(io)?.len() > end {
                part.file.set_len(end).map_err(io)?;
            }
            part.checkpoint(&mut runs[0], &Fault::default())?;
            // Newly created ancestors are durable before any progress is acknowledged.
            for dir in part.dir.ancestors().take(4) {
                File::open(dir).and_then(|f| f.sync_all()).map_err(io)?;
            }
        }
        if corrupt {
            return refuse(Code::OBJECT_CORRUPT, "retained source prefix failed integrity validation and was discarded; retry starts this object at zero");
        }
        Ok((part, runs))
    }
    /// Data first, then the run's state by atomic rename. A crash leaves either the old or
    /// the new state, and both describe bytes already on disk.
    fn checkpoint(&self, run: &mut Run, fault: &Fault) -> Result<()> {
        fault.hit("prefix-written");
        self.file.sync_data().map_err(io)?;
        fault.hit("prefix-fsync");
        let state = State {
            owner: self.owner.clone(),
            object: self.object.clone(),
            start: run.start,
            extent: run.extent,
            prefix_sha256: hex(&run.hash.clone().finish()),
        };
        let name = state_name(run.start);
        let temp = self.dir.join(format!("{name}.tmp"));
        let mut file = File::create(&temp).map_err(io)?;
        std::io::Write::write_all(&mut file, &state.canonical_bytes())
            .and_then(|()| file.sync_data())
            .map_err(io)?;
        fs::rename(temp, self.dir.join(format!("{name}.json"))).map_err(io)?;
        run.durable = run.extent;
        fault.hit("prefix-checkpoint");
        Ok(())
    }
    fn drop_state(&self, start: u64) -> Result<()> {
        match fs::remove_file(self.dir.join(format!("{}.json", state_name(start)))) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io(e)),
            _ => Ok(()),
        }
    }
    /// Forget every run and start the object again at zero.
    fn reset(&self, starts: impl Iterator<Item = u64>) -> Result<Run> {
        for start in starts.filter(|start| *start > 0) {
            self.drop_state(start)?;
        }
        self.file.set_len(0).map_err(io)?;
        let mut prefix = Run::new(0);
        self.checkpoint(&mut prefix, &Fault::default())?;
        Ok(prefix)
    }
}

/// One run's place on the board. `run` is `None` while a worker holds it. `claimed` is set
/// only while that worker is streaming: the byte it has been granted up to, the floor below
/// which nothing may be split off.
struct Slot {
    run: Option<Run>,
    claimed: Option<u64>,
}

#[derive(Default)]
struct Entry {
    part: Option<Arc<Part>>,
    opened: bool,
    runs: BTreeMap<u64, Slot>,
    /// The origin answered this call with a 206, so the object may be split.
    ranged: bool,
    /// The origin ignored a range: fetch this object as one body from zero.
    whole: bool,
    reset_pending: bool,
    done: bool,
    landed: u64,
}
impl Entry {
    fn end(&self, start: u64, length: u64) -> u64 {
        self.runs
            .range(start + 1..)
            .next()
            .map_or(length, |(next, _)| *next)
    }
    fn held(&self) -> bool {
        self.runs.values().any(|slot| slot.run.is_none())
    }
}

struct Board {
    entries: Vec<Entry>,
    failure: Option<Refusal>,
    seq: u64,
}

enum Task {
    Open(usize),
    Fetch(usize, Arc<Part>, Run),
    Absorb(usize, Arc<Part>, Run, Run),
    Finish(usize, Arc<Part>, Run),
}

struct Download<'a> {
    store: &'a Store,
    owner: &'a str,
    jobs: &'a [SourceJob],
    policy: &'a SourcePolicy,
    credentials: &'a dyn CredentialProvider,
    deadline: Deadline,
    ledger: &'a Ledger,
    options: &'a SourceDownload,
    checkpoint: u64,
    board: Mutex<Board>,
    /// Woken whenever work may have appeared: an object opened or ranged, a run checked in.
    changed: Condvar,
    /// The delivery URL each object's source URL last redirected to.
    resolved: Vec<Mutex<Option<String>>>,
    transferred: Vec<AtomicU64>,
    progress: SourceProgress<'a>,
    held: u64,
    total: u64,
    reported: Mutex<u64>,
}

/// Fetch every job into the store, returning the bytes this call moved per job.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fetch_sources(
    store: &Store,
    owner: &str,
    jobs: &[SourceJob],
    policy: &SourcePolicy,
    credentials: &dyn CredentialProvider,
    deadline: Deadline,
    ledger: &Ledger,
    options: &SourceDownload,
    held: u64,
    progress: SourceProgress<'_>,
) -> Result<Vec<u64>> {
    // The whole remaining download against the disk budget, before any byte moves: bytes
    // already retained under this owner are part of the store's occupancy.
    let mut adding = 0u64;
    for job in jobs {
        let part = owner_dir(store, owner)?
            .join(&job.grant.object.sha256)
            .join("bytes.part");
        let retained = fs::metadata(part).map_or(0, |md| allocated(&md));
        adding = adding.saturating_add(job.grant.object.length.saturating_sub(retained));
    }
    store.admits(adding)?;
    let download = Download {
        store,
        owner,
        jobs,
        policy,
        credentials,
        deadline,
        ledger,
        options,
        checkpoint: options.checkpoint_bytes.clamp(1 << 20, 1 << 30),
        board: Mutex::new(Board {
            entries: jobs.iter().map(|_| Entry::default()).collect(),
            failure: None,
            seq: 0,
        }),
        changed: Condvar::new(),
        resolved: jobs.iter().map(|_| Mutex::new(None)).collect(),
        transferred: jobs.iter().map(|_| AtomicU64::new(0)).collect(),
        progress,
        held,
        total: held + jobs.iter().map(|job| job.grant.object.length).sum::<u64>(),
        reported: Mutex::new(0),
    };
    download.report(0, held);
    std::thread::scope(|scope| {
        for _ in 0..options.streams.clamp(1, 128) {
            scope.spawn(|| download.work());
        }
    });
    let board = download.board.into_inner().unwrap();
    if let Some(failure) = board.failure {
        return Err(failure);
    }
    ledger.check_running()?;
    if board.entries.iter().any(|entry| !entry.done) {
        return refuse(
            Code::TRANSFER_FAILED,
            "the source download stopped with objects incomplete",
        );
    }
    Ok(download
        .transferred
        .iter()
        .map(|moved| moved.load(Ordering::Relaxed))
        .collect())
}

impl Download<'_> {
    fn work(&self) {
        while let Some(task) = self.next() {
            if let Err(refusal) = self.run(task) {
                let mut board = self.board.lock().unwrap();
                if board.failure.is_none() {
                    // Keep the actual failure before waking peers with a cancellation.
                    board.failure = Some(refusal);
                    self.ledger.cancel();
                }
                self.changed.notify_all();
                return;
            }
        }
    }

    /// Wait for work while anything in flight could still produce some.
    fn next(&self) -> Option<Task> {
        let mut board = self.board.lock().unwrap();
        loop {
            if board.failure.is_some() || self.ledger.check_running().is_err() {
                return None;
            }
            if let Some(task) = self.pick(&mut board) {
                return Some(task);
            }
            let busy = board.entries.iter().any(|entry| {
                !entry.done && ((entry.opened && entry.part.is_none()) || entry.held())
            });
            if !busy {
                return None;
            }
            board = self
                .changed
                .wait_timeout(board, self.ledger.sample())
                .unwrap()
                .0;
        }
    }

    /// The next thing worth doing: settle a finished object, continue a run, open an
    /// object, and only then split the largest range in flight.
    fn pick(&self, board: &mut Board) -> Option<Task> {
        let checkpoint = self.checkpoint;
        let mut open = None;
        for (index, entry) in board.entries.iter_mut().enumerate() {
            if entry.done || entry.reset_pending {
                continue;
            }
            if !entry.opened {
                open = open.or(Some(index));
                continue;
            }
            // Still being opened by another worker.
            let Some(part) = entry.part.clone() else {
                continue;
            };
            let length = self.jobs[index].grant.object.length;
            let prefix = entry.runs[&0].run.as_ref().map(Run::reached);
            if prefix == Some(length) {
                let run = entry.runs.get_mut(&0).unwrap().run.take().unwrap();
                return Some(Task::Finish(index, part, run));
            }
            if let Some(reached) = prefix {
                let finished = entry.runs.get(&reached).is_some_and(|slot| {
                    slot.run
                        .as_ref()
                        .is_some_and(|run| run.reached() == entry.end(reached, length))
                });
                if reached > 0 && finished {
                    let next = entry.runs.get_mut(&reached).unwrap().run.take().unwrap();
                    let prefix = entry.runs.get_mut(&0).unwrap().run.take().unwrap();
                    return Some(Task::Absorb(index, part, prefix, next));
                }
            }
            let pending = entry.runs.iter().find_map(|(start, slot)| {
                let run = slot.run.as_ref()?;
                (run.reached() < entry.end(*start, length)).then_some(*start)
            });
            if let Some(start) = pending {
                let slot = entry.runs.get_mut(&start).unwrap();
                let run = slot.run.take().unwrap();
                slot.claimed = Some(run.reached());
                return Some(Task::Fetch(index, part, run));
            }
        }
        if let Some(index) = open {
            board.entries[index].opened = true;
            return Some(Task::Open(index));
        }
        let mut widest: Option<(usize, u64, u64)> = None;
        for (index, entry) in board.entries.iter().enumerate() {
            if !entry.ranged || entry.whole || entry.done || entry.reset_pending {
                continue;
            }
            let length = self.jobs[index].grant.object.length;
            for (start, slot) in &entry.runs {
                let Some(claimed) = slot.claimed else {
                    continue;
                };
                let left = entry.end(*start, length).saturating_sub(claimed);
                if widest.is_none_or(|(_, _, most)| left > most) {
                    widest = Some((index, claimed, left));
                }
            }
        }
        let (index, claimed, left) = widest?;
        if left < 2 * checkpoint {
            return None;
        }
        let at = claimed + left / 2;
        let entry = &mut board.entries[index];
        entry.runs.insert(
            at,
            Slot {
                run: None,
                claimed: Some(at),
            },
        );
        let part = entry.part.clone().expect("a split object is open");
        Some(Task::Fetch(index, part, Run::new(at)))
    }

    fn run(&self, task: Task) -> Result<()> {
        match task {
            Task::Open(index) => self.open(index),
            Task::Fetch(index, part, mut run) => {
                let fetched = self.fetch(index, &part, &mut run);
                // Whatever ended the stream, the bytes it landed stay retained.
                let kept = if run.extent > run.durable {
                    self.save(index, &part, &mut run, &self.options.fault)
                } else {
                    Ok(())
                };
                self.check_in(index, &part, run)?;
                fetched.and(kept)
            }
            Task::Absorb(index, part, mut prefix, next) => {
                let whole = hash_range(&part.file, next.start, next.extent, &mut prefix.hash)
                    .map_err(io)?;
                if !whole {
                    return refuse(Code::LENGTH_MISMATCH, "a retained source run was truncated");
                }
                prefix.extent += next.extent;
                part.checkpoint(&mut prefix, &Fault::default())?;
                part.drop_state(next.start)?;
                self.board.lock().unwrap().entries[index]
                    .runs
                    .remove(&next.start);
                self.check_in(index, &part, prefix)
            }
            Task::Finish(index, part, prefix) => self.finish(index, &part, prefix),
        }
    }

    fn open(&self, index: usize) -> Result<()> {
        let object = &self.jobs[index].grant.object;
        let (part, runs) = Part::open(self.store, self.owner, object, true)?;
        let landed = runs.iter().map(|run| run.extent).sum();
        let mut board = self.board.lock().unwrap();
        let entry = &mut board.entries[index];
        entry.part = Some(Arc::new(part));
        for run in runs {
            entry.runs.insert(
                run.start,
                Slot {
                    run: Some(run),
                    claimed: None,
                },
            );
        }
        drop(board);
        self.set_landed(index, landed);
        self.changed.notify_all();
        Ok(())
    }

    /// Return a run to the board. The last one back from an object whose origin stopped
    /// honouring ranges starts it again as one body from zero.
    fn check_in(&self, index: usize, part: &Part, run: Run) -> Result<()> {
        let mut board = self.board.lock().unwrap();
        self.changed.notify_all();
        let entry = &mut board.entries[index];
        let slot = entry.runs.get_mut(&run.start).unwrap();
        slot.claimed = None;
        slot.run = Some(run);
        if !entry.reset_pending || entry.held() {
            return Ok(());
        }
        let prefix = part.reset(entry.runs.keys().copied())?;
        entry.runs.clear();
        entry.runs.insert(
            0,
            Slot {
                run: Some(prefix),
                claimed: None,
            },
        );
        entry.reset_pending = false;
        drop(board);
        self.set_landed(index, 0);
        Ok(())
    }

    /// Clip a read to the run's current end (a split may have moved it) and record the
    /// claim, so no later split can take bytes this run is about to write. 0 is "stop".
    fn claim(&self, index: usize, start: u64, at: u64, n: u64) -> u64 {
        let mut board = self.board.lock().unwrap();
        let entry = &mut board.entries[index];
        if entry.reset_pending {
            return 0;
        }
        let n = n.min(
            entry
                .end(start, self.jobs[index].grant.object.length)
                .saturating_sub(at),
        );
        entry.runs.get_mut(&start).unwrap().claimed = Some(at + n);
        n
    }

    /// One run's range, one GET per ask, re-asked from the byte reached. Asks are spent
    /// only by failures with no byte landed since the last one, never by a stated wait.
    fn fetch(&self, index: usize, part: &Part, run: &mut Run) -> Result<()> {
        let length = self.jobs[index].grant.object.length;
        let mut attempts = 0;
        loop {
            let at = run.reached();
            let end = {
                let board = self.board.lock().unwrap();
                if board.entries[index].reset_pending {
                    return Ok(());
                }
                board.entries[index].end(run.start, length)
            };
            if at >= end {
                return Ok(());
            }
            let asked = format!("bytes={at}-{}", end - 1);
            let failed = match self.ask(index, &asked) {
                Ok(Opened::Body(response)) => {
                    let whole = at == 0 && end == length;
                    match response.status {
                        206 => {
                            if range(response.header("content-range"))
                                != Some((at, end - 1, length))
                            {
                                return refuse(
                                    Code::LENGTH_MISMATCH,
                                    "source range differs from requested immutable extent",
                                );
                            }
                            let mut board = self.board.lock().unwrap();
                            if !board.entries[index].ranged {
                                board.entries[index].ranged = true;
                                self.changed.notify_all();
                            }
                        }
                        200 if whole => {}
                        200 => {
                            let mut board = self.board.lock().unwrap();
                            let entry = &mut board.entries[index];
                            entry.whole = true;
                            entry.reset_pending = true;
                            return Ok(());
                        }
                        _ => {
                            return refuse(
                                Code::TRANSFER_FAILED,
                                "source did not return object bytes",
                            )
                        }
                    }
                    let before = run.extent;
                    let streamed = self.stream(index, part, run, response, end);
                    if run.extent > before {
                        attempts = 0;
                    }
                    match streamed {
                        Ok(()) => continue,
                        Err(refusal) => refusal,
                    }
                }
                Ok(Opened::Retryable(status, stated)) => {
                    let (wait, counted) = self.ledger.retry_wait(stated);
                    if counted {
                        attempts += 1;
                        if attempts >= FETCH_ATTEMPTS {
                            return refuse(
                                Code::TRANSFER_FAILED,
                                format!("the source answered {status} to {asked} on every ask"),
                            );
                        }
                    }
                    self.ledger.pause(wait, self.deadline)?;
                    continue;
                }
                Ok(Opened::Expired(status)) => Refusal {
                    code: Code::TRANSFER_FAILED,
                    detail: format!("the source answered {status} to {asked}"),
                },
                Ok(Opened::Refused(refusal)) => return Err(refusal),
                Err(refusal) if refusal.code == Code::TRANSFER_FAILED => refusal,
                Err(refusal) => return Err(refusal),
            };
            self.ledger.check_running()?;
            attempts += 1;
            if attempts >= FETCH_ATTEMPTS {
                return Err(failed);
            }
            if run.extent > run.durable {
                self.save(index, part, run, &self.options.fault)?;
            }
        }
    }

    /// Open one ask from the delivery URL this object last resolved to, or its source URL.
    /// A refused delivery URL is forgotten and the source URL walked again at once: that is
    /// re-authorization, not a failed ask.
    fn ask(&self, index: usize, asked: &str) -> Result<Opened<'_>> {
        let source = &self.jobs[index].url;
        let cached = self.resolved[index].lock().unwrap().clone();
        let (mut last, mut hops) = (None, 0);
        let opened = pull::open_object(
            pull::shared_client(),
            cached.as_deref().unwrap_or(source),
            self.policy,
            self.credentials,
            self.deadline,
            self.ledger,
            &[("range".into(), asked.into())],
            &mut |checked, _| {
                hops += 1;
                last = Some(checked.url());
            },
        );
        let slot = &self.resolved[index];
        let refused = match &opened {
            Ok(Opened::Body(_)) => {
                if hops > 1 {
                    *slot.lock().unwrap() = last;
                }
                false
            }
            Ok(Opened::Retryable(..)) => false,
            _ => cached.is_some(),
        };
        if refused {
            *slot.lock().unwrap() = None;
            drop(opened);
            return self.ask(index, asked);
        }
        opened
    }

    /// Stream one answer into the run at its offsets, checkpointing on the cadence.
    fn stream(
        &self,
        index: usize,
        part: &Part,
        run: &mut Run,
        response: super::http::Response<'_>,
        end: u64,
    ) -> Result<()> {
        let mut body = response.into_body();
        let mut buffer = vec![0u8; 1 << 20];
        let fault = &self.options.fault;
        loop {
            let at = run.reached();
            if at >= end {
                return Ok(());
            }
            let boundary = run.start + run.durable + self.checkpoint;
            let want = (buffer.len() as u64).min(end - at).min(boundary - at) as usize;
            let n = body.read_refusing(&mut buffer[..want])?;
            if n == 0 {
                return refuse(
                    Code::TRANSFER_FAILED,
                    "source ended before the requested extent",
                );
            }
            let n = self.claim(index, run.start, at, n as u64) as usize;
            if n == 0 {
                return Ok(());
            }
            part.file.write_all_at(&buffer[..n], at).map_err(io)?;
            run.hash.update(&buffer[..n]);
            run.extent += n as u64;
            self.transferred[index].fetch_add(n as u64, Ordering::Relaxed);
            self.store.moved(n as u64);
            if run.extent > self.checkpoint {
                fault.hit("prefix-next-written");
            }
            if run.extent - run.durable >= self.checkpoint {
                self.save(index, part, run, fault)?;
            }
        }
    }

    fn finish(&self, index: usize, part: &Part, mut prefix: Run) -> Result<()> {
        let object = &self.jobs[index].grant.object;
        if hex(&prefix.hash.clone().finish()) != object.sha256 {
            prefix = part.reset(std::iter::empty())?;
            self.check_in(index, part, prefix)?;
            self.set_landed(index, 0);
            return refuse(
                Code::OBJECT_ID_MISMATCH,
                "complete source bytes do not match the pinned SHA-256",
            );
        }
        let fault = &self.options.fault;
        fault.hit("prefix-verified");
        self.store.admit_verified(
            Verified {
                tmp: part.dir.join("bytes.part"),
                file: part.file.try_clone().map_err(io)?,
                sha256: object.sha256.clone(),
                length: object.length,
            },
            fault,
            None,
        )?;
        fault.hit("prefix-admitted");
        // Keep the lock inode stable. An old process must never lock an unlinked predecessor.
        let _ = fs::remove_file(part.dir.join("state.json"));
        let mut board = self.board.lock().unwrap();
        let entry = &mut board.entries[index];
        entry.done = true;
        entry.part = None;
        drop(board);
        self.set_landed(index, object.length);
        self.changed.notify_all();
        Ok(())
    }

    /// Checkpoint a run and count its newly durable bytes.
    fn save(&self, index: usize, part: &Part, run: &mut Run, fault: &Fault) -> Result<()> {
        let before = run.durable;
        part.checkpoint(run, fault)?;
        let delta = run.durable - before;
        self.landed(index, |landed| landed + delta);
        Ok(())
    }

    fn set_landed(&self, index: usize, value: u64) {
        self.landed(index, |_| value);
    }

    /// Update an object's durable bytes and report the call's total.
    fn landed(&self, index: usize, update: impl FnOnce(u64) -> u64) {
        let (seq, present) = {
            let mut board = self.board.lock().unwrap();
            let entry = &mut board.entries[index];
            entry.landed = update(entry.landed);
            board.seq += 1;
            let present: u64 = board.entries.iter().map(|entry| entry.landed).sum();
            (board.seq, present)
        };
        self.report(seq, self.held + present);
    }

    /// Reports leave in the order they were counted; a stale one is dropped.
    fn report(&self, seq: u64, present: u64) {
        let mut reported = self.reported.lock().unwrap();
        if seq < *reported {
            return;
        }
        *reported = seq;
        (self.progress)(present, self.total);
    }
}

fn range(value: Option<&str>) -> Option<(u64, u64, u64)> {
    let (span, total) = value?.strip_prefix("bytes ")?.split_once('/')?;
    let (first, last) = span.split_once('-')?;
    Some((first.parse().ok()?, last.parse().ok()?, total.parse().ok()?))
}

/// Allocated bytes: a sparse part holds only what its runs wrote.
fn allocated(md: &fs::Metadata) -> u64 {
    md.len().min(md.blocks().saturating_mul(512))
}

pub(crate) fn discard_owner(store: &Store, owner: &str) -> Result<()> {
    let dir = owner_dir(store, owner)?;
    match fs::remove_dir_all(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(io(e)),
    }
}
pub(crate) fn bytes(store: &Store) -> Result<u64> {
    let owners = match fs::read_dir(area(store)) {
        Ok(rows) => rows,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(e) => return Err(io(e)),
    };
    let mut total = 0u64;
    for owner in owners {
        for object in fs::read_dir(owner.map_err(io)?.path()).map_err(io)? {
            let path = object.map_err(io)?.path().join("bytes.part");
            match fs::symlink_metadata(path) {
                Ok(md) if md.is_file() => {
                    total = total.checked_add(allocated(&md)).ok_or_else(|| Refusal {
                        code: Code::ARITH_OVERFLOW,
                        detail: "prefix bytes overflow".into(),
                    })?
                }
                Ok(_) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                Err(e) => return Err(io(e)),
            }
        }
    }
    Ok(total)
}
/// Copy an independently owned verified prefix while both source and recipient are fenced.
/// Runs beyond the donor's prefix are not adopted; the recipient fetches those ranges.
pub(crate) fn adopt_owner(
    store: &Store,
    source: &str,
    owner: &str,
    mut landed: impl FnMut(&ObjectRef) -> Result<()>,
) -> Result<()> {
    let entries = match fs::read_dir(owner_dir(store, source)?) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io(e)),
    };
    for entry in entries {
        let dir = entry.map_err(io)?.path();
        let Some(state) = read_state(&dir.join("state.json"))? else {
            continue;
        };
        if state.owner != source
            || dir.file_name().and_then(|name| name.to_str()) != Some(&state.object.sha256)
        {
            return refuse(
                Code::TRANSACTION_CONFLICT,
                "source prefix has another immutable identity",
            );
        }
        // Admission may have linked a read-only CAS inode or removed bytes.part
        // before the source writer recorded this member. Reuse the native verified
        // object and record recipient custody before considering any prefix copy.
        let (plan, _) = crate::fetch::FetchPlan::of_objects(
            store,
            "source-prefix-adoption",
            std::slice::from_ref(&state.object),
        )?;
        if !plan.held.is_empty() {
            landed(&state.object)?;
            continue;
        }
        let (from, runs) = Part::open(store, source, &state.object, false)?;
        let extent = runs[0].extent;
        if extent == state.object.length {
            // A stopped chmod/link admission may leave only a read-only prefix.
            // Copy through normal digest-checked admission; never chmod, truncate,
            // consume or unlink the donor inode before recipient custody is durable.
            let mut reader = std::io::Read::take(&from.file, extent);
            store.put_stream(&mut reader, Some(&state.object), &Fault::default())?;
            landed(&state.object)?;
            continue;
        }
        let (to, to_runs) = Part::open(store, owner, &state.object, true)?;
        if to_runs[0].extent >= extent {
            continue;
        }
        store.admits(extent - to_runs[0].extent)?;
        let mut prefix = to.reset(to_runs.iter().map(|run| run.start))?;
        let mut buffer = vec![0u8; 1 << 20];
        while prefix.extent < extent {
            let n = buffer.len().min((extent - prefix.extent) as usize);
            let n = from
                .file
                .read_at(&mut buffer[..n], prefix.extent)
                .map_err(io)?;
            if n == 0 {
                return refuse(Code::LENGTH_MISMATCH, "adopted prefix was truncated");
            }
            to.file
                .write_all_at(&buffer[..n], prefix.extent)
                .map_err(io)?;
            prefix.hash.update(&buffer[..n]);
            prefix.extent += n as u64;
        }
        to.checkpoint(&mut prefix, &Fault::default())?;
    }
    Ok(())
}
