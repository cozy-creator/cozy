//! One GPU's ledger, held by TensorD: the table executors claim their rows in, and TensorD's own
//! row, whose context is the display floor plus every byte NVML shows that no row explains (other
//! programs, executors still dying). A watch thread reads NVML once a second and on demand: it
//! reaps dead rows, returns their bytes once NVML shows them back, and every wedge floor ends the
//! rows asked for bytes whose progress did not move. NVML is read for capacity and these checks only.
use super::{MemoryLog, Sample};
use crate::process::Liveness;
use cozy_platform::{gpu::Device, process::Birth};
use std::{
    fs::File,
    io,
    os::fd::{AsFd, OwnedFd},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
    time::{Duration, Instant},
};
use varena::ledger::{Admit, Facts, Ledger, Row, RowFacts, State};

/// A CUDA context before any process on this GPU published one (design §7: ≈0.5 GiB).
const UNMEASURED_CONTEXT: u64 = 512 << 20;
/// What a desktop keeps free on its GPU in shared mode, or a sixteenth of a larger card.
const DISPLAY_FLOOR: u64 = 512 << 20;

/// Ends a process the ledger names stalled (its pid).
pub type Ender = Box<dyn Fn(u32) + Send + Sync>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Admitted {
    Now,
    /// Bytes no ask or ended idle executor freed: in use, pinned or in activations. The start
    /// goes ahead and its executor makes what room it can.
    Short(u64),
}

struct Inner {
    entry: String,
    ledger: Ledger,
    fd: OwnedFd,
    own: Row,
    floor: u64,
    /// The largest context a row here published.
    context: AtomicU64,
    log: MemoryLog,
    end: OnceLock<Ender>,
    looked: Mutex<Instant>,
    liveness: Liveness,
}

#[derive(Default)]
struct Ticks {
    latest: Option<Sample>,
    /// Ticks asked for, and the latest ask a finished tick began after.
    asked: u64,
    answered: u64,
    done: u64,
    stopped: bool,
}

pub struct GpuLedger {
    inner: Arc<Inner>,
    ticks: Arc<(Mutex<Ticks>, Condvar)>,
}

fn err(error: varena::Error) -> io::Error {
    io::Error::other(error.to_string())
}

/// Its pid exited (a zombie included) or never was.
fn gone(pid: u32) -> bool {
    Birth::of(pid).map_or(true, |birth| birth.ended().unwrap_or(false))
}

impl GpuLedger {
    /// `entry`'s ledger, sized by NVML's total, else `capacity`; None when neither says. `shared`
    /// leaves a display its floor.
    pub fn start(
        entry: &str,
        shared: bool,
        floor_bytes: u64,
        capacity: Option<u64>,
        liveness: Liveness,
        log: MemoryLog,
    ) -> io::Result<Option<Self>> {
        let device = Device::open(entry)
            .map_err(|error| tracing::warn!("memory: NVML unavailable for {entry:?}: {error}"))
            .ok();
        let reading = device.as_ref().and_then(|d| read(d));
        let Some(total) = reading.as_ref().map(|r| r.total).or(capacity) else {
            return Ok(None);
        };
        let uuid = device.as_ref().and_then(|d| d.key().split('/').next().map(str::to_owned)).unwrap_or_else(|| entry.into());
        let display = reading.as_ref().is_some_and(|r| r.display);
        let floor = match shared && display {
            true => DISPLAY_FLOOR.max(total / 16),
            false => 0,
        }
        .max(floor_bytes);
        let (ledger, fd) = Ledger::create(&uuid, total).map_err(err)?;
        let own = ledger.claim(std::process::id()).map_err(err)?;
        let inner = Arc::new(Inner {
            entry: entry.into(),
            ledger,
            fd,
            own,
            floor,
            context: AtomicU64::new(0),
            log,
            end: OnceLock::new(),
            looked: Mutex::new(Instant::now()),
            liveness,
        });
        let ticks: Arc<(Mutex<Ticks>, Condvar)> = Arc::default();
        let (watched, shared_ticks) = (inner.clone(), ticks.clone());
        std::thread::Builder::new().name("gpu-ledger".into()).spawn(move || {
            watch(device, &watched, &shared_ticks)
        })?;
        let ledger = Self { inner, ticks };
        ledger.now();
        Ok(Some(ledger))
    }

    /// What ends a stalled process; set once, after the pool exists.
    pub fn on_stalled(&self, end: Ender) {
        let _ = self.inner.end.set(end);
    }

    /// A descriptor of the memfd executors claim their rows in, to send one.
    pub fn grant(&self) -> io::Result<File> {
        Ok(File::from(self.inner.fd.as_fd().try_clone_to_owned()?))
    }

    pub fn capacity(&self) -> u64 {
        self.inner.ledger.capacity()
    }

    /// A new process's context here: the largest one measured, else the unmeasured estimate.
    pub fn context(&self) -> u64 {
        match self.inner.context.load(Ordering::Relaxed) {
            0 => UNMEASURED_CONTEXT,
            measured => measured,
        }
    }

    /// The latest NVML reading, taken on the watch thread.
    pub fn sample(&self) -> Option<Sample> {
        self.ticks.0.lock().unwrap().latest.clone()
    }

    /// A tick taken after this call: NVML read, dead rows reaped, TensorD's row republished.
    pub fn now(&self) {
        let (lock, changed) = &*self.ticks;
        let mut ticks = lock.lock().unwrap();
        ticks.asked += 1;
        let ticket = ticks.asked;
        changed.notify_all();
        drop(changed.wait_while(ticks, |t| t.answered < ticket && !t.stopped).unwrap());
    }

    /// The next tick of the watch thread's own cadence.
    fn next(&self) {
        let (lock, changed) = &*self.ticks;
        let ticks = lock.lock().unwrap();
        let ticket = ticks.done + 1;
        drop(changed.wait_while(ticks, |t| t.done < ticket && !t.stopped).unwrap());
    }

    fn rows(&self) -> io::Result<Vec<RowFacts>> {
        self.inner.ledger.snapshot().map_err(err)
    }

    /// Capacity less every byte the rows hold.
    pub fn free(&self) -> io::Result<u64> {
        let held = self.rows()?.iter().fold(0u64, |sum, row| sum.saturating_add(row.held()));
        Ok(self.capacity().saturating_sub(held))
    }

    /// What `pid`'s live row holds, and of that its weights resident.
    pub fn row(&self, pid: u32) -> Option<(u64, u64)> {
        let rows = self.rows().ok()?;
        let row = rows.iter().find(|r| r.pid == pid && r.state == State::Live)?;
        Some((row.held(), row.facts.resident))
    }

    /// Whether `need` fits now (`exact`), or once others' cold and idle bytes are asked for.
    pub fn fits(&self, need: u64, exact: bool) -> bool {
        match self.inner.ledger.admit(need) {
            Ok(Admit::Now) => true,
            Ok(Admit::After { short, .. }) => !exact && short == 0,
            Err(_) => false,
        }
    }

    /// Room for `need` bytes: what is free; else other rows' idle and cold bytes, asked for and
    /// waited on (the watch thread ends a victim that stops answering); else dead rows' bytes,
    /// waited for while each reading returns some; else, while `end_idle` ends one, idle
    /// executors. Never "now" below need.
    pub fn admit(&self, need: u64, end_idle: &mut dyn FnMut() -> bool) -> io::Result<Admitted> {
        let inner = &*self.inner;
        let (mut asking, mut going_before) = (true, None);
        loop {
            self.now();
            let (victims, short) = match inner.ledger.admit(need).map_err(err)? {
                Admit::Now => {
                    inner.log.note(serde_json::json!({"event": "admit", "gpu": inner.entry, "need": need}));
                    return Ok(Admitted::Now);
                }
                Admit::After { victims, short } => (victims, short),
            };
            if asking && !victims.is_empty() {
                let before = self.free()?;
                let bytes = victims.iter().map(|v| v.bytes).sum();
                for victim in inner.ledger.ask(None, bytes).map_err(err)? {
                    inner.ledger.wait(&victim).map_err(err)?;
                }
                self.now();
                let after = self.free()?;
                inner.log.note(serde_json::json!({"event": "ask", "gpu": inner.entry, "need": need,
                    "asked": bytes, "victims": victims.iter().map(|v| v.pid).collect::<Vec<_>>(),
                    "freed": after.saturating_sub(before)}));
                asking = after > before;
                continue;
            }
            let going: u64 = self.rows()?.iter().filter(|r| r.state != State::Live).map(RowFacts::held).sum();
            if going > 0 && going_before.is_none_or(|before| going < before) {
                going_before = Some(going);
                self.next();
                continue;
            }
            if end_idle() {
                (asking, going_before) = (true, None);
                continue;
            }
            let short = short.max(need.saturating_sub(self.free()?));
            inner.log.note(serde_json::json!({"event": "admit_short", "gpu": inner.entry, "need": need, "short": short}));
            return Ok(Admitted::Short(short));
        }
    }
}

impl Drop for GpuLedger {
    fn drop(&mut self) {
        let (lock, changed) = &*self.ticks;
        lock.lock().unwrap().stopped = true;
        changed.notify_all();
    }
}

fn read(device: &Device) -> Option<Sample> {
    device.read().ok().map(|r| Sample { total: r.total, free: r.free, display: r.display, processes: r.processes })
}

/// One tick a second, or sooner when asked; until the ledger's owner drops it.
fn watch(device: Option<Device>, inner: &Inner, ticks: &(Mutex<Ticks>, Condvar)) {
    let (lock, changed) = ticks;
    loop {
        let asked = lock.lock().unwrap().asked;
        let reading = device.as_ref().and_then(read);
        if let Err(error) = inner.settle(reading.as_ref()) {
            tracing::warn!("memory: GPU {} ledger: {error}", inner.entry);
        }
        inner.look();
        let mut state = lock.lock().unwrap();
        state.latest = reading;
        state.answered = asked;
        state.done += 1;
        changed.notify_all();
        if state.stopped {
            return;
        }
        let state = changed
            .wait_timeout_while(state, Duration::from_secs(1), |t| t.asked == asked && !t.stopped)
            .unwrap()
            .0;
        if state.stopped {
            return;
        }
    }
}

impl Inner {
    fn mine(&self, row: &RowFacts) -> bool {
        row.slot == self.own.id() && row.gen == self.own.gen()
    }

    /// Dead rows whose process is gone are reaped; reaping rows whose bytes NVML no longer
    /// charges (or, where it cannot say per process, on this reading after their exit) return,
    /// after TensorD's row took whatever of them the reading still shows.
    fn settle(&self, reading: Option<&Sample>) -> varena::Result<()> {
        for row in self.ledger.snapshot()?.iter().filter(|r| r.state == State::Dead && gone(r.pid)) {
            let held = self.ledger.reap(row.pid)?;
            self.log.note(serde_json::json!({"event": "reap", "gpu": self.entry, "pid": row.pid, "held": held}));
        }
        let rows = self.ledger.snapshot()?;
        let charged = |pid: u32| reading.and_then(|s| s.processes.as_ref()).is_some_and(|p| p.contains_key(&pid));
        let back: Vec<&RowFacts> = rows.iter().filter(|r| r.state == State::Reaping && !charged(r.pid)).collect();
        if let Some(context) = rows.iter().filter(|r| r.state == State::Live && !self.mine(r)).map(|r| r.facts.context).max() {
            self.context.fetch_max(context, Ordering::Relaxed);
        }
        let explained = rows
            .iter()
            .filter(|r| !self.mine(r) && !back.iter().any(|b| b.slot == r.slot))
            .fold(0u64, |sum, r| sum.saturating_add(r.held()));
        let unexplained = reading.map_or(0, |s| s.total.saturating_sub(s.free).saturating_sub(explained));
        self.own.publish(&Facts { context: self.floor + unexplained, ..Facts::default() });
        for row in back {
            self.ledger.returned(row.pid)?;
            self.log.note(serde_json::json!({"event": "returned", "gpu": self.entry, "pid": row.pid, "bytes": row.held()}));
        }
        Ok(())
    }

    /// Once a wedge floor: rows asked for bytes whose progress did not move since the last look
    /// are ended.
    fn look(&self) {
        let mut looked = self.looked.lock().unwrap();
        if looked.elapsed() < self.liveness.floor() {
            return;
        }
        *looked = Instant::now();
        drop(looked);
        match self.ledger.stalled() {
            Ok(pids) => {
                for pid in pids {
                    self.log.note(serde_json::json!({"event": "stalled", "gpu": self.entry, "pid": pid}));
                    if let Some(end) = self.end.get() {
                        end(pid);
                    }
                }
            }
            Err(error) => tracing::warn!("memory: GPU {} stalls: {error}", self.entry),
        }
    }
}
