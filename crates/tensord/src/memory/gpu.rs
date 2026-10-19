//! One GPU's ledger, held by TensorD (design memory-varena §2, §5.6): the table executors claim
//! their rows in, with the eventfd beside it, and TensorD's own row, whose context is the display
//! floor plus every byte NVML shows that no row explains (other programs, executors still dying).
//! A watch thread reads NVML once a second and on demand and wakes on the eventfd: it reaps dead
//! rows, returns them once NVML shows their bytes back, ends the rows the ledger names stalled
//! once a wedge floor, and tells the pool when bytes freed. TensorD decides nothing from bytes:
//! it asks `fits`.
use super::{MemoryLog, Sample};
use crate::process::Liveness;
use cozy_platform::{gpu::Device, process::Birth};
use std::{
    fs::File,
    io,
    os::fd::{AsFd, AsRawFd, FromRawFd, OwnedFd},
    sync::{Arc, Condvar, Mutex, OnceLock},
    time::Instant,
};
use varena::ledger::{Facts, Ledger, Row, RowFacts, State, Status};

/// What a desktop keeps free on its GPU in shared mode, or a sixteenth of a larger card.
const DISPLAY_FLOOR: u64 = 512 << 20;

/// Ends a process the ledger names stalled (its pid).
pub type Ender = Box<dyn Fn(u32) + Send + Sync>;
/// Told when bytes freed on the GPU.
pub type Freed = Box<dyn Fn() + Send + Sync>;

struct Inner {
    entry: String,
    ledger: Ledger,
    fd: OwnedFd,
    own: Row,
    floor: u64,
    log: MemoryLog,
    end: OnceLock<Ender>,
    freed: OnceLock<Freed>,
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
    /// An eventfd that wakes the watch thread for an asked tick or a stop.
    wake: Arc<OwnedFd>,
}

fn err(error: varena::Error) -> io::Error {
    io::Error::other(error.to_string())
}

/// Its pid exited (a zombie included) or never was.
fn gone(pid: u32) -> bool {
    Birth::of(pid).map_or(true, |birth| birth.ended().unwrap_or(false))
}

fn eventfd() -> io::Result<OwnedFd> {
    // SAFETY: eventfd(2) with no shared state; a negative return is an error.
    let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: a descriptor we just created and own.
    Ok(unsafe { OwnedFd::from_raw_fd(fd) })
}

/// Adds one to an eventfd.
fn signal(fd: &OwnedFd) {
    let one = 1u64.to_ne_bytes();
    // SAFETY: an 8-byte write to an eventfd we hold.
    unsafe { libc::write(fd.as_raw_fd(), one.as_ptr().cast(), 8) };
}

/// Reads an eventfd to zero; whether it was signalled.
fn drain(fd: i32) -> bool {
    let mut count = [0u8; 8];
    // SAFETY: an 8-byte read from a nonblocking eventfd we hold.
    unsafe { libc::read(fd, count.as_mut_ptr().cast(), 8) == 8 }
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
        let reading = device.as_ref().and_then(read);
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
            log,
            end: OnceLock::new(),
            freed: OnceLock::new(),
            looked: Mutex::new(Instant::now()),
            liveness,
        });
        let ticks: Arc<(Mutex<Ticks>, Condvar)> = Arc::default();
        let wake = Arc::new(eventfd()?);
        let (watched, shared_ticks, woken) = (inner.clone(), ticks.clone(), wake.clone());
        std::thread::Builder::new().name("gpu-ledger".into()).spawn(move || {
            watch(device, &watched, &shared_ticks, &woken)
        })?;
        let ledger = Self { inner, ticks, wake };
        ledger.now();
        Ok(Some(ledger))
    }

    /// What ends a stalled process; set once, after the pool exists.
    pub fn on_stalled(&self, end: Ender) {
        let _ = self.inner.end.set(end);
    }

    /// What is told when bytes free here (the dispatch queue's wake); set once.
    pub fn on_freed(&self, freed: Freed) {
        let _ = self.inner.freed.set(freed);
    }

    /// An executor's grant: descriptors of the memfd it claims its row in and of the eventfd it
    /// signals when it frees bytes.
    pub fn grant(&self) -> io::Result<[File; 2]> {
        let events = self.inner.ledger.events().ok_or_else(|| io::Error::other("the ledger has no eventfd"))?;
        Ok([File::from(self.inner.fd.as_fd().try_clone_to_owned()?), File::from(events.try_clone_to_owned()?)])
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
        signal(&self.wake);
        drop(changed.wait_while(ticks, |t| t.answered < ticket && !t.stopped).unwrap());
    }

    /// The next tick of the watch thread's own cadence.
    fn next(&self) {
        let (lock, changed) = &*self.ticks;
        let ticks = lock.lock().unwrap();
        let ticket = ticks.done + 1;
        drop(changed.wait_while(ticks, |t| t.done < ticket && !t.stopped).unwrap());
    }

    /// Whether a request of `package` can start here now (`Ledger::fits`): at `shape`, or its
    /// widest measured one before its shape is known; `spawn`: it starts a process too.
    pub fn fits(&self, package: &str, shape: Option<&str>, spawn: bool) -> bool {
        self.inner.ledger.fits(package, shape, spawn).unwrap_or(false)
    }

    /// Whether any request runs here now, by any process's row.
    pub fn running(&self) -> bool {
        self.inner.ledger.snapshot().is_ok_and(|rows| rows.iter().any(|r| !r.running.is_empty()))
    }

    /// Waits while rows of exited processes give their bytes back: a reading at a time, as long
    /// as each returns some.
    pub fn settle(&self) {
        let mut before = usize::MAX;
        loop {
            self.now();
            let going = self.inner.ledger.snapshot().map_or(0, |rows| rows.iter().filter(|r| r.state != State::Live).count());
            if going == 0 || going >= before {
                return;
            }
            before = going;
            self.next();
        }
    }

    /// Whether `pid`'s live row has weights resident here.
    pub fn resident(&self, pid: u32) -> bool {
        let rows = self.inner.ledger.snapshot().unwrap_or_default();
        rows.iter().any(|r| r.pid == pid && r.state == State::Live && r.facts.resident > 0)
    }

    /// Names the key `request` records its peak under, before it runs.
    pub fn dispatch(&self, request: &str, key: &str) -> io::Result<()> {
        self.inner.ledger.dispatch(request, key).map_err(err)
    }

    pub fn undispatch(&self, request: &str) {
        self.inner.ledger.undispatch(request);
    }

    /// The table at a glance (`machine show`, telemetry).
    pub fn status(&self) -> io::Result<Status> {
        self.inner.ledger.status().map_err(err)
    }
}

impl Drop for GpuLedger {
    fn drop(&mut self) {
        let (lock, changed) = &*self.ticks;
        lock.lock().unwrap().stopped = true;
        changed.notify_all();
        signal(&self.wake);
    }
}

fn read(device: &Device) -> Option<Sample> {
    device.read().ok().map(|r| Sample { total: r.total, free: r.free, display: r.display, processes: r.processes })
}

/// One tick a second, or sooner when asked or when the ledger's eventfd says bytes freed; until
/// the ledger's owner drops it. A freed signal reaches the pool after the tick that follows it.
fn watch(device: Option<Device>, inner: &Inner, ticks: &(Mutex<Ticks>, Condvar), wake: &OwnedFd) {
    let (lock, changed) = ticks;
    let events = inner.ledger.events().map(|e| e.as_raw_fd());
    let mut freed = false;
    loop {
        let asked = lock.lock().unwrap().asked;
        let reading = device.as_ref().and_then(read);
        if let Err(error) = inner.settle(reading.as_ref()) {
            tracing::warn!("memory: GPU {} ledger: {error}", inner.entry);
        }
        inner.look();
        if std::mem::take(&mut freed) {
            if let Some(told) = inner.freed.get() {
                told();
            }
        }
        let mut state = lock.lock().unwrap();
        state.latest = reading;
        state.answered = asked;
        state.done += 1;
        changed.notify_all();
        if state.stopped {
            return;
        }
        drop(state);
        let mut fds = [wake.as_raw_fd(), events.unwrap_or(-1)].map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 });
        // SAFETY: poll on two descriptors this thread's owners keep open (-1 is ignored).
        unsafe { libc::poll(fds.as_mut_ptr(), 2, 1000) };
        if fds[0].revents & libc::POLLIN != 0 {
            drain(fds[0].fd);
        }
        if fds[1].revents & libc::POLLIN != 0 {
            freed |= drain(fds[1].fd);
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
