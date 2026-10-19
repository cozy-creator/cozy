//! One GPU's ledger, held by TensorD (design memory-varena §2, §5.6): the table executors claim
//! their rows in, with the eventfd beside it, and TensorD's own row, whose context is the display
//! floor plus every byte NVML shows that no row explains (other programs, executors still dying).
//! A watch thread blocks on the eventfd and wakes the pool the moment varena signals freed bytes.
//! It reads NVML on demand (before a decision, after an executor's exit): it reaps dead rows and
//! returns them once NVML shows their bytes back. Only while a call is in flight, or rows are
//! reaping or asked for bytes, does it sample, at the wedge rule's resolution: their bytes coming
//! back, or a stalled row (progress not moved for a wedge floor), which it ends. TensorD decides nothing from bytes: it
//! asks `fits`.
use super::{MemoryLog, Sample};
use crate::{device_executor::Page, process::Liveness};
use platform::{gpu::Device, process::Birth};
use std::{
    collections::BTreeMap,
    fs::File,
    io,
    os::{
        fd::{AsFd, AsRawFd, FromRawFd, OwnedFd},
        unix::net::UnixStream,
    },
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc, Condvar, Mutex, OnceLock,
    },
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
    own: Arc<Row>,
    floor: u64,
    /// Its row's context as last read: the floor and what no row explains.
    context: AtomicU64,
    /// Kept GPU pages by layout key.
    kept: Mutex<BTreeMap<String, Holding>>,
    log: MemoryLog,
    end: OnceLock<Ender>,
    freed: OnceLock<Freed>,
    looked: Mutex<Instant>,
    liveness: Liveness,
    /// Calls TensorD has in flight on this GPU: executors ask each other for bytes then.
    active: AtomicUsize,
}

/// GPU pages an executor kept of one layout, so its crash or restart leaves them on the GPU.
/// While `pid` lives each fd is a container (a socket whose queue holds the pages' fd) that it
/// drains itself when it unmaps them; once it is gone the fds are taken out, and are TensorD's:
/// counted in its row as kept, handed to the next executor of the layout, closed on an ask.
struct Holding {
    fingerprint: String,
    pid: u32,
    pages: BTreeMap<u32, (u64, OwnedFd)>,
    taken: bool,
    used: Instant,
}

impl Holding {
    fn take(&mut self) {
        if self.taken || !gone(self.pid) {
            return;
        }
        self.taken = true;
        let pages = std::mem::take(&mut self.pages).into_iter();
        let take = |(r, (n, c)): (u32, (u64, OwnedFd))| Some((r, (n, platform::handles::recv_fd(&UnixStream::from(c)).ok()?)));
        self.pages = pages.filter_map(take).collect();
    }

    fn bytes(&self) -> u64 {
        self.pages.values().map(|p| p.0).filter(|_| self.taken).sum()
    }
}

#[derive(Default)]
struct Ticks {
    latest: Option<Sample>,
    /// Ticks asked for, and the latest ask a finished tick began after.
    asked: u64,
    answered: u64,
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
    /// `entry`'s ledger, sized by NVML's total, else `capacity`, and never above `limit` (a larger
    /// card standing in for a smaller one); None when neither says. `shared` leaves a display its
    /// floor.
    pub fn start(
        entry: &str,
        shared: bool,
        floor_bytes: u64,
        capacity: Option<u64>,
        limit: Option<u64>,
        liveness: Liveness,
        log: MemoryLog,
    ) -> io::Result<Option<Self>> {
        let device = Device::open(entry)
            .map_err(|error| tracing::warn!("memory: NVML unavailable for {entry:?}: {error}"))
            .ok();
        let reading = device.as_ref().and_then(read);
        let Some(card) = reading.as_ref().map(|r| r.total).or(capacity) else {
            return Ok(None);
        };
        let total = limit.map_or(card, |limit| card.min(limit));
        log.note(serde_json::json!({"event": "ledger", "gpu": entry, "card": card, "capacity": total}));
        let uuid = device.as_ref().and_then(|d| d.key().split('/').next().map(str::to_owned)).unwrap_or_else(|| entry.into());
        let display = reading.as_ref().is_some_and(|r| r.display);
        let floor = match shared && display {
            true => DISPLAY_FLOOR.max(total / 16),
            false => 0,
        }
        .max(floor_bytes);
        let (ledger, fd) = Ledger::create(&uuid, total).map_err(err)?;
        let own = Arc::new(ledger.claim(std::process::id()).map_err(err)?);
        let inner = Arc::new(Inner {
            entry: entry.into(),
            ledger,
            fd,
            own,
            floor,
            context: AtomicU64::new(floor),
            kept: Mutex::default(),
            log,
            end: OnceLock::new(),
            freed: OnceLock::new(),
            looked: Mutex::new(Instant::now()),
            liveness,
            active: AtomicUsize::new(0),
        });
        let server = inner.clone();
        inner.own.serve(move |want| server.evict_kept(want)).map_err(err)?;
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


    /// A call in flight here until the guard drops: the watch samples meanwhile, for stalled asks.
    pub fn active(&self) -> Active<'_> {
        self.inner.active.fetch_add(1, Ordering::AcqRel);
        signal(&self.wake);
        Active(self)
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

    /// Reads again while rows of exited processes give their bytes back: as long as each reading
    /// returns some.
    pub fn settle(&self) {
        let mut before = usize::MAX;
        loop {
            self.now();
            let going = self.inner.ledger.snapshot().map_or(0, |rows| rows.iter().filter(|r| r.state != State::Live).count());
            if going == 0 || going >= before {
                return;
            }
            before = going;
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

    /// Holds GPU pages `pid` kept of layout `key` (region, bytes, container); a deposit by
    /// another process or of other contents replaces what was held.
    pub fn keep(&self, key: &str, fingerprint: &str, pid: u32, pages: Vec<(Page, File)>) {
        let mut kept = self.inner.kept.lock().unwrap();
        let fresh = || Holding { fingerprint: fingerprint.into(), pid, pages: BTreeMap::new(), taken: false, used: Instant::now() };
        let held = kept.entry(key.into()).or_insert_with(fresh);
        if held.pid != pid || held.fingerprint != fingerprint {
            *held = fresh();
        }
        held.pages.extend(pages.into_iter().map(|(p, f)| (p.region, (p.nbytes, f.into()))));
        held.used = Instant::now();
    }

    /// The pages kept of layout `key` by a process now gone, handed over: the fingerprint and
    /// each page's fd. Whoever maps them keeps them next.
    pub fn take_kept(&self, key: &str) -> Option<(String, Vec<(Page, File)>)> {
        let held = {
            let mut kept = self.inner.kept.lock().unwrap();
            let held = kept.get_mut(key)?;
            held.take();
            match held.taken {
                true => kept.remove(key)?,
                false => return None,
            }
        };
        self.inner.publish();
        let pages = held.pages.into_iter().map(|(region, (nbytes, fd))| (Page { region, nbytes }, File::from(fd)));
        Some((held.fingerprint, pages.collect()))
    }

    /// A GPU fault: no kept page is trusted any more.
    pub fn forget_kept(&self) {
        let dropped = std::mem::take(&mut *self.inner.kept.lock().unwrap()).len();
        self.inner.publish();
        self.inner.log.note(serde_json::json!({"event": "kept_forgotten", "gpu": self.inner.entry, "layouts": dropped}));
    }
}

pub struct Active<'a>(&'a GpuLedger);

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.0.inner.active.fetch_sub(1, Ordering::AcqRel);
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

/// Blocks on the ledger's eventfd and the wake eventfd until the ledger's owner drops it: a
/// freed signal reaches the pool at once; an asked tick reads NVML and settles the rows. While
/// rows are reaping or asked for bytes, it also wakes at the wedge rule's sample to read again.
fn watch(device: Option<Device>, inner: &Inner, ticks: &(Mutex<Ticks>, Condvar), wake: &OwnedFd) {
    let (lock, changed) = ticks;
    let events = inner.ledger.events().map(|e| e.as_raw_fd());
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
        changed.notify_all();
        if state.stopped {
            return;
        }
        drop(state);
        let timeout = match inner.pending() {
            true => inner.liveness.sample.as_millis().min(i32::MAX as u128) as i32,
            false => -1,
        };
        loop {
            let mut fds = [wake.as_raw_fd(), events.unwrap_or(-1)].map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 });
            // SAFETY: poll on two descriptors this thread's owners keep open (-1 is ignored).
            let ready = unsafe { libc::poll(fds.as_mut_ptr(), 2, timeout) };
            if fds[1].revents & libc::POLLIN != 0 && drain(fds[1].fd) {
                if let Some(told) = inner.freed.get() {
                    told();
                }
            }
            let woken = fds[0].revents & libc::POLLIN != 0 && drain(fds[0].fd);
            if woken || ready == 0 {
                break;
            }
        }
    }
}

impl Inner {
    /// Kept pages whose depositor is gone: TensorD's own bytes.
    fn kept_bytes(&self) -> u64 {
        let mut kept = self.kept.lock().unwrap();
        kept.values_mut().map(|h| {
            h.take();
            h.bytes()
        }).sum()
    }

    /// TensorD's row: its context, and its kept pages as warm weights an ask takes last.
    fn publish(&self) {
        let kept = self.kept_bytes();
        self.own.publish(&Facts { context: self.context.load(Ordering::Acquire), resident: kept, cold: kept, kept, ..Facts::default() });
    }

    /// An ask of TensorD's row: kept pages close, least recently kept layout first.
    fn evict_kept(&self, want: u64) -> u64 {
        self.kept_bytes();
        let mut freed = 0;
        {
            let mut kept = self.kept.lock().unwrap();
            while freed < want {
                let Some(key) = kept.iter().filter(|(_, h)| h.taken).min_by_key(|(_, h)| h.used).map(|(k, _)| k.clone()) else { break };
                let held = kept.get_mut(&key).unwrap();
                while freed < want {
                    let Some((_, (n, fd))) = held.pages.pop_last() else { break };
                    drop(fd);
                    freed += n;
                }
                if held.pages.is_empty() {
                    kept.remove(&key);
                }
            }
        }
        if freed > 0 {
            self.publish();
            self.log.note(serde_json::json!({"event": "kept_evicted", "gpu": self.entry, "want": want, "freed": freed}));
            self.ledger.freed();
        }
        freed
    }

    fn mine(&self, row: &RowFacts) -> bool {
        row.slot == self.own.id() && row.gen == self.own.gen()
    }

    /// A call in flight, rows whose bytes are on their way back, or rows asked for bytes: what
    /// the watch samples for.
    fn pending(&self) -> bool {
        let rows = self.ledger.snapshot().unwrap_or_default();
        self.active.load(Ordering::Acquire) > 0 || rows.iter().any(|r| r.state != State::Live || r.want > 0)
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
        let kept = self.kept_bytes();
        let unexplained = reading.map_or(0, |s| s.total.saturating_sub(s.free).saturating_sub(explained).saturating_sub(kept));
        self.context.store(self.floor + unexplained, Ordering::Release);
        self.publish();
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
