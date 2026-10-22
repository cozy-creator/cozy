//! TensorD on varena's ledger (design varena-slim §2–§4). Each envelope GPU the inventory names
//! has a [`Ledger`] (capacity min(card, `gpu_memory_limit`)): an executor is granted it at spawn,
//! its row is reaped once its exit is proven, and admission is [`admit`] (`Ledger::admit`: a
//! group on all of its GPUs or none, waiting while refused). TensorD holds no byte count and
//! decides no eviction: kept-page custody runs inside the ledger. Host RAM is the OS page cache;
//! [`Host`] watches what the running calls must keep free of it. One thread blocks on every
//! ledger's events and the host's short event.
use crate::{execution::Engine, machine::receipt};
use platform::contain::host_memory;
use platform::contain::host_pressure::{self, Growth, HostMode, Nudge};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Read, Write},
    os::{
        fd::{AsRawFd, RawFd},
        unix::net::UnixStream,
    },
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use varena::{ledger::Mode, GpuId, HostCall, Ledger, Peak};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// One envelope GPU's ledger, and the GPU model its peaks are kept under.
pub struct Gpu {
    pub ledger: Ledger,
    pub model: String,
}

impl Gpu {
    /// `entry`'s ledger, sized by the inventory's memory (an integrated GPU's is the host's) and
    /// never above `limit`, in the machine's sharing `mode`, with the peaks this machine measured
    /// on its model. None when the inventory does not name it or its table cannot be made.
    pub fn create(entry: &str, inventory: &[receipt::Gpu], limit: Option<u64>, mode: HostMode, peaks: &Peaks, log: &MemoryLog) -> Option<Gpu> {
        let row = inventory.iter().find(|g| g.device_index.to_string() == entry || g.device_uuid == entry)?;
        let id = GpuId { uuid: row.device_uuid.clone(), model: row.device_name.clone(), memory: row.memory_bytes };
        let ledger = match Ledger::create(&id, limit) {
            Ok(ledger) => ledger,
            Err(error) => {
                tracing::warn!("memory: GPU {entry} has no ledger: {error}");
                return None;
            }
        };
        let mode = match mode {
            HostMode::Shared => Mode::Shared,
            HostMode::Dedicated => Mode::Dedicated,
        };
        if let Err(error) = ledger.set_mode(mode) {
            tracing::warn!("memory: GPU {entry} stays shared: {error}");
        }
        peaks.restore(&id.model, &ledger);
        log.note(serde_json::json!({"event": "ledger", "gpu": entry, "card": id.memory, "capacity": ledger.status().capacity}));
        Some(Gpu { ledger, model: id.model })
    }
}

/// Admits request `id` of `family` at `<package>|<shape>` on every one of `ledgers` or none
/// (`Ledger::admit`, which names the key its peak is recorded under; a yielded request of the
/// same family, waiting on this one, is left out). Refused, it waits for the engine's activity,
/// which every ledger event and every cancel bumps, and asks again: running requests end, and a
/// GPU nothing runs on always admits. False: it was canceled meanwhile. A ledger that cannot
/// answer lets it run unrecorded.
pub fn admit(engine: &Engine, ledgers: &[&Ledger], id: &str, package: &str, shape: &str, family: Option<&str>) -> io::Result<bool> {
    loop {
        let seen = engine.activity_epoch();
        match Ledger::admit(ledgers, id, package, shape, family) {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(error) => {
                tracing::warn!("memory: request {id} runs unrecorded: {error}");
                return Ok(true);
            }
        }
        if engine.get(id)?.cancel_actor.is_some() {
            return Ok(false);
        }
        engine.wait_activity(seen, None);
    }
}

/// The GPU of the group (an index into `ledgers`, GPU order) that keeps each set of `sets`, by
/// `Ledger::place` for `package`'s request at `shape` in `family`. A set's bytes are what its
/// load recorded on a GPU of the group (its layout's, exact; its region split is the model's,
/// known only where it loads): a set never loaded goes where the most room is free. A ledger
/// that cannot answer keeps every set on GPU 0, where it pages.
pub fn homes(ledgers: &[&Ledger], package: &str, shape: &str, family: Option<&str>, sets: &[&str], log: &MemoryLog) -> Vec<u32> {
    let unknown: Vec<(&str, Option<u64>)> = sets.iter().map(|set| (*set, None)).collect();
    match Ledger::place(ledgers, package, shape, family, &unknown) {
        Ok(homes) => homes,
        Err(error) => {
            log.note(serde_json::json!({"event": "homes_on_gpu_0", "package": package, "why": error.to_string()}));
            vec![0; sets.len()]
        }
    }
}

/// Measured peaks, persisted per GPU model as `<dir>/<model>.json`: varena's records, kept and
/// restored whole. A key names its package release (`<construction>|<shape>`, or a set's
/// `<package>@<release>/<component>|…`), so one file holds every release's. Each GPU of the
/// model contributes its records; a key two of them measured is merged by the ledger's restore.
pub struct Peaks {
    dir: PathBuf,
    saved: Mutex<BTreeMap<String, Vec<Peak>>>,
}

impl Peaks {
    pub fn at(dir: PathBuf) -> Self {
        Self { dir, saved: Mutex::default() }
    }

    fn path(&self, model: &str) -> PathBuf {
        let name: String = model.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect();
        self.dir.join(format!("{name}.json"))
    }

    fn restore(&self, model: &str, ledger: &Ledger) {
        let mut saved = self.saved.lock().unwrap();
        let held = saved.entry(model.into()).or_insert_with(|| {
            let read = fs::read(self.path(model)).ok().and_then(|bytes| serde_json::from_slice::<Vec<Peak>>(&bytes).ok());
            read.unwrap_or_default()
        });
        if let Err(error) = ledger.restore(held) {
            tracing::warn!("memory: {model}'s peaks not restored: {error}");
        }
    }

    /// Saves what `ledgers` (every GPU of `model`, each holding what was restored into it) now
    /// hold, when it changed.
    pub fn save<'a>(&self, model: &str, ledgers: impl Iterator<Item = &'a Ledger>) {
        let mut saved = self.saved.lock().unwrap();
        let mut held: Vec<Peak> = vec![];
        for peak in ledgers.flat_map(Ledger::peaks) {
            if !held.contains(&peak) {
                held.push(peak);
            }
        }
        if saved.get(model).is_some_and(|before| before.len() == held.len() && held.iter().all(|peak| before.contains(peak))) {
            return;
        }
        let written = fs::create_dir_all(&self.dir)
            .map_err(platform::Error::from)
            .and_then(|()| platform::durable::replace(&self.path(model), &serde_json::to_vec(&held).unwrap_or_default(), 0o600));
        match written {
            Ok(()) => drop(saved.insert(model.into(), held)),
            Err(error) => tracing::warn!("memory: {model}'s peaks not saved: {error}"),
        }
    }
}

/// What woke the events thread.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Woke {
    /// A ledger: room freed, a row died, a peak was recorded.
    Ledger,
    /// The host is short of what its running calls must keep free ([`Host`]).
    HostShort,
}

/// A source the events thread waits on: a ledger's events, or the host's short event.
pub enum Source {
    Ledger(Arc<Gpu>),
    Host(Arc<Host>),
}

impl Source {
    fn fd(&self) -> RawFd {
        match self {
            Source::Ledger(gpu) => gpu.ledger.events().as_raw_fd(),
            Source::Host(host) => host.short.as_raw_fd(),
        }
    }

    /// Clears what it signalled (a ready event: no wait).
    fn clear(&self) -> Woke {
        match self {
            Source::Ledger(gpu) => {
                gpu.ledger.clear_events();
                Woke::Ledger
            }
            Source::Host(host) => {
                host.short();
                Woke::HostShort
            }
        }
    }
}

/// Blocks on every source until the returned end drops; `wake` runs once per wake with what
/// woke it. It never times out.
pub fn react(sources: Vec<Source>, wake: impl Fn(Woke) + Send + 'static) -> io::Result<UnixStream> {
    let (held, end) = UnixStream::pair()?;
    std::thread::Builder::new().name("memory-events".into()).spawn(move || {
        let watched = std::iter::once(end.as_raw_fd()).chain(sources.iter().map(Source::fd));
        let mut fds: Vec<libc::pollfd> = watched.map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 }).collect();
        loop {
            // SAFETY: poll on descriptors this thread owns; no timeout.
            if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return tracing::warn!("memory: events: {}", io::Error::last_os_error());
            }
            let (pool, events) = fds.split_at_mut(1);
            if pool.iter().any(|fd| fd.revents != 0) {
                return;
            }
            let mut woke = vec![];
            for (fd, source) in events.iter_mut().zip(&sources).filter(|(fd, _)| fd.revents != 0) {
                if fd.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0 {
                    fd.fd = -1; // closed or failed: poll skips it from now on
                    continue;
                }
                let what = source.clear();
                if !woke.contains(&what) {
                    woke.push(what);
                }
            }
            woke.into_iter().for_each(&wake);
        }
    })?;
    Ok(held)
}

/// The host rule (#1851) on the platform's watcher: each pressure event, or a nudge (TensorD's
/// own events, where the kernel gives none), is a look. What the calls running on these GPUs
/// must keep free, each process's measured anonymous peak less its anonymous bytes now plus
/// the growth seen past it ([`Growth`]), against the live budget, page cache counted as room.
/// Short, the events thread hears it ([`Woke::HostShort`]): the pool ends one idle process, and
/// the next event looks again. No timer.
pub struct Host {
    short: UnixStream,
    nudge: Nudge,
    /// Looks begun and made so far, and a signal for each made.
    looks: Arc<(Mutex<(u64, u64)>, std::sync::Condvar)>,
}

impl Host {
    /// Watches while the process lives; None (a memory-log note says why) where the platform
    /// gives no pressure events.
    pub fn watch(mode: HostMode, gpus: Vec<Arc<Gpu>>, log: &MemoryLog) -> Option<Arc<Host>> {
        let armed = host_pressure::Pressure::arm(mode).and_then(|pressure| {
            let (short, signal) = UnixStream::pair()?;
            short.set_nonblocking(true)?;
            Ok((pressure, (short, signal)))
        });
        let (mut pressure, (short, mut signal)) = armed
            .inspect_err(|error| log.note(serde_json::json!({"event": "host_unwatched", "mode": format!("{mode:?}"), "why": error.to_string()})))
            .ok()?;
        let (nudge, looks) = (pressure.nudge(), Arc::new((Mutex::new((0, 0)), std::sync::Condvar::new())));
        let looked = looks.clone();
        let watching = std::thread::Builder::new().name("host-pressure".into()).spawn(move || {
            let mut keep = Keep::default();
            while pressure.wait().is_ok() {
                looked.0.lock().unwrap_or_else(|e| e.into_inner()).0 += 1;
                let need = keep.look(gpus.iter().map(|gpu| gpu.ledger.host_calls()).collect::<Option<Vec<_>>>().map(|calls| calls.concat()));
                let budget = u64::try_from(host_memory::read().available).unwrap_or(u64::MAX);
                if budget < need && signal.write_all(&[1]).is_err() {
                    return;
                }
                looked.0.lock().unwrap_or_else(|e| e.into_inner()).1 += 1;
                looked.1.notify_all();
            }
        });
        if let Err(error) = watching {
            log.note(serde_json::json!({"event": "host_unwatched", "why": error.to_string()}));
            return None;
        }
        Some(Arc::new(Host { short, nudge, looks }))
    }

    /// Something happened (a call began, an executor's frame came): the watcher looks.
    pub fn nudge(&self) {
        self.nudge.send();
    }

    /// Nudges the watcher and blocks until a look begun since has been made.
    pub fn look(&self) {
        let begun = self.looks.0.lock().unwrap_or_else(|e| e.into_inner()).0;
        self.nudge();
        self.await_look(begun);
    }

    /// Looks made so far.
    pub fn looks(&self) -> u64 {
        self.looks.0.lock().unwrap_or_else(|e| e.into_inner()).1
    }

    /// Blocks until more than `made` looks have been made.
    pub fn await_look(&self, made: u64) {
        let (count, signal) = &*self.looks;
        let mut looks = count.lock().unwrap_or_else(|e| e.into_inner());
        while looks.1 <= made {
            looks = signal.wait(looks).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Whether a look found the host short since this was last asked (the events thread asks).
    pub fn short(&self) -> bool {
        let mut buf = [0; 64];
        (&self.short).read(&mut buf).is_ok_and(|n| n > 0)
    }
}

/// What the host keeps free for the calls running on it: each process's anonymous growth still
/// to come (its requests' measured anonymous peak less its anonymous bytes now) and [`Growth`],
/// the most a look saw them grow past what was measured to come (an unmeasured call's every
/// byte). Page cache is never need: the budget holds it.
#[derive(Default)]
struct Keep {
    growth: Growth,
    /// Each process's requests running at the last look: each that ended settles the growth.
    running: BTreeMap<u32, u64>,
}

impl Keep {
    /// A look at the processes running requests (`None`: a GPU's table cannot be read): what
    /// must stay free.
    fn look(&mut self, calls: Option<Vec<HostCall>>) -> u64 {
        // A process with rows on several GPUs is one call.
        let mut by: BTreeMap<u32, HostCall> = BTreeMap::new();
        for call in calls.unwrap_or_default() {
            by.entry(call.pid)
                .and_modify(|seen| {
                    seen.peak = seen.peak.zip(call.peak).map(|(a, b)| a.max(b));
                    seen.running += call.running;
                })
                .or_insert(call);
        }
        let (mut need, mut unforeseen) = (0u64, BTreeMap::new());
        for call in by.values() {
            let peak = call.peak.unwrap_or(0);
            need = need.saturating_add(peak.saturating_sub(call.anon));
            unforeseen.insert(call.pid.to_string(), call.anon.saturating_sub(peak));
        }
        for (pid, was) in &self.running {
            for _ in by.get(pid).map_or(0, |call| call.running)..*was {
                self.growth.settle();
            }
        }
        self.running = by.values().map(|call| (call.pid, call.running)).collect();
        need.saturating_add(self.growth.observe(unforeseen))
    }
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// Where one machine's memory events go: `memory.log` in its store's `logs`, beside TensorFS's
/// transport log, read by `cozy machine logs --memory`. None logs to tracing.
#[derive(Clone, Debug, Default)]
pub struct MemoryLog(Option<Arc<PathBuf>>);

impl MemoryLog {
    pub fn at(dir: PathBuf) -> Self {
        Self(Some(Arc::new(dir)))
    }

    pub(crate) fn note(&self, event: serde_json::Value) {
        let Some(dir) = self.0.as_deref() else {
            return tracing::info!("memory: {event}");
        };
        let path = dir.join("memory.log");
        if fs::metadata(&path).is_ok_and(|meta| meta.len() > ROTATE_BYTES) {
            let _ = fs::rename(&path, dir.join("memory.log.1"));
        }
        let line = format!("{:.3} {event}\n", now_ms() as f64 / 1e3);
        let written = fs::create_dir_all(dir)
            .and_then(|_| OpenOptions::new().create(true).append(true).open(&path))
            .and_then(|mut file| file.write_all(line.as_bytes()));
        if let Err(error) = written {
            tracing::warn!("memory: {event} ({}: {error})", path.display());
        }
    }
}
