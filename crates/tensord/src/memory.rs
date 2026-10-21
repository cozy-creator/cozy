//! TensorD on varena's ledger (design varena-slim §2–§4). Each envelope GPU the inventory names
//! has a [`Ledger`] (capacity min(card, `gpu_memory_limit`)): an executor is granted it at spawn,
//! its row is reaped once its exit is proven, and admission is its `fits` (`admit`: a group on
//! all of its GPUs or none). TensorD holds no byte count and decides no eviction: kept-page
//! custody runs inside the ledger and the RAM tier ([`Tier`]) in this process. One thread blocks
//! on every ledger's events; on each wake the dispatch queue asks again, the rows the ledger names
//! stalled end, and changed peaks are saved.
use crate::machine::receipt;
use platform::contain::host_pressure::{self, HostMode};
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{self, Write},
    os::{
        fd::{AsRawFd, OwnedFd},
        unix::net::UnixStream,
    },
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};
use varena::{GpuId, Ledger, Peak, Pressure, Status, Tier};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// A row's classes in `RowStatus::classes` order (F, O, S, N, D, W, P): from S on, pages it maps.
const MAPPED: usize = 2;

/// One envelope GPU's ledger, and the GPU model its peaks are kept under.
pub struct Gpu {
    pub ledger: Ledger,
    pub model: String,
}

impl Gpu {
    /// `entry`'s ledger, sized by the inventory's memory (an integrated GPU's is the host's) and
    /// never above `limit`, with the peaks this machine measured on its model. None when the
    /// inventory does not name it or its table cannot be made.
    pub fn create(entry: &str, inventory: &[receipt::Gpu], limit: Option<u64>, peaks: &Peaks, log: &MemoryLog) -> Option<Gpu> {
        let row = inventory.iter().find(|g| g.device_index.to_string() == entry || g.device_uuid == entry)?;
        let id = GpuId { uuid: row.device_uuid.clone(), model: row.device_name.clone(), memory: row.memory_bytes };
        let ledger = match Ledger::create(&id, limit) {
            Ok(ledger) => ledger,
            Err(error) => {
                tracing::warn!("memory: GPU {entry} has no ledger: {error}");
                return None;
            }
        };
        peaks.restore(&id.model, &ledger);
        log.note(serde_json::json!({"event": "ledger", "gpu": entry, "card": id.memory, "capacity": ledger.status().capacity}));
        Some(Gpu { ledger, model: id.model })
    }

    /// A request runs here now, by any process's row, or is dispatched to one.
    pub fn running(&self) -> bool {
        let status = self.ledger.status();
        !status.dispatched.is_empty() || status.rows.iter().any(|r| r.live && (r.remaining > 0 || r.unmeasured > 0))
    }

    /// `pid`'s live row maps weights here.
    pub fn maps(&self, pid: u32) -> bool {
        let status = self.ledger.status();
        status.rows.iter().any(|r| r.pid == pid && r.live && r.classes.iter().skip(MAPPED).any(|bytes| *bytes > 0))
    }

    /// The table at a glance, for the memory log.
    pub fn status(&self) -> Status {
        self.ledger.status()
    }
}

/// Measured peaks, persisted per GPU model as `<dir>/<model>.json`. A key names its package
/// release (`<construction>|<shape>`, or a set's `<package>@<release>/<component>|…`), so one
/// file holds every release's; a key measured on several GPUs of the model keeps its largest.
pub struct Peaks {
    dir: PathBuf,
    saved: Mutex<BTreeMap<String, BTreeMap<String, Peak>>>,
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
            read.unwrap_or_default().into_iter().map(|peak| (peak.key.clone(), peak)).collect()
        });
        if let Err(error) = ledger.restore(&held.values().cloned().collect::<Vec<_>>()) {
            tracing::warn!("memory: {model}'s peaks not restored: {error}");
        }
    }

    /// Saves what `ledgers` (all of `model`) measured, when it changed.
    pub fn save<'a>(&self, model: &str, ledgers: impl Iterator<Item = &'a Ledger>) {
        let mut saved = self.saved.lock().unwrap();
        let held = saved.entry(model.into()).or_default();
        let mut changed = false;
        for peak in ledgers.flat_map(Ledger::peaks) {
            if held.get(&peak.key).is_none_or(|kept| kept.bytes < peak.bytes) {
                held.insert(peak.key.clone(), peak);
                changed = true;
            }
        }
        if !changed {
            return;
        }
        let written = fs::create_dir_all(&self.dir).map_err(platform::Error::from).and_then(|()| {
            let peaks: Vec<&Peak> = held.values().collect();
            platform::durable::replace(&self.path(model), &serde_json::to_vec(&peaks).unwrap_or_default(), 0o600)
        });
        if let Err(error) = written {
            tracing::warn!("memory: {model}'s peaks not saved: {error}");
        }
    }
}

/// Blocks on every ledger's events (its own copies of them) until the returned end drops;
/// `wake` runs once per wake. It never times out: a ledger signals each change.
pub fn react(ledgers: &[&Ledger], wake: impl Fn() + Send + 'static) -> io::Result<UnixStream> {
    let events: Vec<OwnedFd> = ledgers.iter().map(|ledger| ledger.events().try_clone()).collect::<io::Result<_>>()?;
    let (held, end) = UnixStream::pair()?;
    std::thread::Builder::new().name("memory-events".into()).spawn(move || {
        let watched = std::iter::once(end.as_raw_fd()).chain(events.iter().map(AsRawFd::as_raw_fd));
        let mut fds: Vec<libc::pollfd> = watched.map(|fd| libc::pollfd { fd, events: libc::POLLIN, revents: 0 }).collect();
        loop {
            // SAFETY: poll on descriptors this thread owns; no timeout.
            if unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) } < 0 {
                if io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return tracing::warn!("memory: events: {}", io::Error::last_os_error());
            }
            let (pool, ledgers) = fds.split_at_mut(1);
            if pool.iter().any(|fd| fd.revents != 0) {
                return;
            }
            for fd in ledgers.iter_mut().filter(|fd| fd.revents != 0) {
                let mut count = [0u8; 8];
                // SAFETY: a read into a local buffer from a descriptor poll found readable.
                let read = fd.revents & libc::POLLIN != 0 && unsafe { libc::read(fd.fd, count.as_mut_ptr().cast(), count.len()) } > 0;
                if !read {
                    fd.fd = -1; // closed or failed: poll skips it from now on
                }
            }
            wake();
        }
    })?;
    Ok(held)
}

/// The machine's RAM tier, served from this process while it lives: on the host's pressure
/// trigger, else (no trigger here) at its headroom check at every host decision.
pub struct HostTier {
    _tier: Option<Tier>,
}

impl HostTier {
    pub fn serve(mode: HostMode) -> Self {
        let pressure = host_pressure::Pressure::arm(mode).map(Pressure::Host).unwrap_or_else(|error| {
            tracing::warn!("memory: no host pressure trigger ({error}); the RAM tier gives way at its headroom checks");
            Pressure::Headroom
        });
        Self { _tier: Tier::serve(pressure).map_err(|error| tracing::warn!("memory: no RAM tier: {error}")).ok().flatten() }
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
