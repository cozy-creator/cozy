//! The host tier's room: the layouts TensorD keeps for later executors (`Deposits`), each live
//! call's measured host need, and two rules on them. A layout seals only into measured surplus:
//! the live budget less what must stay free, and while any live call is unmeasured nothing
//! seals. Under pressure, rungs go back while the budget is below what must stay free.
//!
//! What must stay free is compared like with like. The budget (`host_memory::read`) already
//! holds every resident page, a sealed layout's once however many executors map it. So a call
//! needs only its own anonymous growth to come: its shape's anonymous peak less its processes'
//! anonymous bytes now. To that come `Growth` and the unfilled bytes of admitted fills
//! (`Deposits::reserved`).
//!
//! Linux keeps no anonymous high-water, only a resident one (`VmHWM`). Each look bounds a
//! process's anonymous peak since its last look as that high-water less the smaller of its file
//! bytes then and now and less its shmem bytes now, then starts its high-water over. Shmem is
//! sealed layouts, which only grow while mapped (their fill is admitted and reserved, never a
//! call's need), so it counts at its latest; the bound is exact while a layout stays mapped.
use crate::{deposits::Deposits, memory::MemoryLog};
use platform::contain::{
    host_memory,
    host_pressure::{self, Growth, HostMode, Nudge, Pressure, Reserve},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
};

/// A call in flight: the construction its peak is kept under, its shape once known, and the
/// processes that run it, each with its measure so far.
struct Live {
    construction: String,
    shape: Option<String>,
    pids: BTreeMap<u32, Window>,
}

impl Live {
    fn key(&self) -> Option<String> {
        Some(format!("{}|{}", self.construction, self.shape.as_ref()?))
    }
}

/// One process over a call: the most anonymous bytes it can have held, and its file bytes at
/// its last look.
#[derive(Clone, Copy, Default)]
struct Window {
    peak: u64,
    file: u64,
}

impl Window {
    /// Measured from now.
    fn start(pid: u32) -> Self {
        let now = host_memory::resident(pid).unwrap_or_default();
        restart(pid);
        Self { peak: now.anon, file: now.file }
    }

    /// A look: the bound since the last one is folded in and the high-water starts over. Its
    /// anonymous bytes now; None: it is gone.
    fn look(&mut self, pid: u32) -> Option<u64> {
        let now = host_memory::resident(pid).ok()?;
        let bound = now.peak.saturating_sub(self.file.min(now.file) + now.shmem);
        self.peak = self.peak.max(bound).max(now.anon);
        self.file = now.file;
        restart(pid);
        Some(now.anon)
    }
}

/// `pid`'s resident high-water starts over (where it may not, it bounds from further back).
fn restart(pid: u32) {
    if let Err(error) = host_memory::reset_peak(pid) {
        tracing::debug!("host peak of {pid} not reset: {error}");
    }
}

pub struct HostTier {
    pub deposits: Deposits,
    mode: HostMode,
    notes: MemoryLog,
    live: Mutex<BTreeMap<String, Live>>,
    /// Each `<construction>|<shape>`'s anonymous peak over its processes, kept across restarts.
    peaks: Mutex<BTreeMap<String, u64>>,
    path: PathBuf,
    growth: Mutex<Growth>,
    /// Shared mode's room for the host's other programs, and the shortfalls the watcher saw: the
    /// count at the last bring-back pass says whether one came since.
    reserve: Mutex<Reserve>,
    short: AtomicU64,
    short_seen: AtomicU64,
    /// Whether the last look was short: a shortfall is noted when it starts or gives something.
    was_short: AtomicBool,
    nudge: OnceLock<Nudge>,
}

impl HostTier {
    pub fn new(root: &Path, mode: HostMode, notes: MemoryLog) -> Self {
        // Resident high-waters (0.11.2) counted mapped layouts: they are no anonymous peaks.
        let _ = fs::remove_file(root.join("host-peaks.json"));
        let path = root.join("host-anon.json");
        let peaks = fs::read(&path).ok().and_then(|bytes| serde_json::from_slice(&bytes).ok()).unwrap_or_default();
        Self {
            deposits: Deposits::default(),
            mode,
            notes,
            live: Mutex::default(),
            peaks: Mutex::new(peaks),
            path,
            growth: Mutex::default(),
            reserve: Mutex::default(),
            short: AtomicU64::new(0),
            short_seen: AtomicU64::new(0),
            was_short: AtomicBool::new(false),
            nudge: OnceLock::new(),
        }
    }

    /// Host bytes discretionary holdings may take now (a warm member, an import ahead, a layout):
    /// the tightest limit's live budget, less, in shared mode, what the host's other programs used
    /// recently beyond their use now. None: unreadable.
    pub fn room(&self) -> Option<u64> {
        let host = host_memory::read();
        let available = u64::try_from(host.available).ok()?;
        if self.mode == HostMode::Dedicated {
            return Some(available);
        }
        let ours = host_memory::tree_private(std::process::id());
        let total = u64::try_from(host_memory::total()).ok()?;
        let others = host_pressure::others(&host, total, ours)?;
        Some(available.saturating_sub(self.reserve.lock().unwrap().observe(others)))
    }

    /// A bring-back pass: with no shortfall since the last one, the reserve forgets half.
    pub fn quiet(&self) {
        let short = self.short.load(Ordering::Acquire);
        if self.short_seen.swap(short, Ordering::AcqRel) != short {
            return;
        }
        let host = host_memory::read();
        let ours = host_memory::tree_private(std::process::id());
        let others = u64::try_from(host_memory::total()).ok().and_then(|total| host_pressure::others(&host, total, ours));
        if let Some(others) = others {
            self.reserve.lock().unwrap().quiet(others);
        }
    }

    /// A look at every live call: what its processes must still be free to grow, plus `Growth`,
    /// and whether every live call is measured (an unmeasured one adds nothing it could be
    /// measured to need). Admitted fills' unfilled bytes come on top (`Deposits::reserved`).
    fn need(&self) -> (u64, bool) {
        let mut live = self.live.lock().unwrap();
        let peaks = self.peaks.lock().unwrap();
        let (mut need, mut measured, mut unforeseen) = (0u64, true, BTreeMap::new());
        for (id, call) in live.iter_mut() {
            let now: u64 = call.pids.iter_mut().filter_map(|(pid, window)| window.look(*pid)).sum();
            let prefix = format!("{}|", call.construction);
            let peak = match call.key() {
                Some(key) => peaks.get(&key).copied(),
                None => peaks.iter().filter(|(key, _)| key.starts_with(&prefix)).map(|(_, peak)| *peak).max(),
            };
            measured &= peak.is_some();
            need = need.saturating_add(peak.unwrap_or(0).saturating_sub(now));
            unforeseen.insert(id.clone(), now.saturating_sub(peak.unwrap_or(0)));
        }
        (need.saturating_add(self.growth.lock().unwrap().observe(unforeseen)), measured)
    }

    /// Whether a layout of `bytes` seals beside `reserved` bytes admitted fills have still to
    /// fill. Called under `Deposits::seal`'s lock, so admissions are one at a time. Idle layouts
    /// make room, least recently granted first, only when doing so can satisfy the whole
    /// admission.
    pub fn admit(&self, bytes: u64, reserved: u64) -> Admission {
        let (need, measured) = self.need();
        let want = need.saturating_add(reserved).saturating_add(bytes);
        let why = if !measured {
            "a live call is unmeasured"
        } else {
            if self.room().is_some_and(|room| room < want && room + self.deposits.idle_bytes() >= want) {
                while self.room().is_some_and(|room| room < want) && self.deposits.release_one() {}
            }
            "measured"
        };
        let room = self.room();
        let seal = why == "measured" && room.is_some_and(|room| room >= want);
        Admission { seal, why, room, need, reserved, bytes }
    }

    /// `call` of `construction` runs in `pids` from now until the guard drops.
    pub fn begin(&self, call: &str, construction: &str, pids: Vec<u32>) -> Call<'_> {
        let live = Live { construction: construction.into(), shape: None, pids: BTreeMap::new() };
        self.live.lock().unwrap().insert(call.into(), live);
        let call = Call(self, call.into());
        call.pids(pids);
        self.nudge();
        call
    }

    /// Something happened (a call began, an executor sent a frame): the watcher looks.
    pub fn nudge(&self) {
        if let Some(nudge) = self.nudge.get() {
            nudge.send();
        }
    }

    /// The watcher (`host_pressure`): one thread blocked until the kernel reports pressure or a
    /// nudge comes, then rungs `give` back (None: its owner is gone) while the budget is below
    /// what must stay free. The memory log says what it is armed on.
    pub fn watch(self: &Arc<Self>, give: impl Fn() -> Option<bool> + Send + 'static) {
        let mode = format!("{:?}", self.mode);
        let mut pressure = match Pressure::arm(self.mode) {
            Ok(pressure) => pressure,
            Err(error) => {
                tracing::error!("host pressure: nothing to watch ({error}); memory goes back only as calls plan");
                return self.notes.note(serde_json::json!({"event": "host_pressure_unarmed", "mode": mode, "error": error.to_string()}));
            }
        };
        tracing::info!("host pressure: {}", pressure.how());
        self.notes.note(serde_json::json!({"event": "host_pressure_armed", "mode": mode, "how": pressure.how()}));
        let _ = self.nudge.set(pressure.nudge());
        let tier = Arc::downgrade(self);
        let started = std::thread::Builder::new().name("host-pressure".into()).spawn(move || {
            while pressure.wait().is_ok() {
                let Some(tier) = tier.upgrade() else { break };
                if tier.answer(&give).is_none() {
                    break;
                }
            }
        });
        if let Err(error) = started {
            tracing::error!("host pressure: {error}");
        }
    }

    /// One look: rungs go back while the budget is short of the need.
    fn answer(&self, give: &impl Fn() -> Option<bool>) -> Option<()> {
        let (mut gave, mut short) = (0, None);
        loop {
            let need = self.need().0.saturating_add(self.deposits.reserved());
            let Some(room) = self.room().filter(|room| *room < need) else { break };
            short = Some((room, need));
            if !give()? {
                break;
            }
            gave += 1;
        }
        let was = self.was_short.swap(short.is_some(), Ordering::AcqRel);
        if let Some((room, need)) = short {
            self.short.fetch_add(1, Ordering::AcqRel);
            if gave > 0 || !was {
                self.notes.note(serde_json::json!({"event": "host_pressure", "room": room, "need": need, "gave": gave}));
            }
        }
        Some(())
    }

    fn measured(&self, call: &str) {
        let measured = {
            let mut live = self.live.lock().unwrap();
            live.get_mut(call).and_then(|live| {
                let looked = live.pids.iter_mut().all(|(pid, window)| window.look(*pid).is_some());
                let peak = live.pids.values().map(|window| window.peak).sum::<u64>();
                looked.then_some((live.key()?, peak))
            })
        };
        let Some((key, peak)) = measured else { return };
        self.growth.lock().unwrap().settle();
        let mut peaks = self.peaks.lock().unwrap();
        if peaks.get(&key).is_some_and(|kept| *kept >= peak) {
            return;
        }
        peaks.insert(key.clone(), peak);
        let written = serde_json::to_vec(&*peaks).map_err(std::io::Error::other).and_then(|bytes| {
            let partial = self.path.with_extension("partial");
            fs::write(&partial, bytes).and_then(|()| fs::rename(&partial, &self.path))
        });
        drop(peaks);
        if let Err(error) = written {
            tracing::warn!("{}: {error}", self.path.display());
        }
        self.notes.note(serde_json::json!({"event": "host_peak", "key": key, "peak": peak}));
    }
}

/// One read_grant's verdict on sealing a layout, as the memory log records it.
#[derive(Debug, serde::Serialize)]
pub struct Admission {
    pub seal: bool,
    /// "measured": room against need, reserved and bytes decided it.
    pub why: &'static str,
    /// The live budget it was measured against, what live calls must stay free to grow
    /// (`need`), and what admitted fills have still to fill (`reserved`), beside `bytes`.
    pub room: Option<u64>,
    pub need: u64,
    pub reserved: u64,
    pub bytes: u64,
}

/// A live call (`HostTier::begin`).
pub struct Call<'a>(&'a HostTier, String);

impl Call<'_> {
    /// The processes that run it; a process new to it is measured from now.
    pub fn pids(&self, pids: Vec<u32>) {
        if let Some(live) = self.0.live.lock().unwrap().get_mut(&self.1) {
            live.pids.retain(|pid, _| pids.contains(pid));
            for pid in pids {
                live.pids.entry(pid).or_insert_with(|| Window::start(pid));
            }
        }
    }

    pub fn shape(&self, shape: &str) {
        if let Some(live) = self.0.live.lock().unwrap().get_mut(&self.1) {
            live.shape = Some(shape.into());
        }
    }

    /// Its processes answered: their anonymous peaks are its shape's.
    pub fn measured(&self) {
        self.0.measured(&self.1);
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        self.0.live.lock().unwrap().remove(&self.1);
    }
}
