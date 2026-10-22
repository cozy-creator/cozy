//! The host tier's room (MEM/HOST-PRESSURE.md): the layouts TensorD keeps for later executors
//! (`Deposits`), each live call's measured host need, and two rules on them. A layout seals only
//! into measured surplus: the live budget less what must stay free, which is each live call's
//! measured peak less its anonymous bytes now, plus `Growth`; while any live call is unmeasured,
//! nothing seals. Under pressure, rungs go back while the budget is below that same need.
use crate::{deposits::Deposits, memory::MemoryLog};
use platform::contain::{
    host_memory,
    host_pressure::{self, Growth, HostMode, Pressure, Reserve},
};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex,
    },
};

/// A call in flight: the construction its peak is kept under, its shape once known, and the
/// processes that run it.
struct Live {
    construction: String,
    shape: Option<String>,
    pids: Vec<u32>,
}

pub struct HostTier {
    pub deposits: Deposits,
    mode: HostMode,
    notes: MemoryLog,
    live: Mutex<BTreeMap<String, Live>>,
    /// Each `<construction>|<shape>`'s measured host peak: its processes' anonymous high-water over
    /// a call. Kept in `host-peaks.json` across restarts, under the GPU ledger's peak keys.
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
}

impl HostTier {
    pub fn new(root: &Path, mode: HostMode, notes: MemoryLog) -> Self {
        let path = root.join("host-peaks.json");
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

    /// What must stay free, and whether every live call is measured (an unmeasured one adds
    /// nothing it could be measured to need).
    fn need(&self) -> (u64, bool) {
        let live = self.live.lock().unwrap();
        let peaks = self.peaks.lock().unwrap();
        let mut need = self.growth.lock().unwrap().most();
        let mut measured = true;
        for call in live.values() {
            let prefix = format!("{}|", call.construction);
            let peak = match &call.shape {
                Some(shape) => peaks.get(&format!("{prefix}{shape}")).copied(),
                None => peaks.iter().filter(|(key, _)| key.starts_with(&prefix)).map(|(_, peak)| *peak).max(),
            };
            let Some(peak) = peak else {
                measured = false;
                continue;
            };
            let now: u64 = call.pids.iter().filter_map(|pid| host_memory::anon(*pid).ok()).map(|anon| anon.now).sum();
            need = need.saturating_add(peak.saturating_sub(now));
        }
        (need, measured)
    }

    /// Whether `call` may seal layouts of `bytes` (every component it selects of their checkpoint,
    /// so its layouts of it fit together): no other call's layout fills now, every live call is
    /// measured, and the budget holds them beside what must stay free. Layouts no live executor
    /// maps make room for it, least recently granted first, when that is enough.
    pub fn admit(&self, call: &str, bytes: u64) -> bool {
        if self.deposits.filling_beside(call) {
            return false;
        }
        let (need, measured) = self.need();
        let want = need.saturating_add(bytes);
        let fits = || self.room().is_some_and(|room| room >= want);
        if measured && !fits() && self.room().is_some_and(|room| room + self.deposits.idle_bytes() >= want) {
            while !fits() && self.deposits.release_one() {}
        }
        measured && fits()
    }

    /// `call` of `construction` runs in `pids` from now until the guard drops.
    pub fn begin(&self, call: &str, construction: &str, pids: Vec<u32>) -> Call<'_> {
        let live = Live { construction: construction.into(), shape: None, pids: vec![] };
        self.live.lock().unwrap().insert(call.into(), live);
        let call = Call(self, call.into());
        call.pids(pids);
        call
    }

    /// The watcher (`host_pressure`): one thread blocked until the kernel reports pressure, then
    /// rungs `give` back (None: its owner is gone) while the budget is below what must stay free.
    /// The memory log says what it is armed on, or why nothing could be.
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
        self.growth.lock().unwrap().observe(host_memory::read());
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
        self.growth.lock().unwrap().observe(host_memory::read());
        let (mut gave, mut short) = (0, None);
        loop {
            let (need, _) = self.need();
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
            let live = self.live.lock().unwrap();
            live.get(call).and_then(|live| {
                let peak = live.pids.iter().map(|pid| host_memory::anon(*pid).map(|anon| anon.peak)).sum::<std::io::Result<u64>>();
                Some((format!("{}|{}", live.construction, live.shape.as_ref()?), peak.ok()?))
            })
        };
        let Some((key, peak)) = measured else { return };
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

/// A live call (`HostTier::begin`).
pub struct Call<'a>(&'a HostTier, String);

impl Call<'_> {
    /// The processes that run it; each one's high-water starts over now.
    pub fn pids(&self, pids: Vec<u32>) {
        for pid in &pids {
            if let Err(error) = host_memory::reset_peak(*pid) {
                tracing::debug!("host peak of {pid} not reset: {error}");
            }
        }
        if let Some(live) = self.0.live.lock().unwrap().get_mut(&self.1) {
            live.pids = pids;
        }
    }

    pub fn shape(&self, shape: &str) {
        if let Some(live) = self.0.live.lock().unwrap().get_mut(&self.1) {
            live.shape = Some(shape.into());
        }
    }

    /// Its processes answered: their high-water over the call is its shape's host peak.
    pub fn measured(&self) {
        self.0.measured(&self.1);
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        self.0.live.lock().unwrap().remove(&self.1);
    }
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    /// A first run is measured, never admitted; once measured, a layout seals where the budget
    /// holds it beside the call's need, and the measure survives a restart.
    #[test]
    fn a_layout_seals_only_into_measured_surplus() {
        let dir = tempfile::tempdir().unwrap();
        let tier = HostTier::new(dir.path(), HostMode::Dedicated, MemoryLog::default());
        let mut executor = std::process::Command::new("sleep").arg("600").spawn().unwrap();
        {
            let call = tier.begin("run-a", "plan-x", vec![executor.id()]);
            assert!(!tier.admit("run-a", 1 << 20), "its host need is unmeasured");
            call.shape("s1");
            call.measured();
        }
        let peaks: BTreeMap<String, u64> = serde_json::from_slice(&fs::read(dir.path().join("host-peaks.json")).unwrap()).unwrap();
        assert!(peaks["plan-x|s1"] > 0, "{peaks:?}");
        let tier = HostTier::new(dir.path(), HostMode::Dedicated, MemoryLog::default());
        let _call = tier.begin("run-b", "plan-x", vec![executor.id()]);
        assert!(tier.admit("run-b", 1 << 20), "measured, and the budget holds it");
        assert!(!tier.admit("run-b", u64::MAX / 2), "more than the budget");
        let other = tier.begin("run-c", "plan-y", vec![]);
        assert!(!tier.admit("run-b", 1 << 20), "a live call is unmeasured");
        drop(other);
        assert_eq!(tier.deposits.seal("m", "l", "run-c#0", true), Ok(true));
        assert!(!tier.admit("run-b", 1 << 20), "another call's layout seals now");
        tier.deposits.ended("run-c");
        assert!(tier.admit("run-b", 1 << 20));
        executor.kill().unwrap();
        executor.wait().unwrap();
    }
}
