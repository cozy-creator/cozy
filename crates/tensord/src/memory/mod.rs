//! The one owner of every GPU memory decision (PLAN decision 3, design rust-machine.md §2.1–2.2).
//! `policy` decides from samples and executor facts; `nvml` samples on its own thread; this
//! module joins them and keeps the floor while a call runs.
pub mod call;
pub mod nvml;
pub mod policy;

use policy::{Ask, Facts, Gpu, Holding, Round, Sample, Step};
use serde::Deserialize;
use std::{
    fs::{self, File, OpenOptions},
    io::{self, Write},
    os::unix::fs::FileExt,
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

/// Budget-cell `wanted` at or below this asks for a process cap of `CAP - wanted` bytes
/// (Runtime `budget_cell.CAP`, `process_cap/1`).
const CELL_CAP: i64 = -2;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Raises the free floor kept on the GPU (never lowers the default).
    pub floor_bytes: u64,
    /// Append every one-second reading here (JSON lines), for evidence.
    pub sample_log: Option<PathBuf>,
}

/// Ask a running executor for a smaller (or larger) process cap at its next block boundary.
/// Wanted first, then the counter it guards, as the executor reads them.
pub fn ask_cap(cell: &File, cap: u64) -> io::Result<i64> {
    let mut word = [0u8; 8];
    cell.read_exact_at(&mut word, 0)?;
    let asked = i64::from_le_bytes(word) + 1;
    cell.write_all_at(&(CELL_CAP - cap as i64).to_le_bytes(), 8)?;
    cell.write_all_at(&asked.to_le_bytes(), 0)?;
    Ok(asked)
}

/// Each running call's plan and its cap cell: a served call and its child may run on one GPU.
type Running = std::collections::BTreeMap<String, Option<File>>;

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

/// Where one machine's memory decisions go: `memory.log` in its store's `logs`, beside
/// TensorFS's transport log, read by `cozy machine logs --memory`. Each GPU pool owns its own,
/// so two machines in one process never write into each other's store. None logs to tracing.
#[derive(Clone, Debug, Default)]
pub struct MemoryLog(Option<Arc<PathBuf>>);

impl MemoryLog {
    pub fn at(dir: PathBuf) -> Self {
        Self(Some(Arc::new(dir)))
    }

    /// One memory decision, a line of the machine's memory log.
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

/// One GPU's ledger, sampler and floor watchdog. Without NVML nothing is decided here and
/// executors read the driver alone.
pub struct GpuMemory {
    ledger: Arc<Mutex<Gpu>>,
    running: Arc<Mutex<Running>>,
    sampler: Option<nvml::Sampler>,
    device: String,
    log: MemoryLog,
}

impl GpuMemory {
    pub fn start(device: &str, config: &MemoryConfig, notes: MemoryLog) -> Self {
        let ledger = Arc::new(Mutex::new(Gpu::with_floor(config.floor_bytes)));
        let running: Arc<Mutex<Running>> = Arc::default();
        let log = config.sample_log.as_ref().and_then(|path| {
            OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .map_err(|error| tracing::warn!("memory: sample log {path:?} unavailable: {error}"))
                .ok()
        });
        let mut key = device.to_string();
        let sampler = match cozy_platform::gpu::Device::open(device) {
            Ok(opened) => {
                key = opened.key().to_string();
                let (watched, cells, notes, gpu) = (ledger.clone(), running.clone(), notes.clone(), key.clone());
                let log = Mutex::new(log);
                nvml::Sampler::start(opened, move |sample| {
                    let mut ledger = watched.lock().unwrap();
                    if let Some(file) = log.lock().unwrap().as_mut() {
                        let _ = writeln!(
                            file,
                            "{}",
                            serde_json::json!({"t_ms": now_ms() as u64, "total": sample.total,
                                "free": sample.free, "floor": ledger.floor(sample), "display": sample.display})
                        );
                    }
                    let Some((plan, cap)) = ledger.below_floor(sample) else {
                        return;
                    };
                    let running = cells.lock().unwrap();
                    let Some(Some(cell)) = running.get(&plan) else {
                        return;
                    };
                    match ask_cap(cell, cap) {
                        Ok(_) => {
                            ledger.set_cap(&plan, cap);
                            notes.note(serde_json::json!({"event":"floor","plan":plan,"cap":cap,
                                "gpu":gpu,"free":sample.free,"floor":ledger.floor(sample)}));
                        }
                        Err(error) => notes.note(serde_json::json!({"event":"floor_cell_failed",
                            "plan":plan,"error":error.to_string()})),
                    }
                })
                .map_err(|error| tracing::warn!("memory: sampler did not start: {error}"))
                .ok()
            }
            Err(error) => {
                tracing::warn!("memory: NVML unavailable for {device:?}: {error}; executors read the driver alone");
                None
            }
        };
        Self {
            ledger,
            running,
            sampler,
            device: key,
            log: notes,
        }
    }

    pub fn sample(&self) -> Option<Sample> {
        self.sampler.as_ref().and_then(nvml::Sampler::now)
    }

    /// Device bytes in use on this GPU now, by anyone. None: no NVML.
    pub fn used(&self) -> Option<u64> {
        self.sample().map(|s| s.total.saturating_sub(s.free))
    }

    pub fn with<T>(&self, f: impl FnOnce(&mut Gpu) -> T) -> T {
        f(&mut self.ledger.lock().unwrap())
    }

    /// Idle tenants give room for `plan` (`ask` says whose) until `short`, the bytes still
    /// missing by each new reading, is 0 or none is left: `holdings` reads custody before each
    /// look, `carry_out` performs each step. Never refuses. The bytes the GPU freed meanwhile,
    /// what is still missing, and the last reading; None without NVML (nothing is decided here).
    pub fn make_room(
        &self,
        plan: &str,
        ask: Ask,
        short: impl Fn(&Gpu, &Sample) -> u64,
        holdings: impl Fn() -> Vec<Holding>,
        mut carry_out: impl FnMut(&Step) -> io::Result<bool>,
    ) -> io::Result<Option<(u64, u64, Sample)>> {
        let mut round = Round::default();
        let mut first = None;
        loop {
            let Some(sample) = self.sample() else {
                return Ok(None);
            };
            let first = *first.get_or_insert(sample.free);
            let held = holdings();
            let (missing, step) = self.with(|gpu| {
                gpu.holdings = held;
                let missing = short(gpu, &sample);
                (missing, (missing > 0).then(|| gpu.step(plan, ask, missing, &mut round)).flatten())
            });
            self.log.note(serde_json::json!({"event": "room", "plan": plan, "ask": format!("{ask:?}"),
                "gpu": self.device, "free": sample.free, "short": missing, "step": format!("{step:?}")}));
            match step {
                Some(step) => {
                    carry_out(&step)?;
                }
                None => return Ok(Some((sample.free.saturating_sub(first), missing, sample))),
            }
        }
    }

    /// A running call's `DeviceRoom`: idle tenants give room until `free_bytes` are free beside
    /// the floor or none is left, then `plan`'s cap rises into what is there, through its rank
    /// cell when it has one (a cap the floor watchdog lowered comes back). The bytes freed and
    /// the cap; None without NVML.
    pub fn room_for(
        &self,
        plan: &str,
        free_bytes: u64,
        cell: Option<&File>,
        holdings: impl Fn() -> Vec<Holding>,
        carry_out: impl FnMut(&Step) -> io::Result<bool>,
    ) -> io::Result<Option<(u64, u64)>> {
        let short = |gpu: &Gpu, s: &Sample| (free_bytes + gpu.floor(s)).saturating_sub(s.free);
        let Some((freed, _, sample)) = self.make_room(plan, Ask::Call, short, holdings, carry_out)? else {
            return Ok(None);
        };
        Ok(Some((freed, self.grant_room(plan, &sample, cell)?)))
    }

    /// Grant the room there is and notify this rank through its budget cell: what the process
    /// holds and everything free above the floor. The ledger lock serializes the grant and its
    /// publication with the watchdog.
    fn grant_room(&self, plan: &str, sample: &Sample, cell: Option<&File>) -> io::Result<u64> {
        self.with(|gpu| {
            let cap = gpu.room(plan, sample);
            if let Some(cell) = cell {
                ask_cap(cell, cap)?;
            }
            gpu.set_cap(plan, cap);
            Ok(cap)
        })
    }

    /// Whether `plan` takes Degree 2 now (`Gpu::fits_resident`). False when NVML cannot say.
    pub fn fits_resident(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        let Some(sample) = self.sample() else {
            return false;
        };
        let held = holdings();
        let (fits, want) = self.with(|gpu| {
            gpu.holdings = held;
            (gpu.fits_resident(plan, &sample), gpu.want(plan))
        });
        self.log.note(serde_json::json!({"event": "degree2", "plan": plan, "fits": fits, "want": want}));
        fits
    }

    /// Whether `plan`'s executor and first working set fit in the room there is now, with no
    /// step on another tenant (a prefetch never makes room). True without NVML.
    pub fn admits(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        self.within(holdings, |gpu, sample| gpu.room(plan, sample) >= gpu.spawn_need(plan))
    }

    /// Whether a call of `plan` fits here now with no step on another tenant (`Gpu::fits`).
    /// True without NVML.
    pub fn fits(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        self.within(holdings, |gpu, sample| gpu.fits(plan, sample))
    }

    fn within(&self, holdings: impl Fn() -> Vec<Holding>, test: impl FnOnce(&Gpu, &Sample) -> bool) -> bool {
        let Some(sample) = self.sample() else {
            return true;
        };
        let held = holdings();
        self.with(|gpu| {
            gpu.holdings = held;
            test(gpu, &sample)
        })
    }

    /// The floor this GPU keeps now.
    pub fn floor(&self) -> Option<u64> {
        let sample = self.sample()?;
        Some(self.with(|gpu| gpu.floor(&sample)))
    }

    pub fn observe(&self, plan: &str, facts: Facts, mapped: Option<bool>) {
        self.with(|gpu| gpu.observe(plan, facts, mapped));
    }

    /// A call starts uncapped; its cell (if any) is what the floor watchdog lowers.
    pub fn running(&self, plan: &str, cell: Option<File>) {
        self.with(|gpu| gpu.active(plan));
        self.running.lock().unwrap().insert(plan.into(), cell);
    }

    pub fn finished(&self, plan: &str) {
        self.running.lock().unwrap().remove(plan);
        self.with(|gpu| gpu.idle(plan));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use policy::{HEADLESS_FLOOR, MIB};
    const GIB: u64 = 1 << 30;

    fn memory() -> GpuMemory {
        let mut gpu = Gpu::default();
        gpu.starting("plan");
        gpu.active("plan");
        GpuMemory {
            ledger: Arc::new(Mutex::new(gpu)),
            running: Arc::default(),
            sampler: None,
            device: String::new(),
            log: MemoryLog::default(),
        }
    }

    fn sample(room: u64) -> Sample {
        Sample {
            total: room + HEADLESS_FLOOR,
            free: room + HEADLESS_FLOOR,
            ..Default::default()
        }
    }

    fn read_cell(cell: &File) -> [i64; 4] {
        let mut bytes = [0u8; 32];
        cell.read_exact_at(&mut bytes, 0).unwrap();
        std::array::from_fn(|i| i64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap()))
    }

    #[test]
    fn room_grants_reach_each_rank_without_borrowing_another_gpus_cap() {
        let leader = cozy_platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        leader.set_len(32).unwrap();
        let follower = cozy_platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        follower.set_len(32).unwrap();
        let leader_memory = memory();
        let follower_memory = memory();
        assert_eq!(
            leader_memory
                .grant_room("plan", &sample(30), Some(&leader))
                .unwrap(),
            30
        );
        assert_eq!(
            follower_memory
                .grant_room("plan", &sample(20), Some(&follower))
                .unwrap(),
            20
        );
        assert_eq!(read_cell(&leader), [1, CELL_CAP - 30, 0, 0]);
        assert_eq!(read_cell(&follower), [1, CELL_CAP - 20, 0, 0]);
        assert_eq!(
            leader_memory.with(|gpu| gpu.tenant("plan").unwrap().cap),
            30
        );
        assert_eq!(
            follower_memory.with(|gpu| gpu.tenant("plan").unwrap().cap),
            20
        );

        // The floor watchdog's later reduction uses the same sequence. A following
        // room grant starts from the reduced authority, not the previous grant.
        follower_memory.with(|gpu| {
            ask_cap(&follower, 5).unwrap();
            gpu.set_cap("plan", 5);
        });
        assert_eq!(read_cell(&follower), [2, CELL_CAP - 5, 0, 0]);
        assert_eq!(
            follower_memory
                .grant_room("plan", &sample(10), Some(&follower))
                .unwrap(),
            10
        );
        assert_eq!(read_cell(&follower), [3, CELL_CAP - 10, 0, 0]);
    }

    #[test]
    fn a_room_grant_below_the_floor_never_raises_the_cap_the_watchdog_lowered() {
        // Run 5424 (an 8 GB laptop card): the card was below its floor, and the grant answered
        // with what the process held, above the watchdog's cap.
        let cell = cozy_platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        cell.set_len(32).unwrap();
        let memory = memory();
        memory.observe(
            "plan",
            policy::Facts {
                process: Some(7 * GIB),
                ..Default::default()
            },
            None,
        );
        let below = Sample {
            total: 8 * GIB,
            free: HEADLESS_FLOOR - 100 * MIB,
            ..Default::default()
        };
        let (_, lowered) = memory.with(|gpu| gpu.below_floor(&below).unwrap());
        memory.with(|gpu| gpu.set_cap("plan", lowered));
        let granted = memory.grant_room("plan", &below, Some(&cell)).unwrap();
        assert_eq!(granted, 7 * GIB - 100 * MIB);
        assert!(granted <= lowered + policy::MARGIN);
        assert_eq!(read_cell(&cell), [1, CELL_CAP - granted as i64, 0, 0]);
    }

    #[test]
    fn a_missing_room_cell_keeps_the_scalar_grant_available() {
        let memory = memory();
        assert_eq!(memory.grant_room("plan", &sample(30), None).unwrap(), 30);
        assert_eq!(memory.with(|gpu| gpu.tenant("plan").unwrap().cap), 30);
    }

    #[test]
    fn an_unmeasured_gpu_publishes_no_room_grant() {
        let cell = cozy_platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        cell.set_len(32).unwrap();
        let memory = memory();
        assert_eq!(
            memory
                .room_for("plan", 30, Some(&cell), Vec::new, |_| {
                    panic!("an unmeasured GPU must not reclaim another tenant")
                })
                .unwrap(),
            None
        );
        assert_eq!(memory.with(|gpu| gpu.tenant("plan").unwrap().cap), u64::MAX);
        assert_eq!(read_cell(&cell), [0; 4]);
    }

    #[test]
    fn a_failed_room_cell_does_not_record_an_unpublished_cap() {
        let cell = cozy_platform::shm::memfd_named("tensord-weights").map(std::fs::File::from).unwrap();
        cell.set_len(32).unwrap();
        cozy_platform::shm::seal_with(std::os::fd::AsFd::as_fd(&cell), cozy_platform::shm::NO_WRITE_SEALS).unwrap();
        let memory = memory();
        assert!(memory.grant_room("plan", &sample(30), Some(&cell)).is_err());
        assert_eq!(memory.with(|gpu| gpu.tenant("plan").unwrap().cap), u64::MAX);
        assert_eq!(read_cell(&cell), [0; 4]);
    }
}
