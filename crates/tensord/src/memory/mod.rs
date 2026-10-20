//! TensorD holds process identities and forwards placement grants. varena owns GPU policy,
//! custody and RAM fills; the Runtime publishes measured working sets to the shared ledger.
pub mod call;
pub mod learned;
pub mod policy;
pub mod slim;

use policy::{Facts, Gpu, Holding, Step};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    pub floor_bytes: u64,
    pub sample_log: Option<PathBuf>,
}
#[derive(Clone, Debug, Default)]
pub struct MemoryLog(Option<Arc<PathBuf>>);
impl MemoryLog {
    pub fn at(dir: PathBuf) -> Self {
        Self(Some(Arc::new(dir)))
    }
    pub(crate) fn note(&self, event: serde_json::Value) {
        let Some(dir) = self.0.as_deref() else {
            tracing::info!("memory: {event}");
            return;
        };
        let write = fs::create_dir_all(dir)
            .and_then(|_| {
                OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(dir.join("memory.log"))
            })
            .and_then(|mut file| writeln!(file, "{event}"));
        if let Err(error) = write {
            tracing::warn!(%error,"memory log");
        }
    }
}
pub struct GpuMemory {
    records: Mutex<Gpu>,
    placement: Option<slim::Memory>,
    log: MemoryLog,
}
impl GpuMemory {
    pub fn start(device: &str, _: &MemoryConfig, root: &std::path::Path, log: MemoryLog) -> Self {
        let mut records = Gpu::default();
        records.device = device.into();
        records.learned = learned::Learned::open(&root.join("memory-learned.json"));
        Self {
            records: Mutex::new(records),
            placement: None,
            log,
        }
    }
    pub fn with_placement(mut self, placement: Option<slim::Memory>) -> Self {
        self.placement = placement;
        self
    }
    pub fn placement(&self) -> std::io::Result<&slim::Memory> {
        self.placement
            .as_ref()
            .ok_or_else(|| std::io::Error::other("GPU inventory is required for placement"))
    }
    pub fn capacity(&self) -> Option<u64> {
        self.placement.as_ref().map(|m| m.ledger().capacity())
    }
    pub fn used(&self) -> Option<u64> {
        self.placement.as_ref()?.status().ok().map(|s| {
            s.rows
                .iter()
                .map(|r| {
                    r.snapshot
                        .context
                        .saturating_add(r.snapshot.classes.iter().copied().sum::<u64>())
                })
                .sum()
        })
    }
    pub fn with<T>(&self, f: impl FnOnce(&mut Gpu) -> T) -> T {
        f(&mut self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner))
    }
    pub fn fits(&self, plan: &str, _: impl Fn() -> Vec<Holding>) -> bool {
        self.placement.as_ref().is_some_and(|m| {
            m.fits(plan, &self.with(|g| g.shape_key(plan).to_owned()))
                .unwrap_or(false)
        })
    }
    pub fn admits(&self, plan: &str, holdings: impl Fn() -> Vec<Holding>) -> bool {
        self.fits(plan, holdings)
    }
    pub fn admit_member(
        &self,
        plan: &str,
        _: bool,
        holdings: impl Fn() -> Vec<Holding>,
        _: impl FnMut(&Step) -> std::io::Result<bool>,
    ) -> std::io::Result<Option<Option<u64>>> {
        Ok(self.fits(plan, holdings).then(|| self.capacity()))
    }
    pub fn floor(&self) -> Option<u64> {
        None
    }
    pub fn fits_resident(&self, _: &str, _: impl Fn() -> Vec<Holding>) -> bool {
        false
    }
    pub fn learn_holding(&self, plan: &str, id: &str) {
        self.with(|g| {
            g.learned.holding(plan, policy::holding_name(id));
        });
    }
    fn measured(&self, change: impl FnOnce(&mut learned::Learned)) {
        self.with(|g| {
            change(&mut g.learned);
            if let Err(error) = g.learned.save() {
                self.log.note(
                    serde_json::json!({"event":"measurements_unsaved","error":error.to_string()}),
                );
            }
        });
    }
    pub fn learn_call(
        &self,
        plan: &str,
        shape: &str,
        peak: u64,
        methods: &BTreeMap<String, u64>,
        context: Option<u64>,
        mapped: Option<u64>,
    ) {
        self.measured(|g| {
            g.call(plan, shape, peak, methods);
            if let Some(mapped) = mapped {
                g.mapped(plan, mapped);
            }
            if let Some(context) = context {
                g.context("reported", context);
            }
        });
    }
    pub fn learn_squeezed(&self, plan: &str, shape: &str, rooms: &BTreeMap<String, u64>) {
        self.measured(|g| g.squeezed(plan, shape, rooms));
    }
    pub fn learn_load(&self, plan: &str, weights: Option<u64>, floor: Option<u64>) {
        if let Some(weights) = weights {
            self.measured(|g| g.load(plan, weights, floor.unwrap_or(0)));
        }
    }
    pub fn learn_host(&self, plan: &str, bytes: u64) {
        self.measured(|g| g.host(plan, bytes));
    }
    pub fn observe(&self, plan: &str, facts: Facts, mapped: Option<bool>) {
        self.with(|g| g.observe(plan, facts, mapped));
    }
    pub fn running(&self, plan: &str, _: u64, _: Option<std::fs::File>) {
        self.with(|g| g.active(plan));
    }
    pub fn finished(&self, plan: &str) {
        self.with(|g| g.idle(plan));
    }
}
