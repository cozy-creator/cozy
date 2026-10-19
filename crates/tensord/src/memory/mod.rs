//! TensorD on varena's ledger (design memory-varena §1, §5.6): each GPU's table, TensorD's own row
//! of what no executor explains, start admission, reaping and stalls. Executors decide their own
//! bytes and make room through the ledger's asks; nothing here models GPU memory.
pub mod call;
#[cfg(unix)]
mod gpu;
#[cfg(not(unix))]
#[path = "gpu_off.rs"]
mod gpu;

pub use gpu::{Admitted, Ender, GpuLedger};
use serde::Deserialize;
use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::Write,
    path::PathBuf,
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

/// Past this the memory log rotates to `memory.log.1`, so it holds at most twice this.
const ROTATE_BYTES: u64 = 4 << 20;

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Raises the free floor kept on each GPU (never lowers the default).
    pub floor_bytes: u64,
}

/// One NVML reading of a GPU.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Sample {
    pub total: u64,
    pub free: u64,
    pub display: bool,
    /// Device bytes NVML charges each process; None where it cannot attribute them.
    pub processes: Option<BTreeMap<u32, u64>>,
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
