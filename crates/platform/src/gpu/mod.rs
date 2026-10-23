//! The NVIDIA GPUs a machine reports in `StatusFrame.gpus[]`, read through NVML, which is opened
//! at run time (`libnvidia-ml.so.1`, `nvml.dll`). No driver is "no NVIDIA GPU", never an error.
#[cfg(windows)]
mod budget;
mod cuda;

#[cfg(windows)]
pub use budget::{Adapter, Budget, adapters};
pub use cuda::memory_classes;

use crate::{Error, Result};
use nvml_wrapper::{
    Nvml,
    enum_wrappers::device::{ComputeMode, Sampling},
    enums::device::{SampleValue, UsedGpuMemory},
    error::NvmlError,
    struct_wrappers::device::{MemoryInfo, ProcessInfo},
};
use std::collections::BTreeMap;

/// One GPU: machine.proto `Gpu` fields 1-6, plus the memory class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub pci_bus_id: String,
    pub memory_bytes: u64,
    pub driver: String,
    /// (major, minor), `(9, 0)` on a Hopper; `None` when the driver does not say.
    pub compute_capability: Option<(u32, u32)>,
    /// [`memory_classes`] fills it from CUDA's device attributes; the probe leaves it `None`.
    pub memory_class: Option<MemoryClass>,
}

/// How a GPU reaches host memory (design §6.2), which picks Runtime's memory policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryClass {
    /// Its own memory across PCIe or SXM: today's paging.
    Discrete,
    /// Its own memory plus coherent host memory (GH200, GB200, GB300).
    Coherent,
    /// One pool with the CPU (GB10, Jetson, Apple Silicon).
    Unified,
}

/// The oldest NVIDIA driver the stack's CUDA 13.0 runs on. The cozy-kernels wheel declares the same
/// (`cozy_kernels.DRIVER_FLOOR`), and cozy-runtime's `kernels.yaml` checks the two are equal.
pub const DRIVER_FLOOR: &str = if cfg!(windows) { "580.88" } else { "580.65.06" };

/// Why a GPU on `driver` cannot serve: it is older than [`DRIVER_FLOOR`]. `None` when it meets the
/// floor, or when the driver's version does not parse (CUDA then answers for itself).
pub fn below_floor(driver: &str) -> Option<String> {
    let version = |text: &str| {
        text.trim()
            .split('.')
            .map(|part| part.parse::<u32>().ok())
            .collect::<Option<Vec<u32>>>()
    };
    (version(driver)? < version(DRIVER_FLOOR)?).then(|| {
        format!(
            "NVIDIA driver {driver} is below {DRIVER_FLOOR}, the oldest CUDA 13.0 runs on: \
             update the driver to {DRIVER_FLOOR} or newer"
        )
    })
}

/// `visible` is `CUDA_VISIBLE_DEVICES` as the caller's config read it: indices or UUIDs, and an
/// empty value hides every GPU.
pub fn probe(visible: Option<&str>) -> Result<Vec<Gpu>> {
    if visible.is_some_and(|visible| visible.trim().is_empty()) {
        return Ok(vec![]);
    }
    let nvml = match Nvml::init() {
        Ok(nvml) => nvml,
        Err(
            NvmlError::LibloadingError(_) | NvmlError::DriverNotLoaded | NvmlError::LibraryNotFound,
        ) => {
            return Ok(vec![]);
        }
        Err(error) => return Err(failed("init")(error)),
    };
    let driver = nvml
        .sys_driver_version()
        .map_err(failed("driver version"))?;
    let mut gpus = vec![];
    for index in 0..nvml.device_count().map_err(failed("device count"))? {
        let device = nvml.device_by_index(index).map_err(failed("device"))?;
        gpus.push(Gpu {
            index,
            name: device.name().map_err(failed("name"))?,
            uuid: device.uuid().map_err(failed("uuid"))?,
            pci_bus_id: device.pci_info().map_err(failed("PCI info"))?.bus_id,
            memory_bytes: memory(device.memory_info())?.0,
            driver: driver.clone(),
            compute_capability: device
                .cuda_compute_capability()
                .ok()
                .map(|cc| (cc.major as u32, cc.minor as u32)),
            memory_class: None,
        });
    }
    if let Some(visible) = visible {
        let named: Vec<&str> = visible.split(',').map(str::trim).collect();
        gpus.retain(|gpu| {
            named.contains(&gpu.index.to_string().as_str()) || named.contains(&gpu.uuid.as_str())
        });
    }
    Ok(gpus)
}

/// One GPU opened for repeated reads (TensorD's memory sampler reads it every second). NVML
/// stays loaded while it lives.
pub struct Device {
    nvml: Nvml,
    index: u32,
    key: String,
}

/// One reading of a GPU's memory.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reading {
    pub total: u64,
    pub free: u64,
    /// A desktop is initialized on it.
    pub display: bool,
    /// Device bytes the driver charges each process (pid -> bytes), compute and graphics (a
    /// desktop, a game) alike; `None` where it cannot account every one (a container whose pids
    /// are not the driver's, WDDM).
    pub processes: Option<BTreeMap<u32, u64>>,
}

impl Device {
    /// `entry` as `CUDA_VISIBLE_DEVICES` spells it: an index, or a `GPU-…` UUID.
    pub fn open(entry: &str) -> Result<Self> {
        let nvml = Nvml::init().map_err(failed("init"))?;
        let device = match entry.parse::<u32>() {
            Ok(index) => nvml.device_by_index(index),
            Err(_) => nvml.device_by_uuid(entry),
        }
        .map_err(failed("open device"))?;
        let index = device.index().map_err(failed("device index"))?;
        let uuid = device.uuid().unwrap_or_else(|_| entry.into());
        let driver = nvml.sys_driver_version().unwrap_or_default();
        Ok(Self {
            key: format!("{uuid}/{driver}"),
            index,
            nvml,
        })
    }

    /// `<GPU UUID>/<driver version>`: what a learned context is valid for.
    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn read(&self) -> Result<Reading> {
        let device = self
            .nvml
            .device_by_index(self.index)
            .map_err(failed("open device"))?;
        let (total, free) = memory(device.memory_info())?;
        let charged = |rows: Vec<ProcessInfo>| -> Option<Vec<(u32, u64)>> {
            rows.into_iter()
                .map(|row| match row.used_gpu_memory {
                    UsedGpuMemory::Used(bytes) => Some((row.pid, bytes)),
                    UsedGpuMemory::Unavailable => None,
                })
                .collect()
        };
        // A process in both lists (a CUDA program that also draws) is charged once: its larger.
        let graphics = device
            .running_graphics_processes()
            .map_or(Some(vec![]), charged);
        let processes = device
            .running_compute_processes()
            .ok()
            .and_then(charged)
            .zip(graphics)
            .map(|(compute, graphics)| {
                compute
                    .into_iter()
                    .chain(graphics)
                    .fold(BTreeMap::new(), |mut by, (pid, bytes)| {
                        let held: &mut u64 = by.entry(pid).or_default();
                        *held = (*held).max(bytes);
                        by
                    })
            });
        Ok(Reading {
            total,
            free,
            display: device.is_display_active().unwrap_or(false),
            processes,
        })
    }

    /// Its free bytes now (an integrated GPU's: the host's available memory), read alone.
    pub fn free(&self) -> Result<u64> {
        let device = self.nvml.device_by_index(self.index).map_err(failed("open device"))?;
        memory(device.memory_info()).map(|(_, free)| free)
    }

    /// Whether device memory moved (kernels' traffic, copies, peer writes) in NVML's samples
    /// after `seen` (its timestamp, µs), which it advances.
    pub fn moved_since(&self, seen: &mut u64) -> Result<bool> {
        let device = self.nvml.device_by_index(self.index).map_err(failed("open device"))?;
        // No sample newer than `seen` is no work seen: NVML answers NotFound while the GPU is idle,
        // which is a reading, not an unreadable GPU: a call stuck there was never judged.
        let samples = match device.samples(Sampling::MemoryUtilization, *seen) {
            Err(NvmlError::NotFound) => return Ok(false),
            samples => samples.map_err(failed("utilization samples"))?,
        };
        Ok(samples.iter().fold(false, |moved, sample| {
            *seen = (*seen).max(sample.timestamp);
            moved || match sample.value {
                SampleValue::U32(value) => value > 0,
                SampleValue::U64(value) => value > 0,
                SampleValue::I64(value) => value > 0,
                SampleValue::F64(value) => value > 0.0,
            }
        }))
    }

    /// Whether its compute mode admits one context at a time (exclusive-process) or none
    /// (prohibited): a second process holding a context locks every other out. A driver that
    /// does not say shares it.
    pub fn exclusive(&self) -> Result<bool> {
        match self.nvml.device_by_index(self.index).and_then(|device| device.compute_mode()) {
            Ok(mode) => Ok(mode != ComputeMode::Default),
            Err(NvmlError::NotSupported) => Ok(false),
            Err(error) => Err(failed("compute mode")(error)),
        }
    }

    pub fn display_active(&self) -> Result<bool> {
        self.nvml
            .device_by_index(self.index)
            .and_then(|device| device.is_display_active())
            .map_err(failed("display active"))
    }
}

/// A GPU's (total, free) memory from NVML's answer. An integrated GPU (GB10) has no
/// framebuffer, so NVML answers NotSupported: its memory is the host's (`MemTotal`,
/// `MemAvailable`).
fn memory(answer: std::result::Result<MemoryInfo, NvmlError>) -> Result<(u64, u64)> {
    match answer {
        Ok(memory) => Ok((memory.total, memory.free)),
        Err(NvmlError::NotSupported) => {
            let total = crate::contain::host_memory::total();
            let free = crate::contain::host_memory::read().mem_available;
            match total > 0 && free >= 0 {
                true => Ok((total as u64, free as u64)),
                false => Err(failed("memory info")(NvmlError::NotSupported)),
            }
        }
        Err(error) => Err(failed("memory info")(error)),
    }
}

fn failed(op: &'static str) -> impl FnOnce(NvmlError) -> Error {
    move |error| Error::Gpu {
        op,
        detail: error.to_string(),
    }
}
