//! The NVIDIA GPUs a machine reports in `StatusFrame.gpus[]`, read through NVML, which is opened
//! at run time (`libnvidia-ml.so.1`, `nvml.dll`). No driver is "no NVIDIA GPU", never an error.
use crate::{Error, Result};
use nvml_wrapper::{Nvml, error::NvmlError};

/// One GPU: machine.proto `Gpu` fields 1-6, plus the memory class.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub uuid: String,
    pub pci_bus_id: String,
    pub memory_bytes: u64,
    pub driver: String,
    /// Unit T fills it from CUDA's `INTEGRATED` and `PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES`
    /// device attributes; the probe leaves it `None`.
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
            memory_bytes: device.memory_info().map_err(failed("memory info"))?.total,
            driver: driver.clone(),
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

fn failed(op: &'static str) -> impl FnOnce(NvmlError) -> Error {
    move |error| Error::Gpu {
        op,
        detail: error.to_string(),
    }
}
