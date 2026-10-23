//! One call's memory as its executor's varena measured it (design memory-varena §5.8): the
//! request's `memory` object per model per step, kept as the run's `memory` measurements and
//! printed by `cozy run show`.
use crate::device_executor::Frame;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One pass of a model: bytes at home and resident, copied home and streamed through slots,
/// the GPU-timed prefetch stall, torch's allocated peak above the pass's start, and torch's
/// allocated bytes at that start (a leak shows as their growth).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct Step {
    pub model: String,
    pub gpu: u32,
    pub step: u64,
    pub home: u64,
    pub resident: u64,
    pub paged_in: u64,
    pub streamed: u64,
    pub stall_ns: u64,
    pub activations: u64,
    pub allocated: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct Eviction {
    pub model: String,
    pub gpu: u32,
    pub rung: u32,
    pub reason: String,
    pub nbytes: u64,
}

/// The measured activation peak of a model at one shape key, and the cuts in force there.
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(default)]
pub struct Peak {
    pub model: String,
    pub shape: String,
    pub peak: u64,
    pub cuts: Vec<String>,
}

/// The executor's `memory` object for one request, kept whole as `stats`: what the merge gate
/// reads (`measurements.memory.stats`).
#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct Stats {
    /// Torch bytes still allocated at the request's end beyond its start, on every GPU.
    pub leftover: u64,
    /// Weight bytes copied to the device during the request, home or through slots.
    pub paged_in: u64,
    /// Per model (`<construction>/<component>`), the bytes copied in each of its passes.
    pub steps: BTreeMap<String, Vec<u64>>,
    pub rows: Vec<Step>,
    pub evictions: Vec<Eviction>,
    pub peaks: Vec<Peak>,
    pub leftover_by_gpu: Vec<(u32, u64)>,
    /// What this machine does not read, kept as sent.
    #[serde(flatten)]
    pub other: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct CallMemory {
    /// torch's high-water marks over the call: live tensors, and the segments reserved for them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_allocated_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_reserved_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<Stats>,
}

impl CallMemory {
    /// None: the executor measured nothing.
    pub fn of(reply: &Frame) -> Option<Self> {
        let known = |v: Option<i64>| v.and_then(|v| u64::try_from(v).ok());
        if reply.memory.is_none() && reply.metrics.is_none() {
            return None;
        }
        let metrics = reply.metrics.clone().unwrap_or_default();
        Some(Self {
            peak_allocated_bytes: known(metrics.peak_vram_bytes),
            peak_reserved_bytes: known(metrics.peak_reserved_bytes),
            stats: reply.memory.clone(),
        })
    }

    /// Models that ran more than one pass in this call: the host cache keeps their layouts
    /// past run-once ones' (`<construction>/<component>`, a weight set's grant name).
    pub fn per_step(&self) -> Vec<String> {
        let steps = self.stats.iter().flat_map(|stats| &stats.steps);
        steps.filter(|(_, passes)| passes.len() > 1).map(|(model, _)| model.clone()).collect()
    }
}
