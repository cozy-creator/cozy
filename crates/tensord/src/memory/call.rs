//! One call's memory as its executor's varena measured it (design memory-varena §5.8): the
//! request's `memory` object per model per step, kept as the run's `memory` measurements and
//! printed by `cozy run show`.
use crate::device_executor::Frame;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// One pass of a model: bytes at home and resident, copied home and streamed through slots,
/// the GPU-timed prefetch stall, and torch's allocated peak above the pass's start.
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
    pub peak_allocated_bytes: Option<u64>,
    pub peak_reserved_bytes: Option<u64>,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_calls_memory_keeps_its_executors_stats_whole() {
        let reply: Frame = serde_json::from_value(serde_json::json!({
            "metrics": {"peak_vram_bytes": 900, "peak_reserved_bytes": 1000},
            "memory": {
                "leftover": 12,
                "paged_in": 25,
                "steps": {"h3/te": [5], "h3/dit": [20, 0]},
                "rows": [
                    {"model": "h3/te", "gpu": 0, "step": 0, "home": 5, "resident": 5, "paged_in": 5},
                    {"model": "h3/dit", "gpu": 0, "step": 0, "home": 20, "resident": 20, "paged_in": 20, "stall_ns": 7},
                    {"model": "h3/dit", "gpu": 0, "step": 1, "home": 20, "resident": 20, "activations": 4}
                ],
                "evictions": [{"model": "h3/te", "gpu": 0, "rung": 2, "reason": "asked", "nbytes": 5}],
                "peaks": [{"model": "h3/dit", "shape": "832x480x121", "peak": 4, "cuts": ["ffn:2"]}],
                "leftover_by_gpu": [[0, 12]],
                "newer": {"kept": true}
            }
        }))
        .unwrap();
        let memory = CallMemory::of(&reply).unwrap();
        assert_eq!((memory.peak_allocated_bytes, memory.peak_reserved_bytes), (Some(900), Some(1000)));
        assert_eq!(memory.per_step(), ["h3/dit"]);
        let kept = serde_json::to_value(&memory).unwrap();
        let sent = serde_json::to_value(&reply.memory).unwrap();
        assert_eq!(kept["stats"], sent, "the gate reads the executor's object as it sent it");
        assert_eq!(kept["stats"]["steps"]["h3/dit"], serde_json::json!([20, 0]));
        assert_eq!(kept["stats"]["newer"]["kept"], true);
        assert_eq!(kept["stats"]["rows"][1]["stall_ns"], 7);
        assert!(CallMemory::of(&Frame::default()).is_none());
    }
}
