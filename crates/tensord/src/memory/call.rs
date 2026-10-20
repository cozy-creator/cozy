//! One call's device memory as its executor measured it: the run's `memory` measurements,
//! which `cozy run show` prints. Peaks are the call's own; holdings are its process's at the
//! call's end; counts are what its plane and torch did since the executor's previous call (since
//! it started, for its first).
use crate::device_executor::{Frame, PlaneFacts, Shortfall};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Per-request facts emitted by Varena, distinct from legacy process counters.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct RequestMemory {
    pub leftover: u64,
    pub paged_in: u64,
    pub steps: BTreeMap<String, Vec<u64>>,
    pub rows: Vec<MemoryStep>,
    pub evictions: Vec<MemoryEviction>,
    pub peaks: Vec<MemoryPeak>,
    pub retries: u64,
    pub leftover_by_gpu: Vec<(i32, u64)>,
    pub plan: Vec<Vec<String>>,
    pub copies: Vec<MemoryCopy>,
    pub proofs: Vec<TransferProof>,
    // Serialized once at CallMemory.trace, not duplicated inside stats.
    #[serde(skip_serializing)]
    pub trace: Vec<TraceEvent>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MemoryStep {
    pub model: String,
    pub gpu: i32,
    pub step: u64,
    pub home: u64,
    pub resident: u64,
    pub paged_in: u64,
    pub streamed: u64,
    pub stall_ns: u64,
    pub activations: i64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MemoryEviction {
    pub model: String,
    pub gpu: i32,
    pub rung: u64,
    pub reason: String,
    pub nbytes: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MemoryPeak {
    pub model: String,
    pub shape: String,
    pub peak: i64,
    pub cuts: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct MemoryCopy {
    pub kind: String,
    pub gpu: i32,
    pub bytes: u64,
    pub seconds: f64,
    pub gbps: f64,
    pub ceiling: f64,
    pub pct: Option<f64>,
    pub link: String,
    pub busy_s: f64,
    pub pin_s: f64,
    pub staged: u64,
    pub cached: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct TraceEvent {
    pub sequence: u64,
    pub request: String,
    pub step: u64,
    pub pass_index: u64,
    pub page: u64,
    pub gpu: i32,
    pub rank: u32,
    pub kind: String,
    pub reason: String,
    pub bytes: u64,
    pub capacity: u64,
    pub resident: u64,
    pub orphan_bytes: u64,
    pub model: Option<String>,
    #[serde(rename = "class")]
    pub class: String,
    pub working_set: Option<u64>,
    pub speculative_budget: Option<u64>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct ProofPage {
    pub page: u64,
    pub content: String,
    pub payload: u64,
    pub mapped: u64,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct TransferProof {
    pub request: String,
    pub gpu: i32,
    pub rank: u32,
    pub version: u32,
    pub complete: bool,
    pub isolated: bool,
    pub capacity: u64,
    pub pages: Vec<ProofPage>,
    pub initial: Vec<u64>,
    pub invalid: Vec<String>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct CallMemory {
    pub stats: Option<RequestMemory>,
    pub trace: Vec<TraceEvent>,
    /// torch's high-water marks: live tensors, and the segments it reserved for them.
    pub peak_allocated_bytes: Option<u64>,
    pub peak_reserved_bytes: Option<u64>,
    /// The peak above the call's start (activations, derived weights), and each stage method's.
    pub activation_peak_bytes: Option<u64>,
    pub stage_peaks: BTreeMap<String, u64>,
    /// The process on its GPU at the end: CUDA context + torch reserved + the plane's maps.
    pub process_bytes: Option<u64>,
    pub context_bytes: Option<u64>,
    pub reserved_bytes: Option<u64>,
    /// Weights mapped on the GPU (its own and custody's), the plane's budget, the process cap.
    pub weights_bytes: Option<u64>,
    pub budget_bytes: Option<u64>,
    pub cap_bytes: Option<u64>,
    /// Weight bytes copied to the GPU; of them, read from disk and bounced through staging.
    pub paged_in_bytes: Option<u64>,
    pub disk_bytes: Option<u64>,
    pub bounce_bytes: Option<u64>,
    pub register_failed: Option<u64>,
    /// Weight regions evicted, and components given back whole to make torch room.
    pub evictions: Option<u64>,
    pub paged_out: Option<u64>,
    pub paged_out_bytes: Option<u64>,
    /// torch short of room: whole-cache drops to place one allocation, OOMs it raised, the
    /// executor's retries after one, and its own cache releases.
    pub alloc_retries: Option<u64>,
    pub alloc_ooms: Option<u64>,
    pub oom_retries: Option<u64>,
    pub cache_releases: Option<u64>,
    pub cache_released_bytes: Option<u64>,
    /// How each stage ran (decode mode, CFG layout), and what a refused stage lacked.
    pub modes: BTreeMap<String, String>,
    pub shortfall: Option<Shortfall>,
}

impl CallMemory {
    /// `since` is the plane's counts at the executor's previous call. None: nothing measured.
    pub fn of(reply: &Frame, since: &PlaneFacts) -> Option<Self> {
        if reply.metrics.is_none()
            && reply.plane.is_none()
            && reply.shortfall.is_none()
            && reply.memory.is_none()
        {
            return None;
        }
        let metrics = reply.metrics.clone().unwrap_or_default();
        let plane = reply.plane.clone().unwrap_or_default();
        let moved = |of: fn(&PlaneFacts) -> Option<u64>| {
            of(&plane).map(|now| now.saturating_sub(of(since).unwrap_or(0)))
        };
        Some(Self {
            stats: reply.memory.clone(),
            trace: reply
                .memory
                .as_ref()
                .map(|memory| memory.trace.clone())
                .unwrap_or_default(),
            peak_allocated_bytes: known(metrics.peak_vram_bytes),
            peak_reserved_bytes: known(metrics.peak_reserved_bytes),
            activation_peak_bytes: known(plane.activation_peak_bytes)
                .or(known(metrics.working_peak_vram_bytes)),
            stage_peaks: metrics
                .activation_peaks
                .iter()
                .filter_map(|(stage, bytes)| Some((stage.clone(), known(Some(*bytes))?)))
                .collect(),
            process_bytes: known(plane.process_bytes),
            context_bytes: known(plane.context_bytes),
            reserved_bytes: known(plane.reserved_bytes),
            weights_bytes: known(plane.committed_bytes)
                .map(|own| own + known(plane.shared_bytes).unwrap_or(0)),
            budget_bytes: known(plane.budget_bytes),
            cap_bytes: known(plane.cap_bytes),
            paged_in_bytes: moved(|p| p.h2d_bytes),
            disk_bytes: moved(|p| p.disk_copy_bytes),
            bounce_bytes: moved(|p| p.bounce_copy_bytes),
            register_failed: moved(|p| p.register_failed),
            evictions: moved(|p| p.evictions),
            paged_out: moved(|p| p.paged_out),
            paged_out_bytes: moved(|p| p.paged_out_bytes),
            alloc_retries: moved(|p| p.alloc_retries),
            alloc_ooms: moved(|p| p.alloc_ooms),
            oom_retries: moved(|p| p.oom_retries),
            cache_releases: moved(|p| p.cache_releases),
            cache_released_bytes: moved(|p| p.cache_released_bytes),
            modes: plane.modes,
            shortfall: reply.shortfall.clone(),
        })
    }
}

/// The executor's signed byte counts: -1 is unreadable.
fn known(value: Option<i64>) -> Option<u64> {
    value.and_then(|v| u64::try_from(v).ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_measurements_and_native_trace_reach_the_public_result() {
        let measured = serde_json::json!({
            "leftover": 0, "paged_in": 8192,
            "steps": {"dit": [4096, 4096]},
            "rows": [{"model": "dit", "gpu": 0, "step": 1, "home": 4096,
                "resident": 4096, "paged_in": 4096, "streamed": 0,
                "stall_ns": 25, "activations": 2048}],
            "evictions": [{"model": "vae", "gpu": 0, "rung": 0,
                "reason": "orphan", "nbytes": 4096}],
            "peaks": [{"model": "dit", "shape": "64x64", "peak": 2048, "cuts": []}],
            "retries": 1, "leftover_by_gpu": [[0, 0]], "plan": [["dit"], ["vae"]],
            "proofs": [{"request": "request-1", "gpu": 0, "rank": 1, "version": 1,
                "complete": true, "isolated": true, "capacity": 8192,
                "pages": [{"page": 3, "content": "fingerprint/0", "payload": 3072, "mapped": 4096}],
                "initial": [3], "invalid": []}],
            "copies": [{"kind": "h2d", "gpu": 0, "bytes": 8192, "seconds": 0.1,
                "gbps": 0.00008192, "ceiling": 0.0001, "pct": 81.92, "link": "PCIe"}],
            "trace": [{"sequence": 1, "request": "request-1", "step": 0,
                "pass_index": 0,
                "page": 3, "gpu": 0, "rank": 1, "kind": "bind", "reason": "working",
                "bytes": 4096, "capacity": 8192, "resident": 4096, "orphan_bytes": 0,
                "model": "dit", "class": "working", "working_set": 4096,
                "speculative_budget": 0}]
        });
        let reply: Frame = serde_json::from_value(serde_json::json!({"memory": measured})).unwrap();
        let result = CallMemory::of(&reply, &PlaneFacts::default()).unwrap();
        let public = serde_json::to_value(result).unwrap();
        assert_eq!(public["stats"]["paged_in"], 8192);
        assert_eq!(public["stats"]["steps"], measured["steps"]);
        assert_eq!(public["stats"]["rows"], measured["rows"]);
        assert_eq!(public["stats"]["evictions"], measured["evictions"]);
        assert_eq!(public["stats"]["peaks"], measured["peaks"]);
        assert_eq!(public["stats"]["plan"], measured["plan"]);
        assert_eq!(public["stats"]["copies"][0]["pct"], 81.92);
        assert_eq!(public["stats"]["proofs"], measured["proofs"]);
        assert_eq!(public["trace"], measured["trace"]);
        assert!(public["stats"].get("trace").is_none());
    }

    #[test]
    fn request_byte_counts_cannot_wrap_negative_input() {
        for memory in [
            serde_json::json!({"paged_in": -1}),
            serde_json::json!({"copies": [{"bytes": -1}]}),
            serde_json::json!({"trace": [{"bytes": -1}]}),
        ] {
            assert!(
                serde_json::from_value::<Frame>(serde_json::json!({"memory": memory})).is_err()
            );
        }
    }

    #[test]
    fn a_call_counts_what_moved_since_the_executors_previous_call() {
        let reply: Frame = serde_json::from_value(serde_json::json!({
            "metrics": {"peak_vram_bytes": 900, "peak_reserved_bytes": 1000,
                "activation_peaks": {"vae.decode": 300}},
            "plane": {"h2d_bytes": 5000, "evictions": 7, "committed_bytes": 400,
                "shared_bytes": 100, "context_bytes": -1, "activation_peak_bytes": 350},
            "shortfall": {"resource": "vram", "needed_bytes": 1200, "available_bytes": 1150,
                "request_shape": "vae.decode"}
        }))
        .unwrap();
        let before = PlaneFacts {
            h2d_bytes: Some(3000),
            evictions: Some(2),
            ..Default::default()
        };
        let memory = CallMemory::of(&reply, &before).unwrap();
        assert_eq!(memory.peak_reserved_bytes, Some(1000));
        assert_eq!(memory.activation_peak_bytes, Some(350));
        assert_eq!(memory.stage_peaks["vae.decode"], 300);
        assert_eq!(
            (memory.paged_in_bytes, memory.evictions),
            (Some(2000), Some(5))
        );
        assert_eq!(
            (memory.weights_bytes, memory.context_bytes),
            (Some(500), None)
        );
        assert_eq!(memory.paged_out, None);
        assert_eq!(memory.shortfall.unwrap().needed_bytes, 1200);
        assert!(CallMemory::of(&Frame::default(), &before).is_none());
    }
}
