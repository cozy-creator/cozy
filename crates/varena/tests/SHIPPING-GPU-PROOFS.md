# Shipping memory GPU proof port — incomplete handoff

Owner: `/root/ram_completion`. Base: shipping master `fc40f4dddaaea13ae776ca1744b7f9044f4ec719`.
Paul stopped implementation before any test code was written. **No ported GPU cases exist or have run.**
This is a partial source/API mapping, not qualification or a statement that higher layers lack a feature.

The inspected native API is `Plane`, `Ticket`, `Lease`, `LeaseView`, per-region budgets,
RO `export_device`/`attach_device`, host tiers and cursors. `tests/gpu.rs` already has real
CUDA fixtures and stream helpers. Reuse those; do not backport Slim implementation.
No root AGENTS.md or CodeGraph index was present; TensorD's AGENTS.md was read.

| Original native20 case | Inspected shipping seam / remaining work |
| --- | --- |
| `pages_pin_evict_refill_and_survive_clean_owner_exit` | Plane want/acquire/drop + RO export/attach. Existing GPU tests cover portions; actual subprocess clean-exit port not written. |
| `escaped_writable_view_retains_pages_and_recovers` | LeaseView explicitly retains VA, not valid bytes after lease release. Original fill-writable-view contract has no direct Plane equivalent; inspect actual caller ownership before porting. |
| `read_only_fault_is_contained_in_its_process` | RO export/attach exists. Needs real child CUDA write fault plus healthy replacement process; not written. |
| `cuda_oom_is_recovered_in_process` | Public CUDA driver and Plane fills exist. Must force actual physical OOM, then verify same-process exact content; budget refusal alone is insufficient. |
| `killed_owner_keeps_read_only_pages` | Exported owned FDs + existing platform SCM_RIGHTS helpers can exercise actual SIGKILL. Custodian subprocess port not written. |
| `retired_page_releases_custody_after_last_view` | Inspect Plane close_ws/lease/view lifetime semantics; do not substitute simple clean close for deferred retirement. |
| `placement::gpu_tests::failed_begin_rolls_back_and_reuses_the_request_identity` | Plane exposes no request begin/identity transaction. Locate real Runtime/TensorD request boundary; not audited/ported. |
| `placement::gpu_tests::failed_direct_ram_fill_retains_its_source_until_checked_recovery` | Existing copy_faults module and staged/sealed sources are candidate seams; retain actual async-fence/source-lifetime assertions. |
| `placement::gpu_tests::ram_warm_returns_before_export_and_close_prevents_stale_registration` | Plane asynchronous tickets/open_sealed exist; close/cancel semantics require inspection. No stale-registration port yet. |
| `placement::gpu_tests::idle_worker_retires_after_pending_consumer_finishes` | Plane owns movers; actual close/consumer fence plus process-thread retirement observation needs port. |
| `placement::gpu_tests::idle_cached_weight_pins_cannot_block_their_own_warm_admission` | No native Plane cache callback/request admission API. Locate shipping caller/service boundary; not audited/ported. |
| `placement::gpu_tests::actual_readiness_survives_concurrent_fill_and_unmap` | Acquire/drop and queued/filling/ready states exist. Needs deterministic real concurrent fill/lease/unmap assertion; not written. |
| `placement::gpu_tests::failed_fills_require_real_room_and_recover_in_the_same_placement` | Existing failed-demand-fill test checks own IO error and same-Plane recovery. Original bounded physical-OOM/progress property still needs an honest port. |
| `placement::gpu_tests::repeated_component_forward_keeps_its_declared_step` | Plane has no model enter/remaining ordered plan API. Locate shipping Runtime/caller planner; do not emulate Slim in tests. |
| `placement::gpu_tests::lower_relief_keeps_fitting_home_and_requires_real_reclaim` | Per-region budget/priority/lease mechanisms exist; the original lower-target policy is outside Plane. Locate actual authority first. |
| `placement::gpu_tests::a_fill_can_bind_another_model_under_pressure_without_waiting_for_itself` | No arbitrary fill-transform callback in inspected Plane registration API. Inspect actual Python/source transform boundary before deciding applicability. |
| `placement::gpu_tests::self_region_fill_refuses_and_the_same_request_recovers` | Same callback-boundary issue; no direct native equivalent identified. No fake self-recursion simulator or weakened assertion. |
| `placement::gpu_tests::allocation_oom_records_zero_io_before_the_source_is_entered` | Plane counters/events and allocation order are candidate seams. Exact source-never-entered/phase attribution remains unproven; not ported. |
| `placement::gpu_tests::idle_observed_peak_allows_adaptive_warm_use_without_dropping_live_pins` | No native Plane observed-peak admission API. Inspect actual TensorD/Runtime policy; not audited/ported. |
| `placement::gpu_tests::activation_oom_waits_for_foreign_idle_reclaim_and_keeps_live_pins` | Cross-process admission/reclaim is outside native Plane. TensorD memory policy exists but was not fully audited; requires real service/process test, not a CPU simulation. |

Existing candidate helpers: `tests/common/mod.rs`, `tests/gpu.rs` (Stream/Fixture and real
CUDA tests), `platform::handles::{send_fd,recv_fd}`. Existing sharing test returns early
when device sharing is unsupported; any new required proof must fail explicitly instead.

No rentals, GPU operations, builds, source validation, daemon changes or product edits were
performed in this branch. Initial estimate sent to Root was 2–4h for an honest compiled port
and explicit remaining gaps, before Paul stopped work; that estimate is not completion evidence.
