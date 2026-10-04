# Bounded progress stage edges

The real 3 GiB Anima4271 observer emitted positions 1–29, then decoding and a successful
30-step terminal/artifact. Its native work completed 30 steps. Source confirms two latest-only
windows: Rust has one ProgressSnapshot per execution, and Creator keeps latest by event kind
plus the first overall fraction. Back-to-back genuine final denoise/decode frames can replace
a stage endpoint before a reader polls. This is advisory observability, not request failure.

Retain at most 8 genuine previous-stage endpoints plus the current latest sample per observed
execution/run, with existing per-sample byte limits. Stage changes capture the previously
emitted sample unchanged; no step, timestamp or completed count is invented. Machine event
pages expose those bounded endpoints by their actual observation sequences. Creator keeps the
received endpoints for live SSE and flushes them only with existing real state/product/outcome
transactions. No telemetry event creates a disk write or advances a durable cursor alone.

The same-stage flood remains latest-only, and count/byte caps bound arbitrary stage names.
A CPU burst must emit denoise29/30/decoding before the observer gets a turn, then prove both
actual machine and CLI/record consumers preserve emitted30 without per-event writes. Native
completion remains the work evidence; advisory delivery never becomes a success gate.
Creator's stage projection has at most eight previous genuine progress samples plus the
latest, alongside the existing first fraction and latest preparation/log samples. Each is
still bounded to 16 KiB. The real SSE stream emits edges in their observed sequence order
with sequence zero, while its per-run/type high-water mark deduplicates already seen live
samples. Terminal/product/state transactions flush the bounded projection before their
durable lifecycle events. An attempt change or detached observer drops its live projection.

Prepared records tests hold the sole database connection across a denoise 29/30/decode burst,
check live frames and durable-cursor neutrality, then reopen the actual terminal journal.
Count/byte/flood/attempt tests and a real SSE subscription cover bounds and presentation.
`TestMachineV1ProgressStageEndpointSurvivesOrdinaryCLIAndRestart` uses the real machine and
`-cpu-progress-burst=<machine repo>/tests/fixtures/cpu_progress_burst`, then reads ordinary
CLI watch events before and after restarting its own isolated controller.

Source checkpoint only: new proof is pending the host resource gate or coordinated rented
CPU execution. Rented model delivery, congested-disk timing and GPU throughput remain open.
