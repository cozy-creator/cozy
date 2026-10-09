Measured stage ETA
==================

Run 5020 reported denoise step 1/30 taking 35.541 seconds, but the persisted list estimator projected only 223,558 ms remaining by treating overall_fraction=0.173333 as elapsed-time weighting. That fraction is authored progress allocation, not a prediction of future stage cost.

The API and CLI now leave whole-run time unknown and expose optional stage_remaining_ms. Both persisted progress and terminal rendering use the latest advancing count interval measured by Runtime timestamps, or an explicit producer unit rate. Only the first counted unit can use step_ms without prior history. A decode callback can count 17 frames, so its step_ms is never multiplied by the number of remaining frames. Missing source timestamps, repeated/reset counts, changed stages/totals and attempt changes do not establish a measured interval. Completed stages have zero stage time remaining while later work stays unknown.

The first denoise sample implies 1,030,689 ms for its 29 remaining units at that measured pace. Later 20-second sparse samples revise only the current stage estimate. Runtime source timestamps are retained separately from receipt-time fallback, and the API reads the latest two progress samples in one database snapshot. Preparation and model-download estimates are unchanged. Historical event recordings without source timestamps retain counts and fractions but cannot manufacture interval timing.

Validation
----------

- The original ordinary-CLI regression fails with exactly 223,558 ms on the base code.
- The fixed regression imports the real 30-step event sequence through persisted Runtime telemetry, reads ordinary run list through a separate daemon, then covers rate changes, completed stages, untimed stage transitions, 17-frame chunks, coalesced chunks and missing Runtime timestamps.
- Shared estimator and live rendering tests cover invalid timing, repeated/reset counts, changed totals/stages, explicit producer rates, duration overflow and unknown future work. Existing retry-attempt tests verify no prior attempt estimate survives.
- Focused and broader progress tests, historical live/piped golden recordings, internal tests, full build and vet pass.
- The separate local Runtime TestRunProgressSurfaces test skipped because no standalone -machine-host was supplied. It is not counted as executed end-to-end inference qualification.

Evidence is retained under outputs/measured-stage-eta-20261009. Installation stays held until the root agent's idle window; this task does not change remote workers or GPU execution.
