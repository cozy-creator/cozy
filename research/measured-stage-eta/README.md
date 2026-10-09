Measured stage ETA
==================

Run5020 reported denoise step1/30 taking35.541s, but the persisted list estimator projected only223558ms remaining by treating overall_fraction0.173333 as elapsed-time weighting. That fraction is authored progress allocation, not a model of future stage cost.

This task will keep whole-run remaining time unknown and expose an explicit stage_remaining_ms estimate from counted measured step timing or measured unit rate. The first recorded denoise sample implies about1030689ms for its remaining29steps at that pace; it does not predict sparse-phase acceleration or later decode/encode work. The live terminal's analogous overall-fraction extrapolation also needs removal while preserving its existing stage-local ETA.

Required proof: the real30-step event sequence through persisted Runtime telemetry and ordinary CLI list, stage transitions/retries, invalid or missing measurements, and existing progress display tests. No live install while native benchmark work is active.
