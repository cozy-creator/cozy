# Provider and price components per rental width

Owner: Paul Fidika; implementation agent: linear_resume.
Branch: fix/rental-provider-price-display-20261009.
Base: origin/master 8abcceee894fdb79c6d824c08430807586141e6e.

The catalog now keeps the provider attached to each GPU count and shows it beside the price. A two-card RunPod offer and a four-card Vast offer can no longer appear to be the same provider's price ladder. Vast's disk component is separated when the Hub supplies it; otherwise the CLI says storage is included and compute is not itemized, rather than falsely showing zero storage.

Before creating a rental, the CLI reads the refreshed exact-request quote and saves its confirmed total hourly limit in the durable paid request. Replays retain that original body and ceiling. A Hub that cannot confirm that limit is refused before any paid request; no generic approval prompt is added. Explicit provider choices remain unchanged. Existing rentals and requests are preserved.

Validation uses ordinary CLI subprocesses over real HTTP fixtures: mixed-width providers, itemized and unknown bundled storage, no paid request to an unaware Hub, durable ceiling propagation, existing locked-rate/disk-quote behavior, and machine-exclusion replay. A one-second exclusion test timed out during the initial shared test run and passed its isolated rerun unchanged. Builds and vet pass. No rented machine was ended or changed.

A retained unsent request re-confirms price-limit support before replay, including after a Hub rollback, while keeping its original body and ceiling. The final client guard compares reported total with the retained total limit; a cheaper storage-inclusive offer is not mistaken for an increase over an earlier compute-only field. Both cases have ordinary CLI regressions.
