Accepted-run observer recovery
==============================

An accepted run whose stream loses owner authorization can stop being observed if the subsequent authority check briefly loses transport or rental resolution returns a maintenance hold. The remote run continues, but the former helper returned immediately and required a later `cozy run show` to collect its result.

Recovery now backs off through temporary Status and resolution failures using the existing1-to30-second delay. A fresh successful Status for the same machine is still required before reopening the same run with a nil spec and its retained cursor. Revoked keys remain refused; no new submission or control is authorized. Terminal rentals stop recovery, and changed probe failures are logged so the intermediate cause is visible.

Validation: a real pinned-TLS daemon regression fails on the original code after one transient Status Unavailable (13.027s). The final26.452s suite passes transport Unavailable/deadline recovery, real rental maintenance-resolution recovery, original auth-renewal recovery, permanent revocation with backoff, observer detach, first-submission rejection, and ending-rental stop. Successful recovery proves exactly one submitted spec, one same-ID/cursor nil-spec reattachment and zero control RPCs. Internal CLI/transport/records tests, full module build and vet pass.

Run5016 demonstrated observation stopping after an authorization error and later manual recovery. Its intermediate recovery-probe error was not retained, so that exact error remains unknown; the separate baseline-failing regression establishes this fix. Evidence lives under `outputs/run-observer-transient-recovery-20261008/`. No live daemon or Hub was changed during validation.

Additional boundary check: `scripts/fence.py` reports33pre-existing violations on both origin/master9f16c351 and this candidate, with no differences after normalizing shifted line numbers. The fence is not green; its unrelated baseline findings were preserved. This patch is limited to recovery after an authorization-lost stream; it does not change the separate policy that parks a directly unavailable remote stream.
