# Private native recovery qualification

Owner: `/root/astra_memoization_finish`.
Branch: `qualify/memoization-private-recovery-20260925`.
Base: `df335c97ac55f1b234aa8f6d4f114b04714b53dc`.
Purpose: ordinary global CLI private native checkpoint and commit-before-ACK recovery.

Use only a separately owned CPU worker. Never restart a shared daemon or another
worker. Native callback instrumentation records actual payload writes, completed
parts and receipt/checkpoint facts. It does not fabricate native or journal state.
The private worker has Runtime only; the normal global CLI retains submission history.

Finite proof order: private compatible partial recovery; changed-callee partial
rejection; native committed/unacknowledged recovery; completed caller reuse; exact
native readback and independent retention. Broader source/quantizer, Eval, effects,
and H3 gates remain separate until their recorded assertions pass.

Retained first proof: `dae87667` supplies the original candidate arithmetic.
The final candidate adds1 without changing package0.1.0, for the incompatible
partial control. Evidence snapshots retain before/after source hashes.
`verify_partial.py EVIDENCE_DIRECTORY` and `verify_changed.py EVIDENCE_DIRECTORY`
check the recorded normal global CLI runs and native events. Expected controls
are source/a/b written once with explicit partial adoption, and a/a/b written
across the incompatible partial followed by the changed implementation.

Actual private Runtime24 automatic recovery exposed a transient queued admission
refusal becoming final; Runtime PR611 supplies the narrow retry policy fix.
Explicit retry success is distinct from automatic retry qualification. The CPU
updater interpreter prerequisite shipped as Creator PR672; no manual SDK overlay
or fabricated cache row is used to claim the updater path.

Private completion: explicit partial adoption and changed-callee rejection passed
on Runtime0.18.24/TensorFS0.3.51. Commit-before-ACK and automatic partial recovery
passed on the distinct24+PR611 native dev wheel; both encountered the real worker
stale-admission refusal and automatically recovered. A forged output slot was
refused before candidate execution. The genuine shared quantizer and its source
execute [1,1] initially and [0,0] for the edited caller. Final ordinary prune removes
two unused memo entries while preserving all six returned artifacts exactly.
Silky was ended via normal CLI and the live list confirms no remaining rentals.

All seven `verify_*.py EVIDENCE_DIRECTORY` checks pass for retained evidence under
`outputs/memoization-private-recovery-20260925/` in the workspace. Full checkpoint,
source/download conversion parity, broader authority/offline/capacity cases,
trained quality, learned Eval and positive H3 generator-bank gates are separate.
