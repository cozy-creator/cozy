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
