# Local transaction package capture

Machine issue101, CLI partB; paired machinePR103. Owner Codex /root.

The current v1 submit refuses a capture containing multiple installations. This
prevents an ordinary unpublished script from calling another unpublished package.
The machine also lacked published callee routing; that is being fixed separately.

Keep the existing captured root and ordered callable graph. Send the root source
and all required unpublished callee wheels as one LocalSource, with an optional
callees map from normalized distribution to logical package identity. All code
runs in the caller's locked environment; each child retains its callee application,
model namespace and memo identity. Published dependency identity comes from the
locked installed source on the machine, not a fresh per-run Hub lookup.

Conflicting versions/distribution identities must refuse concretely before
submission; capture replay must preserve the submitted bytes. Do not mutate
source projects or old Opus worktrees. No protocol or exact-source-SHA gate.

Proof will use the ordinary CLI and actual Rust machine for two-package local
transactions, nested calls and model-less jobs, plus preserved existing single-root
behavior. A model-bearing callee and H3 automatic references are following paired
consumer gates. No candidate or publication claim comes from component tests.
