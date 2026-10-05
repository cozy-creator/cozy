# Local transaction package capture

Machine issue #101, CLI part B; paired machine PR #103. Owner Codex `/root/bench_takeover`.

The current v1 submit refuses a capture containing multiple installations. This
prevents an ordinary unpublished script from calling another unpublished package.
The machine also lacked published callee routing; that is being fixed separately.

Keep the existing captured root and ordered callable graph. During intake, retain
the root and callable dependencies as wheels from the already frozen source copy.
Send that wheel-only closure as one LocalSource, with an optional
callees map from normalized distribution to logical package identity. All code
runs in the caller's locked environment; each child retains its callee application,
model namespace and memo identity. Published dependency identity comes from the
locked Hub index captured at intake, carried beside any copied wheel, not a fresh
per-run Hub lookup. A single source-only root keeps its existing source transport.

The paired installer refuses a frozen source lock mixed with wheels/requirements.
The wheel-only multi-package path avoids rewriting that lock at machine admission
and avoids rebuilding a child from mutable source after capture.

Conflicting versions/distribution identities must refuse concretely before
submission; capture replay must preserve the submitted bytes. Do not mutate
source projects or old Opus worktrees. No protocol or exact-source-SHA gate.

Proof will use the ordinary CLI and actual Rust machine for two-package local
transactions, nested calls and model-less jobs, plus preserved existing single-root
behavior. A model-bearing callee and H3 automatic references are following paired
consumer gates. No candidate or publication claim comes from component tests.
