# Scoped source/open handoff for cr-174

Owner: Codex /root/tensorfs_native_resume.
Worktree: ~/cozy/.worktrees/tensorfs/cr174-scoped-source-open.
Branch: feat/cr174-scoped-source-open.
Base: 91ee78442a956a27a788c666a9e41de4f0c38287; reconciled with the released
0.3.40 tree at 8d9c5c0. This increment prepares the distinct 0.3.41 release.

DerivedTransaction.accept(fd) receives either the live writer or the retained native
receipt selected by the storage owner. serve_derived_replay records the receipt before
success. SourceCapability.from_fd(manifest, length, fd) opens only that exact granted
source's native metadata/configs through serve_source. Configs use bounded 1 MiB frames;
no executor Store or Runtime tensor spool is needed. All source selection, identity,
custody and geometry stay in TensorFS. Inspection preserves the header's construction
order and rejects every absent requested component before projecting selected rows.

The source/open/replay functions are additive to 0.3.40. Existing writer calls and
protocol framing remain valid; no artifact format, schema or journal migration is added.

Validation: PR169 native/format/conformance checks and installed manylinux wheel smoke
pass. The isolated UID65533 client cannot open the privileged Store but can inspect its
exact source, transfer a 2 MiB config, write its declared output and receive recorded
replay. Native source and FD tests preserve nonlexical component/tensor order and reject
unknown selections on both sides. Actual H3 source-order repair, restamp, shared VAE and
lane byte controls use source853ccdc62e2e, wheel SHA256
52faba71511c4930adb7963152b162f8c97c33ca25008b8a90bf1757af88fb99.

The ordinary Creator CLI already exercised worker-owned source/output/adoption on the
prior df816bf API build. An updated exact-wheel CLI run and final 0.3.41 CI/release wheel
verification are still required before publishing this component. The Runtime0.18
integration and its disconnected/private/default-home qualification are separate gates.
