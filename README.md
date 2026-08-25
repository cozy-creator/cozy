# cozy-creator (v2)

The local-first product: `cozy` CLI + LocalService/LocalCoordinator (Go).
Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2) `cozy-creator.md`; issues `tracker/cozy-creator/` (`cl-*`).
No automated tests (decisions.md #160): verification = live runs + benchmarks.

## CLI foundation (cl-002)

- `internal/manifest` is the ONE command surface — data. It drives dispatch, help,
  capability tokens and the docs inventory (`cozy commands`). A startup self-check
  refuses (exit 1) unless every implemented row names a registered handler and every
  registered handler is advertised.
- `internal/exit` is the ONE error/exit layer: the shared matrix from
  cozy-runtime-cli.md, frozen in `docs/exit-matrix.md` and fence-checked. No local codes.
- `internal/render` is the ONE output layer: compact text, `--json`, `--full`,
  `--fields`, truncation, aggregates, `next:` lines.
- `internal/service` is the ONE seam onto the LocalService (cl-001/cl-006). Until it
  lands, server-backed verbs refuse typed-unavailable (9) with `next: cozy up`.
- AXI: bare `cozy` is live status (exit 0, never help), `-h/--help` is concise and
  never mutates or dials, and no prompt exists anywhere — destructive verbs refuse
  without `--yes` (exit 7).

Gates run most-durable-refusal-first: confirm (7) → service (9) → not-implemented (2).

`scripts/fence.py` enforces four families: forbidden deps, byte-plane vocabulary
(TensorFS owns storage/chunking/residency), interactive prompts, and exit-matrix parity.
Doors are greppable: `//cozy:allow`, `//cozy:stdin-value`.
