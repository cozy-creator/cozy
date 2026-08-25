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

## Install, pins and maintenance (cl-009)

An install is an **immutable generation** plus a **pin per `(endpoint, major)`**, built
through one staged transaction:

    stage → verify source → build venv → descriptor → activate

`internal/records` is the ONE local lifecycle authority: `install_generations` and
`pins` rows in one local Turso/libSQL database (`$COZY_HOME/records.db`, default
`~/.cozy`). No `state.json`, no second store. cl-001's LocalService adopts this package
and adds its own tables to the same database. Driver: `github.com/tursodatabase/go-libsql`
(embedded libSQL, **CGO_ENABLED=1**) — never vanilla SQLite.

`internal/install` owns the pipeline. **stage** extracts a `.tar.gz` release under
compressed/expanded/file-count/path-length bounds and refuses traversal, absolute paths,
links, devices, duplicate and case-folding names, and entries the release does not
declare; the declaration is the archive's *first* entry, so an undeclared entry refuses
before it is written. **verify** settles source identity before any code executes —
`--digest` by default, `--allow-unsigned` the development door, `--dir` the editable
trust path with a source-snapshot digest. **venv** runs `uv sync --locked`: no resolve,
no relaxed fallback, no lock rewrite. **descriptor** runs `cozy-runtime describe --check`
in the generation's own venv (cr-003): the release's own runtime derives its surface
without loading weights and compares it against the committed `endpoint.descriptor.json`,
so a descriptor that disagrees with the code refuses the install (exit 13, naming which
pair diverged) and a passing check records its `surface_digest`. There is no door.
**activate** inserts the generation row and swaps
the pin in ONE transaction, so a kill at any earlier stage leaves the prior pin runnable.

Hardlink dedup is uv's link mode, not a second pool; it degrades to copies with a loud
warning across mounts. Disk is measured once at install — `ls` reads the record and never
walks the tree. `rm --yes` drops the pin and venv; `gc` is the plan without `--yes` and
executes with it, reclaiming only what nothing references.

`--from <archive>` is the pre-hub source door standing in for the hub resolve (cl-011);
`scripts/pack.py` is the pre-hub packager `cozy deploy` (cl-012) replaces.

## The LocalService and LocalCoordinator (cl-001)

`cozy up` starts the ONE long-lived process; `cozy down` stops it. Its scheduling role is
the **LocalCoordinator** (`internal/coord`): the worker protocol's **SERVER** over a unix
socket — the cozy-runtime supervisor dials a stable address — plus local dispatch, the
device ledger, and output publication authority.

- **Identity is canonical bytes.** `internal/canonical` is the document plane: the writer
  is adapted from worker-protocol's own independent Go canonicalizer, the reader is this
  repo's and refuses everything the writer could not have produced. `protocol/` carries
  the generated `cozy.worker.v1` bindings, COPIED (never imported as a module).
- **One transaction.** "terminal accepted + output visible" commits together, and only
  then is `TerminalAck` sent. A crash between them replays; it cannot half-apply.
- **The recovered-attempts law** (worker-protocol/02 §6.2): the attempts a restarted
  supervisor reports on `Register` are OPEN OBLIGATIONS, and no next ordinal for their
  request ids is minted until each is closed by its own journaled terminal.
- **Device grants** are an attribute of the worker process row, admitted by one atomic
  statement. Two concurrent starts cannot both consume an envelope.
- **Liveness is an OS fact.** The service holds an exclusive `flock` on
  `$COZY_HOME/service.lock` for its life; a reader that can TAKE the lock has proof of
  absence. No pidfile, no heartbeat, no grace period. Worker adoption uses the protocol's
  own identities plus an OS process-birth identity: only the process this service started
  may bind a worker slot (`SO_PEERCRED`), and a reused pid is never signalled.
- **No credential is minted anywhere.** A local grant is a CAS root plus an output
  directory; the payload rides it as the input `payload`, and one `OutputDestination` is
  named per result FIELD PATH.
- `internal/config` is the ONE environment reader: `Load()` reads once and freezes, and
  it is also the child-env allowlist.

## Verification

No automated tests. Verification is running the real thing:

- `scripts/redarm.py` builds each hostile input for real, runs the real binary and
  observes the typed refusal (16 arms).
- `cmd/cozy-live` drives the REAL coordinator against real peers:
  `canonical` (worker-protocol's frozen corpus, byte-for-byte, plus every semantic twin
  refused by its own code), `arms` (the refusal matrix against `fakeworker`, a second
  independent Go implementation of the worker side), `attempt` and `recovered` (the real
  cozy-runtime supervisor + executor on a real GPU).
- `scripts/fence.py` enforces eight families: forbidden deps, byte-plane vocabulary
  (TensorFS owns storage/residency), interactive prompts, exit-matrix parity, a manifest
  lint, the env-read fence (one reader), no lifecycle sidecar, and no cloud emulation or
  minted credential. Doors are greppable: `//cozy:allow`, `//cozy:stdin-value`.
