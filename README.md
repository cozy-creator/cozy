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
`scripts/pack.py` is the pre-hub packager an endpoint-release publish (th-003/th-004) replaces.

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

## Catalog and first-party publish (cl-011)

`internal/hub` is the ONE client onto tensorhub's HTTP API. Launch-1 tensorhub has **no
identity plane** (owner ruling, decisions #229): there are no accounts, no sessions and
no orgs, so the surface splits exactly two ways.

- **Reads are public.** `cozy search [<query>] [--kind]` and `cozy repo show <org/name>`
  carry no credential at all. A miss is exit 0 with an empty state; an unknown ref is
  exit 4.
- **First-party writes carry ONE static admin token.** `cozy repo create <org/name>
  --kind <model|endpoint> --reason <why>` and `cozy hub config` send
  `Authorization: Bearer $TENSORHUB_TOKEN`. `--reason` is required because the hub
  records why every admin mutation happened *before* it performs it.
- **`cozy hub status`** names the configured URL and the credential's digest with the
  provenance of each, and reports reachability as a STATE (exit 0) — an unreachable hub
  is what you asked about, not a refusal.
- **`cozy login` / `logout` are DEFERRED to th-031**, not forgotten: the manifest rows
  say so, and there is nothing for a PKCE flow to authenticate against. The env carries
  a value, never a login act.

Configuration is the frozen `internal/config` value: `TENSORHUB_URL` (default
`http://127.0.0.1:8080`) and `TENSORHUB_TOKEN`. The token is a `secret.Value` — it
renders as `sha256:<12 hex>`, the same spelling the hub uses for its own redacted keys,
so `cozy hub status` and `cozy hub config` can be compared without either end printing
it. `Reveal()` has exactly one caller: the line that builds the Authorization header.

Hub refusals reach the user **verbatim** — the hub's `{code, message, remedy}` envelope
under a code from the shared exit matrix: 401/403 → 5, 404 → 4, 409 → 13, 4xx → 3,
5xx/unreachable → 9, no answer at all → 10. An answer that is not a typed envelope
(an unknown route, a proxy, a login page) says so rather than being invented into one.

Not built here, and named: release/lane resolution and the `--lane` grammar (th-003),
per-source credential rows for `hf`/`civitai`/`comfy` (they need a source that is not
the hub, which arrives with cr-016's fetcher).

## Transfer: push and pull (cl-012)

Two verbs move a canonical checkpoint between the local store and the hub. Neither owns
a byte or a protocol — `internal/tfs` is the ONE door to a byte-plane fact and
`internal/hub` the ONE door to the hub — so `internal/transfer` owns only the SEQUENCE.

**`cozy push <org/repo> <sha256:snapshot> --family <f> --reason <why>`** drives th-002's
declare-first protocol: declare the exact object set before a byte moves → the hub
answers what it lacks → write only those, straight to their FINAL content keys under
every condition the grant signed → the hub streams each object back and hashes it
ITSELF → complete runs the hermetic verifier and installs one root. A path refuses by
name: only a canonical snapshot is publishable, and the border runs where the bytes are
(`tfs ingest`). `--dry-run` declares and stops — the plan is the hub's answer, not a
local guess.

**`cozy pull <org/repo>[@sha256:…]`** is the inverse, and it needs no route that lists a
checkpoint's objects, because the artifact declares itself: the manifest arrives first
and is admitted only if it hashes to the id asked for; then everything the manifest
names directly; then the transitive closure the byte plane computes from those
documents. `tfs fill` admits every object under its declared identity, and the snapshot
becomes a named local root only after `tfs snapshot verify` proves every declared byte
— cl-009's transactional install with a different source.

- **Resumability is not a feature, it is the absence of one.** There is no client-side
  journal. A publish resumes because the hub re-plans a re-declared closure server-side
  (`Begin` is idempotent on the closure digest); a fetch resumes because the store's own
  verification records already say which objects are good. A third opinion about what is
  done is always the wrong one.
- **Accounting separates MOVED from DEDUPED** on every line, and an object is counted
  once per run even though the fetch rounds overlap by construction.
- **The transfer plane hashes nothing** (fence family `cas`). A digest computed while
  moving bytes could only become a client receipt, and a client receipt substitutes for
  nothing — the hub re-verifies every object it already held, and says how many.
- `--token-stdin` takes this invocation's credential as a VALUE on stdin. It is not a
  prompt and never argv.

**`cozy pull` needs one hub route that does not exist.** th-002 landed the whole write
side and no read side: the hub signs PUTs at final keys and streams objects back for its
own verification, but exposes no route handing a client a presigned GET. Against such a
hub `pull` refuses `hub.no_read_plane` by name rather than inventing a bucket URL or
reaching storage with a credential of its own — a second custody authority is exactly
what law 2 forbids. `scripts/xfer-live.py` supplies that one route (and nothing else) so
the rest of the path can be proved on real bytes.

`deploy` and `promote` stay advertised-not-built: releases arrive with th-003 (the hub's
own promote route answers `promote.not_armed` today) and the leased build that turns a
source snapshot into a release with th-004. `datasets push|pull` waits on th-035.

## Verification

No automated tests. Verification is running the real thing:

- `scripts/redarm.py` builds each hostile input for real, runs the real binary and
  observes the typed refusal (16 arms).
- `scripts/hub-live.py` drives the real `cozy` against a REAL tensorhub: public reads,
  admin writes, every refusal arm (wrong token, absent repo, closed port, a hub that
  accepts and never answers, a 200 that is not our document), and a cumulative secrecy
  check that the raw credential appears in no byte the session printed.
- `cmd/cozy-live` drives the REAL coordinator against real peers:
  `canonical` (worker-protocol's frozen corpus, byte-for-byte, plus every semantic twin
  refused by its own code), `arms` (the refusal matrix against `fakeworker`, a second
  independent Go implementation of the worker side), `attempt` and `recovered` (the real
  cozy-runtime supervisor + executor on a real GPU).
- `scripts/verify-cl012.sh` + `scripts/xfer-live.py` drive the real `cozy` against a
  real tensorhub (built from a PINNED commit through a read-only `git archive`, because
  that repo has a concurrent writer), its own Postgres container, real R2 under
  `v2/cl-012/<run>/`, and real artifacts written by `tfs`: the publish round trip, the
  0-byte dedup publish, an interrupted publish and an interrupted fetch each converging
  on a re-run, the whole refusal matrix, and the benchmarks. Everything it writes to R2
  is swept and the sweep is re-listed to prove it.
- `scripts/fence.py` enforces eleven families: forbidden deps, byte-plane vocabulary
  (TensorFS owns storage/residency), interactive prompts, exit-matrix parity, a manifest
  lint, the env-read fence (one reader), no lifecycle sidecar, no cloud emulation or
  minted credential, the secret fence (no credential-shaped flag takes an argv value;
  `Reveal()` only where the value becomes a header), the `cas` fence (the transfer plane
  hashes nothing; nobody composes a store path) and the `tensor` fence (the tensorfs CLI
  has one caller). Doors are greppable: `//cozy:allow`, `//cozy:stdin-value`.
