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

## The local client API (cl-006)

`cozy up` serves the **shared client contract's CORE** on loopback — the same surface
Tensorhub (th-021) and the private-deployment pod (cl-014) serve, byte for byte. Full
contract: [`docs/client-contract.md`](docs/client-contract.md), which
`scripts/fence.py`'s `contract` family checks against `internal/api/routes.go` row for
row. `internal/api` is a CLIENT of `internal/coord` and opens no second door into the
runtime: every submission still flows coordinator → worker protocol → runtime.

- **Submit / status / cancel** (`/v1/requests`). The idempotency key names one request
  forever; the body digest covers the WHOLE submission, so one key naming a different
  function conflicts as loudly as one with different input. 202 started it, 200 is a
  replay, 409 is a changed body.
- **Two SSE routes over one event model.** Durable lifecycle rows live in the same
  authority as the requests they describe — monotonic, totally ordered, resumable from
  `?cursor=` across a restart — and the terminal event is appended INSIDE the terminal
  transaction. Live progress is the lossy lane: never durable, never replayed, latest
  tick replayed on connect. `/v1/events` multiplexes every request onto one connection
  (a browser caps six per origin); `/v1/requests/{id}/events` terminal-stops.
  **An attempt ending is not a request ending**: an attempt the coordinator will requeue
  emits `request.attempt_failed`, deliberately outside the terminal set.
- **Media by OPAQUE id only.** The id is minted inside the terminal transaction, so an
  output no terminal published has no id at all — the ComfyUI `/view` traversal class is
  structurally absent rather than defended against. MIME allowlist, no inline SVG,
  `nosniff`, one bounded `Range`.
- **Triage** (`/v1/local/attempts/{key}/triage`). cl-006 owns what cr-011 deliberately did
  not: the bundle is COPIED out of the worker root at terminal time, verified against the
  digest the accepted TERMINAL declared (stronger than the worker's own journal), and
  re-verified on every read — an edited bundle is `bundle_corrupt`, never a story.
- **A loopback bind is not a boundary.** Loopback-only, IPv4 and IPv6, with no flag that
  widens it; a `Host` allowlist (the DNS-rebinding kill switch); `Origin` checks on every
  mutation and stream open; **bearer only, no cookie read anywhere**; no CORS header at
  all; a strict CSP. Two per-launch credentials die with the process — the browser's rides
  `--open`'s URL FRAGMENT (never the query string, so it reaches neither the server nor a
  log nor a `Referer`), the CLI's a 0600 file. Neither ever enters argv, a log, or an
  error. The `api` fence family proves the cookie, CORS and single-bind-site invariants.
- The **stub page** (cl-007's unblocked half) is embedded with `go:embed`, three files so
  its CSP needs no inline script.

Live drivers: `./cozy-live apiarms` (35 door/refusal checks, no GPU), `./cozy-live api`
(51 checks, real SDXL end to end), `./cozy-live apicrash` (23 checks — `kill -9` the
coordinator mid-attempt and watch the recovered-attempts law from the client's seat). All
three install the endpoint through `cozy install`; the runtime they run is the one the
RELEASE pinned (`scripts/sdxl-release.sh` builds it from a read-only `git archive`), so a
verification's peer cannot move underneath it.

## Lifecycle and request verbs (cl-010)

`start` / `stop` / `logs` / `run` / `describe` / `doctor` / `fit` — the CLI as the local
client API's **first client**. `internal/client` is the ONE way any of them reaches the
service, so there is no temporary direct-Go path for the HTTP API to later wrap:

    cozy run  = POST /v1/requests + GET /v1/requests/{id}/events + GET /v1/media/{id}
    cozy start/stop = POST / DELETE /v1/local/workers
    cozy logs <attempt> = GET /v1/local/attempts/{key}/triage
    cozy doctor = GET /v1/local/doctor

- **ONE EXECUTION PATH, and cold and warm traverse the same states.** A request whose
  binding no live worker advertises does not queue for capacity that nothing would create:
  the coordinator SELECTS-OR-STARTS (`coord.selectOrStart`), the worker's READY drains the
  queue, and dispatch stays the one placement path. `cozy start` is the same act made
  explicit for prewarming — never a second invocation mechanism. A worker that cannot
  become dispatchable settles the request FAILED rather than leaving it queued forever.
- **The CLI reads a 0600 credential**, never argv and never an env value. The file's mode
  is CHECKED: one that became group- or world-readable is exit 5, because reading it anyway
  would be the client agreeing to a leak the server tried to prevent.
- **Every API refusal renders typed.** `api.CodeOf` is `statusOf`'s inverse and lives
  beside it, so the server's status maps back to the shared matrix code while the
  envelope's own NAME survives (`override_unresolved`, `bundle_corrupt`).
- **An installed generation is the only source of launch facts** (`internal/launch`), and
  `--dev-endpoint` is DELETED with its loader and its writer. A generation carries the
  venv that runs it, the descriptor its own runtime derived and the install verified, and
  the `endpoint.toml` binding table that selects its artifact; the local artifact index
  answers where the bytes are. cozy-creator mints the local pinned-binding record from
  those three (th-004 owns the real EntrypointBindingPlan document — the named seam).
- **The payload is typed against the recorded schema**, client-side, before a request
  exists: `steps=2` is an int because the surface says int, an undeclared key is exit 3
  naming what the release declares, and none of it costs a subprocess or a round trip.
- **`--out` writes by DECLARED FIELD PATH** (`image`, `detail.thumb`) — a consequence of
  the endpoint's declared result, not a convention. The file gets no invented extension:
  the worker's manifest hard-codes `application/octet-stream` today, so the run degrades
  loudly and names the seam instead of sniffing the bytes.
- **SIGINT cancels, it does not abandon.** The client asks the coordinator to cancel and
  keeps watching, because the attempt's own journaled terminal settles the request; the
  exit code is the terminal's, through `exit.JobTerminal` (0 · 11 · 12 · 10).
- `describe` renders the surface the install already verified (re-deriving it per
  invocation would import the endpoint's whole module graph for a proven fact); `fit`
  always delegates, because a verdict prices against the card's measured free bytes now.

Live drivers: `./cozy-live verbs` (25 refusal arms, no GPU) and `./cozy-live journey`
(the whole user path on the real card). `scripts/sdxl-release.sh` builds the SDXL endpoint
RELEASE they install — source, a lock over its whole closure, and the two wheels that are
not on an index yet, with cozy-runtime built from a pinned read-only `git archive`.

## Bounded jobs and the durable publication root (cl-004)

`job submit` / `status` / `ls` / `follow` / `cancel` — the bounded job surface, LOCAL ONLY
(the hub's job plane is th-008's; a manually provisioned pod is not the loopback
coordinator). A job is an **attempt class on the one coordinator**, not a second scheduler:
same request row, same ordinal law, same terminal transaction, same durable event stream
(`cozy job follow` is `GET /v1/requests/{id}/events`'s client). What differs is what cr-009
says differs — a `JobDirective`, a `JobExecutionSpec`, job capacity instead of serving
capacity, and a grant that writes somewhere a reclaim cannot reach.

- **The publication root is a real place, and a job never writes into it.**
  `<COZY_HOME>/publications/<org>/_job-<job-id>` is the addressable publication; an attempt
  in flight is granted `<that>/.staging/a<N>` instead, and the coordinator PROMOTES the
  landed writes across by rename after the terminal document has been verified. So the
  addressable path only ever holds bytes some terminal vouched for. The fence is the
  resolution, not a scan: a destination that does not resolve INSIDE the grant's root
  cannot be granted, so no job ever holds a capability to write outside its own stage.
  Neither path is under `workers/` or `outputs/` — a bounded job is reclaimed at its
  terminal, and a bundle inside what reclaim removes would be destroyed by the act that
  ends the job that produced it.
- **The publication commits with the terminal.** The row rides `AcceptTerminal`'s
  transaction beside the outputs, so "a bundle is visible" and "a terminal was accepted"
  are one fact. The verdict is STAMPED as metadata and gates nothing: a failed run's landed
  writes still land (jobs.md), and a REQUEUEING attempt writes no publication at all.
- **The CHECKPOINT half is a runtime-border SEAM, not a Creator protocol.** A job that
  produces canonical bytes writes them through the one TensorFS border, and the publication
  TRANSACTION is the runtime's (cr-005/cr-009, jobs.md). What this host owes is to validate
  ONE typed durable publication receipt from that border, root what it names, and record
  the catalog projection. That receipt does not exist yet, so nothing here interprets a
  job's result to guess at one — the asset plane above is the whole of what ships.
- **A job worker is terminal-and-reclaim, and the QUEUE is the coordinator's.**
  `reclaim_on_terminal` is true: one immutable build, one bounded attempt, terminal,
  reclaim. Deep queueing (owner directive, decisions #394 / cr-019) is the dispatch queue
  plus select-or-start, not a warm worker — warm persistence is a serving concern.
  Submission never answers "busy": no capacity is a queued STATE, `queue_position` says
  where, and a submission with a non-empty queue JOINS it rather than overtaking it.
- **`job_descriptor_id` is derived, never read.** It is absent from the descriptor by
  design (a digest in a source-stable surface has no clock), so this host reproduces
  `internal/descriptor.py::job_descriptor_id` under tfs-013's writer rules — verified
  byte-identical against the runtime's own derivation, and checked again every run because
  the worker resolves its local plan record BY that string.
- **No fabricated `$0.00`.** The bill exists only where `COZY_LOCAL_RATE_MICRO_USD_PER_HOUR`
  configured one; otherwise the field is absent and the rendering says why.
- **The durable checkpoint exchange** is journaled here as a SECOND observation (law 9):
  same identity replays the receipt, same keys with different bytes conflict, and nothing
  on this side can un-write what the worker already made durable.

Live drivers: `./cozy-live jobarms` (refusal arms, no store), `./cozy-live jobs` (the real
census job end to end over the 6.9 GB bench store, the depth pass, kill/resume, the retry
projection, benchmarks) and `./cozy-live jobcrash` (the coordinator killed at the two
lifecycle points that decide whether a publication is a promise or a fact).
`scripts/job-release.sh` builds the release they install — cr-009's own
`structural_census.py`, verbatim from a pinned `git archive`.

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

**`cozy push <org/repo> <sha256:snapshot> --reason <why>`** drives th-002's
declare-first protocol: declare the exact object set before a byte moves → the hub
answers what it lacks → write only those, straight to their FINAL content keys under
every condition the grant signed → the hub streams each object back and hashes it
ITSELF → complete runs the hermetic verifier and installs one root. **The model family is
not declared** — th-003's hub CLASSIFIES it from the artifact's topology digest and refuses
an unknown field, so `--family` is gone (cl-010 closed cl-006's finding). A path refuses by
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

**The read plane is real.** th-003 landed `POST …/checkpoints/{snapshot}/reads` and
`scripts/xfer-realhub.sh` proves push → pull end to end against a hub built from that
commit, with no shim anywhere. Against an OLDER hub `pull` still refuses
`hub.no_read_plane` by name rather than inventing a bucket URL or reaching storage with a
credential of its own — a second custody authority is exactly what law 2 forbids.

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
  cozy-runtime supervisor + executor on a real GPU), `api`/`apiarms`/`apicrash` (the local
  client API from a client's seat), and `verbs`/`journey` (the CLI as a user types it,
  against an INSTALLED endpoint — no hand-written spec document anywhere).
- `scripts/verify-cl012.sh` + `scripts/xfer-live.py` drive the real `cozy` against a
  real tensorhub (built from a PINNED commit through a read-only `git archive`, because
  that repo has a concurrent writer), its own Postgres container, real R2 under
  `v2/cl-012/<run>/`, and real artifacts written by `tfs`: the publish round trip, the
  0-byte dedup publish, an interrupted publish and an interrupted fetch each converging
  on a re-run, the whole refusal matrix, and the benchmarks. Everything it writes to R2
  is swept and the sweep is re-listed to prove it.
- `scripts/fence.py` enforces thirteen families: forbidden deps, byte-plane vocabulary
  (TensorFS owns storage/residency), interactive prompts, exit-matrix parity, a manifest
  lint, the env-read fence (one reader), no lifecycle sidecar, no cloud emulation or
  minted credential, the secret fence (no credential-shaped flag takes an argv value;
  `Reveal()` only where the value becomes a header), the `cas` fence (the transfer plane
  hashes nothing; nobody composes a store path) and the `tensor` fence (the tensorfs CLI
  has one caller). Doors are greppable: `//cozy:allow`, `//cozy:stdin-value`.
