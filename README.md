# cozy-creator (v2)

The local-first product: `cozy` CLI + LocalService/LocalOrchestrator (Go).
Design authority: [tracker-v2](https://github.com/cozy-creator/tracker-v2) `cozy-creator.md`; issues `tracker/cozy-creator/` (`cl-*`).
Verification is LIVE (decisions.md #160): `go test ./internal/live` starts the real system
and observes it — no mocks, no unit-test layer. #634 amends #160 for this repo only.

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

Gates run help-first, then most-durable-refusal-first: `-h` (0) → confirm (7) →
not-implemented (2) → arity (2) → service (9). Arity is a gate, not a parse step, so
`-h` and a planned row's owning issue are never masked by an argument count.

## Install, pins and maintenance (cl-009)

An install is an **immutable generation** plus a **pin per `(endpoint, major)`**, built
through one staged transaction:

    stage → verify source → build venv → descriptor → activate

`internal/records` is the ONE local lifecycle authority: `install_generations` and
`pins` rows in one local SQLite database (`$COZY_HOME/records.db`, default `~/.cozy`).
No `state.json`, no second store. cl-001's LocalService adopts this package and adds its
own tables to the same database. Driver: `modernc.org/sqlite` — SQLite transpiled to Go,
so the binary is **CGO_ENABLED=0** and cross-compiles everywhere. The store uses no
engine-specific feature, so the driver is a distribution decision and pure Go settles it;
the file is plain `SQLite format 3`, so a `records.db` an earlier libSQL-linked build
wrote opens here untouched. Pragmas ride the DSN (`busy_timeout`, `foreign_keys`,
`journal_mode=WAL`) because a pragma is a property of a connection and `database/sql`
may redial one at any moment.

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
pair diverged) and a passing check records its canonical semantic `descriptor_digest`.
Whitespace and object-key order are not surface meaning. There is no door.
**activate** inserts the generation row and swaps
the pin in ONE transaction, so a kill at any earlier stage leaves the prior pin runnable.

Hardlink dedup is uv's link mode, not a second pool; it degrades to copies with a loud
warning across mounts. Disk is measured once at install — `ls` reads the record and never
walks the tree. `rm --yes` drops the pin and venv; `gc` is the plan without `--yes` and
executes with it, reclaiming what nothing references plus retained outputs past the
retention horizon (cl-033).

`--from <archive>` is the pre-hub source door standing in for the hub resolve (cl-011);
`scripts/pack.py` is the pre-hub packager an endpoint-release publish (th-003/th-004) replaces.

## The deterministic wheel packer (th-039)

`internal/wheel` is the ONE way an endpoint project becomes an importable unit, and it
lives here because both seats that need it share this build seat: the tensorhub env-lane
build stage (tensorhub-build.md §1 stage 2e) and a local `cozy install` must produce the
SAME `project_wheel_digest` for the same tree, and they can only do that by running the
same code. `wheel.Pack` is that seam; `cozy pack <tree>` is its local door.

It is FIXED first-party code. It does not consult the repo's `[build-system]`, does not
import the project, and fires no PEP 517 hook: it copies the canonical path-sorted tree,
synthesizes `METADATA`/`WHEEL`/`RECORD` from declared metadata read as DATA, and emits one
`py3-none-any` wheel. Distribution identity comes from `[project]` when the tree declares
it and from the caller otherwise — an endpoint's identity is its RELEASE, not a line in
its tree.

**Determinism is structural, not hoped for.** The wheel bytes are a function of (canonical
tree, distribution identity, packer version) and nothing else. Two consequences are chosen
rather than inherited: the zip container is written by hand — the two seats do not share a
build of Go, so a library's framing choices must not enter the digest — and entries are
STORED, not deflated, because a compressor's output is a property of whoever compiled the
packer. Every entry carries 1980-01-01 and mode 0644; no path inside is absolute.

Refusals are typed and ordered: `build_backend_unsupported`, `compiled_extension`,
`project_code_execution`, `metadata_malformed`/`metadata_dynamic`, `unsafe_entry`,
`application_module_absent`. The first three each name the FUTURE sandboxed class —
custom PEP 517 backends and compiled project wheels execute tenant code at assembly, so
they land on the VM-class sandbox posture (tensorhub-build.md §1.1) or they do not land.

## The LocalService and LocalOrchestrator (cl-001)

`cozy up` starts the ONE long-lived process; `cozy down` stops it. Its scheduling role is
the **LocalOrchestrator** (`internal/orchestrator`): local placement, dispatch, and the
device ledger. The same service is the worker protocol's **record owner**: it DIALS each
worker's own socket (#436), owns durable attempt ordinals and terminal acceptance, and
publishes accepted outputs. The wire is `cozy.worker.v1` at **schema rev 8** — rev-2's
dynamic-serving shape with honest control-Runtime provenance.

- **Identity is canonical bytes.** `internal/canonical` is the document plane: the writer
  is adapted from worker-protocol's own independent Go canonicalizer, the reader is this
  repo's and refuses everything the writer could not have produced. `protocol/` carries
  the generated `cozy.worker.v1` bindings, COPIED (never imported as a module) and
  byte-checked against the schema by `ci.yaml`'s `vendored-protocol` job.
- **The schema fences itself** (#530-A1). `wire_schema_digest` rides `Claim`/`ClaimAck`,
  is checked BEFORE any other body field, and refuses the handshake on a mismatch or an
  absence — absence being precisely the pre-rev-2 signal, since an older binding cannot
  spell the field. `WIRE_MINOR` is 2 and fences nothing: the minor is the additive
  train and cannot honestly move for a breaking in-place revision.
- **Actor names match their current responsibilities.** Cause origin 4 is `WORKER`, the
  machine-local peer; origin 7 is `RECORD_OWNER`, the durable-attempt authority. Rev 3
  retains their wire numbers and hardcuts the old names with no aliases.
- **The desired set is BYTES.** A local target authors its `PlacementSet` once. A rented
  target instead persists and validates Tensorhub's acquisition-attempt control snapshot,
  then relays that exact `PlacementSet/2`; it never resolves or renders it from this
  machine's install. After the snapshot ack, the RecordOwner obtains one scoped artifact
  grant and sends it before desired state. Plans and the required model-object set are
  ordinary grant subjects fetched by the worker; refreshing their expiring locations never
  changes the desired revision. The invocation media plane carries no endpoint distribution
  route. The worker recomputes every subject digest before using its bytes.
- **A rental is a reusable pod, not an endpoint-shaped image.** `cozy rent revise` asks
  Tensorhub to author one immutable active placement revision, then the existing
  RecordOwner sends that revision's grant before its set on the already-claimed stream.
  The worker instance and boot identity do not change; a different base-worker-image family
  refuses at Tensorhub before Creator can relay it.
- **Acquisition is observable, not authoritative.** Schema rev 7 carries endpoint/model
  monotonic intervals plus downloaded/reused byte counters. Creator persists those facts
  by worker boot + placement spec and exposes them on the worker listing, but no timing or
  counter participates in identity, convergence, readiness, or admission.
- **Two axes, one admission fence.** A placement's convergence is
  materialization × serving; DISPATCHABLE gates dispatch. Capacity is a WORKER property —
  `admission_state` + `admission_generation` + `available_attempt_slots` — and every offer
  echoes the generation it observed, so a stale one refuses deterministically.
- **Acceptance and convergence are two facts.** `accepted_desired_state_revision` moves on
  acceptance; `converged_revision` moves only when observed state satisfies it. The old
  `applied_revision` is retired as dishonest.
- **One transaction.** "outcome accepted + output visible" commits together, and only
  then is `AttemptOutcomeAck` sent. A crash between them replays; it cannot half-apply.
  An outcome this record owner holds unacked is counted against the worker's own free
  seats, so a record owner that stops acking starves its own admission.
- **Every offer gets a journaled outcome**, including a refusal. The REFUSED projection
  SPLITS by (cause, origin): author/runtime causes settle; WORKER pre-execution causes
  consume the ordinal, spend zero execution budget (`execution_started` is a structural
  bit, never a cause-code allowlist) and are immediately re-dispatchable.
- **The recovered-attempts law** (worker-protocol/02 §6.2): the `held_attempts` a
  restarted worker reports in its ONE digest-acked snapshot are OPEN OBLIGATIONS, and no
  next ordinal for their request ids is minted until each is closed by its own journaled
  outcome. The snapshot is bounded by its own digest — a truncated one cannot match — and
  dispatch stays closed until this record owner acks the exact (id, digest).
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

`cozy up` serves Creator's local client API on loopback. Its CORE routes are the proposed
common request-level API for future Tensorhub and private-rental servers; no cross-host
parity is claimed until those servers pass shared conformance tests. The current contract
is [`docs/client-contract.md`](docs/client-contract.md). `scripts/fence.py` checks its
route method, path, scope, and order against `internal/api/routes.go`; payload and behavior
remain implementation-and-test contracts. `internal/api` is a CLIENT of
`internal/orchestrator` and opens no second door into the runtime: every submission still
flows orchestrator → worker protocol → runtime.

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
  **An attempt ending is not a request ending**: an attempt the orchestrator will requeue
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
  mutation and stream open; **bearer only, no cookie read anywhere**; **no CORS header
  here, absolutely and without a door** (#628) — a CORS header on a loopback bind is what
  makes it readable by any page the browser loads; a strict CSP. Elsewhere a CORS header
  needs a `//cozy:allow` door naming where its origins come from. Two per-launch credentials die with the process — the browser's rides
  `--open`'s URL FRAGMENT (never the query string, so it reaches neither the server nor a
  log nor a `Referer`), the CLI's a 0600 file. Neither ever enters argv, a log, or an
  error. The `api` fence family proves the cookie, CORS and single-bind-site invariants.
- The **stub page** (cl-007's unblocked half) is embedded with `go:embed`, three files so
  its CSP needs no inline script.

Verified by `go test ./internal/live -run TestLocalAPIDoor`: the door matrix against a
real `cozy up` process — no credential, a wrong one, a rebinding Host, a cross-origin
mutation, a credential presented as a cookie, a path where a media id belongs, and the
credential absent from every byte the service writes.

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
  the orchestrator SELECTS-OR-STARTS (`orchestrator.selectOrStart`), the placement becoming
  DISPATCHABLE drains the
  queue, and dispatch stays the one placement path. `cozy start` is the same act made
  explicit for prewarming — never a second invocation mechanism. A worker that cannot
  become dispatchable settles the request FAILED rather than leaving it queued forever.
- **The CLI reads a 0600 credential**, never argv and never an env value. The file's mode
  is CHECKED: one that became group- or world-readable is exit 5, because reading it anyway
  would be the client agreeing to a leak the server tried to prevent.
- **Every API refusal renders typed.** `api.CodeOf` is `statusOf`'s inverse and lives
  beside it, so the server's status maps back to the shared matrix code while the
  envelope's own NAME survives (`override_unresolved`, `bundle_corrupt`).
- **The runtime answers questions about itself.** `describe`, `list`, `doctor`, `fit
  --json` and `bindings --json` are the runtime's own verbs, and cozy-creator reads their
  documents rather than growing second readers of the same grammars: cl-010's
  `[bindings]` table reader and its fit-row parser are both DELETED (cr-016 built the
  replacements for exactly that), and so are the last two: the supervisor is entered
  through the PUBLIC verb `cozy-runtime serve` rather than a private
  `internal.worker.session:main` import (runtime `db4ab8a` passes its already-read config
  into `supervise`), so the orchestrator speaks the verb's own closed launch grammar
  (`--socket`/`--out`); for a wholly weightless release, `bindings --json` also supplies
  exact canonical plan subjects before spawn and `serve --weightless-endpoint` privately
  stages those same bytes; and `job_descriptor_id` is READ from `describe <job> --json`
  (runtime `4485f27`) rather than re-derived, which deletes cl-004's Go reimplementation
  of the canonical form along with the refusals it owed for values the protocol profile
  cannot spell.
- **An installed generation is the only source of launch facts** (`internal/launch`), and
  `--dev-endpoint` is DELETED with its loader and its writer. A generation carries the
  venv that runs it, the descriptor its own runtime derived and the install verified, and
  the `endpoint.toml` binding table that selects its artifact; the local artifact index
  answers where the bytes are. The modeled path still carries its explicit local record.
  The weightless path consumes Runtime-authored plan subjects and never recreates its
  canonical closure in Go.
- **The payload is typed against the recorded schema**, client-side, before a request
  exists: `steps=2` is an int because the surface says int, an undeclared key is exit 3
  naming what the release declares, and none of it costs a subprocess or a round trip.
- **`--out` writes by DECLARED FIELD PATH** (`image`, `detail.thumb`) — a consequence of
  the endpoint's declared result, not a convention. The extension comes from the type the
  MANIFEST declares and from nothing else: the client sniffs no bytes, and an output that
  declares no type is written without a suffix and says so. (cl-010 landed against a
  runtime that hard-coded `application/octet-stream`; the pinned peer declares the real
  type, so the file is `image.png`.)
- **SIGINT cancels, it does not abandon.** The client asks the orchestrator to cancel and
  keeps watching, because the attempt's own journaled terminal settles the request; the
  exit code is the terminal's, through `exit.JobTerminal` (0 · 11 · 12 · 10).
- `describe` renders the surface the install already verified (re-deriving it per
  invocation would import the endpoint's whole module graph for a proven fact); `fit`
  always delegates, because a verdict prices against the card's measured free bytes now.

Verified by `go test ./internal/live -run TestProductPath`: install -> up -> invoke ->
typed result -> the file on disk, both terminal verdicts, the client-side payload grammar,
and a `kill -9` of the record owner mid-attempt. It runs on `fixtures/weightless/` — a real
endpoint with no model and no card — so it is the whole product path minus the GPU.

## Cozy Video

`cozy video compose` turns an editable `cozy.video/1` YAML source into an ordinary
ordered workflow without starting it. `cozy video submit` submits either that source or a
retained creative-plan digest through the same workflow owner. The source grammar,
examples, asset rules, and deliberate omissions are documented in
[`docs/cozy-video.md`](docs/cozy-video.md).

## Bounded jobs and the durable publication root (cl-004)

`job submit` / `status` / `ls` / `follow` / `cancel` — the bounded job surface, LOCAL ONLY
(the hub's job plane is th-008's; a manually provisioned pod is not the loopback
orchestrator). A job is an **attempt class on the one orchestrator**, not a second scheduler:
same request row, same ordinal law, same terminal transaction, same durable event stream
(`cozy job follow` is `GET /v1/requests/{id}/events`'s client). What differs is what cr-009
says differs — a `JobDirective`, a `JobExecutionSpec`, job capacity instead of serving
capacity, and a grant that writes somewhere a reclaim cannot reach.

- **The publication root is a real place, and a job never writes into it.**
  `<COZY_HOME>/publications/<org>/_job-<job-id>` is the addressable publication; an attempt
  in flight is granted `<that>/.staging/a<N>` instead, and the orchestrator PROMOTES the
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
- **A job worker is terminal-and-reclaim, and the QUEUE is the orchestrator's.**
  `reclaim_on_terminal` is true: one immutable build, one bounded attempt, terminal,
  reclaim. Deep queueing (owner directive, decisions #394 / cr-019) is the dispatch queue
  plus select-or-start, not a warm worker — warm persistence is a serving concern.
  Submission never answers "busy": no capacity is a queued STATE, `queue_position` says
  where, and a submission with a non-empty queue JOINS it rather than overtaking it.
- **`job_descriptor_id` is read from its owner.** It is absent from the descriptor by
  design (a digest in a source-stable surface has no clock), so cl-004 reproduced
  `internal/descriptor.py::job_descriptor_id` here in Go. cr-016's `describe <job> --json`
  now carries it, so the second implementation is DELETED: the id comes from the runtime
  that derives it, and a release pinning a runtime too old to say it refuses by name
  rather than being handed a plausible digest.
- **No fabricated `$0.00`.** The bill exists only where `COZY_LOCAL_RATE_MICRO_USD_PER_HOUR`
  configured one; otherwise the field is absent and the rendering says why.
- **The durable checkpoint exchange** is journaled here as a SECOND observation (law 9):
  same identity replays the receipt, same keys with different bytes conflict, and nothing
  on this side can un-write what the worker already made durable.

`scripts/job-release.sh` builds the census release a live job run installs — cr-009's own
`structural_census.py`, verbatim from a pinned `git archive`.

## Local output retention (cl-033)

An attempt's outputs are mirrored onto this host BEFORE its terminal is acknowledged and
are kept when the pod that produced them is destroyed — a rented GPU costs dollars an hour
and `cozy rent release` must not take its results with it. That makes
`<COZY_HOME>/outputs/` the one plane here that grows forever, so it has a policy, and the
policy is ONE rule: an output becomes reclaimable when it is older than the horizon
(`--keep-media`, default 30d). Nothing else. In particular it is not an unreferenced
sweep — every retained output belongs to a settled request by construction, so "nothing
references it" is true of all of them and would delete exactly what this plane exists to
keep.

`cozy media ls` is what the host is storing, read straight out of the one local database
so it answers with the service down. `cozy gc` without `--yes` prints what would go and
removes nothing; with `--yes` it unlinks the bytes and stamps the row. The ROW survives
its bytes on purpose: `GET /v1/media/{id}` then answers 410 `media_reclaimed`, which is a
different fact from the 404 an id that never existed gets. A job's bytes are excluded —
they are its durable publication, not a mirror — and the deleter refuses any recorded path
that does not resolve under the local output namespace.

KNOWN LIMITATION, recorded rather than fixed: the output fetch is SYNCHRONOUS at terminal
time, so a pod that dies before its terminal takes those outputs with it. There is no late
fetch.

## The durable artifact transaction (cl-023)

A job may declare **artifact outputs** — TensorFS snapshot slots, stated explicitly beside
the InvocationSpec because rev-5's `OutputBinding` carries no kind. The declaration decides
what an artifact slot is; an arriving receipt never does, and this lane is artifact-only so
a missing receipt can be classified without guessing about asset slots.

For each declared slot the record owner records exactly **one** durable final intent inside
the terminal transaction — `ADOPT` the returned `ArtifactReceipt/1` into this job's own
scratch root, `ABANDON` a committed receipt, or `ABANDON_UNCOMMITTED` an open slot — then
asks the runtime to execute it over th-049's one `ArtifactFinalizeRequest`/`Result`
exchange and journals the exact result BEFORE `AttemptOutcomeAck`. First intent wins; the
opposite decision conflicts forever; a lost result replays the identical request.

- **Creator authors no storage identity.** It never mints a receipt, never derives a
  snapshot id, and refuses a bare snapshot id, a job-authored authority, a cross-job or
  cross-spec receipt, an undeclared slot, and any local path — the closed `ArtifactReceipt`
  document has no slot for one.
- **The scratch root is Creator's own derivation** inside its job namespace: an opaque
  semantic id, never a path and never the public `<org>/_job-*` repository name.
- **Nothing becomes public.** An adopted artifact job writes no output row and no
  publication; promotion out of the scratch root is a later, explicit act.
- `GET /v1/requests/<id>` grows an `artifacts` array — slot, disposition, outcome, receipt
  digest, scratch root id. No path, no TensorFS internals.

The runtime half of the exchange is cr-030's and does not exist yet, so this plane has no
standing verification: its adversary matrix was deleted with `cmd/cozy-live` (decisions
#634) rather than kept green against a peer that cannot answer.

## Catalog and first-party publish (cl-011)

`internal/hub` is the ONE client onto tensorhub's HTTP API. Launch-1 tensorhub has **no
identity plane** (owner ruling, decisions #229): there are no accounts, no sessions and
no orgs, so the surface splits exactly two ways.

- **Reads are public.** `cozy endpoint search|show` and `cozy model search|show`
  carry no credential at all. A search miss is exit 0 with an empty state; an unknown
  typed ref is exit 4. There is no generic `repo` or root `search` command.
- **First-party writes carry ONE static admin token.** `cozy endpoint create
  <org/name> --reason <why>`, `cozy model create <org/name> --reason <why>`, and
  `cozy hub config` send
  `Authorization: Bearer $TENSORHUB_TOKEN`. `--reason` is required because the hub
  records why every admin mutation happened *before* it performs it.
- **`cozy hub status`** names the configured URL and the credential's digest with the
  provenance of each, and reports reachability as a STATE (exit 0) — an unreachable hub
  is what you asked about, not a refusal.
- **`cozy login` / `logout` are DEFERRED to th-031**, not forgotten: the manifest rows
  say so, and there is nothing for a PKCE flow to authenticate against. The env carries
  a value, never a login act.

Configuration is the frozen `internal/config` value, composed from LAYERED SOURCES
(cl-029), later layers winning: defaults → `$COZY_HOME/config.yaml` → `.env` in the
working directory → the process environment → the manifest's own CLI flags. The FILE is
the primary home for a local product's values — `tensorhub_url`, `tensorhub_token`,
`tfs`, `local_rate_micro_usd_per_hour`, `port`, `yield`, flat `key: value` scalars with
a CLOSED key set (an unknown key refuses naming the known six). The env spellings stay
admitted above it (`TENSORHUB_URL`, `TENSORHUB_TOKEN`, `COZY_TFS`,
`COZY_LOCAL_RATE_MICRO_USD_PER_HOUR`, `COZY_HOME` — the recorded implementer's call for
the rate: the file is its primary home and the env name survives, because dev roots
already speak it), and every value keeps its source (`default|file|dotenv|env`) so
`cozy hub status` says which layer spoke last. The token is a `secret.Value` — it
renders as `sha256:<12 hex>`, the same spelling the hub uses for its own redacted keys,
so `cozy hub status` and `cozy hub config` can be compared without either end printing
it. `Reveal()` has exactly one caller: the line that builds the Authorization header.

Hub refusals reach the user **verbatim** — the hub's `{code, message, remedy}` envelope
under a code from the shared exit matrix: 401/403 → 5, 404 → 4, 409 → 13, 4xx → 3,
5xx/unreachable → 9, no answer at all → 10. An answer that is not a typed envelope
(an unknown route, a proxy, a login page) says so rather than being invented into one.

Not built here, and named: per-source credential rows for `hf`/`civitai`/`comfy`
(they need a source that is not the hub, which arrives with cr-016's fetcher).

## Model publication and download (cl-012)

Two verbs move a canonical checkpoint between the local store and the hub. Neither owns
a byte or a protocol — `internal/tfs` is the ONE door to a byte-plane fact and
`internal/hub` the ONE door to the hub — so `internal/transfer` owns only the SEQUENCE.

**`cozy model publish <org/model> <sha256:snapshot> --reason <why>`** drives the
incremental publication protocol: open under a stable operation id → claim sorted known
ObjectRefs → obtain grants only for unsettled transfer ids → report exact received bytes
→ have Tensorhub stream and hash each transfer → seal the exact closure, topology, and
manifest documents. The seal runs the hermetic verifier and installs one root. **The model family is
not declared** — th-003's hub CLASSIFIES it from the artifact's topology digest and refuses
an unknown field, so `--family` is gone (cl-010 closed cl-006's finding). A path refuses by
name: only a canonical snapshot is publishable, and the border runs where the bytes are
(`tfs ingest`). `--dry-run` opens and claims the transfer rows, then stops before grants
or bytes — the plan is the hub's durable state, not a local guess.

**`cozy model download <org/model>[@release|@sha256:…] [--lane <selector>]`** first
resolves through `GET /v1/models/resolve`; there is no client-side default release or
checkpoint guess. It then needs no route that lists a checkpoint's objects, because the artifact declares itself: the manifest arrives first
and is admitted only if it hashes to the id asked for; then everything the manifest
names directly; then the transitive closure the byte plane computes from those
documents. `tfs fill` admits every object under its declared identity, and the snapshot
becomes a named local root only after `tfs snapshot verify` proves every declared byte
— cl-009's transactional install with a different source.

- **There is no client-side journal.** A publication resumes from Tensorhub's durable
  transfer rows under the same operation id; a download resumes because the store's own
  verification records already say which objects are good. A third opinion about what is
  done is always the wrong one.
- **Accounting separates MOVED from DEDUPED** on every line, and an object is counted
  once per run even though the fetch rounds overlap by construction.
- **Tensorhub accepts only scoped custody.** An already-accepted immutable transfer can
  reuse that exact fact without another HEAD/hash; sealing still hydrates and verifies the
  complete checkpoint before catalog commit. A client digest is never custody proof.
- `--token-stdin` takes this invocation's credential as a VALUE on stdin. It is not a
  prompt and never argv.

**The read plane is real.** `POST …/checkpoints/{snapshot}/reads` and
`scripts/xfer-realhub.sh` proves model publish → model download end to end against a hub
built from the current contract, with no shim or retired route anywhere.

`cozy endpoint publish` and `cozy endpoint promote` stay advertised-not-built: releases
arrive with th-003 (the hub's own promotion route answers `promote.not_armed` today) and
the leased proof that turns endpoint source into a release with th-004. `datasets
push|pull` waits on th-035.

## Release and distribution (cl-013)

Distribution is ACCEPTANCE, not packaging residue. Source state, tracker state,
installed-binary state and release state are four distinct evidence axes, so a green
`master` is not a shipped product and CI says so with two separate jobs.

- **`scripts/release.sh` refuses a dirty tree.** A release binary must provably carry its
  commit, and a stamp naming a commit whose bytes were not the bytes built is a lie the
  installer would propagate. `--allow-dirty` is the development door and puts `+dirty` in
  the tag, so the artifact confesses. Every target is built TWICE into separate directories
  and compared; a platform that does not reproduce is reported as UNBUILT rather than
  shipped. The tarball is deterministic (sorted, epoch mtimes, `gzip -n`) — otherwise the
  checksum would be a fact about the clock.
- **UNBUILT is part of the output, and the C wall is gone from it.** `RELEASE.json` names
  every platform that was not built and why, because an absent artifact and an artifact
  nobody attempted look identical in a directory listing. `cozy` is pure Go, so
  `linux/amd64`, `linux/arm64`, `darwin/arm64` and `darwin/amd64` are four cross-compiles
  from one host. The per-OS facts that used to be C are Go build tags now —
  `internal/flock` (flock vs `LockFileEx`), `internal/install` (statfs vs
  `GetDiskFreeSpaceEx`, `st_dev`/`st_nlink` vs the volume serial), and `internal/orchestrator`'s
  peer credential (`SO_PEERCRED`, Darwin's `LOCAL_PEERCRED`, and no answer at all
  elsewhere, which `peerPID` already reads as 0). `windows/amd64` is the one row left, and
  it is no longer a dependency: `internal/orchestrator/worker.go` spells the worker's process
  group and its kills as `syscall.SysProcAttr{Setpgid}` and `syscall.Kill`, which Windows
  answers with Job Objects instead. The UNBUILT reason quotes the compiler naming those
  five lines, which is a port of one file rather than a wall.
- **A build is not a run, and `RELEASE.json` says which it was.** Each artifact row carries
  `binary_sha256`, `binary_bytes`, the `format` line `file` printed, and `executed` — and
  on a Linux builder only the `linux/amd64` row can say it ran. The macOS and Windows
  binaries are static evidence until a runner of that platform drives `scripts/accept.sh`
  against them. `scripts/install.sh` is a POSIX installer and the Windows tarball carries
  `cozy.exe`, so Windows also still owes an installer of its own.
- **`scripts/install.sh` verifies the checksum BEFORE anything is replaced**, then stages
  inside the target directory so the final move is a rename on one filesystem. A corrupted
  asset is exit 13 with both digests named and the working installation untouched — the
  matrix's own reading of 13, "failed replacement kept the working state". Install and
  upgrade are ONE act: point it at a different asset. There is no download and no channel;
  `latest` and rollback are the first-external-user tier.
- **`scripts/accept.sh` is the fixture the guard runs**, and it drives the release asset and
  nothing else — no Go toolchain, no repository, no build tree. `scripts/clean-machine.sh`
  is that file plus a throwaway container with an empty home to run it on.
- **`fixtures/weightless/`** is an endpoint with no `Model` parameter, so `gpu` derives
  False and its release depends on the BASE cozy-runtime wheel alone — no torch anywhere in
  its venv. It **serves**: the installed runtime authors one canonical provisioned closure,
  reports exact plan subjects through `bindings --json`, and privately stages the identical
  bytes under `serve --weightless-endpoint`; Creator only carries those identities as the
  record owner's desired state. There is no flat Record/2 fallback. The fixture drives the whole
  product path — install, `start`, invoke, typed result, published PNG, and the failure
  terminal on the same path — with nothing to load, which is the only claim a machine with
  no card can make for itself. `no_servable_function` survives for its one remaining case:
  a release whose descriptor registers no entrypoint at all.

## Remote pod media client

Creator owns only the renter-side HTTPS client in `internal/media`. Tensorhub owns
`pod-supervisor`, the pod-side media server, its readiness handoff, hardening fences, and
the base worker image recipe that compiles it. No pod executable or server implementation
is built from this repository.

The two repositories ship independently, so `internal/mediawire` declares the service name,
revision, and health shape this client expects. `Client.Health` authenticates the request and
refuses a missing or different revision before any input or output byte moves. Creator live
tests use an independent minimal protocol peer; they do not retain Tensorhub server code as a
test helper.

## Verification

Creator is v2's test-suite exception; its suite and scripts drive real processes and protocol peers:

- `scripts/redarm.py` builds each hostile input for real, runs the real binary and
  observes the typed refusal (16 arms).
- `scripts/wheel-live.sh` drives the real `cozy pack` against the real H3 and SDXL
  endpoint trees: the same tree packed under four environments (umask, TZ, locale, cwd,
  HOME, rewritten mtimes and modes, a different absolute path) and inside two containers
  (glibc and musl, another uid, no `$HOME`) yields ONE `project_wheel_digest`; eight
  planted trees each refuse by their own name, observed red, and pack green once the
  planted file is removed; and each real wheel is `pip install`ed into a throwaway venv
  where `cozy-runtime describe --check` runs THROUGH THE INSTALLED WHEEL — site-packages
  as the project root, no source tree on `sys.path`.
- `scripts/hub-live.py` drives the real `cozy` against a REAL tensorhub: public reads,
  admin writes, every refusal arm (wrong token, absent model, closed port, a hub that
  accepts and never answers, a 200 that is not our document), and a cumulative secrecy
  check that the raw credential appears in no byte the session printed.
- `go test ./...` is the permitted Creator suite; `internal/live` starts the real
  system and observe it, with no mocks anywhere. `TestCanonicalDocuments` writes and reads
  worker-protocol's frozen corpus byte-for-byte and refuses every semantic twin by its own
  code; `TestNumberProfile` walks Runtime's 4,561-row ES6 float oracle, which is the
  cross-language hazard behind every canonical digest; `TestEndpointDescriptor` fences the
  grammar Creator consumes at install; `TestWorkerRefusals` and `TestDroppedOutcomeAck` are
  the interop proof against `internal/live/fakeworker`, a second independent Go implementation of
  the worker side that the orchestrator was not co-developed against; `TestLocalAPIDoor` is
  the loopback door matrix; `TestOutputRetention` is the plan/perform discipline on the one
  verb that removes a user's bytes; `TestProductPath` is install -> up -> invoke -> result ->
  crash on the weightless fixture. The remote-placement tests drive the real owner client
  against an independent media-protocol peer rather than importing Tensorhub's server.
  A test whose external peer is absent skips by name.
- `scripts/xfer-realhub.sh` drives the real `cozy` against a pinned current Tensorhub,
  its own Postgres container, four freshly bootstrapped storage prefixes, real R2, and
  a real TensorFS checkpoint. It proves typed endpoint/model create/show/search,
  incremental model publication, zero-byte committed replay, typed resolution and
  download, whole-checkpoint verification, warm zero-byte download, retired-command
  refusals, teardown, and exact R2-prefix cleanup—with no read shim or retired route.
- `scripts/clean-machine.sh` runs `scripts/accept.sh` inside a throwaway container with a
  fresh user, an empty home, no toolchain and no mount of this repository but `scripts/`:
  the corrupted-asset red arm, the checksum-verified install, the tag and commit the
  binary carries, the service-down refusals, `cozy up` building its records database on a
  machine that never had one, the weightless endpoint install and its whole serve —
  a READY worker with no weights to fill, a typed `completed` terminal, a published PNG,
  zero reserved VRAM and an empty construction digest, plus the typed failure terminal —
  `cozy down`'s proof of absence, and the verified upgrade.
- `scripts/fence.py` enforces fourteen families: forbidden deps, the canonical tensor
  carriers TensorFS parses, interactive prompts, exit-matrix parity, a manifest lint (gate
  declaration, no `--version` global, and a `Next:`/`Examples:` disclosure on every
  implemented row), the `render` fence (one stream — an error is data and leaves on
  stdout), the env-read fence (the CLI's one reader),
  no lifecycle sidecar, the secret fence (no
  credential-shaped flag takes an argv value; `Reveal()` only where the value becomes a
  header), the `cas` fence (nobody composes a store path), the `api` fence (no cookie, one
  loopback bind, and no CORS inside `internal/api` per #628), client-contract parity, the
  media-client contract expectation declared once in `internal/mediawire`, and
  the `tensor` fence (the tensorfs CLI has one caller). Doors are greppable:
  `//cozy:allow`, `//cozy:stdin-value`. The `cloud` family and the transfer-plane digest ban were deleted
  2026-08-28 as vocabulary rules with no failure behind them.
