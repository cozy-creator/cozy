# Cozy local client API — v1

This document describes the HTTP API implemented by the Cozy daemon. The routes in the
CORE module are the proposed common request-level API for
future Tensorhub and private-rental servers, but no cross-host parity is claimed until
those servers exist and pass shared conformance tests. The LOCAL module is Cozy-only.

`internal/api/routes.go` is the route registry. `scripts/fence.py` checks this document's
route method, path, scope, and order against that registry. Authentication, idempotency,
payloads, and behavior are documented below but are outside that row-level fence.

## 1. Transport and identity

| | |
|---|---|
| version | `cozy.client.v1` (`GET /v1/capabilities` → `core`) |
| encoding | JSON, UTF-8. SSE for streams. |
| auth | `Authorization: Bearer <token>`. Cozy never accepts a cookie. |
| idempotency | `Idempotency-Key` header on submit; required. |
| errors | one envelope, §6 |

Bearer-only prevents a cross-site request from carrying ambient authority into the local
daemon.

## 2. Core routes

| route | scope | auth | notes |
|---|---|---|---|
| `POST /v1/requests` | core | yes | submit; `Idempotency-Key` required; 202 fresh / 200 replay |
| `GET /v1/requests` | core | yes | listing, newest first; `?status=`, `?package=`, `?limit=` |
| `GET /v1/requests/{id}` | core | yes | the lifecycle document |
| `POST /v1/requests/{id}/cancel` | core | yes | requests cancellation; `?grace_ms=` |
| `GET /v1/requests/{id}/events` | core | yes | SSE, one request, terminal-stop; `?cursor=` |
| `GET /v1/media/{media_id}` | core | yes | bytes by opaque id; `HEAD`; one `Range` |

### Submit

```json
POST /v1/requests
Idempotency-Key: <caller's key>

{"package": "org/name", "function": "denoise", "input": {…},
 "outputs": ["image"]}
```

`input` is the package's own typed payload and is carried verbatim to the runtime.
`outputs` names result FIELD PATHS (`image`, `detail.thumb`) and defaults to the
entrypoint's declared set — binding by field path is what makes a two-output result
unswappable.

`install_id` is Cozy's local, opaque exact-install selector. The CLI resolves the
package's installed release and sends its id; the daemon verifies that the install
belongs to `package` and records it with the request. Other clients may omit it and use
the active package pointer.

The `model`, `lane`, and `adapter` fields are reserved but not resolved by Cozy yet.
Any non-empty value refuses as `501 override_unresolved`; it is never silently ignored.

`rental: true` permits remote placement; `rental_required: true` excludes local capacity.
The CLI's `--rental-only` uses both. CLI-authenticated callers may additionally specify
`requested_rental`, the existing rental ID resolved from `--rental=<name-or-id>`. This
immutable constraint participates in submission identity and survives child calls, retries
and restarts. It cannot be unpinned or replaced after loss. `worker` remains the scheduler's
current assignment. Default execution is local-only.

Creator reconciles every locally known rental with Tensorhub and sums the immutable Cozy
retail hourly rate locked into each rental. Every potentially billing manual or managed
rental counts until Tensorhub confirms release or absence. A new managed rental is allowed
only when its locked rate fits under `rentals.max_hourly_spend_usd`. Creator reuses the
cheapest idle rental first, otherwise buys the cheapest offered SKU. Every rental, manual or
Creator-managed, is released once nothing has been queued, running, or owed on it for
`rentals.idle_release_s`; a managed rental whose assigned requests are terminal, their output
bytes mirrored, and their outcome acknowledgements sent is released at once when its work was
a job. `cozy rental end` releases one now.

A generic rental becomes attachable when Tensorhub publishes its pinned worker location.
The paid request contains only the SKU, media-token hash, and Creator public key; package and
model choices remain local until Creator attaches. Creator reuses an idle rental whose observed
hardware fits the request, then sends the exact published release as a signed logical
`package_set`, or an exact local revision as `local_package_set`; the worker's observed-state
stream supplies the derived placement, dispatchable binding, and exact release/environment/config
identities. Runtime observes the actual worker environment and refuses protected platform-package
conflicts before offline installation. Tensorhub's immutable release detail supplies the verified
request/result PackageInterface.

For a callable with an explicit `assets` descriptor, repeated `--asset <file>`
flags append files in attachment order. `--asset 'alice=~/Pictures/alice.png'`
adds an optional, exact label; nonempty labels must be unique. Repeating the same
file keeps separate occurrences while reusing its content identity. Explicit
paths such as `~/Pictures/a=b.png` remain filenames. Labels and files do not
rewrite the prompt or infer model-specific roles. The package decides how to use
them. For example:

```sh
cozy run paul/minimax-h3/ref2va --await --rental-only \
  prompt='<Picture1> and <Picture2> walk through a garden' \
  --asset='alice=~/Pictures/alice.png' --asset='bob=~/Pictures/bob.png'
```

The descriptor names one request field containing `{asset, label?, fidelity?}` records.
Bindings use `<parameter>.<index>.asset`, with labels carried in the ordinary
request payload and memo identity. Missing declared collections default to `[]`;
the authored minimum/maximum count decides whether that is valid. Each media
kind's encoded byte limit applies per file; Runtime owns decoded byte limits.
Optional kind counts are compiled from the author's argument-level `AssetLimits`
annotation. The CLI, API admission and inherited child inputs use the same count
check over observed MIME metadata.
An exact named payload asset still uses `--asset <field-path>=<file>`, including
on jobs, and takes precedence over a same-spelled label. Named fields and the
declared collection can coexist.

`--asset-fidelity alice=high` or `--asset-fidelity 0=low` records an optional
per-occurrence hint: `auto` (default), `low`, `medium`, or `high`. Exact labels
take precedence over canonical nonnegative indexes. Duplicate selectors for the
same occurrence and unknown selectors refuse before file reads. Fidelity is part
of the payload and memo identity; model adapters interpret it. The shared media
loader preserves the input's resolution.

`local_assets` is the CLI-only local extension carrying those attachments.
Each row names the exact request-schema field path plus a source
path, digest, length, and detected media type. The daemon verifies those claims and
copies the bytes into its private content-addressed input store before recording the
request. Only an opaque digest reference enters `input`; only the field-path identity,
digest, length, media type, and ordered occurrence enter the invocation spec. A client
filesystem path is never portable authority. This field requires the OS-protected CLI
bearer. A future web UI must upload selected bytes through its own reviewed surface.

The private staged object remains only while some unsettled request references its
digest. The daemon serializes the filesystem-to-request-row ownership handoff with the
same guard used by terminal cleanup, so deduplication cannot turn cleanup into a race.
Per-attempt input directories are removed after the outcome is mirrored, committed, and
acknowledged; locally mirrored output bytes remain addressable by their media ids. A
failed pod deletion remains a durable attempt-row obligation and is retried on subsequent
worker reports; it is never converted into a successful cleanup claim.

Local worker grants currently refuse on Windows as
`local_file_grant_unsupported` (structural). Remote pod execution remains supported.
This is an explicit boundary until cozy-runtime and Cozy share one Windows file-URL
encoder/authorizer; Cozy does not emit malformed `file://C:\\…` capabilities.

A pinned request crosses a REAL byte boundary, and the crossing is the pod's own media
server: the payload and every input asset are uploaded as separate
objects before the attempt is dispatched, the worker reads them off the pod's disk, and
the outputs are fetched back and verified against the terminal's manifest before anything
is acked. A rental whose media
plane does not answer is `media_unreachable` — this host never falls back to granting a
path on its own disk, because the pod cannot reach one.

**The idempotency key names one request forever.** The host records the key beside a
digest of the whole submission — package, function, input, asset identities, outputs —
so the same key with a different FUNCTION conflicts as loudly as one with different input.
Answers:

| | |
|---|---|
| `202` | this call started the work; `idempotent_replay: false` |
| `200` | the key was already recorded; the SAME request, nothing started |
| `409 idempotency_conflict` | the key names a request with a different body |

```json
{"request_id": "req-…", "status": "queued", "attempt": 1,
 "status_url": "/v1/requests/req-…", "response_url": "/v1/requests/req-…",
 "cancel_url": "/v1/requests/req-…/cancel", "events_url": "/v1/requests/req-…/events",
 "idempotent_replay": false}
```

`status`: `queued` · `in_progress` · `completed` · `failed` · `canceled`.

Status and list responses expose `overall_fraction` when whole-run progress is
known. Completed runs report `1`; failed and canceled runs retain the last
measured fraction of their current attempt across daemon restarts. The field is
omitted when no overall measurement was recorded. Stage coordinates and ETA
remain live-only, and stage progress has its own denominator.

### Status

```json
{"request_id": "req-…", "status": "completed", "package": "org/name",
 "function": "denoise", "attempt": 1, "attempts": 1, "overall_fraction": 1,
 "metrics": {"runtime_ms": 401, "handler_ms": 398, "peak_vram_bytes": 6…},
 "result": {…the package's typed result…},
 "outputs": [{"output_id": "image", "media_id": "med-…", "url": "/v1/media/med-…",
              "mime_type": "image/png", "length": 11134, "digest": "sha256:…"}],
 "triage": {"attempt_key": "att-…", "subject_id": "trb-…",
            "url": "/v1/local/attempts/att-…/triage", "length": 4994, "kept": true}}
```

A failed request carries `error_type` (the terminal's cause code) and `error` (the
worker's safe message) instead of `result`.

### Cancel

`202 {"status": "cancel_requested"}`. A cancel is a REQUEST for cancellation, never a
verdict: the attempt's own journaled terminal settles it, and a non-cooperative handler
may still succeed. A cancel arriving after the terminal is an idempotent `200` with the
lifecycle document.

## 3. Events

One envelope for every event, durable or live:

```json
{"type": "request.accepted", "request_id": "req-…", "attempt": 1,
 "event_id": 42, "at": "2026-08-25T…", "payload": {…}}
```

**Durable** events are rows in the host's authority. They are monotonic, totally ordered
across all requests, and replayable: reconnect with `?cursor=<event_id>` and everything
after it arrives in order, across a host restart. `id:` carries the cursor on the wire.

| type | payload |
|---|---|
| `request.submitted` | `package`, `function`, `body_digest`, `plan_id`, `outputs` |
| `request.queued` | `reason` (verbatim diagnostic), `wait` (stable cause: `worker_start` · `worker_warming` · `slot_busy` · `queue_ahead` · `rental` · `model_transfer`), `waiting_on` (machine word, when known), `package`, `position` |
| `request.routed` | the routing decision (residency-aware-routing.md §3.1): `candidates[]` (`worker`, `lane`, `held`, `cost`, `score`, `resident[]`, `manifests_missing`, `rental`), `pick` (`worker`, `lane`, `rental`, `pinned`) |
| `request.placement` | the placement decision (placement-economics.md, attempt 0): no worker held the placement, so the fleet placed it on an attached rental or a bought pod — or waits on a fitting rental whose worker has not attached (`rental` absent, one candidate `attaching`, re-decided on the fleet's next observation) — `tier`, `config_digest`, `ladder[]` (the owner's fit map, absent when none is bound), `override` (an explicit `model.<param>=…/lane`, which is not a rung), `throughput[]` (every row used), `candidates[]` (`rental`, `machine`, `sku`, `rung` (absent under an override), `lane`, `fit`: `components …` · `rung_asserted` · `lane_bytes …` · `derive_only` (a JOB's models are never resident), `ahead`, `rate_usd_micros_per_hour`, `measured`, `time_s`, `cost_usd_micros`, `score`, `verdict`: `chosen` · `slower` · `dearer` · `unmeasured` · `attaching` · `no_rung` · `no_stock` · `excluded:<reason>`), `rental`, `bought`, `line` |
| `request.parked` | the drain skipped a waiting request: the `request.queued` fields plus `lanes`, `overtaken`, `budget`, `claims` |
| `request.dispatch_aborted` | pre-offer preparation failed; `cause`, `error`; no worker saw this ordinal |
| `request.dispatched` | `instance_id`, `invocation_digest` |
| `request.accepted` | `plan_digest`, `construction_digest`, `plan` |
| `request.attempt_failed` | one ATTEMPT ended and the request did NOT — `status`, `cause`, `requeuing: true`, optional last measured `overall_fraction` |
| `request.requeued` | `cause`, `requeues`, `budget` |
| `request.completed` | `status`, `cause`, `outputs[]` (media ids), `triage_subject`, optional last measured `overall_fraction` |
| `request.failed` | the above plus `error_type`, `error` |
| `request.canceled` | the above |

**Live** events (`event_id: 0`, `payload.live: true`) are the lossy lane: never durable,
never replayed from a cursor. The LATEST tick is replayed immediately on connect so a
mid-run subscriber renders current state. `request.progress` carries `value` (a fraction)
and `seq`; when Runtime reports measured production coordinates, `value` is an object with
`stage`, `stage_fraction`, `overall_fraction`, `position`, `total`, and `step_ms`
(with unmeasured coordinates omitted). `request.log`, `request.stage` and
`request.metric` carry the worker's own.

Three rules a client may rely on:

1. **Terminal-stop.** On the per-request stream a terminal event is ABSORBING: the host
   closes and the client must not reconnect. The terminal set is exactly
   `request.completed` · `request.failed` · `request.canceled`. **An attempt ending is
   not a request ending**: an attempt the host will requeue emits `request.attempt_failed`
   — deliberately outside that set — because a client that stopped there would report a
   failure for a request that goes on to succeed.
2. **A close is not a verdict.** A stream that ends without a terminal means reconnect
   from the cursor. It never means the request failed.
3. **A flood cannot delay a terminal.** Live frames are shed at a bounded buffer; durable
   rows are drained to exhaustion before a terminal is written.

**Media is announced, never pushed.** A terminal event carries `media_id`s. Bytes are
fetched from the media route. No frame on any stream carries base64 image data, and the
event envelope has no field that could.

## 4. Media

`GET /v1/media/{media_id}` and nothing else. The id is minted inside the terminal
transaction, so an output no terminal published has no id at all.

**There is no path form.** Not a rejected one — none. The route's only input is an opaque
id, which is what makes the ComfyUI `/view` traversal class structurally absent rather
than defended against.

Responses carry `X-Content-Type-Options: nosniff`, `X-Cozy-Digest` (the digest the
manifest declared, so a client verifies rather than trusts), and
`Content-Disposition` with a filename derived from the id. An explicit MIME allowlist is
served inline; anything else becomes `application/octet-stream; attachment`.
`image/svg+xml` is never served inline — an SVG is an executable document. `HEAD` gives
the size. `Range` honours exactly ONE range; a multi-range request is `416`.

## 5. Capabilities

```json
{"core": "cozy.client.v1", "host": "cozy.local", "tokens": ["api.…", …]}
```

Feature presence is a TOKEN. A client never infers a feature from a version string.

## 6. The error envelope

```json
{"error": {"code": "idempotency_conflict", "message": "…", "remedy": "…",
           "request_id": "req-…"}}
```

`code` is the detailed name from `docs/exit-matrix.md`. The CLI keeps that name in its
structured error document while projecting process exits to 0/1/2. Every refusal renders
through one envelope, including an unknown route.

| matrix | HTTP |
|---|---|
| usage · validation | 400 |
| credential | 401 |
| not_found · offline_miss | 404 |
| structural | 422 |
| conflict | 409 |
| capacity | 507 |
| unavailable | 503 |
| deadline | 504 |
| failed · canceled | **200** — a terminal is an answer, not a transport failure |

## 7. Local extension module

Cozy-only control routes use `/v1/local/`. Uploads and embedded web assets use their product
URLs but remain local-scope rows in the same guarded route table.

| route | scope | auth | notes |
|---|---|---|---|
| `GET /v1/local/attempts/{attempt_key}/triage` | local | yes | one attempt's kept triage bundle from its own row; 404 when none was kept |
| `POST /v1/local/rentals/{rental_id}/claim` | local | yes | attach the daemon to one generic empty private worker and directly claim WorkerControl |
| `DELETE /v1/local/rentals/{rental_id}/claim` | local | yes | detach that worker and wait for its control loop before rental credentials are removed |
| `POST /v1/local/rentals/{rental_id}/prune` | local | yes | prune unused operation cache roots on the claimed Host; report `removed_entries`, `reclaimed_bytes`, and whether native GC is still `store_busy` |
| `POST /v1/local/cache/prune` | local | yes | prune unused operation cache roots in this machine's Runtime workspace; report `removed_entries`, `reclaimed_bytes`, and `store_busy` |
| `POST /v1/local/daemon/unload` | local | yes | stop definitely-idle local serving workers; never touch active work, jobs, rentals, or installed bytes |
| `POST /v1/local/daemon/down` | local | yes | safe down fence; `{all:true}` requests local cancellation and returns paid obligations that must be confirmed absent before retrying |
| `POST /v1/local/jobs` | local | yes | submit one bounded job (CLI-authenticated `local_assets` use the same immutable staging and input grants as requests); `Idempotency-Key`; 202 with the handle and its publication repo |
| `GET /v1/local/jobs/{id}` | local | yes | one job: state, queue position, retry budget, publication, checkpoints, bill where a rate exists; `model_sources` names every selected source file that has NOT verified, with the worker's own `safe_code`/`safe_detail` |
| `POST /v1/local/jobs/{id}/pause` | local | yes | fence active attempts while preserving the same request and retained work |
| `POST /v1/local/jobs/{id}/resume` | local | yes | queue the same paused request with its captured execution inputs |
| `POST /v1/local/jobs/{id}/retry-publication` | local | yes | retry failed output publication using the retained successful attempt and receipts; never rerun the producer |
| `POST /v1/local/jobs/{id}/cancel` | local | yes | request cancellation; a queued job leaves the queue, a running one gets its terminal |
| `GET /{$}` | local | no | embedded localhost web UI entrypoint |
| `GET /app.css` | local | no | embedded localhost web UI stylesheet |
| `GET /app.js` | local | no | embedded localhost web UI script |

Uploads are capped at 64 MiB, hashed while streaming, atomically published under their sha256,
deduplicated, and served only by opaque `upl-<sha256>` id. The future browser flow will bind those
ids into invocation assets; it will never send a host filesystem path.

### The job family is local

A job is an ATTEMPT CLASS in the same Cozy daemon, not a second scheduler: the same
orchestrator places and dispatches it, and the same record owner gives it an ordinal,
settles its terminal transaction, and streams it over the same durable event route
(`GET /v1/requests/{id}/events` — there is no second event authority anywhere, and
`cozy run` follows it).

The family is local because a job's typed input trees are directories the caller already
owns (`{"trees":["<ref>=<dir>"]}`). Client filesystem paths are not part of the proposed
common core.

Two answers a job carries that a request does not:

- **`publication`** — the durable publication the job's landed writes became:
  `{repo, root, status, entries, bytes, committed_at}` under the scratch repo
  `<org>/_job-<job-id>`. An attempt writes into a per-attempt STAGE and the record owner
  promotes it into the addressable root after the terminal is verified; the row is written
  INSIDE the terminal transaction, so a publication a terminal did not commit does not
  exist. `status` is the terminal's verdict STAMPED as metadata; a failed run's landed
  writes still land. Model-producing jobs instead expose `checkpoints` after their
  typed receipts and destination publication are verified.
- **`bill`** — ABSENT unless the host was configured with an explicit local rate. There is
  no `$0.00`: a fabricated zero is a claim about money nobody measured.

A CLI-authenticated job may attach output publication independently of input acquisition:
`"model_transfer":{"kind":"model-upload","destination":"org/model","outputs":[{"name":"model"}]}`.
Its `models` remain the exact ModelRefs validated against the job's declared parameters.
The source, source_selection, source_license, source_files, input_lane, source_profiles
and local_only fields are absent in this form; partial acquisition declarations refuse.
The existing receipt, upload, finalization and cancellation paths handle its outputs.

Remote output-only publication requires `worker` to pin an existing, suitably prepared
rental. Exact manifest lengths are metadata sizes and cannot size the input closure or
the producer's output working set, so this form does not purchase a new rental. A normal
source-transfer intent still declares its source and selection, profile bindings and
inspected headers, and uses the existing source preparation and checkpoint-custody path.

A successful model producer remains `finalizing` while output publication is incomplete,
including after an upload failure. Its receipts, worker custody and rental remain held;
no outcome acknowledgement or automatic rental release occurs before publication succeeds
or explicit cancellation finishes cleanup. `cozy run retry-publication <job>` (or the
retry-publication route with an optional `actor`) retries the failed publication with the
same request, attempt, receipts and per-object transfer identities. It does not rerun the
producer. `cozy run cancel <job>` abandons unfinished destination holds and then settles
cancellation; already completed checkpoints remain published. Both decisions survive a
daemon restart. A failed request from an older build is not silently resurrected.

`queue_position` and `queue_depth` are one atomic orchestrator scheduling snapshot;
`requeues`/`retry_budget` are the record owner's durable-attempt facts. Several jobs submitted at once queue against one
worker and drain in submission order, while the retry projection over neutral outcomes
spends a durable budget that the settlement names when it is exhausted.

### Private transactions and child calls

`retain_work:true` makes an ordinary private job retain its exact inputs, implementation
and native intermediate work across failed or paused attempts. `pause` durably fences
admission and reaches `paused` only after its writers and child writers stop. `resume`
continues the same captured request. A deterministic `blocked` failure requires a
corrected new request with `retry_of`, which accepts the prior public run number or ID.
The new request selects the same physical store; run lineage scopes artifact grants and
history, while completed computation is cached by the Host independently of run IDs.
Fresh unrelated runs on that workspace can reuse compatible completed operations. The old
payload and attempt outcomes stay immutable. Cancellation releases only the caller's
ownership. Explicit rental release also abandons its dependent transactions.

Private source/weights checkpoints stay on that worker; they are not implicitly uploaded
to Tensorhub. `--publish-to` remains an explicit final-output publication choice. A job
document's `retaining` boolean reports continuing resource custody, including successful
private model outputs. Such a completed job can be explicitly canceled to release custody
without rewriting its successful execution outcome.

Child calls use the current parent attempt's authenticated worker stream. The owner joins
the requested interface/module/export to an immutable dependency captured in that parent's
install, then creates an ordinary job. The child status includes `parent_request_id`,
`parent_call_index`, and, for an acquired previous result, `reused_from`. A reused child has
no invented execution attempt. Scalar results can complete immediately; model results stay
`finalizing` until native independent retention is confirmed.

Cross-request reuse requires an explicit `memoize:true` callable declaration, identical
target implementation/dependencies, typed inputs/model references, and the current
privileged numerical environment measurement. An unavailable measurement disables reuse
while allowing execution. Exact same-parent call replay continues the original recorded
child. Known egress/secret capabilities and non-reusable child effects cannot enter a
reusable operation. This is an author contract, not a proof of arbitrary Python purity.

The Runtime workspace records cache results from verified successful terminals before outcome ACKs.
Lookup uses the computation key and an already-recorded consumer request; a HIT acquires
independent native holds before returning. Creator persists a pending lookup obligation
before the RPC, so pause/cancel and owner restart reconcile even a lost response. Once the
request owns a result, resuming it does not depend on the cache entry still existing.
Local and rented execution address this same journal through RuntimePreparation or the
authenticated Host. Request history never becomes a second cache authority. Declined
optional cache admission leaves the successful result unchanged; an unresolved journal
RPC remains an obligation until it can be reconciled.

`cozy cache prune` or `cozy rental prune <rental>` removes unused cache roots and attempts
native garbage collection in the selected workspace. Local pruning can start an empty
control Runtime when no package worker remains. Request-owned results and unresolved lookup recipients are
preserved. A busy Store reports deferred byte collection; repeating prune can collect
those bytes even when no further cache entries are removed. Cache roots do not count as
unfinished rental work, and pruning neither ends the rental nor deletes run history.

Model artifacts are closed typed references with `producer_request_id`, `output_slot`,
`manifest:{digest,length}`, and `tensorfs_receipt_digest`. The last field hashes the native
TensorFS receipt, not its protocol envelope. Creator verifies provenance and scope, then
binds the same manifest through the existing ModelRef input contract. Owner-selected native
retention IDs remain outside package payloads. Only schema-declared model artifact result
fields acquire custody; matching JSON field spellings grant nothing.

Reuse granularity is the exact child package closure and entrypoint. Changing a helper
within that closure invalidates its reusable operations; the owner never guesses a
function-only dependency hash. Native source acquisition remains independently keyed by
the selected source and profile, so a producer implementation edit can reuse that work.

## 8. Local host security posture

A loopback bind is not a boundary — Ollama's DNS-rebinding CVE is the standing proof.
This host's chain, in order:

1. **Loopback-only bind**, IPv4 and IPv6, with no flag that widens it. The LAN door is
   deferred behind TLS, durable auth and its own threat review.
2. **Host allowlist** — three exact spellings; a rebinding request carries the attacker's
   hostname and is refused before routing.
3. **Origin check** on every mutation and every stream open.
4. **Bearer auth** on every state/byte API route; only embedded static assets are public. No cookie
   is read anywhere.
5. **No CORS headers at all**, and a strict CSP on every response.

One credential is minted per daemon launch and handed to the CLI through a 0600 file.
It never appears in argv, inherited environment, a log line, or a rendered error. A future
web UI owns its own authentication design rather than inheriting this filesystem authority.

## 9. The status-to-exit projection

Every refusal envelope carries the refusal's own name; the exit-matrix code comes from
the HTTP status. `api.CodeOf` is the inverse of `statusOf` and lives beside it so the two
cannot drift:

| status | matrix code |
|---|---|
| 200 · 201 · 202 | 0 ok |
| 400 | 3 validation |
| 401 · 403 | 5 credential |
| 404 · 410 | 4 not_found |
| 409 | 13 conflict |
| 413 · 422 | 6 structural |
| 428 | 7 confirm |
| 501 | 2 usage (`not_implemented`) |
| 503 | 9 unavailable |
| 504 · 408 | 10 deadline |
| 507 | 14 capacity |

**A FAILED or CANCELED terminal never comes back through this table.** It is answered
`200` on purpose—a terminal is an answer, not a transport failure. The CLI preserves the
terminal's detailed symbolic status and returns shell exit 1 for any non-successful outcome.


Private invocable dependencies preserve standard Python extra selection. For example,
a dependency declared as `cozy-eval[jobs]` activates that library's declared
job requirements in the immutable child capture. The original editable pyproject
and lock remain unchanged. The captured pyproject, resolved uv lock, wheel and execution
environment identify the selected closure; equivalent normalized extra sets capture
identically. Unknown or ambiguous local extra groups refuse. This uses the existing
private installation and invocation flow, without a wrapper package or another command.

Installed App wheels use the selected environment's exact dependency graph, including
active extras. A callee's identity excludes its caller, unrelated dependencies and the
wheel's local-file versus registry origin. Creator retains the original wheel and its
digest, then uses the same private requirement sealing as source packages: only METADATA
and RECORD change in the executable derivative. The derivative and selected closure bind
the caller interface and computation key. The worker does not execute a byte-identical
copy of the downloaded archive, and mismatched protected base versions must refuse.
Wheel discovery reads metadata and source without importing the package; ordinary
non-App dependencies and Runtime's native facade keep their existing behavior.
