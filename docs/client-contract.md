# The shared client contract — core v1

**One contract, three hosts.** `cozy-creator` local (this host, the reference
implementation), Tensorhub cloud (th-021), and the private-deployment pod profile
(cl-014, `private-deployments.md` §1). The CORE below is byte-identical on all three.
Extension modules are not: this host adds a LOCAL module, Tensorhub adds catalog,
billing and org routes, the pod adds owner-token and output-store routes — and none of
them ever pretends to be the core.

Not invented here. This is the `/v1/requests` submit / status / cancel + SSE surface
cozy.art already speaks against Tensorhub, taken verbatim. Every divergence from it is a
recorded decision, and the ones this host made are in §7.

`internal/api/routes.go` is the machine-readable form of the tables below and
`scripts/fence.py`'s `contract` family checks them against each other, row for row.
A route in one and not the other is CI-red.

## 1. Transport and identity

| | |
|---|---|
| version | `cozy.client.v1` (`GET /v1/capabilities` → `core`) |
| encoding | JSON, UTF-8. SSE for streams. |
| auth | `Authorization: Bearer <token>`. **Never a cookie**, on any host. |
| idempotency | `Idempotency-Key` header on submit; required. |
| errors | one envelope, §6 |

Bearer-only is a contract property, not a local choice: it is what makes CORS simple on
the pod host (a browser talks to the pod directly) and what keeps a cross-site request
from carrying ambient authority on the local host.

## 2. Core routes

| route | scope | auth | notes |
|---|---|---|---|
| `POST /v1/requests` | core | yes | submit; `Idempotency-Key` required; 202 fresh / 200 replay |
| `GET /v1/requests` | core | yes | listing, newest first; `?status=`, `?limit=` |
| `GET /v1/requests/{id}` | core | yes | the lifecycle document |
| `POST /v1/requests/{id}/cancel` | core | yes | requests cancellation; `?grace_ms=` |
| `GET /v1/requests/{id}/events` | core | yes | SSE, one request, terminal-stop; `?cursor=` |
| `GET /v1/events` | core | yes | SSE, multiplexed, never terminal; `?cursor=`, `?from=now` |
| `GET /v1/media/{media_id}` | core | yes | bytes by opaque id; `HEAD`; one `Range` |
| `GET /v1/capabilities` | core | yes | feature tokens |

### Submit

```json
POST /v1/requests
Idempotency-Key: <caller's key>

{"endpoint": "org/name", "function": "denoise", "input": {…},
 "outputs": ["image"], "model": "org/repo@release", "lane": "auto"}
```

`input` is the endpoint's own typed payload and is carried verbatim to the runtime.
`outputs` names result FIELD PATHS (`image`, `detail.thumb`) and defaults to the
entrypoint's declared set — binding by field path is what makes a two-output result
unswappable.

`worker` is a **LOCAL ADDITION** (cl-015), not part of the shared core: it pins the
request to a rented pod this host has attached, by rental id. The cloud host places work
itself and has no rental for a client to name, so it does not carry the field. An id this
host does not hold is `404` **before** a request row exists — a pin that cannot be
resolved now cannot be resolved on a later attempt either. The pin is deliberately NOT in
the idempotency digest: it says WHERE the same work runs, not what the work is.

`local_assets` is the CLI-only local extension for `--asset
<field-path>=<file>`. Each row names the exact request-schema field path plus a source
path, digest, length, and detected media type. The service verifies those claims and
copies the bytes into its private content-addressed input store before recording the
request. Only an opaque digest reference enters `input`; only the field-path identity,
digest, length, media type, and ordered occurrence enter the invocation spec. A network
host uses an upload/asset-id surface instead — a client filesystem path is never portable
authority. This field requires the OS-protected CLI bearer; the browser bearer is refused
before any path is read, and this local host does not yet expose a browser asset-upload
route.

The private staged object remains only while some unsettled request references its
digest. The service serializes the filesystem-to-request-row ownership handoff with the
same guard used by terminal cleanup, so deduplication cannot turn cleanup into a race.
Per-attempt input directories are removed after the outcome is mirrored, committed, and
acknowledged; locally mirrored output bytes remain addressable by their media ids. A
failed pod deletion remains a durable attempt-row obligation and is retried on subsequent
worker reports; it is never converted into a successful cleanup claim.

Local worker grants currently refuse on Windows as
`local_file_grant_unsupported` (structural). Remote pod execution remains supported.
This is an explicit boundary until cozy-runtime and Creator share one Windows file-URL
encoder/authorizer; Creator does not emit malformed `file://C:\\…` capabilities.

A pinned request crosses a REAL byte boundary, and the crossing is the pod's own media
server (cl-014, ruled #506b): the payload and every input asset are uploaded as separate
objects before the attempt is dispatched, the worker reads them off the pod's disk, and
the outputs are fetched back and verified against the terminal's manifest before anything
is acked. A rental whose media
plane does not answer is `media_unreachable` — this host never falls back to granting a
path on its own disk, because the pod cannot reach one.

**The idempotency key names one request forever.** The host records the key beside a
digest of the whole submission — endpoint, function, input, asset identities, outputs —
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

### Status

```json
{"request_id": "req-…", "status": "completed", "endpoint": "org/name",
 "function": "denoise", "attempt": 1, "attempts": 1,
 "metrics": {"runtime_ms": 401, "handler_ms": 398, "peak_vram_bytes": 6…},
 "result": {…the endpoint's typed result…},
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
| `request.submitted` | `endpoint`, `function`, `body_digest`, `plan_id`, `outputs` |
| `request.queued` | `reason` |
| `request.dispatch_aborted` | pre-offer preparation failed; `cause`, `error`; no worker saw this ordinal |
| `request.dispatched` | `instance_id`, `exec_spec_digest` |
| `request.accepted` | `plan_digest`, `construction_digest`, `plan` |
| `request.attempt_failed` | one ATTEMPT ended and the request did NOT — `status`, `cause`, `requeuing: true` |
| `request.requeued` | `cause`, `requeues`, `budget` |
| `request.completed` | `status`, `cause`, `outputs[]` (media ids), `triage_subject` |
| `request.failed` | the above plus `error_type`, `error` |
| `request.canceled` | the above |

**Live** events (`event_id: 0`, `payload.live: true`) are the lossy lane: never durable,
never replayed from a cursor. The LATEST tick is replayed immediately on connect so a
mid-run subscriber renders current state. `request.progress` carries `value` (a fraction)
and `seq`; `request.log`, `request.stage` and `request.metric` carry the worker's own.

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
{"core": "cozy.client.v1", "host": "cozy-creator.local", "tokens": ["api.…", …]}
```

Feature presence is a TOKEN. A client never infers a feature from a version string.

## 6. The error envelope

```json
{"error": {"code": "idempotency_conflict", "message": "…", "remedy": "…",
           "request_id": "req-…"}}
```

`code` is the name from the shared exit matrix (`docs/exit-matrix.md`), which is also
what the CLI exits with — one vocabulary from HTTP status to shell exit code. Every
refusal renders through it, including an unknown route: a client that parses one envelope
parses every answer.

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

Mounted under `/v1/local/` so the boundary is visible in the URL. The cloud host serves
none of these.

| route | scope | auth | notes |
|---|---|---|---|
| `GET /v1/local/endpoints` | local | yes | installed endpoints and their functions |
| `GET /v1/local/workers` | local | yes | live workers: protocol identities, devices, worker phase, the placement's two axes, the admission fence |
| `POST /v1/local/workers` | local | yes | ensure one local endpoint resident or claim one exact rental without invoking a model; exactly one of `endpoint` or `rental` |
| `DELETE /v1/local/workers/{instance_id}` | local | yes | drain and stop the process group |
| `GET /v1/local/doctor` | local | yes | host facts, bound families, counts |
| `POST /v1/local/service/shutdown` | local | yes | ask the service to drain every worker and exit (`cozy down`'s cooperative tier); exit is proved by the service lock, never this reply |
| `GET /v1/local/attempts/{attempt_key}/triage` | local | yes | the retained WorkerTriageBundle |
| `POST /v1/local/jobs` | local | yes | submit one bounded job; `Idempotency-Key`; 202 with the handle and its publication repo |
| `GET /v1/local/jobs` | local | yes | jobs newest-first with per-state counts; `?status=`, `?endpoint=` |
| `GET /v1/local/jobs/{id}` | local | yes | one job: state, queue position, retry budget, publication, checkpoints, bill where a rate exists |
| `POST /v1/local/jobs/{id}/cancel` | local | yes | request cancellation; a queued job leaves the queue, a running one gets its terminal |
| `POST /v1/local/workflows` | local | yes | submit one canonical ordered workflow; `Idempotency-Key`; 202 fresh or 200 replay |
| `GET /v1/local/workflows/{id}` | local | yes | one workflow with ordinary child request states and accepted outputs projected from the records authority |
| `GET /v1/local/workflows/{id}/receipt` | local | yes | canonical plan, materialized submissions, and workflow-owned exact rental controls, independent of live dial authority |
| `POST /v1/local/workflows/{id}/cancel` | local | yes | persist cancellation, cancel the active child, and prevent every later child |
| `POST /v1/local/video-compositions` | local | CLI only | compose exact source bytes or one retained creative-plan digest into the existing workflow submission shape; local base paths are resolution-only and never returned or stored |
| `GET /healthz` | local | no | liveness ONLY; says nothing about any request |
| `GET /{$}` | local | no | the embedded stub page |
| `GET /app.js` | local | no | the stub's script — a FILE, so no inline-script CSP |
| `GET /app.css` | local | no | the stub's style, for the same reason |

Triage is served by the OPAQUE attempt key and verified on read against the digest its
terminal declared. A bundle whose bytes moved is `409 bundle_corrupt`, never a plausible
story.

### The job family is LOCAL, deliberately (cl-004)

A job is an ATTEMPT CLASS in the same local service, not a second scheduler: the same
orchestrator places and dispatches it, and the same record owner gives it an ordinal,
settles its terminal transaction, and streams it over the same durable event route
(`GET /v1/requests/{id}/events` — there is no second event authority anywhere, and
`cozy job follow` is that route's client).

The FAMILY is nonetheless local rather than core, for two reasons that are about honesty:
the hub's job plane is th-008's, so a core route only one of the three hosts served would
make this document a description of this host; and a job's typed input TREES are
directories the caller already owns (`{"trees":["<ref>=<dir>"]}`), which is exactly the
parameter shape the core must never take. A cloud host materializes trees from digests.

Two answers a job carries that a request does not:

- **`publication`** — the durable publication the job's landed writes became:
  `{repo, root, status, entries, bytes, committed_at}` under the scratch repo
  `<org>/_job-<job-id>`. An attempt writes into a per-attempt STAGE and the record owner
  promotes it into the addressable root after the terminal is verified; the row is written
  INSIDE the terminal transaction, so a publication a terminal did not commit does not
  exist. `status` is the terminal's verdict STAMPED as metadata; a failed run's landed
  writes still land. Published CHECKPOINTS are absent by design: the publication
  transaction for canonical bytes is the runtime's, and this host will root and project
  ONE typed receipt from that border when it exists.
- **`bill`** — ABSENT unless the host was configured with an explicit local rate. There is
  no `$0.00`: a fabricated zero is a claim about money nobody measured.

`queue_position` is the orchestrator's scheduling fact; `requeues`/`retry_budget` are the
record owner's durable-attempt facts. Several jobs submitted at once queue against one
worker and drain in submission order, while the retry projection over neutral outcomes
spends a durable budget that the settlement names when it is exhausted.

### The workflow family is LOCAL and ordered (cl-018)

A workflow is one immutable maximum-16 ordered plan over ordinary serving requests. Only backward
output bindings exist, so a cycle is unrepresentable. Creator persists exact materialization before
submitting each child, derives its ordinary idempotency key from workflow/ordinal/release/action/
materialized identity, and mints at most one current child. Request attempts, retries, terminals,
outputs, media and cancellation remain the existing request authority; the workflow status document
projects those rows and copies none of their lifecycle state.

There is no workflow SSE route, list route, retry verb, cursor, resume verb, endpoint callback,
YAML parser or timeline object. Re-submitting the same idempotency key is recovery. `workflow
follow` samples the structured status document and has no elapsed-time verdict; worker liveness and
measured no-progress facts remain the stall authority. Cancellation persists before it touches the
active ordinary child.

`POST /v1/local/workflows` carries `{plan, assets?, targets?}`. `assets` contains staged
digest/length/type resolutions and therefore requires the OS-protected CLI credential; no path is
accepted. `targets` maps a one-based step to an attached rental id. Targets and local install ids
do not enter creative or model-execution meaning, but rental targets do enter the submission
identity so one idempotency key cannot silently move a workflow to another paid machine. The frozen workflow status
set is `running | canceling | succeeded | failed | canceled`; ordinary child rows are projected as
`queued | in_progress | completed | failed | canceled` plus workflow-only `pending | prepared`.
Structured status also projects the existing materialized-submission digest, child key, exact
resolved output bindings, materialized asset identities, rental id, and media refs. `workflow
download` obtains a workflow-owned receipt through the local API, independent of the current
rental table. It writes the canonical workflow and creative plans, each exact materialized
submission, child lifecycle/event/triage receipts, one content-addressed copy of Tensorhub's exact
control snapshot, actual ClaimAck GPU identity, and digest-verified media into a sibling staging
tree. Every path in a receipt is bundle-relative; the destination appears only by one final rename.

### The video composer is a LOCAL authoring boundary (cl-024)

`POST /v1/local/video-compositions` is the only YAML-reading route. It requires the OS-protected
CLI credential because a fresh composition reads caller-owned local paths. Its strict request names
exactly one of bounded source bytes or a retained `creative_plan_digest`, plus exactly one local H3
endpoint ref or attached H3 rental id. A rental already fixes its endpoint, so the API refuses a
second endpoint truth beside it. The fixed CPU assembler is a Creator-owned
product dependency, not a source or CLI selection. The rental is a Tensorhub-selected
result handle, never provider or placement policy. The absolute source directory resolves
relative asset spellings but is never stored, hashed, echoed, or added to a receipt.

The caller retains the editable YAML; Creator retains its exact-byte digest rather than a hidden
second copy. The response separates that source identity, path-free/deployment-free creative identity, and
the existing deployment-resolved `cozy.workflow.Plan/1`. Composition starts no workflow. `cozy video
submit` passes that plan and its staged content identities to `POST /v1/local/workflows`; workflow
status, cancellation, recovery, attempts, outputs, and events remain cl-018's one authority. There
is no video-specific status route, retry policy, cursor, provider selector, endpoint callback, or second
lifecycle table.

Composition rows are durable project records in v1 and deliberately retain their staged content
identities after any workflow settles. There is no implicit expiry or GC guess about whether
creative work is disposable. A future explicit forget verb may release a composition; until then,
retention is permanent and visible. The entire local root is mode 0700, protecting prompts in the
records database and SQLite's lazily-created WAL/SHM files from other OS users.

Only `cozy video compose` and `cozy video submit` exist. A formatter would rewrite the very source
bytes whose identity is being preserved, while an honest validator must resolve schemas and stage
content and is therefore composition with its result discarded. The source parser lives only in
Creator's video package; endpoints, Runtime, Tensorhub, worker protocol, and workflow records never
parse YAML or acquire shot semantics.

`cozy video submit` requires an explicit idempotency key. A client that loses the POST response can
therefore retry the same key and recover the same cl-018 workflow rather than minting a duplicate.

## 8. Local host security posture

A loopback bind is not a boundary — Ollama's DNS-rebinding CVE is the standing proof.
This host's chain, in order:

1. **Loopback-only bind**, IPv4 and IPv6, with no flag that widens it. The LAN door is
   deferred behind TLS, durable auth and its own threat review.
2. **Host allowlist** — three exact spellings; a rebinding request carries the attacker's
   hostname and is refused before routing.
3. **Origin check** on every mutation and every stream open.
4. **Bearer auth**, no cookie read anywhere.
5. **No CORS headers at all**, and a strict CSP on every response.

Two per-launch credentials, both dying with the process: the BROWSER token rides
`--open`'s URL FRAGMENT (never the query string — a fragment reaches neither the server
nor an access log nor a `Referer`), and the CLI credential is handed over through a 0600
file. Neither ever appears in argv, a log line, or a rendered error.

## 9. Divergences from the cozy.art/Tensorhub surface

Recorded, not accidental.

1. **`GET /v1/events`, the multiplexed stream, is NEW.** cozy.art opens one EventSource
   per request; a browser caps concurrent connections per origin at six, so a gallery
   watching ten requests silently stops receiving four of them. Same envelope, same
   cursor, one connection. The per-request route is unchanged and still terminal-stops.
2. **`events_url` is added to the submit handle.** cozy.art derives it by string
   concatenation; naming it is what lets a host move the stream later.
3. **`idempotent_replay` is added, and 200-vs-202 is load-bearing.** A client that
   retried a timed-out POST must be able to tell "I started it" from "this key was
   already mine" without comparing ids.
4. **The body digest covers the whole submission**, not the payload alone.
5. **Media is served by opaque id rather than a presigned URL.** The cloud host presigns
   because its bytes are in object storage; the local host's are on this disk. The
   CLIENT-VISIBLE shape is the same — an opaque handle in a document, a fetch to get
   bytes — which is what the contract actually requires.
6. **No estimate surface.** th-021's price-and-wait quote has no local meaning: nothing
   bills and the queue depth is one card.
7. **Model/lane/adapter overrides are ADMISSIBLE here and refuse `501
   override_unresolved`.** The local binding resolver is cl-005's. An override that is
   silently ignored is the worse bug, so the field refuses rather than being dropped.

## 10. The status→exit projection (cl-010)

Every host's refusal envelope carries the refusal's own NAME; the shared exit-matrix code
comes from the HTTP status. `api.CodeOf` is the inverse of `statusOf` and lives beside it
so the two cannot drift, and a client on any of the three hosts reads it the same way:

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
`200` on purpose — a terminal is an answer, not a transport failure — and a client maps
the terminal's own status with the job-terminal mapping instead (succeeded 0 · failed 11 ·
canceled 12 · deadline 10).
