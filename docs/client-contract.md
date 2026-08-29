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
| `GET /v1/requests` | core | yes | listing, newest first; `?status=`, `?limit=` |
| `GET /v1/requests/{id}` | core | yes | the lifecycle document |
| `POST /v1/requests/{id}/cancel` | core | yes | requests cancellation; `?grace_ms=` |
| `GET /v1/requests/{id}/events` | core | yes | SSE, one request, terminal-stop; `?cursor=` |
| `GET /v1/media/{media_id}` | core | yes | bytes by opaque id; `HEAD`; one `Range` |
| `POST /v1/uploads` | local | yes | admit bounded content-addressed bytes; no caller path |
| `GET /v1/uploads/{upload_id}` | local | yes | opaque upload bytes with digest and Range support |

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

The `model`, `lane`, and `adapter` fields are reserved but not resolved by Cozy yet.
Any non-empty value refuses as `501 override_unresolved`; it is never silently ignored.

`worker` is a **LOCAL ADDITION**: it pins the request to an attached rental by id. An id
this Cozy daemon does not hold is `404` before a request row exists. The pin is not in
the idempotency digest because it selects where the same work runs, not what the work is.
A new rental cannot be pinned until its exact desired placement has been accepted,
converged, and reported dispatchable; otherwise submission refuses as
`rental.convergence_pending`.

`local_assets` is the CLI-only local extension for `--asset
<field-path>=<file>`. Each row names the exact request-schema field path plus a source
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

### Status

```json
{"request_id": "req-…", "status": "completed", "package": "org/name",
 "function": "denoise", "attempt": 1, "attempts": 1,
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
| `request.queued` | `reason` |
| `request.dispatch_aborted` | pre-offer preparation failed; `cause`, `error`; no worker saw this ordinal |
| `request.dispatched` | `instance_id`, `invocation_digest` |
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
| `POST /v1/local/rentals/{rental_id}/claim` | local | yes | attach the daemon to one already-provisioned private worker |
| `POST /v1/local/daemon/unload` | local | yes | stop definitely-idle local serving workers; never touch active work, jobs, rentals, or installed bytes |
| `POST /v1/local/daemon/down` | local | yes | safe down fence; `{all:true}` requests local cancellation and returns paid obligations that must be confirmed absent before retrying |
| `POST /v1/local/jobs` | local | yes | submit one bounded job; `Idempotency-Key`; 202 with the handle and its publication repo |
| `GET /v1/local/jobs/{id}` | local | yes | one job: state, queue position, retry budget, publication, checkpoints, bill where a rate exists |
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
`cozy invoke run` follows it).

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
  writes still land. Published CHECKPOINTS are absent by design: the publication
  transaction for canonical bytes is the runtime's, and this host will root and project
  ONE typed receipt from that border when it exists.
- **`bill`** — ABSENT unless the host was configured with an explicit local rate. There is
  no `$0.00`: a fabricated zero is a claim about money nobody measured.

`queue_position` is the orchestrator's scheduling fact; `requeues`/`retry_budget` are the
record owner's durable-attempt facts. Several jobs submitted at once queue against one
worker and drain in submission order, while the retry projection over neutral outcomes
spends a durable budget that the settlement names when it is exhausted.

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
