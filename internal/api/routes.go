package api

import "github.com/cozy-creator/cozy-creator/internal/manifest"

// The route table is DATA, and it is the same kind of object cl-002 made the CLI surface:
// one declarative registry that drives dispatch, the capability tokens, and the contract
// document. `docs/client-contract.md` is generated from these rows and fence-checked
// against them, so a route that exists in code and not in the document — or a document
// that describes a route nobody serves — is CI-red rather than a discovery six months
// later (th-021's "the table is a MANIFEST" item, built here first because this host is
// the reference).
//
// The split this file exists to make VISIBLE:
//
//	Core   — the shared client contract (th-021). Byte-identical across the three hosts:
//	         cozy-creator local, Tensorhub cloud, and the private-deployment pod. A change
//	         here is a change to all three and is a recorded decision, never a local
//	         convenience.
//	Local  — the LOCAL extension module. Installed endpoints, workers, host doctor,
//	         triage. These do not exist on the cloud host and never pretend to: they are
//	         mounted under /v1/local/ so a client can see the boundary in the URL.
//
// Cloud-only families (catalog, billing, orgs) are Tensorhub extensions and have no row
// here at all.

// Scope is which module a route belongs to.
type Scope string

const (
	Core  Scope = "core"
	Local Scope = "local"
)

// Route is one row of the surface.
type Route struct {
	Method string
	Path   string // the Go 1.22 mux pattern, with {wildcards}
	Scope  Scope
	// Auth is false only for the two routes a credential-less caller may reach: the
	// liveness probe and the stub page. Neither answers with a fact about any request.
	Auth bool
	// Mutation marks a route that changes state. Mutations carry the Origin check.
	Mutation bool
	// Stream marks an SSE route. Stream OPENS carry the Origin check too — a stream is
	// how a cross-site page would exfiltrate, and it is a GET.
	Stream bool
	// Idempotency names the header a caller must present, or "" when the route needs none.
	Idempotency string
	Summary     string
	// Consumer is the named first consumer (law 13). No row may exist without one.
	Consumer string
}

// Routes is THE surface. Order is the document's order.
var Routes = []Route{
	// ---- the shared client contract CORE (th-021) ----
	{"POST", "/v1/requests", Core, true, true, false, "Idempotency-Key",
		"submit one request; 202 with the request handle",
		"cl-010 `cozy run`, cl-007's UI, cozy.art"},
	{"GET", "/v1/requests", Core, true, false, false, "",
		"list requests newest-first, optionally filtered by status",
		"cl-010 `cozy status`"},
	{"GET", "/v1/requests/{id}", Core, true, false, false, "",
		"one request: status, attempt, metrics, typed result, media refs",
		"cl-010 `cozy run`"},
	{"POST", "/v1/requests/{id}/cancel", Core, true, true, false, "",
		"request cancellation of the live attempt; the terminal still arrives",
		"cl-010 `cozy run` on SIGINT"},
	{"GET", "/v1/requests/{id}/events", Core, true, false, true, "",
		"SSE for ONE request: durable lifecycle from a cursor plus live progress; terminal-stop",
		"cozy.art's SSE client, cl-010 `--stream`"},
	{"GET", "/v1/events", Core, true, false, true, "",
		"the MULTIPLEXED SSE stream: every request, one connection, one cursor",
		"cl-007's UI (the browser connection cap)"},
	{"GET", "/v1/media/{media_id}", Core, true, false, false, "",
		"one output's bytes by OPAQUE id; bounded Range; never a path",
		"cl-010 `--out`, cl-007's UI"},
	{"GET", "/v1/capabilities", Core, true, false, false, "",
		"feature tokens; presence is a token, never a version string",
		"cl-002 `cozy capabilities`"},

	// ---- the LOCAL extension module ----
	{"GET", "/v1/local/endpoints", Local, true, false, false, "",
		"installed endpoints: pin, generation, release, ready plans",
		"cl-010 `cozy ls`, cl-007's UI"},
	{"GET", "/v1/local/workers", Local, true, false, false, "",
		"live endpoint workers: protocol identities, devices, intake state",
		"cl-010 `cozy status`"},
	{"POST", "/v1/local/workers", Local, true, true, false, "",
		"make one local endpoint resident or claim one exact rental without invoking a model",
		"cl-010 `cozy start`, cl-021 `cozy rent probe`"},
	{"DELETE", "/v1/local/workers/{instance_id}", Local, true, true, false, "",
		"drain and stop one worker's whole process group",
		"cl-010 `cozy stop`"},
	{"POST", "/v1/local/rentals/{rental_id}/placement-revisions", Local, true, true, false, "Idempotency-Key",
		"author one Tensorhub placement revision and relay grant-before-set on the claimed worker",
		"`cozy rent revise`"},
	{"GET", "/v1/local/doctor", Local, true, false, false, "",
		"host facts and the service's own state",
		"cl-010 `cozy doctor`"},
	{"POST", "/v1/local/service/shutdown", Local, true, true, false, "",
		"ask the LocalService to drain every worker and exit; exit is proved by the service lock",
		"cl-010 `cozy down` (#449's cooperative tier)"},
	{"GET", "/v1/local/attempts/{attempt_key}/triage", Local, true, false, false, "",
		"the retained WorkerTriageBundle by OPAQUE attempt key, verified against its terminal",
		"cl-010 `cozy logs <attempt>`"},

	// ---- the JOB family (cl-004), LOCAL by design: the hub's job plane is th-008's,
	// and a job's typed input trees are directories only a local caller owns.
	{"POST", "/v1/local/jobs", Local, true, true, false, "Idempotency-Key",
		"submit one bounded job; 202 with the job handle and its publication repo",
		"cl-004 `cozy job submit`"},
	{"GET", "/v1/local/jobs", Local, true, false, false, "",
		"list jobs newest-first with per-state counts, optionally filtered",
		"cl-004 `cozy job ls`"},
	{"GET", "/v1/local/jobs/{id}", Local, true, false, false, "",
		"one job: state, queue position, retry budget, publication, checkpoints, bill where a rate exists",
		"cl-004 `cozy job status`"},
	{"POST", "/v1/local/jobs/{id}/cancel", Local, true, true, false, "",
		"request cancellation; a queued job leaves the queue, a running one gets its terminal",
		"cl-004 `cozy job cancel`"},

	// ---- Creator-owned ordered workflows. Children are ordinary /v1/requests rows. ----
	{"POST", "/v1/local/workflows", Local, true, true, false, "Idempotency-Key",
		"submit one canonical ordered workflow; 202 fresh or 200 replay",
		"cl-018 `cozy workflow submit`, cl-024"},
	{"GET", "/v1/local/workflows/{id}", Local, true, false, false, "",
		"one workflow with child request states and accepted outputs projected from authority",
		"cl-018 `cozy workflow status`, cl-024"},
	{"GET", "/v1/local/workflows/{id}/receipt", Local, true, false, false, "",
		"durable canonical plan, materialized submissions, and exact frozen rental controls",
		"cl-024 `cozy workflow download`"},
	{"POST", "/v1/local/workflows/{id}/cancel", Local, true, true, false, "",
		"persist workflow cancellation, cancel the active child, and mint nothing later",
		"cl-018 `cozy workflow cancel`"},
	{"POST", "/v1/local/video-compositions", Local, true, true, false, "",
		"compose an exact Cozy Video source or retained creative plan into an ordinary workflow",
		"cl-024 `cozy video compose`, `cozy video submit`"},

	// ---- unauthenticated: liveness and the stub page ----
	{"GET", "/healthz", Local, false, false, false, "",
		"liveness only — it answers `up` and nothing about any request",
		"cl-002 `internal/service.Probe`"},
	{"GET", "/{$}", Local, false, false, false, "",
		"the embedded stub page (cl-007's stub half)",
		"cl-007"},
	{"GET", "/app.js", Local, false, false, false, "",
		"the stub's script — a FILE so the page needs no inline-script CSP",
		"the stub page"},
	{"GET", "/app.css", Local, false, false, false, "",
		"the stub's style — a FILE for the same reason",
		"the stub page"},
}

// Tokens are the capability tokens this API advertises. Presence is a token; a client
// never infers a feature from a version string (cl-002's discipline, same registry).
var Tokens = []string{
	"api.contract.core.v1",       // the shared client-contract core, this host
	"api.requests.submit",        // idempotency key + body digest
	"api.requests.cancel",        // explicit, digest-fenced cancellation
	"api.events.sse",             // durable lifecycle + cursor resume + terminal-stop
	"api.events.multiplex",       // one connection for every request
	"api.media.opaque",           // media by opaque id; no path shape exists
	"api.errors.envelope",        // one typed {error:{code,message,remedy,request_id}}
	"api.auth.bearer",            // bearer only; no cookie is read anywhere
	"api.bind.loopback",          // loopback-only bind, IPv4 and IPv6
	"api.local.extension",        // the LOCAL module, explicitly not the core
	"api.local.triage",           // retained bundles by opaque attempt key
	"api.local.jobs",             // the bounded job family: submit/list/status/cancel
	"api.jobs.publication",       // a job's landed writes are a durable publication root
	"api.local.workflows",        // ordered ordinary children, Creator-owned recovery/cancellation
	"api.local.video",            // strict source composition into the workflow form
	"api.local.rental-revisions", // same-pod dynamic endpoint convergence
	"api.stub.embedded",          // the go:embed stub page
}

// ContractVersion is the shared client contract's core version this host serves. It
// replaces cl-002's `pending-cl-006` placeholder in `cozy version`.
const ContractVersion = "cozy.client.v1"

// ONE registry. `cozy capabilities` and `GET /v1/capabilities` read the same slice, so
// the CLI cannot advertise a feature the server does not serve.
func init() { manifest.APITokens = Tokens }
