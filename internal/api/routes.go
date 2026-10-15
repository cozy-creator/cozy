package api

// The route table is DATA: one declarative registry drives dispatch and capability
// tokens. `scripts/fence.py` checks its method, path, scope, and order against
// `docs/client-contract.md`, so a route present on only one side is CI-red.
//
// The split this file exists to make VISIBLE:
//
//	Core  — Cozy's implemented request-level API and the proposed common core for
//	        future servers. Cross-host parity requires conformance proof; it is not
//	        asserted by this registry.
//	Local — the Cozy-only extension module: jobs, rentals, lifecycle, and
//	        localhost web assets. Its URL mount makes
//	        that boundary visible in the URL.

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
	// Auth is false only for embedded static web assets. They disclose no user state.
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
	// ---- Cozy's implemented request-level CORE ----
	{"GET", "/v1/capabilities", Core, true, false, false, "",
		"advertise supported request selection semantics",
		"`cozy run` model overrides"},
	{"POST", "/v1/requests", Core, true, true, false, "Idempotency-Key",
		"submit one request; 202 with the request handle",
		"`cozy run`"},
	{"GET", "/v1/requests", Core, true, false, false, "",
		"list requests newest-first, optionally filtered by status",
		"`cozy run list`"},
	{"GET", "/v1/requests/{id}", Core, true, false, false, "",
		"one request: status, attempt, metrics, typed result, media refs",
		"`cozy run`"},
	{"POST", "/v1/requests/{id}/cancel", Core, true, true, false, "",
		"request cancellation of the live attempt; the terminal still arrives",
		"`cozy run cancel` and run SIGINT"},
	{"GET", "/v1/requests/{id}/events", Core, true, false, true, "",
		"SSE for ONE request: durable lifecycle from a cursor plus live progress; terminal-stop",
		"`cozy run` follow"},
	{"GET", "/v1/media/{media_id}", Core, true, false, false, "",
		"one output's bytes by OPAQUE id; bounded Range; never a path",
		"`cozy run --out`"},
	// ---- the LOCAL extension module ----
	{"GET", "/v1/local/attempts/{attempt_key}/triage", Local, true, false, false, "",
		"one attempt's kept triage bundle from its own attempt row; 404 when none was kept",
		"run/job failure remedies"},
	{"POST", "/v1/local/requests/{id}/abandon", Local, true, true, false, "",
		"abandon local Runtime-run tracking without confirming remote stop or releasing a rental",
		"`cozy run cancel --abandon`"},
	{"GET", "/v1/local/requests/{id}/evidence", Local, true, false, false, "",
		"one run's durable events and its last attempt's kept triage bundle",
		"`cozy run show`"},
	{"GET", "/v1/local/rentals", Local, true, false, false, "",
		"reconciled rental inventory, account spend, pending acquisitions, and activity",
		"`cozy rental list`"},
	{"POST", "/v1/local/rentals/{rental_id}/keepalive", Local, true, true, false, "",
		"reset the rental's fifteen-minute idle clock once, after its machine acknowledges",
		"`cozy rental keepalive`"},
	{"GET", "/v1/local/machines/{machine}/status", Local, true, false, false, "",
		"one machine's picture as it reports it: software, GPUs, live runs, environments, disk, idle deadline",
		"`cozy rental show`"},
	{"POST", "/v1/local/machines/{machine}/ready", Local, true, true, false, "",
		"notify the controller of an availability edge so accepted work can resume",
		"`cozy machine install` and `cozy machine start`"},
	{"GET", "/v1/local/machines/{machine}/logs/{log}", Local, true, false, false, "",
		"one log a machine keeps, oldest line first; an older machine answers a note",
		"`cozy rental logs --tensorfs`, `cozy machine logs --tensorfs`"},
	{"POST", "/v1/local/machines/{machine}/operations", Local, true, true, false, "",
		"record one package or model operation on a machine as a numbered run and start it",
		"`cozy package install|update|remove`, `cozy model download|upload`, `cozy run upload`, `cozy rental update`"},
	{"POST", "/v1/local/calcifer/down", Local, true, true, false, "",
		"stop, leaving work in flight running; under explicit --all request cancellation before confirmed rental teardown",
		"cl-045 `cozy down [--all]`"},

	{"GET", "/{$}", Local, false, false, false, "",
		"embedded localhost web UI entrypoint",
		"`cozy up`"},
	{"GET", "/app.css", Local, false, false, false, "",
		"embedded localhost web UI stylesheet",
		"web/index.html"},
	{"GET", "/app.js", Local, false, false, false, "",
		"embedded localhost web UI script",
		"web/index.html"},
}
