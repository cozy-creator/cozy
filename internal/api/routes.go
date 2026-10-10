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
	{"GET", "/v1/local/machines/{machine}/logs/{log}", Local, true, false, false, "",
		"one log a machine keeps, oldest line first; an older machine answers a note",
		"`cozy rental logs --tensorfs`, `cozy machine logs --tensorfs`"},
	{"POST", "/v1/local/rentals/{rental_id}/prepare", Local, true, true, false, "",
		"queue exact package or model installation on an existing rental",
		"`cozy package install --rental` and `cozy model download --rental`"},
	{"GET", "/v1/local/rentals/{rental_id}/installs/{id}", Local, true, false, false, "",
		"one queued installation's state and, while it runs, the machine's latest progress",
		"`cozy model download --await`"},
	{"POST", "/v1/local/rentals/{rental_id}/runtime-update", Local, true, true, false, "",
		"Update one private worker's Runtime under a maintenance hold.", "`cozy rental update`"},
	{"GET", "/v1/local/rentals/{rental_id}/runtime-update", Local, true, false, false, "",
		"Observe one private worker's recorded Runtime update.", "`cozy rental update`"},
	{"POST", "/v1/local/daemon/down", Local, true, true, false, "",
		"stop, leaving work in flight running; under explicit --all request cancellation before confirmed rental teardown",
		"cl-045 `cozy down [--all]`"},

	// ---- the JOB family (cl-004), LOCAL by design: the hub's job plane is th-008's,
	// and a job's typed input trees are directories only a local caller owns.
	{"POST", "/v1/local/jobs", Local, true, true, false, "Idempotency-Key",
		"submit one bounded job; 202 with the job handle and its publication repo",
		"`cozy run` for a job callable"},
	{"GET", "/v1/local/jobs/{id}", Local, true, false, false, "",
		"one job: state, queue position, publication, checkpoints, bill where a rate exists",
		"`cozy run` follow and cancel"},
	{"POST", "/v1/local/jobs/{id}/pause", Local, true, true, false, "",
		"stop attempts while retaining the exact request and its work",
		"`cozy run pause`"},
	{"POST", "/v1/local/jobs/{id}/resume", Local, true, true, false, "",
		"resume the same paused request with its captured inputs",
		"`cozy run resume`"},
	{"POST", "/v1/local/jobs/{id}/cancel", Local, true, true, false, "",
		"request cancellation; a queued job leaves the queue, a running one gets its terminal",
		"`cozy run cancel`"},
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
