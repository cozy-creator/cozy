// Package api is Cozy's local client API server: the request-level CORE served on
// loopback by the one Cozy daemon, plus an explicitly Cozy-only extension module.
//
// It is a client of internal/orchestrator and nothing else. Every submission still flows
// orchestrator → worker protocol → runtime; this package adds an HTTP shape, a typed error
// envelope, and the event/media planes a browser needs. It opens no second door into the
// runtime, and it cannot: it holds a *orchestrator.Orchestrator, and the orchestrator's exported
// surface has no "run this" that bypasses the request record.
//
// # A loopback bind is not a boundary
//
// Ollama's DNS-rebinding CVE is the standing proof: a web page the user visits can drive a
// loopback-only API, because the browser resolves the attacker's hostname to 127.0.0.1 and
// then talks to whatever answers. Every defense below exists for that adversary, in this
// order, and each one alone would be insufficient:
//
//  1. LOOPBACK-ONLY BIND, IPv4 and IPv6, and there is NO flag that widens it. The
//     non-localhost door is deferred behind TLS, durable auth and its own threat review —
//     a LAN bind is not a switch someone can flip in a config file.
//  2. HOST ALLOWLIST. A rebinding request arrives with the ATTACKER's hostname in `Host`,
//     because that is what the browser was told to visit. Three exact spellings are
//     admitted and everything else is 403 before routing.
//  3. ORIGIN CHECK on every mutation and every stream open. A cross-site `fetch`, form
//     POST, or EventSource sends `Origin`; a foreign one refuses. Absent `Origin` is a
//     non-browser client (the CLI), which is the only caller that legitimately omits it.
//  4. BEARER AUTH, and NO COOKIE IS READ ANYWHERE. Cookies are ambient authority — the
//     browser attaches them to a cross-site request whether or not the page meant it.
//     A bearer is not ambient: the page must possess the token and choose to send it.
//     (cl-007's session-cookie ceremony lands behind its real-UI gate; the invariant that
//     nothing here reads a cookie is launch surface now, and the `api` fence proves it.)
//  5. NO CORS HEADERS AT ALL. Not a narrow allowlist — none. A browser page from another
//     origin cannot read a response it is not allowed to read.
//
// # Tokens
//
// Two credentials are minted per launch and neither is ever printed, logged, or placed on
// argv. The BROWSER token rides the `--open` URL's FRAGMENT (fragments are not sent to the
// server, do not enter access logs, and do not leak through `Referer`); the CLI credential
// is handed over through a 0600 file in the local root. Both are `secret.Value`, compared
// by full-digest constant time — this package never calls Reveal, so there is no code path
// through which a raw token could reach a log line or a rendered error.
package api

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// MaxBody caps a submitted request body. Asset bytes never ride JSON: the local extension
// names files the daemon ingests, while a network host uses its own upload surface.
const MaxBody = 8 << 20

// Server is the local client API. One per Cozy daemon.
type Server struct {
	scopedEndpoint    string
	machineExecutions MachineExecutions
	orchestrator      *orchestrator.Orchestrator
	store             *records.Store
	layout            home.Layout
	cfg               config.Config
	creds             Credentials
	addr              string
	log               io.Writer
	web               http.Handler
	// packages resolves a package ref to a spec the orchestrator can start. It is the
	// LOCAL module's resolver; the pod profile (cl-014) supplies its own.
	packages Resolver

	// rentals resolves only the non-secret, attempt-bound desired placement. The
	// credential and dial triple remain orchestrator-only and are obtained at dial time.
	rentals             func(id string) (*orchestrator.DesiredPlacement, *exit.Error)
	rentalInventory     func(string, bool, bool) (RentalInventory, *exit.Error)
	rentalKeepalive     func(context.Context, string) (RentalKeepaliveResult, *exit.Error)
	rentalInstall       func(string, records.RentalInstallSelection) (*records.RentalInstall, *exit.Error)
	rentalInstallStatus func(string, string) (*RentalInstallStatus, *exit.Error)
	runtimeUpdate       func(string, RuntimeUpdateRequest) (*records.RuntimeUpdate, *exit.Error)

	// shutdown asks the process that owns this server to drain and stop — `cozy down`'s
	// cooperative tier (#449). The route refuses when the builder wired none.
	shutdown func()

	// shutdownAdmission closes one race: a mutation may not commit between `down`'s
	// active-work read and listener shutdown. Handlers hold the shared side only for
	// bounded admission work; provider acquisition and execution start afterward.
	shutdownAdmission sync.RWMutex
	shuttingDown      bool
	// reportedDownRentals is the paid-pod set the last `--all` pass handed back for the
	// caller to end. Seeing the SAME set again with nothing else changed is how the
	// teardown knows the caller could not end them, and that asking a third time would
	// produce the same answer forever.
	reportedDownRentals []string
}

// Resolver exposes control-plane placement facts separately from a local worker launch.
// A remote request and package listing use ResolvePlacement; only an explicit local
// start may require the target environment through Resolve.
type Resolver interface {
	RefreshEditable(pkg string) (installID string, editable, changed bool, problem *exit.Error)
	PrepareLocal(context.Context, string) (localpackage.Installation, *exit.Error)
	LocalInstallation(string, string) (localpackage.Installation, *exit.Error)
	ResolvePlacement(pkg string) (orchestrator.DesiredPlacement, *exit.Error)
	ResolveInstall(installID string, models []orchestrator.ModelRef) (orchestrator.WorkerLaunchSpec, *exit.Error)
	ResolveRemoteRelease(hub, pkg, release, function string, models []orchestrator.ModelRef) (
		orchestrator.LogicalPackage, *launch.Entrypoint, *exit.Error)
	ResolveRemoteJob(hub, pkg, release, function string, models []orchestrator.ModelRef,
		deferredModels bool) (orchestrator.LogicalJob, *launch.Entrypoint, *exit.Error)
	Entrypoint(installID, name string) (*launch.Entrypoint, bool, *exit.Error)
	// Jobs names the `@job` functions one installed package registers, with the
	// descriptor id each resolves to. The job submit route resolves a function to its
	// digest through this and never lets a client name one (cl-004).
	Jobs(pkg string) ([]launch.JobFacts, *exit.Error)
	JobsInstall(installID string) ([]launch.JobFacts, *exit.Error)
}

// Options is the frozen input to one API server.
type Options struct {
	ScopedEndpoint    string // foreground controller admits only this explicitly pinned target
	MachineExecutions MachineExecutions
	Orchestrator      *orchestrator.Orchestrator
	Cfg               config.Config
	Creds             Credentials
	Addr              string
	Log               io.Writer
	Web               http.Handler
	Packages          Resolver
	// Rentals validates one attached generic worker id; desired package/model state is
	// sent separately over WorkerControl.
	Rentals func(id string) (*orchestrator.DesiredPlacement, *exit.Error)
	// RentalInventory answers one hub's fleet, or with allHubs every hub's.
	RentalInventory func(hub string, allHubs, reconcile bool) (RentalInventory, *exit.Error)
	RentalKeepalive func(context.Context, string) (RentalKeepaliveResult, *exit.Error)
	RentalInstall   func(string, records.RentalInstallSelection) (*records.RentalInstall, *exit.Error)
	// RentalInstallStatus reads one queued installation on one machine, with its progress.
	RentalInstallStatus func(machine, id string) (*RentalInstallStatus, *exit.Error)
	RuntimeUpdate       func(string, RuntimeUpdateRequest) (*records.RuntimeUpdate, *exit.Error)
	// Shutdown is the cooperative-down hook the shutdown route calls (#449).
	Shutdown func()
}

// New builds the server and its route table. It binds nothing; Listeners does that.
func New(opt Options) *Server {
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	return &Server{
		scopedEndpoint:    opt.ScopedEndpoint,
		machineExecutions: opt.MachineExecutions,
		orchestrator:      opt.Orchestrator, store: opt.Orchestrator.Store(),
		layout: opt.Orchestrator.Layout(), cfg: opt.Cfg, creds: opt.Creds,
		addr: opt.Addr, log: opt.Log, web: opt.Web, packages: opt.Packages,
		rentals: opt.Rentals, rentalInventory: opt.RentalInventory, rentalKeepalive: opt.RentalKeepalive, rentalInstall: opt.RentalInstall, rentalInstallStatus: opt.RentalInstallStatus, runtimeUpdate: opt.RuntimeUpdate, shutdown: opt.Shutdown,
	}
}

// HubHeader names the Tensorhub origin a client command addresses. The daemon serves
// every hub at once: work it records keeps this origin, and reads are scoped to it.
const HubHeader = "Cozy-Tensorhub"

// hubOf is the request's Tensorhub origin; a client that names none uses the
// daemon's configured default.
func (s *Server) hubOf(r *http.Request) (string, *exit.Error) {
	value := strings.TrimSpace(r.Header.Get(HubHeader))
	if value == "" {
		return s.cfg.HubURL, nil
	}
	return config.HubOrigin(value)
}

// requestHub is the hub a recorded request belongs to: its own, else its rental's, else
// this daemon's configured hub.
func (s *Server) requestHub(row records.Request) string {
	if row.Hub != "" {
		return row.Hub
	}
	if hub, problem := s.store.RequestHub(row.ID, s.cfg.HubURL); problem == nil {
		return hub
	}
	return s.cfg.HubURL
}

// submissionHub is the hub new work belongs to. Work on a named rental that the client
// did not place on a hub itself belongs to that rental's hub.
func (s *Server) submissionHub(r *http.Request, rentalID string) (string, *exit.Error) {
	selected, problem := s.hubOf(r)
	if problem != nil || rentalID == "" || strings.TrimSpace(r.Header.Get(HubHeader)) != "" {
		return selected, problem
	}
	row, problem := s.store.RentalRow(rentalID)
	if problem != nil || row == nil {
		return selected, problem
	}
	if origin, invalid := config.HubOrigin(row.Hub); invalid == nil {
		return origin, nil
	}
	return selected, nil
}

// activateRecorded starts capacity selection only after the HTTP authority has durably
// recorded the request. Provider acquisition and worker preparation may take minutes; neither
// belongs under the short shutdown-admission gate.
func (s *Server) activateRecorded(request records.Request) {
	go func() {
		if _, problem := s.orchestrator.ActivateRecordedRequest(request); problem != nil {
			fmt.Fprintf(s.log, "request %s activation: %s\n", request.ID, problem.Message)
		}
	}()
}

// recordSubmission is the complete shutdown race boundary. Descriptor resolution and
// payload validation happen before it; provider work happens after it. The lock protects
// only the durable row insertion against `down`'s final active-work snapshot.
func (s *Server) recordSubmission(spec orchestrator.Submission) (records.Request, bool, *exit.Error) {
	s.shutdownAdmission.RLock()
	defer s.shutdownAdmission.RUnlock()
	if s.shuttingDown {
		return records.Request{}, false, exit.Named(exit.Unavailable,
			"daemon_shutting_down", "the daemon has accepted shutdown and no longer admits work")
	}
	return s.orchestrator.RecordSubmission(spec)
}

// Handler is the whole surface, wrapped in the guard chain. Routes are registered FROM
// the route table, so a row without a handler — or a handler without a row — is a startup
// refusal exactly as cl-002's manifest self-check is.
func (s *Server) Handler() (http.Handler, *exit.Error) {
	mux := http.NewServeMux()
	handlers := map[string]http.HandlerFunc{
		"GET /v1/capabilities":                              s.capabilities,
		"POST /v1/requests":                                 s.submit,
		"GET /v1/requests":                                  s.listRequests,
		"GET /v1/requests/{id}":                             s.getRequest,
		"POST /v1/local/requests/{id}/abandon":              s.abandonRequest,
		"POST /v1/requests/{id}/cancel":                     s.cancelRequest,
		"GET /v1/requests/{id}/events":                      s.requestEvents,
		"GET /v1/media/{media_id}":                          s.media,
		"GET /v1/local/attempts/{attempt_key}/triage":       s.attemptTriage,
		"GET /v1/local/requests/{id}/evidence":              s.requestEvidence,
		"POST /v1/local/rentals/{rental_id}/keepalive":      s.keepRentalAlive,
		"GET /v1/local/rentals":                             s.listRentals,
		"DELETE /v1/local/rentals/{rental_id}/claim":        s.detachRental,
		"GET /v1/local/machines/{machine}/status":           s.machineStatus,
		"GET /v1/local/machines/{machine}/logs/{log}":       s.machineLog,
		"POST /v1/local/rentals/{rental_id}/prepare":        s.prepareRentalPackage,
		"GET /v1/local/rentals/{rental_id}/installs/{id}":   s.rentalInstallState,
		"POST /v1/local/rentals/{rental_id}/runtime-update": s.startRuntimeUpdate,
		"GET /v1/local/rentals/{rental_id}/runtime-update":  s.readRuntimeUpdate,
		"POST /v1/local/daemon/down":                        s.downDaemon,
		"POST /v1/local/jobs":                               s.submitJob,
		"GET /v1/local/jobs/{id}":                           s.getJob,
		"POST /v1/local/jobs/{id}/pause":                    s.pauseJob,
		"POST /v1/local/jobs/{id}/resume":                   s.resumeJob,
		"POST /v1/local/jobs/{id}/uploads":                  s.uploadJobOutput,
		"POST /v1/local/jobs/{id}/cancel":                   s.cancelJob,
		"GET /{$}":                                          s.webUI,
		"GET /app.css":                                      s.webUI,
		"GET /app.js":                                       s.webUI,
	}
	registered := map[string]bool{}
	for _, r := range Routes {
		pattern := r.Method + " " + r.Path
		h, ok := handlers[pattern]
		if !ok {
			return nil, exit.Internalf("the route table advertises %q with no handler", pattern)
		}
		// LAW 13 AT STARTUP. Every route names one real consumer and says what it does,
		// or this binary refuses to serve. It is the same discipline cl-002's manifest
		// self-check applies to the CLI surface, and it is why the table's `Consumer`
		// column is a checked field rather than a comment someone stops updating.
		if r.Consumer == "" || r.Summary == "" {
			return nil, exit.Internalf(
				"route %q names no consumer or no summary — a surface with no consumer is machinery",
				pattern)
		}
		mux.Handle(pattern, s.guard(r, h))
		registered[pattern] = true
	}
	for pattern := range handlers {
		if !registered[pattern] {
			return nil, exit.Internalf("handler %q is registered but no route advertises it", pattern)
		}
	}
	// The CATCH-ALL, method-agnostic and unauthenticated: an unknown path must answer
	// with the TYPED envelope, not Go's plain-text `404 page not found` (cl-011's
	// finding on the hub side, decisions #315 — a client that parses one envelope
	// everywhere should never meet a bare string). It is unauthenticated deliberately:
	// "this route does not exist" is not a fact worth a credential, and gating it would
	// make an unknown route indistinguishable from an unauthorized one.
	mux.Handle("/", s.guard(Route{Scope: Core}, s.unknownRoute))
	return s.baseline(mux), nil
}

// baseline is what EVERY response carries, including refusals from the guards themselves.
func (s *Server) baseline(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h := w.Header()
		h.Set("X-Content-Type-Options", "nosniff")
		// No response is cacheable: media is opaque-keyed and everything else is live.
		h.Set("Cache-Control", "no-store")
		// A referrer would carry the URL to whatever the page links to next. Nothing in
		// this API's URLs is a secret today (the token rides a header and a fragment),
		// and this keeps it true if a future URL ever carries one.
		h.Set("Referrer-Policy", "no-referrer")
		// Nothing an API response renders may load, connect, or execute anything. The stub
		// page replaces this with its own slightly wider policy (still no inline script).
		h.Set("Content-Security-Policy",
			"default-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
		// NO Access-Control-Allow-* header is set anywhere in this package, deliberately.
		next.ServeHTTP(w, r)
	})
}

// guard is the ordered defense chain. The order is the point: the most structural refusal
// first, so a rebinding page is turned away before the server considers whether it holds a
// credential — and an unauthenticated caller learns nothing about which requests exist.
func (s *Server) guard(route Route, h http.HandlerFunc) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// 1. The peer must be on the loopback interface. The listener already guarantees
		//    it; this is the assertion that says so out loud, and it is what keeps the
		//    guarantee if a future profile ever binds elsewhere.
		if !loopbackPeer(r.RemoteAddr) {
			s.refuse(w, r, http.StatusForbidden, "not_loopback",
				"this API serves the loopback interface only",
				"the non-localhost door is deferred behind TLS and durable auth; it is not a flag")
			return
		}
		// 2. The Host allowlist — the DNS-rebinding kill switch.
		if !s.allowedHost(r.Host) {
			s.refuse(w, r, http.StatusForbidden, "host_not_allowed",
				fmt.Sprintf("Host %q is not one this server answers to", r.Host),
				"reach it as 127.0.0.1, localhost or [::1] on its own port")
			return
		}
		// 3. Origin, on mutations and stream opens. A same-origin page and a non-browser
		//    client both pass; a cross-site page does not.
		if route.Mutation || route.Stream {
			if origin := r.Header.Get("Origin"); origin != "" && !s.allowedOrigin(origin) {
				s.refuse(w, r, http.StatusForbidden, "origin_not_allowed",
					fmt.Sprintf("Origin %q may not reach this route", origin),
					"only a page served by this Cozy daemon may mutate or open a stream")
				return
			}
		}
		// 4. Bearer. No cookie is read: `r.Cookie` appears nowhere in this package and
		//    the `api` fence family proves the absence.
		if route.Auth && !s.authenticated(r) {
			w.Header().Set("WWW-Authenticate", `Bearer realm="cozy"`)
			s.refuse(w, r, http.StatusUnauthorized, "unauthenticated",
				"this route needs the local client credential",
				"the CLI reads it from the local root; the UI is handed one at launch")
			return
		}
		if route.Idempotency != "" && strings.TrimSpace(r.Header.Get(route.Idempotency)) == "" {
			s.refuse(w, r, http.StatusBadRequest, "idempotency_key_required",
				"submission needs an "+route.Idempotency+" header",
				"one key names one request forever; retrying under it is safe by construction")
			return
		}
		if route.Mutation && route.Path != "/v1/local/daemon/down" &&
			route.Path != "/v1/requests" && route.Path != "/v1/local/jobs" {
			s.shutdownAdmission.RLock()
			defer s.shutdownAdmission.RUnlock()
			if s.shuttingDown {
				s.refuse(w, r, http.StatusServiceUnavailable, "daemon_shutting_down",
					"the daemon has accepted shutdown and no longer admits mutations",
					"wait for the daemon to finish stopping")
				return
			}
		}
		h(w, r)
	})
}

func loopbackPeer(remote string) bool {
	host, _, err := net.SplitHostPort(remote)
	if err != nil {
		host = remote
	}
	ip := net.ParseIP(strings.Trim(host, "[]"))
	return ip != nil && ip.IsLoopback()
}

// allowedHost admits the three exact spellings of "this server, on its own port". A
// hostname is never resolved: resolution is precisely what rebinding subverts.
func (s *Server) allowedHost(host string) bool {
	_, port, err := net.SplitHostPort(s.addr)
	if err != nil {
		return false
	}
	switch host {
	case "127.0.0.1:" + port, "localhost:" + port, "[::1]:" + port:
		return true
	}
	return false
}

func (s *Server) allowedOrigin(origin string) bool {
	_, port, err := net.SplitHostPort(s.addr)
	if err != nil {
		return false
	}
	switch origin {
	case "http://127.0.0.1:" + port, "http://localhost:" + port, "http://[::1]:" + port:
		return true
	}
	return false
}

// authenticated compares the presented bearer against BOTH per-launch credentials in
// constant time. The raw values are never read here — `secret.Value.Equal` hashes both
// sides — so no formatting verb in this package can print one.
func (s *Server) authenticated(r *http.Request) bool {
	header := r.Header.Get("Authorization")
	presented, ok := strings.CutPrefix(header, "Bearer ")
	if !ok {
		return false
	}
	return s.creds.Admits(presented)
}

func (s *Server) cliAuthenticated(r *http.Request) bool {
	header := r.Header.Get("Authorization")
	presented, ok := strings.CutPrefix(header, "Bearer ")
	return ok && s.creds.AdmitsCLI(presented)
}

// ---------------------------------------------------------------- the error envelope

// envelope is the contract's one error shape: {"error": {code, message, ...}}. Every
// refusal in this package renders through it, including the guards' own and the
// unknown-route catch-all.
type envelope struct {
	Code      string `json:"code"`
	Message   string `json:"message"`
	Remedy    string `json:"remedy,omitempty"`
	RequestID string `json:"request_id,omitempty"`
}

func (s *Server) refuse(w http.ResponseWriter, r *http.Request, status int, code, message, remedy string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]envelope{"error": {
		Code: code, Message: message, Remedy: remedy,
		RequestID: r.PathValue("id"),
	}})
	s.logf("%s %s -> %d %s", r.Method, r.URL.Path, status, code)
}

// refuseTyped renders a orchestrator/records refusal without inventing a second
// vocabulary: the typed error's own name is the envelope code, and the shared exit
// matrix decides the HTTP status. One mapping, in one place.
func (s *Server) refuseTyped(w http.ResponseWriter, r *http.Request, e *exit.Error) {
	s.refuse(w, r, statusOf(e.Code), e.ErrName(), e.Message, e.Remedy)
}

// statusOf maps the SHARED exit matrix onto HTTP. It is a projection of the matrix, not a
// second table: a client that reads the envelope's `code` gets the matrix name, and a
// client that only reads HTTP still gets a sane status.
func statusOf(code exit.Code) int {
	switch code {
	case exit.OK:
		return http.StatusOK
	case exit.Usage, exit.Validation:
		return http.StatusBadRequest
	case exit.NotFound:
		return http.StatusNotFound
	case exit.Credential:
		return http.StatusUnauthorized
	case exit.Structural:
		return http.StatusUnprocessableEntity
	case exit.Confirm:
		return http.StatusPreconditionRequired
	case exit.OfflineMiss:
		return http.StatusNotFound
	case exit.Unavailable:
		return http.StatusServiceUnavailable
	case exit.Deadline:
		return http.StatusGatewayTimeout
	case exit.Failed, exit.Canceled:
		return http.StatusOK // a terminal is an ANSWER, never a transport failure
	case exit.Conflict:
		return http.StatusConflict
	case exit.Capacity:
		return http.StatusInsufficientStorage
	}
	return http.StatusInternalServerError
}

// CodeOf is statusOf's INVERSE, and it lives here so the two cannot drift: a client that
// reads this API's refusals turns the status back into the shared matrix code the server
// mapped out of. The envelope's `code` stays the refusal's own NAME (`override_unresolved`,
// `bundle_corrupt`) — a name is more specific than a matrix row and is never flattened
// into one. cl-010's CLI is the consumer; th-021's other hosts get the same projection.
//
// 200 for a FAILED or CANCELED terminal is deliberate on the way out (a terminal is an
// answer, not a transport failure) and therefore never comes back through here: a client
// reads the terminal's own status and maps it with exit.JobTerminal.
func CodeOf(status int) exit.Code {
	switch status {
	case http.StatusOK, http.StatusAccepted, http.StatusCreated:
		return exit.OK
	case http.StatusBadRequest:
		return exit.Validation
	case http.StatusUnauthorized, http.StatusForbidden:
		return exit.Credential
	case http.StatusNotFound, http.StatusGone:
		return exit.NotFound
	case http.StatusConflict:
		return exit.Conflict
	case http.StatusPreconditionRequired:
		return exit.Confirm
	case http.StatusRequestEntityTooLarge, http.StatusUnprocessableEntity:
		return exit.Structural
	case http.StatusInsufficientStorage:
		return exit.Capacity
	case http.StatusServiceUnavailable:
		return exit.Unavailable
	case http.StatusNotImplemented:
		// cl-002's spelling: "advertised, not carried by this build" is a USAGE
		// refusal naming the issue that lands it, never an outage.
		return exit.Usage
	case http.StatusGatewayTimeout, http.StatusRequestTimeout:
		return exit.Deadline
	}
	return exit.Internal
}

func (s *Server) unknownRoute(w http.ResponseWriter, r *http.Request) {
	s.refuse(w, r, http.StatusNotFound, "unknown_route",
		fmt.Sprintf("this host serves no %s %s", r.Method, r.URL.Path),
		"GET /v1/capabilities lists what it does serve")
}

// ---------------------------------------------------------------------------- output

func (s *Server) ok(w http.ResponseWriter, r *http.Request, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(body); err != nil {
		s.logf("%s %s: the response could not be written: %v", r.Method, r.URL.Path, err)
	}
}

func (s *Server) logf(format string, args ...any) {
	fmt.Fprintf(s.log, "[api] "+format+"\n", args...)
}
