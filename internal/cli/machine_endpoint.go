package cli

import (
	"cmp"
	"io"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// startEndpointController composes the existing business API and machine observer for
// one explicit target. It does not take the daemon lock, reconcile/sweep other work,
// register a rental, change the Hub or start this computer's machine.
func startEndpointController(ctx *Context, path string) (func(), *exit.Error) {
	ep, err := machineendpoint.Read(path)
	if err != nil {
		return nil, exit.New(exit.Validation, "invalid explicit machine endpoint: %s", err)
	}
	return endpointController(ctx, ep)
}

func endpointController(ctx *Context, ep *machineendpoint.Endpoint) (func(), *exit.Error) {
	l := home.Paths(ctx.Cfg.Home)
	st, problem := records.Open(l.DB)
	if problem != nil {
		return nil, problem
	}
	runs := endpointRuns(ctx, ep, l, st)
	resolver, fleet := runs.resolver, runs.fleet
	owner, problem := orchestrator.Open(orchestrator.Options{StartMachineExecution: runs.Start, Cfg: ctx.Cfg, Layout: l, Store: st, Log: io.Discard})
	if problem != nil {
		runs.cancel()
		st.Close()
		return nil, problem
	}
	fleet.owner = owner
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		runs.cancel()
		owner.Close(0)
		st.Close()
		return nil, exit.Internalf("cannot open foreground controller: %s", err)
	}
	credential := secret.Mint() // Process-local API authority; never a persisted or printed key.
	server := api.New(api.Options{ScopedEndpoint: ep.Name(), MachineExecutions: runs, Orchestrator: owner, Cfg: ctx.Cfg, Creds: api.Credentials{CLI: credential}, Addr: listener.Addr().String(), Log: io.Discard, Packages: resolver})
	handler, problem := server.Handler()
	if problem != nil {
		listener.Close()
		runs.cancel()
		owner.Close(0)
		st.Close()
		return nil, problem
	}
	// This listener is private to the process-local client. Other requests cannot
	// borrow its target or cause another machine to wake.
	scoped := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		presented, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer ")
		if !ok || !credential.Equal(presented) {
			http.Error(w, "foreground controller authentication required", http.StatusUnauthorized)
			return
		}
		// A run of this target is observed and controlled by id, a request's or a job's.
		if id, ok := runIDOf(r.URL.Path); ok {
			row, problem := st.RequestByReference(id)
			if problem != nil || row == nil {
				http.Error(w, "foreground operation cannot observe or control another target", http.StatusForbidden)
				return
			}
			link, problem := st.MachineExecution(row.ID)
			if problem != nil || link == nil || link.MachineID != ep.Name() {
				http.Error(w, "foreground operation cannot observe or control another target", http.StatusForbidden)
				return
			}
		}
		_, run := runIDOf(r.URL.Path)
		submit := r.URL.Path == "/v1/requests" || r.URL.Path == "/v1/local/jobs"
		if r.Method == http.MethodPost && !submit && !run || r.Method == http.MethodDelete && r.URL.Path == "/v1/local/daemon" {
			http.Error(w, "foreground operation is scoped to this request lifecycle", http.StatusForbidden)
			return
		}
		handler.ServeHTTP(w, r)
	})
	httpServer := &http.Server{Handler: scoped}
	go func() { _ = httpServer.Serve(listener) }()
	ctx.endpoint = ep
	ctx.foregroundClient = localapi.At(ctx.Cfg, listener.Addr().String(), credential)
	return func() {
		// Observer teardown cannot cancel a durable machine execution.
		runs.cancel()
		owner.Close(0)
		_ = httpServer.Close()
		st.Close()
		ctx.foregroundClient = nil
		ctx.endpoint = nil
	}, nil
}

// endpointRuns are the machine runs of one explicit target and nothing else: no other machine,
// this computer's included, is reached through them.
func endpointRuns(ctx *Context, ep *machineendpoint.Endpoint, l home.Layout, st *records.Store) *machineRuns {
	background := *ctx
	background.Out = io.Discard // API progress remains normal typed lifecycle events.
	fleet := &managedRentals{ctx: &background, layout: l, store: st}
	found := &machines.Resolver{Host: machines.NewHost(l.Machine, ctx.Cfg.TensorFSRoot, nil), Hub: func(origin string) *hub.Client { return client(ctx.forHub(origin)) }, Endpoint: func(name string) (*machineendpoint.Endpoint, *exit.Error) {
		if name == ep.Name() {
			return ep, nil
		}
		return nil, exit.Named(exit.Unavailable, "machine.endpoint_scope", "this foreground operation cannot access another machine")
	}}
	found.EndpointRental = rentalEndpoint(l, st)
	found.RentalHub = func(id string) *hub.Client { return client(fleet.atRental(id)) }
	found.Only = ep.Name()
	return newMachineRuns(&background, l, st, NewResolver(st, ctx.Cfg), fleet, found)
}

// endpointForRecordedRun keeps watch/cancel on the recorded target after the
// selector file or original observer is gone. The existing daemon remains usable.
func endpointForRecordedRun(ctx *Context, id string) (func(), *exit.Error) {
	st, problem := records.Open(filepath.Join(ctx.Cfg.Home, "creator.sqlite"))
	if problem != nil {
		return nil, problem
	}
	defer st.Close()
	row, problem := st.RequestByReference(id)
	if problem != nil || row == nil {
		return nil, problem
	}
	link, problem := st.MachineExecution(row.ID)
	if problem != nil || link == nil {
		return nil, problem
	}
	if !machineendpoint.IsName(link.MachineID) {
		return nil, nil
	}
	ep, problem := st.RequestMachineEndpoint(row.ID, link.MachineID)
	if problem != nil {
		return nil, problem
	}
	if ep == nil {
		return nil, exit.New(exit.NotFound, "recorded endpoint is absent")
	}
	return endpointController(ctx, ep)
}

// rentalEndpoint answers the rental an explicit endpoint reaches (its recorded worker or
// address), with the creator key its machine authorizes, not this computer's machine owner
// key. Nil for an endpoint that is no rental of this host.
func rentalEndpoint(l home.Layout, st *records.Store) func(*machineendpoint.Endpoint) (*machines.EndpointRental, *exit.Error) {
	return func(ep *machineendpoint.Endpoint) (*machines.EndpointRental, *exit.Error) {
		rows, problem := st.Rentals()
		if problem != nil {
			return nil, problem
		}
		for _, row := range rows {
			if row.ExpectedWorkerID != "" && row.ExpectedWorkerID == ep.WorkerID || row.Address != "" && row.Address == ep.Address {
				key, problem := rental.CreatorIdentityFor(l, row.ID)
				if problem != nil {
					return nil, problem
				}
				return &machines.EndpointRental{ID: row.ID, Key: key}, nil
			}
		}
		return nil, nil
	}
}

// runIDOf is the run a path observes or controls: `/v1/requests/<id>/…` or
// `/v1/local/jobs/<id>/…`.
func runIDOf(path string) (string, bool) {
	for _, prefix := range []string{"/v1/requests/", "/v1/local/jobs/"} {
		if rest, ok := strings.CutPrefix(path, prefix); ok && rest != "" {
			return strings.Split(rest, "/")[0], true
		}
	}
	return "", false
}

// foregroundRental is a named rental's machine as an explicit endpoint, for a run the running
// daemon cannot carry (it predates cozy.machine.v1): nil when no daemon runs (this CLI starts
// its own) or the daemon runs v1 work itself.
func foregroundRental(ctx *Context, name string) (*machineendpoint.Endpoint, *exit.Error) {
	state := daemon.Probe(ctx.Cfg)
	if !state.Up || state.OperatorOwned {
		return nil, nil
	}
	c, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return nil, problem
	}
	if caps, problem := c.Capabilities(); problem != nil || caps.MachineV1 {
		return nil, problem
	}
	return rentalMachineEndpoint(ctx, name)
}

// rentalMachineEndpoint is a rental's machine as this command reaches it itself, by the
// address and pinned certificate its record holds.
func rentalMachineEndpoint(ctx *Context, name string) (*machineendpoint.Endpoint, *exit.Error) {
	st, problem := records.Open(home.Paths(ctx.Cfg.Home).DB)
	if problem != nil {
		return nil, problem
	}
	defer st.Close()
	row, problem := st.RentalByMachine(name)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.Address == "" || row.CertPath == "" || row.ExpectedWorkerID == "" {
		return nil, exit.Named(exit.Unavailable, "rental.not_attached", "rental %s has no machine this host can reach", name)
	}
	leaf, err := os.ReadFile(row.CertPath)
	if err != nil {
		return nil, exit.New(exit.Credential, "rental %s's pinned certificate is unreadable: %s", name, err)
	}
	ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: row.Address, WorkerID: row.ExpectedWorkerID,
		WorkerBootID: cmp.Or(row.ExpectedWorkerBootID, row.ID), CertificatePEM: string(leaf), WorkspaceID: row.ID}
	if err := ep.Validate(); err != nil {
		return nil, exit.New(exit.Validation, "rental %s is not reachable as a machine endpoint: %s", name, err)
	}
	return ep, nil
}
