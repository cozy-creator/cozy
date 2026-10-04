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
	background := *ctx
	background.Out = io.Discard // API progress remains normal typed lifecycle events.
	resolver := NewResolver(st, ctx.Cfg)
	fleet := &managedRentals{ctx: &background, layout: l, store: st}
	found := &machines.Resolver{Host: machines.NewHost(l.Machine, ctx.Cfg.TensorFSRoot, nil), Hub: func(origin string) *hub.Client { return client(ctx.forHub(origin)) }, Endpoint: func(name string) (*machineendpoint.Endpoint, *exit.Error) {
		if name == ep.Name() {
			return ep, nil
		}
		return nil, exit.Named(exit.Unavailable, "machine.endpoint_scope", "this foreground operation cannot access another machine")
	}}
	found.EndpointRented = rentedEndpoint(st)
	found.Only = ep.Name() // a foreground run never reaches this computer's machine
	runs := newMachineRuns(&background, l, st, resolver, fleet, found)
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

// rentedEndpoint reports whether an explicit endpoint is one of this host's own rentals (its
// recorded worker or address): such a machine reads its own Hub as the pod. A rental another
// account shared with this one is not, so its runs carry this account's own Hub access.
func rentedEndpoint(st *records.Store) func(*machineendpoint.Endpoint) (bool, *exit.Error) {
	return func(ep *machineendpoint.Endpoint) (bool, *exit.Error) {
		rows, problem := st.Rentals()
		for _, row := range rows {
			if row.ExpectedWorkerID != "" && row.ExpectedWorkerID == ep.WorkerID || row.Address != "" && row.Address == ep.Address {
				return true, problem
			}
		}
		return false, problem
	}
}

// sharedRental is a rental another account shared with this one, as an explicit endpoint. This
// host never bought it and holds no record of it; the Hub names its machine, which admits this
// install's key from the renter's share. Nil when name is one of this host's own rentals or
// no shared rental.
func sharedRental(ctx *Context, name string) (*machineendpoint.Endpoint, *exit.Error) {
	st, problem := records.Open(home.Paths(ctx.Cfg.Home).DB)
	if problem != nil {
		return nil, problem
	}
	defer st.Close()
	if row, problem := st.RentalByMachine(name); problem != nil || row != nil {
		return nil, problem
	}
	if row, problem := st.RentalRow(name); problem != nil || row != nil {
		return nil, problem
	}
	hctx, cancel := hub.Context()
	rentals, problem := client(ctx).Rentals(hctx)
	cancel()
	if problem != nil {
		return nil, problem
	}
	for _, shared := range rentals {
		if !shared.Shared || (shared.ID != name && shared.Name != name) {
			continue
		}
		if !shared.Ready() {
			return nil, exit.Named(exit.Unavailable, "rental.not_ready", "shared rental %s is %s", name, shared.State)
		}
		ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: shared.Address, WorkerID: shared.WorkerID,
			WorkerBootID: shared.WorkerBootID, CertificatePEM: shared.CertPEM, WorkspaceID: shared.ID}
		if err := ep.Validate(); err != nil {
			return nil, exit.New(exit.Validation, "shared rental %s is not reachable as a machine endpoint: %s", name, err)
		}
		return ep, nil
	}
	return nil, nil
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
