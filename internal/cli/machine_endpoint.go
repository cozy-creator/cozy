package cli

import (
	"cmp"
	"context"
	"io"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/calcifer"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/rentalid"
	"github.com/cozy-creator/cozy/internal/secret"
)

// startEndpointController composes the existing business API and machine observer for
// one explicit target. It does not take the Calcifer lock, reconcile/sweep other work,
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
		// A run of this target is observed and controlled by its id.
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
		submit := r.URL.Path == "/v1/requests"
		if r.Method == http.MethodPost && !submit && !run || r.Method == http.MethodDelete && r.URL.Path == "/v1/local/calcifer" {
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
	found.EndpointRental = rentalEndpoint(ctx, l, st)
	found.RentalHub = func(id string) *hub.Client { return client(fleet.atRental(id)) }
	found.Only = ep.Name()
	runs := newMachineRuns(&background, l, st, NewResolver(st, ctx.Cfg), fleet, found)
	runs.foreground = true
	return runs
}

// endpointForRecordedRun keeps watch/cancel on the recorded target after the
// selector file or original observer is gone. The existing Calcifer remains usable.
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
	// Settled before anything starts: a run whose rental is known to have ended has nothing to reach.
	if problem := st.ReconcileEndedMachineExecution(row.ID); problem != nil {
		return nil, problem
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
func rentalEndpoint(ctx *Context, l home.Layout, st *records.Store) func(*machineendpoint.Endpoint) (*machines.EndpointRental, *exit.Error) {
	return func(ep *machineendpoint.Endpoint) (*machines.EndpointRental, *exit.Error) {
		// Workspace identity survives a missing inventory projection. A rental
		// signing key is custody evidence; endpoint state still comes from its Hub.
		if !rentalid.Valid(ep.WorkspaceID) {
			return nil, nil
		}
		if _, err := os.Stat(l.RentalCreatorIdentity(ep.WorkspaceID)); os.IsNotExist(err) {
			return nil, nil
		}
		row, problem := observedRental(ctx, st, ep.WorkspaceID)
		if problem != nil {
			return nil, problem
		}
		key, problem := rental.CreatorIdentityFor(l, row.ID)
		if problem != nil {
			return nil, problem
		}
		ep.Address, ep.CertificatePEM = row.Address, row.CertificatePEM
		ep.WorkerID, ep.WorkerBootID = row.ExpectedWorkerID, row.ExpectedWorkerBootID
		return &machines.EndpointRental{ID: row.ID, Key: key}, nil
	}
}

// runIDOf is the run a path observes or controls: `/v1/requests/<id>/…`.
func runIDOf(path string) (string, bool) {
	if rest, ok := strings.CutPrefix(path, "/v1/requests/"); ok && rest != "" {
		return strings.Split(rest, "/")[0], true
	}
	return "", false
}

// foregroundRental is a named rental's machine as an explicit endpoint, for a run the running
// Calcifer cannot carry (it predates cozy.machine.v1): nil when no Calcifer runs (this CLI starts
// its own) or Calcifer runs v1 work itself.
func foregroundRental(ctx *Context, name string) (*machineendpoint.Endpoint, *exit.Error) {
	state := calcifer.Probe(ctx.Cfg)
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
	live, problem := resolveLiveRental(ctx, st, name)
	if problem != nil {
		return nil, problem
	}
	row, problem := observedRental(ctx, st, live.Rental.ID)
	if problem != nil {
		return nil, problem
	}
	if row == nil || row.Address == "" || row.CertificatePEM == "" || row.ExpectedWorkerID == "" {
		return nil, exit.Named(exit.Unavailable, "rental.not_attached", "rental %s has no machine this host can reach", name)
	}
	ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: row.Address, WorkerID: row.ExpectedWorkerID,
		WorkerBootID: cmp.Or(row.ExpectedWorkerBootID, row.ID), CertificatePEM: row.CertificatePEM, WorkspaceID: row.ID}
	if err := ep.Validate(); err != nil {
		return nil, exit.New(exit.Validation, "rental %s is not reachable as a machine endpoint: %s", name, err)
	}
	return ep, nil
}

// awaitAcceptance follows a run until its machine's durable receipt, or until it settles. A
// signal detaches only this caller; it never cancels the run nor stands in for acceptance.
func awaitAcceptance(client *localapi.Client, life api.Lifecycle) (api.Lifecycle, *exit.Error) {
	wait, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	for life.MachineExecution != nil && !life.MachineExecution.Accepted && !invocationSettled(life.Status) {
		select {
		case <-wait.Done():
			return life, nil
		case <-time.After(200 * time.Millisecond):
		}
		next, problem := client.Request(life.RequestID)
		if problem != nil {
			return life, problem
		}
		life = next
	}
	return life, nil
}
