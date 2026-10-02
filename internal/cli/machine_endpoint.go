package cli

import (
	"io"
	"net"
	"net/http"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
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
		if strings.HasPrefix(r.URL.Path, "/v1/requests/") {
			id := strings.Split(strings.TrimPrefix(r.URL.Path, "/v1/requests/"), "/")[0]
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
		if r.Method == http.MethodPost && r.URL.Path != "/v1/requests" && !strings.HasPrefix(r.URL.Path, "/v1/requests/") || r.Method == http.MethodDelete && r.URL.Path == "/v1/local/daemon" {
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
