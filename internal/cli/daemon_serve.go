package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	machineset "github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	cozyweb "github.com/cozy-creator/cozy/web"
)

// serveDaemon is the private process entrypoint shared by explicit `up` and
// commands that ensure the daemon is running.
func serveDaemon(ctx *Context) *exit.Error {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return e
	}
	port := ctx.Cfg.Port
	if v := ctx.Inv.Value("--port"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n < 0 || n > 65535 {
			return exit.Usagef("--port %q is not a port number or zero for automatic selection", v)
		}
		port = n
	}
	yield := ctx.Inv.Value("--yield")
	switch yield {
	case "", "smart", "always", "never":
	default:
		return exit.Usagef("--yield %q is not one of smart|always|never", yield)
	}

	// Already running is idempotent 0 printing the live status.
	if st := daemon.Probe(ctx.Cfg); st.Up {
		return emit(ctx, output.Record{Fields: []output.Field{
			{K: "daemon", V: "running"}, {K: "address", V: st.Addr},
			{K: "socket", V: st.Socket}, {K: "pid", V: st.PID}, {K: "since", V: st.Since},
		}, Notes: []string{"already running: `cozy run list` is idempotent"}})
	}

	// Arm shutdown before publishing the daemon or starting owned work. Clients
	// can send SIGTERM as soon as the lock/credential or API becomes visible.
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(stop)

	socket := l.Root + "/worker.sock"

	// LOOPBACK-ONLY, and the bind happens before anything durable does. internal/api
	// owns the only net.Listen("tcp", …) in this binary; there is no flag that widens
	// it, because the LAN door is deferred behind TLS and its own threat review.
	listen := api.Listeners
	if ctx.Cfg.PortSource == "default" && port == config.DefaultPort {
		listen = api.PreferredListeners
	}
	v4, v6, addr, e := listen(port)
	if e != nil {
		return e
	}
	bound := []string{"ipv4"}
	if v6 != nil {
		bound = append(bound, "ipv6")
	}
	closeListeners := func() {
		v4.Close()
		if v6 != nil {
			v6.Close()
		}
	}

	// The claim: a second daemon on this root fails here. It comes after the bind so a
	// port conflict is reported as a port conflict, and before anything is written.
	held, e := daemon.Hold(l, addr, socket)
	if e != nil {
		closeListeners()
		return e
	}
	defer held.Release()

	st, e := records.OpenForDaemon(l.DB)
	if e != nil {
		closeListeners()
		return e
	}
	defer st.Close()

	// The resolver is built BEFORE the orchestrator, because the orchestrator holds it:
	// select-or-start is the scheduler's act, and a request whose binding no live worker
	// advertises makes one rather than queueing for capacity nothing would create.
	// The device envelope every local worker is granted; its reported lanes are ordinals
	// into it, and dispatch draws a seat from the placement's lane (proto-024).
	resolver := NewResolver(st, ctx.Cfg)
	// Two questions, deliberately not one object: the API resolves the persisted,
	// non-secret attempt control (so a bad pin refuses before a request row), and the
	// orchestrator resolves the dial triple at dial time. Only the second reads the token.
	rentals := rental.Resolver(l, st)
	knownRentals := rental.Known(st)

	fleet := &managedRentals{ctx: ctx, layout: l, store: st}
	// The daemon starts this computer's machine as `cozy machine start` does.
	localMachine, problem := localMachineHost(ctx)
	if problem != nil {
		return problem
	}
	found := &machineset.Resolver{
		Endpoint: st.MachineEndpoint,
		Host:     localMachine, HubOrigin: ctx.Cfg.HubURL,
		Hub:     func(origin string) *hub.Client { return client(ctx.forHub(origin)) },
		Rentals: rentals, RentalHub: func(id string) *hub.Client { return client(fleet.atRental(id)) },
		UseRental:      func(id, holder string) (func(), *exit.Error) { return fleet.owner.UseRental(id, holder) },
		RentalKey:      func(id string) (rental.CreatorIdentity, *exit.Error) { return rental.CreatorIdentityFor(l, id) },
		EndpointRental: rentalEndpoint(l, st),
		Log:            ctx.Out,
	}
	machines := newMachineRuns(ctx, l, st, resolver, fleet, found)
	defer machines.cancel()
	updates := &rentalRuntimeUpdates{machines: machines}
	machines.updates = updates
	fleet.boot = updates.Boot
	transfers := NewModelTransferOwner(ctx.Cfg, st, ctx.Out, ctx.AccountAuth)
	c, e := orchestrator.Open(orchestrator.Options{
		StartMachineExecution: machines.Start,
		Cfg:                   ctx.Cfg, Layout: l, Store: st, Log: ctx.Out,
		Rentals:        rentals,
		ModelTransfers: transfers,
		ReclaimInstall: func(id string) *exit.Error {
			// Background cleanup and editable refresh are mutations by this same daemon.
			// Serialize them before the cross-process writer claim so cleanup cannot
			// make an otherwise valid submission fail with a writer conflict.
			resolver.refreshMu.Lock()
			defer resolver.refreshMu.Unlock()
			writer, problem := install.Lock(l)
			if problem != nil {
				return problem
			}
			defer writer.Unlock()
			_, problem = install.Reclaim(l, st, id)
			return problem
		},
	})
	if e != nil {
		closeListeners()
		return e
	}
	fleet.owner = c
	fleet.forget = found.Forget
	// Rental readiness is a queue-capacity edge. The fleet reconciler wakes
	// pinned machine executions as soon as the hub publishes an attachable
	// worker, including after a daemon restart.
	fleet.wakeQueue = c.WakeQueue
	installContext, cancelInstalls := context.WithCancel(context.Background())
	installs := machineset.NewInstalls(st, machines.Prewarm, ctx.Out)
	fleet.installs = installs
	installsStopped := make(chan struct{})
	defer cancelInstalls()

	e = c.Reconcile()
	if e != nil {
		closeListeners()
		return e
	}
	go fleet.releaseOrphaned()

	// cl-076: an install directory is reachable only through its record, so a records.db
	// that was lost or rebuilt strands every one of them on disk with no verb able to
	// touch it. The sweep runs here — after reconcile has closed the worker rows that
	// would otherwise still reference an install, and before the API is served — and it
	// is never fatal: what it removes is a venv `cozy package install` rebuilds.
	swept, sweepNote := sweepInstalls(l, st)
	// The two roots that used to grow without bound. The writer removes its own bytes
	// when they die; these sweeps are the backstop for a crashed one, and each lets the
	// records authority — or a live process's kernel lock — decide what still has a claim.
	workers, workersNote := reclaimNote(reclaim.Workers(l, st))
	tmp, tmpNote := reclaimNote(reclaim.Tmp(l, st))
	publications, publicationsNote := reclaimNote(reclaim.Publications(l, st))
	rentalSecrets, rentalSecretsNote := reclaimNote(reclaim.RentalSecrets(l, st))
	reclaim.EmptyRoots(l)

	// One per-launch CLI credential rides the daemon's own held 0600 record. It is never
	// printed, logged, or placed on argv, and rotates with every launch. The public web
	// stub exposes no state; full browser authorization is a later, separately reviewed
	// door.
	creds, e := api.Mint(l)
	if e != nil {
		closeListeners()
		return e
	}

	server := api.New(api.Options{
		RuntimeUpdate:       updates.Start,
		RentalKeepalive:     fleet.keepalive,
		RentalInstall:       installs.Accept,
		RentalInstallStatus: installs.Status,
		MachineExecutions:   machines,
		Orchestrator:        c, Cfg: ctx.Cfg, Creds: creds, Addr: addr,
		Log: ctx.Out, Web: cozyweb.Handler(), Packages: resolver, Rentals: knownRentals,
		RentalInventory: func(hub string, allHubs, reconcile bool) (api.RentalInventory, *exit.Error) {
			return readRentalInventory(st, fleet, hub, allHubs, reconcile)
		},
		Shutdown: func() { stop <- syscall.SIGTERM },
	})
	handler, e := server.Handler()
	if e != nil {
		closeListeners()
		return e
	}
	updates.Resume()
	machines.Resume()
	go func() { defer close(installsStopped); installs.Run(installContext) }()
	defer func() { cancelInstalls(); <-installsStopped }()

	fmt.Fprintf(ctx.Out, "Cozy daemon up: api %s (%s, loopback only) · worker socket %s\n",
		addr, strings.Join(bound, "+"), socket)
	fmt.Fprintf(ctx.Out, "  install sweep: reclaimed %d of %d director(ies), freed %s exclusive%s\n",
		swept.Removed, swept.Scanned, output.Bytes(swept.Bytes), sweepNote)
	fmt.Fprintf(ctx.Out, "  worker sweep: reclaimed %d of %d director(ies), freed %s%s\n",
		workers.Removed, workers.Scanned, output.Bytes(workers.Bytes), workersNote)
	fmt.Fprintf(ctx.Out, "  tmp sweep: reclaimed %d of %d entr(y|ies), freed %s%s\n",
		tmp.Removed, tmp.Scanned, output.Bytes(tmp.Bytes), tmpNote)
	if publications.Scanned+rentalSecrets.Scanned > 0 {
		fmt.Fprintf(ctx.Out, "  lifecycle sweep: %d publication root(s) settled%s · %d stale rental secret(s) erased%s\n",
			publications.Removed, publicationsNote, rentalSecrets.Removed, rentalSecretsNote)
	}
	fmt.Fprintf(ctx.Out, "  records %s · yield %s\n", l.DB, yield)
	fmt.Fprintf(ctx.Out, "  client credential %s (carried in %s, mode 0600)\n", creds.CLI.Digest(), l.Daemon)
	fmt.Fprintln(ctx.Out, "  rentals: released after 15 minutes without active work; manual cozy rental keepalive resets once")
	if ctx.Cfg.DaemonIdleShutdown > 0 {
		fmt.Fprintf(ctx.Out, "  next: cozy run list · stop with cozy down · exits on its own after %s with nothing to manage\n",
			ctx.Cfg.DaemonIdleShutdown)
	} else {
		fmt.Fprintf(ctx.Out, "  next: cozy run list · stop with cozy down\n")
	}
	fmt.Fprintf(ctx.Out, "  claim: %s; this daemon stops if that record stops naming it\n", l.Daemon)

	httpServer := daemon.NewHTTPServer(handler)
	go func() { _ = httpServer.Serve(v4) }()
	if v6 != nil {
		go func() { _ = httpServer.Serve(v6) }()
	}
	go func() { _ = c.Serve() }()

	// The idle exit takes exactly the path a SIGTERM takes: the server's shutdown hook
	// feeds the same channel, and everything after `<-stop` is shared.
	quit := make(chan struct{})
	go fleet.watch(quit)
	// Every editable install's source tree is watched for as long as this daemon runs
	// (cl-097): an edit is rebuilt and every worker holding the package re-prepared before
	// the next run asks. A watch that cannot be set up is reported, not fatal — runs still
	// refresh at submission.
	if _, e := startEditableSync(l, st, resolver, ctx.Out, quit); e != nil {
		fmt.Fprintf(ctx.Out, "editable watch unavailable: %s\n", e.Message)
	}
	idle := idleWatch{debounce: ctx.Cfg.DaemonIdleShutdown, store: st, owner: c,
		server: server, log: ctx.Out}
	if ctx.Cfg.DaemonIdleShutdown > 0 {
		go idle.run(quit)
	}
	// Unconditional, and deliberately not behind `daemon.idle_shutdown_s`: a root that
	// stops carrying this daemon's claim has taken away every way to reach it or stop it.
	go claimWatch{held: held, root: l.Root, managed: idle.managed, log: ctx.Out,
		stop: func() { stop <- syscall.SIGTERM }}.run(quit)

	<-stop
	close(quit)
	cancelInstalls()
	<-installsStopped
	// Disconnect uploads/observation immediately. Runtime owns accepted execution;
	// canceling this client context never sends an execution-cancel command.
	machines.cancel()
	fmt.Fprintln(ctx.Out, "draining package processes…")
	drain, cancelDrain := context.WithTimeout(context.Background(), orchestrator.StopGrace)
	if err := httpServer.Shutdown(drain); err != nil {
		fmt.Fprintf(ctx.Out, "HTTP response drain incomplete: %s\n", err)
	}
	cancelDrain()
	fleet.close()
	c.Close(orchestrator.StopGrace)
	return nil
}

// reclaimNote turns a reclaim sweep's first refusal into a banner note; like the install
// sweep, nothing here may keep the daemon down.
func reclaimNote(swept reclaim.Swept, problem *exit.Error) (reclaim.Swept, string) {
	if problem != nil {
		return swept, " · incomplete: " + problem.Message
	}
	return swept, ""
}

// sweepInstalls takes the single-writer lock the install transaction takes, sweeps, and
// answers a note for the startup banner instead of an error. Every outcome here is
// recoverable by reinstalling one package, so none of them may keep the daemon down: a
// concurrent writer defers the sweep to the next start, and a directory that refuses to
// go is reported and left where it is.
func sweepInstalls(l home.Layout, st *records.Store) (install.Swept, string) {
	writer, problem := install.Lock(l)
	if problem != nil {
		return install.Swept{}, " · deferred: " + problem.Message
	}
	defer writer.Unlock()
	swept, problem := install.Sweep(l, st)
	if problem != nil {
		return swept, " · incomplete: " + problem.Message
	}
	return swept, ""
}
