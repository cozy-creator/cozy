package cli

import (
	"fmt"
	"net/http"
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
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
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

	// THE HOST RUNTIME IS ADMITTED BEFORE THE DAEMON EXISTS. A cozy-runtime on PATH that
	// speaks an older wire minor cannot bring a worker READY, and a daemon that starts over
	// one queues every local run silently (cl-086). A host with NO tool starts: rentals and
	// the hub need none, and the launch itself refuses `host_runtime_missing` by name.
	if _, e := launch.HostRuntime(ctx.Cfg.Tool()); e != nil && e.ErrName() != "host_runtime_missing" {
		return e
	}

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
	// The held daemon lock proves any prior handoff is stale. Only the winning
	// daemon removes it, so concurrent auto-start callers cannot erase a live token.
	_ = os.Remove(l.Client)

	st, e := records.Open(l.DB)
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
	resolver := NewResolver(st, ctx.Cfg, orchestrator.LocalDeviceEnvelope())
	// Two questions, deliberately not one object: the API resolves the persisted,
	// non-secret attempt control (so a bad pin refuses before a request row), and the
	// orchestrator resolves the dial triple at dial time. Only the second reads the token.
	rentals := rental.Resolver(l, st)
	knownRentals := rental.Known(st)

	// The two per-daemon identity digests every local InvocationSpec rides (cl-022's
	// guard): left unset, dispatch froze empty strings into every persisted invocation.
	configDigest := localConfigDigest(ctx.Cfg)
	fleet := &managedRentals{ctx: ctx, layout: l, store: st}
	transfers := newModelTransferOwner(ctx.Cfg, st, ctx.Out, ctx.AccountAuth)
	privateWheels := newPrivateWheelOwner(ctx.Cfg, ctx.Out, ctx.AccountAuth)
	defects := newDefectReporter(ctx.Cfg, ctx.Out, ctx.AccountAuth)
	c, e := orchestrator.Open(orchestrator.Options{
		Cfg: ctx.Cfg, Layout: l, Store: st, Yield: yield, Log: ctx.Out,
		Packages: resolver, Rentals: rentals, ObserveRental: rental.ObserveWorker(st),
		RentalClaimProof: rental.ClaimProof(l), RentalPackageSet: rental.PackageSetSigner(l),
		RentalFleet: fleet.status, AcquireManagedRental: fleet.acquire,
		ReleaseManagedRental: fleet.release,
		ModelTransfers:       transfers,
		PrivateWheels:        privateWheels.grants,
		ReportReleaseDefect:  defects.report,
		ConfigDigest:         configDigest,
	})
	if e != nil {
		closeListeners()
		return e
	}
	fleet.owner = c
	killed, forgotten, e := c.Reconcile()
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

	// One per-launch CLI credential is handed over through a 0600 file. It is never
	// printed, logged, or placed on argv, and dies with this process. The public web stub
	// exposes no state; full browser authorization is a later, separately reviewed door.
	creds, e := api.Mint(l)
	if e != nil {
		closeListeners()
		return e
	}
	defer os.Remove(l.Client)

	// The stop channel exists before the server so the shutdown route can feed it: the
	// route is the ask a platform with no process signal still has (#449), and it takes
	// exactly the path a SIGTERM takes.
	stop := make(chan os.Signal, 1)
	server := api.New(api.Options{
		Orchestrator: c, Cfg: ctx.Cfg, Creds: creds, Addr: addr,
		Log: ctx.Out, Web: cozyweb.Handler(), Packages: resolver, Rentals: knownRentals,
		Shutdown: func() { stop <- syscall.SIGTERM },
	})
	handler, e := server.Handler()
	if e != nil {
		closeListeners()
		return e
	}

	fmt.Fprintf(ctx.Out, "Cozy daemon up: api %s (%s, loopback only) · worker socket %s\n",
		addr, strings.Join(bound, "+"), socket)
	fmt.Fprintf(ctx.Out, "  install sweep: reclaimed %d of %d director(ies), freed %s exclusive%s\n",
		swept.Removed, swept.Scanned, output.Bytes(swept.Bytes), sweepNote)
	fmt.Fprintf(ctx.Out, "  records %s · yield %s · reconcile killed %d orphan(s), forgot %d stale row(s)\n",
		l.DB, yield, killed, forgotten)
	fmt.Fprintf(ctx.Out, "  client credential %s (%s, mode 0600)\n", creds.CLI.Digest(), l.Client)
	if ctx.Cfg.RentalsIdleRelease > 0 {
		fmt.Fprintf(ctx.Out, "  rentals: released after %s with nothing queued, running, or owed (rentals.idle_release_s)\n",
			ctx.Cfg.RentalsIdleRelease)
	} else {
		fmt.Fprintln(ctx.Out, "  rentals: never idle-released (rentals.idle_release_s=0); each ends with cozy rental end")
	}
	if ctx.Cfg.DaemonIdleShutdown > 0 {
		fmt.Fprintf(ctx.Out, "  next: cozy run list · stop with cozy down · exits on its own after %s with nothing to manage\n",
			ctx.Cfg.DaemonIdleShutdown)
	} else {
		fmt.Fprintf(ctx.Out, "  next: cozy run list · stop with cozy down\n")
	}

	go func() { _ = http.Serve(v4, handler) }()
	if v6 != nil {
		go func() { _ = http.Serve(v6, handler) }()
	}
	go func() { _ = c.Serve() }()

	// The idle exit takes exactly the path a SIGTERM takes: the server's shutdown hook
	// feeds the same channel, and everything after `<-stop` is shared.
	quit := make(chan struct{})
	go fleet.watch(quit)
	if ctx.Cfg.DaemonIdleShutdown > 0 {
		go idleWatch{debounce: ctx.Cfg.DaemonIdleShutdown, store: st, owner: c,
			server: server, log: ctx.Out}.run(quit)
	}

	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	close(quit)
	fmt.Fprintln(ctx.Out, "draining package processes…")
	closeListeners()
	fleet.close()
	c.Close(orchestrator.StopGrace)
	return nil
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
