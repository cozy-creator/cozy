package app

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	localapi "github.com/cozy-creator/cozy-creator-v2/internal/client"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/rental"
	"github.com/cozy-creator/cozy-creator-v2/internal/service"
)

// `cozy up` / `cozy down`: the LocalService's own lifecycle (cl-001). The endpoint-process
// verbs (`start`, `stop`, `run`, `logs`) are clients of the running service and land with
// cl-010 — two process levels, never conflated.

func handleUp(ctx *Context) *exit.Error {
	l, e := home.Open(ctx.Cfg.Home)
	if e != nil {
		return e
	}
	port := ctx.Cfg.Port
	if v := ctx.Inv.Value("--port"); v != "" {
		n, err := strconv.Atoi(v)
		if err != nil || n <= 0 || n > 65535 {
			return exit.Usagef("--port %q is not a port number", v)
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
	if st := service.Probe(ctx.Cfg); st.Up {
		return emit(ctx, render.Record{Kind: "service", Fields: []render.Field{
			{K: "service", V: "up"}, {K: "address", V: st.Addr},
			{K: "socket", V: st.Socket}, {K: "pid", V: st.PID}, {K: "since", V: st.Since},
		}, Notes: []string{"already running: `cozy up` is idempotent"}})
	}

	if ctx.Inv.Bool("--detach") {
		return detach(ctx, port, yield)
	}

	socket := l.Root + "/worker.sock"

	// LOOPBACK-ONLY, and the bind happens before anything durable does. internal/api
	// owns the only net.Listen("tcp", …) in this binary; there is no flag that widens
	// it, because the LAN door is deferred behind TLS and its own threat review.
	v4, v6, addr, e := api.Listeners(port)
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

	// The claim: a second service on this root fails here. It comes after the bind so a
	// port conflict is reported as a port conflict, and before anything is written.
	held, e := service.Hold(l, addr, socket)
	if e != nil {
		closeListeners()
		return e
	}
	defer held.Release()

	st, e := records.Open(l.DB)
	if e != nil {
		closeListeners()
		return e
	}
	defer st.Close()

	// The resolver is built BEFORE the orchestrator, because the orchestrator holds it:
	// select-or-start is the scheduler's act, and a request whose binding no live worker
	// advertises makes one rather than queueing for capacity nothing would create.
	resolver := NewResolver(st, ctx.Cfg)
	// Two questions, deliberately not one object: the API resolves the persisted,
	// non-secret attempt control (so a bad pin refuses before a request row), and the
	// orchestrator resolves the dial triple at dial time. Only the second reads the token.
	rentals := rental.Resolver(l, st)

	c, e := orchestrator.Open(orchestrator.Options{
		Cfg: ctx.Cfg, Layout: l, Store: st, Yield: yield, Log: ctx.Out,
		Endpoints: resolver, Rentals: rentals,
	})
	if e != nil {
		closeListeners()
		return e
	}
	killed, forgotten, e := c.Reconcile()
	if e != nil {
		closeListeners()
		return e
	}

	// Two per-launch credentials: one for the browser (handed over in the `--open` URL's
	// FRAGMENT) and one for CLI clients (handed over through a 0600 file). Neither is
	// ever printed, logged, or placed on argv, and both die with this process.
	creds, e := api.Mint(l)
	if e != nil {
		closeListeners()
		return e
	}

	// The stop channel exists before the server so the shutdown route can feed it: the
	// route is the ask a platform with no process signal still has (#449), and it takes
	// exactly the path a SIGTERM takes.
	stop := make(chan os.Signal, 1)
	server := api.New(api.Options{
		Orchestrator: c, Cfg: ctx.Cfg, Creds: creds, Addr: addr,
		Log: ctx.Out, Endpoints: resolver, Bound: bound, Rentals: rental.Known(st),
		Shutdown: func() { stop <- syscall.SIGTERM },
	})
	handler, e := server.Handler()
	if e != nil {
		closeListeners()
		return e
	}

	fmt.Fprintf(ctx.Out, "cozy LocalService up: api %s (%s, loopback only) · worker socket %s\n",
		addr, strings.Join(bound, "+"), socket)
	fmt.Fprintf(ctx.Out, "  records %s · yield %s · reconcile killed %d orphan(s), forgot %d stale row(s)\n",
		l.DB, yield, killed, forgotten)
	fmt.Fprintf(ctx.Out, "  client credential %s (%s, mode 0600)\n", creds.CLI.Digest(), l.Client)
	fmt.Fprintf(ctx.Out, "  next: cozy status · stop with cozy down\n")

	go func() { _ = http.Serve(v4, handler) }()
	if v6 != nil {
		go func() { _ = http.Serve(v6, handler) }()
	}
	go func() { _ = c.Serve() }()

	if ctx.Inv.Bool("--open") {
		openStub(ctx, creds, addr)
	}

	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	fmt.Fprintln(ctx.Out, "draining endpoint processes…")
	c.Close(orchestrator.StopGrace)
	closeListeners()
	return nil
}

// open hands the browser the stub page with the per-launch credential in the URL's
// FRAGMENT. The URL is never printed in full and never logged — the fragment IS the
// credential, and a token in a terminal scrollback is a token in a terminal scrollback.
func openStub(ctx *Context, creds api.Credentials, addr string) {
	if !api.StubReachable() {
		fmt.Fprintln(ctx.Out, "  --open: this build embeds no stub page")
		return
	}
	url := creds.OpenURL(addr)
	cmd := exec.Command("xdg-open", url)
	cmd.Env = ctx.Cfg.Tool()
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(ctx.Out, "  --open: no browser could be launched (%s)\n", err)
		return
	}
	go func() { _ = cmd.Wait() }()
	fmt.Fprintf(ctx.Out, "  --open: the stub page was opened with the browser credential %s\n",
		creds.Browser.Digest())
}

// detach re-runs this binary as the service in its own session, then waits for the
// LOCK to be held — the same proof every other reader uses, never a sleep.
func detach(ctx *Context, port int, yield string) *exit.Error {
	args := []string{"up", "--port", strconv.Itoa(port)}
	if yield != "" {
		args = append(args, "--yield", yield)
	}
	self, err := os.Executable()
	if err != nil {
		return exit.Internalf("cannot find this binary: %s", err)
	}
	logPath := ctx.Cfg.Home + "/service.log"
	logFile, err := os.Create(logPath)
	if err != nil {
		return exit.Internalf("cannot open %s: %s", logPath, err)
	}
	defer logFile.Close()
	cmd := exec.Command(self, args...)
	cmd.Env = ctx.Cfg.Child("COZY_HOME=" + ctx.Cfg.Home)
	cmd.Stdout, cmd.Stderr = logFile, logFile
	detachProcess(cmd)
	if err := cmd.Start(); err != nil {
		return exit.Internalf("cannot start the detached LocalService: %s", err)
	}
	deadline := time.Now().Add(15 * time.Second)
	for time.Now().Before(deadline) {
		if st := service.Probe(ctx.Cfg); st.Up {
			return emit(ctx, render.Record{Kind: "service", Fields: []render.Field{
				{K: "service", V: "up"}, {K: "address", V: st.Addr},
				{K: "socket", V: st.Socket}, {K: "pid", V: st.PID}, {K: "log", V: logPath},
			}, Next: []string{"cozy status"}})
		}
		if cmd.ProcessState != nil {
			break
		}
		time.Sleep(50 * time.Millisecond)
	}
	return exit.New(exit.Conflict, "the detached LocalService did not take the service lock").
		WithRemedy("its log is %s", logPath)
}

// handleDown is the three-tier stop (#449): ask cooperatively, wait under a bounded
// deadline, and only then force. The ask has two spellings — the authenticated shutdown
// route (exists on every platform), and the process SIGTERM as the fallback where the
// route cannot be reached (a wedged API on a platform that has signals). Windows has no
// cooperative signal, so there the route is the only ask and `signalProcess` says so.
func handleDown(ctx *Context) *exit.Error {
	st := service.Probe(ctx.Cfg)
	if !st.Up {
		return emit(ctx, render.Record{Kind: "service",
			Fields: []render.Field{{K: "service", V: "down"}},
			Notes:  []string{"not running: `cozy down` is idempotent"}})
	}
	if st.PID <= 0 {
		return exit.Internalf("the service lock is held but names no pid")
	}
	// The wait bound is the caller's --timeout when given; otherwise it is DERIVED from
	// the service's own published cancellation budget (#434): workers drain in parallel
	// under one StopGrace, and the service's own teardown plus lock release is bounded by
	// the same budget again — so two StopGrace windows, not a number invented here.
	timeout := 2 * orchestrator.StopGrace
	if v := ctx.Inv.Value("--timeout"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return exit.Usagef("--timeout %q is not a duration", v)
		}
		timeout = d
	}
	asked := ""
	if cl, e := localapi.Open(ctx.Cfg, st); e == nil {
		if e := cl.ShutdownService(); e == nil {
			asked = "the shutdown route"
		}
	}
	if asked == "" {
		if err := signalProcess(st.PID, syscall.SIGTERM); err == nil {
			asked = "SIGTERM"
		}
	}
	if asked != "" {
		// The proof of exit is the LOCK becoming free, not the pid disappearing and not
		// a timer: the kernel drops the lock when the process dies, whatever killed it.
		deadline := time.Now().Add(timeout)
		for time.Now().Before(deadline) {
			if !service.Probe(ctx.Cfg).Up {
				return emit(ctx, render.Record{Kind: "service",
					Fields: []render.Field{{K: "service", V: "down"}, {K: "stopped_pid", V: st.PID}},
					Notes: []string{fmt.Sprintf(
						"asked over %s; the service lock is free again — the process is provably gone", asked)}})
			}
			time.Sleep(50 * time.Millisecond)
		}
	}
	// The forced tier. After a KILL the kernel frees the lock at process death, so a
	// further wait observes only that; one more StopGrace bounds it, and a process that
	// survives a KILL that long is REPORTED, not waited on forever.
	_ = signalProcess(st.PID, syscall.SIGKILL)
	hard := time.Now().Add(orchestrator.StopGrace)
	for time.Now().Before(hard) && service.Probe(ctx.Cfg).Up {
		time.Sleep(50 * time.Millisecond)
	}
	why := fmt.Sprintf("the cooperative drain over %s exceeded %s; the process was killed", asked, timeout)
	if asked == "" {
		why = "no cooperative channel could be reached (route unreachable, no signal on this platform); " +
			"the process was killed"
	}
	if service.Probe(ctx.Cfg).Up {
		return exit.Internalf("the LocalService (pid %d) survived a kill; its lock is still held", st.PID)
	}
	return emit(ctx, render.Record{Kind: "service",
		Fields: []render.Field{{K: "service", V: "down"}, {K: "stopped_pid", V: st.PID}},
		Notes:  []string{why}})
}
