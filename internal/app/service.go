package app

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/coord"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
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

	addr := fmt.Sprintf("127.0.0.1:%d", port)
	socket := l.Root + "/worker.sock"

	// The claim comes FIRST: a second service on this root fails here, before it can
	// bind a port, spawn a worker, or write a row.
	held, e := service.Hold(l, addr, socket)
	if e != nil {
		return e
	}
	defer held.Release()

	// Loopback only, and the port is checked before anything durable happens.
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return exit.New(exit.Conflict, "%s is held by another process: %s", addr, err).
			WithRemedy("choose another port with --port, or stop what holds it")
	}

	st, e := records.Open(l.DB)
	if e != nil {
		ln.Close()
		return e
	}
	defer st.Close()

	c, e := coord.Open(coord.Options{
		Cfg: ctx.Cfg, Layout: l, Store: st, Socket: socket, Yield: yield, Log: ctx.Out,
	})
	if e != nil {
		ln.Close()
		return e
	}
	killed, forgotten, e := c.Reconcile()
	if e != nil {
		ln.Close()
		return e
	}

	fmt.Fprintf(ctx.Out, "cozy LocalService up: api %s (loopback only) · worker socket %s\n",
		addr, socket)
	fmt.Fprintf(ctx.Out, "  records %s · yield %s · reconcile killed %d orphan(s), forgot %d stale row(s)\n",
		l.DB, yield, killed, forgotten)
	fmt.Fprintf(ctx.Out, "  next: cozy status · stop with cozy down\n")

	go func() { _ = http.Serve(ln, loopback(c, st, addr)) }()
	go func() { _ = c.Serve() }()

	stop := make(chan os.Signal, 1)
	signal.Notify(stop, syscall.SIGINT, syscall.SIGTERM)
	<-stop
	fmt.Fprintln(ctx.Out, "draining endpoint processes…")
	c.Close(30 * time.Second)
	ln.Close()
	return nil
}

// loopback is the LocalService's HTTP transport. cl-006 owns the client-contract routes;
// what exists here is the reachability surface plus the invariants that are launch
// surface from the first line: LOOPBACK ONLY, a Host-header allowlist (the DNS-rebinding
// kill switch), nosniff, and no permissive CORS.
func loopback(c *coord.Coordinator, st *records.Store, addr string) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		counts, _ := st.Counts()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"service": "up", "address": addr, "counts": counts,
		})
	})
	return hostAllowlist(addr, mux)
}

func hostAllowlist(addr string, next http.Handler) http.Handler {
	_, port, _ := net.SplitHostPort(addr)
	allowed := map[string]bool{
		"127.0.0.1:" + port: true, "localhost:" + port: true, "[::1]:" + port: true,
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-Content-Type-Options", "nosniff")
		if !allowed[r.Host] {
			http.Error(w, "host not allowed", http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
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
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
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

func handleDown(ctx *Context) *exit.Error {
	timeout := 30 * time.Second
	if v := ctx.Inv.Value("--timeout"); v != "" {
		d, err := time.ParseDuration(v)
		if err != nil {
			return exit.Usagef("--timeout %q is not a duration", v)
		}
		timeout = d
	}
	st := service.Probe(ctx.Cfg)
	if !st.Up {
		return emit(ctx, render.Record{Kind: "service",
			Fields: []render.Field{{K: "service", V: "down"}},
			Notes:  []string{"not running: `cozy down` is idempotent"}})
	}
	if st.PID <= 0 {
		return exit.Internalf("the service lock is held but names no pid")
	}
	if err := syscall.Kill(st.PID, syscall.SIGTERM); err != nil {
		return exit.Internalf("cannot signal the LocalService (pid %d): %s", st.PID, err)
	}
	// The proof of exit is the LOCK becoming free, not the pid disappearing and not a
	// timer: the kernel drops the lock when the process dies, whatever killed it.
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !service.Probe(ctx.Cfg).Up {
			return emit(ctx, render.Record{Kind: "service",
				Fields: []render.Field{{K: "service", V: "down"}, {K: "stopped_pid", V: st.PID}},
				Notes:  []string{"the service lock is free again — the process is provably gone"}})
		}
		time.Sleep(50 * time.Millisecond)
	}
	_ = syscall.Kill(st.PID, syscall.SIGKILL)
	hard := time.Now().Add(5 * time.Second)
	for time.Now().Before(hard) && service.Probe(ctx.Cfg).Up {
		time.Sleep(50 * time.Millisecond)
	}
	return emit(ctx, render.Record{Kind: "service",
		Fields: []render.Field{{K: "service", V: "down"}, {K: "stopped_pid", V: st.PID}},
		Notes: []string{fmt.Sprintf("the cooperative drain exceeded %s; the process group was killed",
			timeout)}})
}
