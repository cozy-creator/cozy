package producttest

import (
	"bytes"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	cozyweb "github.com/cozy-creator/cozy/web"
)

// integration marks a test that builds real Python and Runtime environments. `-short`
// (the default dispatched suite) skips it; a dispatch with `full=true` runs it.
func integration(t *testing.T) {
	t.Helper()
	if testing.Short() {
		t.Skip("integration: builds real Runtime environments; runs with full=true")
	}
}

// fullRun marks a test the default dispatched suite skips because it outwaits a real-time
// bound or reads a live third-party service. A dispatch with `full=true` runs it.
func fullRun(t *testing.T, why string) {
	t.Helper()
	if testing.Short() {
		t.Skip(why + "; runs with full=true")
	}
}

// The two binaries the suite drives as real processes, built once by TestMain.
var cozyBin string

// runtimeWireProtocol is what a current cozy-runtime's `version` names.
const runtimeWireProtocol = "cozy.machine.v1"

// scratchBase is this process's own scratch root. Named roots live under it, so two
// concurrent runs of the same test never share a COZY_HOME or its database.
var scratchBase string

func TestMain(m *testing.M) {
	// The reaper is this same binary under another argv (see reap_test.go). It must be
	// recognised before anything else happens: it builds nothing and runs no test.
	if len(os.Args) == 2 && strings.HasPrefix(os.Args[1], reapMode) {
		runDaemonReaper(strings.TrimPrefix(os.Args[1], reapMode), os.Stdin) //cozy:stdin-value owned reaper subprocess liveness pipe
		os.Exit(0)
	}
	// So is a root's machine agent (local_machine_lifecycle_test.go).
	if filepath.Base(os.Args[0]) == "cozy-machine" {
		serveFakeMachineAgent()
		os.Exit(0)
	}
	// Reap the roots the previous run abandoned before claiming disk of our own,
	// and refuse to start at all on a box that has no fork headroom left — a
	// suite that dies partway through leaks a scratch root per killed test.
	if reaped, failed := reapAbandonedScratch(os.TempDir(), scratchReapMinAge); reaped+failed > 0 {
		fmt.Fprintf(os.Stderr, "reaped %d abandoned scratch roots (%d could not be removed)\n", reaped, failed)
	}
	if ok, detail := forkHeadroom(); !ok {
		fmt.Fprintf(os.Stderr, "refusing to start: the box is out of fork headroom (%s).\n"+
			"Reap orphaned daemons and scratch roots before running the suite.\n", detail)
		os.Exit(1)
	}
	dir, err := os.MkdirTemp("", "cozy-product-test-bin")
	if err != nil {
		panic(err)
	}
	claimScratch(dir)
	// Short: worker control sockets live under it, and a Unix socket path is at most
	// 107 bytes (scripts/product-tests.sh gives each process its own short TMPDIR).
	if scratchBase, err = os.MkdirTemp("", "cozyp-"); err != nil {
		panic(err)
	}
	claimScratch(scratchBase)
	// The suite must never reach the user's ~/.tensorfs: config freezes on its first
	// in-process Load, so the isolated TensorFS home is pinned before any test runs.
	if err := os.Setenv("TENSORFS_HOME", filepath.Join(dir, "tensorfs")); err != nil {
		panic(err)
	}
	// Nor production Tensorhub, the product default: in-process owners and CLI children
	// (childEnv) default to the same unanswered loopback hub.
	if err := os.Setenv("TENSORHUB_URL", testDefaultHub); err != nil {
		panic(err)
	}
	if conn, err := net.Dial("tcp", strings.TrimPrefix(testDefaultHub, "http://")); err == nil {
		conn.Close()
		panic("the suite's unanswered Tensorhub " + testDefaultHub + " answers on this box")
	}
	cozyBin = filepath.Join(dir, "cozy")
	build := exec.Command("go", "build", "-o", cozyBin, ".")
	build.Dir = "../.."
	if out, err := build.CombinedOutput(); err != nil {
		fmt.Fprintf(os.Stderr, "building cozy: %v\n%s", err, out)
		os.Exit(1)
	}
	// Every daemon this run causes to exist dies with it. Layer 3 first, so the roots a
	// test registers are known to a process that outlives a SIGKILL of this one.
	startDaemonReaper()
	code := m.Run()
	for _, root := range trackedRoots() {
		reapDaemonRoot(root)
	}
	stopDaemonReaper()
	_ = os.RemoveAll(dir)
	_ = removeAllForce(scratchBase)
	os.Exit(code)
}

// ---------------------------------------------------------------- the in-process owner

// owner is the REAL orchestrator and record owner on a fresh root — everything `cozy run list`
// hosts, minus the HTTP transport a test that drives the orchestrator directly has no use
// for.
type owner struct {
	root   string
	cfg    config.Config
	l      home.Layout
	store  *records.Store
	c      *orchestrator.Orchestrator
	once   sync.Once
	closer func()
}

// hostOwner builds the real orchestrator on a fresh root. `with` mutates the options
// before Open, for the wiring a particular product path needs (the rental callbacks, say).
func hostOwner(t *testing.T, name string, with ...func(*orchestrator.Options)) *owner {
	t.Helper()
	root := filepath.Join(scratchBase, name)
	must(t, removeAllForce(root))
	must(t, os.MkdirAll(root, 0o755))
	claimScratch(root)
	must(t, os.Setenv("COZY_HOME", root))

	cfg, e := config.Load()
	fatal(t, e)
	cfg.Home = root // Load freezes on first call; a suite that reuses the process re-derives
	cfg.TensorFSRoot = filepath.Join(root, "tensorfs")
	l, e := home.Open(cfg.Home)
	fatal(t, e)
	st, e := records.Open(l.DB)
	fatal(t, e)
	log, err := os.Create(filepath.Join(root, "orchestrator.log"))
	must(t, err)
	options := orchestrator.Options{
		Cfg: cfg, Layout: l, Store: st, Log: log,
	}
	for _, mutate := range with {
		mutate(&options)
	}
	c, e := orchestrator.Open(options)
	fatal(t, e)
	go func() { _ = c.Serve() }()
	o := &owner{root: root, cfg: cfg, l: l, store: st, c: c}
	o.closer = func() {
		c.Close(20 * time.Second)
		st.Close()
		log.Close()
	}
	t.Cleanup(o.close)
	return o
}

// publicationControlAPI serves an in-process owner through the ordinary authenticated
// API and daemon record, so the built CLI reaches it exactly as it reaches a daemon.
func publicationControlAPI(t *testing.T, o *owner, configure ...func(*api.Options)) func() {
	t.Helper()
	v4, v6, addr, problem := api.Listeners(0)
	fatal(t, problem)
	if v6 != nil {
		_ = v6.Close()
	}
	held, problem := daemon.Hold(o.l, addr, "")
	fatal(t, problem)
	creds, problem := api.Mint(o.l)
	fatal(t, problem)
	options := api.Options{Orchestrator: o.c, Cfg: o.cfg, Creds: creds, Addr: addr, Web: cozyweb.Handler()}
	for _, apply := range configure {
		apply(&options)
	}
	handler, problem := api.New(options).Handler()
	fatal(t, problem)
	server := &http.Server{Handler: handler, ReadHeaderTimeout: 5 * time.Second}
	go func() { _ = server.Serve(v4) }()
	return func() { _ = server.Close(); held.Release() }
}

// awaitMachineReceipt waits for Runtime's durable acceptance of a machine execution. A
// detached `cozy run` returns once the run is queued; the daemon records the receipt when
// the machine accepts it. It fails as soon as the run settles without one.
func awaitMachineReceipt(t *testing.T, store *records.Store, requestID string) *records.MachineExecution {
	t.Helper()
	for {
		link, problem := store.MachineExecution(requestID)
		fatal(t, problem)
		if link != nil && len(link.Receipt) > 0 {
			return link
		}
		row, problem := store.RequestRow(requestID)
		fatal(t, problem)
		if row != nil && (records.Settled(row.State) || records.RetainedState(row.State)) {
			t.Fatalf("run %s became %s before Runtime accepted it", requestID, row.State)
		}
		time.Sleep(50 * time.Millisecond)
	}
}

// close releases the root so a later `cozy run list` on the same root is the only owner of it.
func (o *owner) close() { o.once.Do(o.closer) }

func planIDOf(t *testing.T, spec orchestrator.WorkerLaunchSpec) string {
	t.Helper()
	return spec.Placement.Entrypoints[0].Digest
}

func submission(planID, pkg, idem string, body map[string]any) orchestrator.Submission {
	data, _ := json.Marshal(body)
	return orchestrator.Submission{
		IdemKey: idem, Package: pkg, Entrypoint: "fake", PlanID: planID,
		Payload: data, Outputs: []string{"image"},
	}
}

// ------------------------------------------------------------- a real hidden daemon process

// daemonProcess is one Cozy daemon this suite owns. The separate process is the point:
// the suite is a CLIENT, which is the seat a real client occupies.
type daemonProcess struct {
	root, addr, token string
	cmd               *exec.Cmd
	// exited delivers the daemon's own exit code once, for the arms where leaving is
	// the behaviour under test rather than the suite's cleanup.
	exited <-chan int
}

func startDaemonProcess(t *testing.T, root string, imposed ...string) *daemonProcess {
	return startDaemonBinary(t, cozyBin, root, imposed...)
}

func startDaemonBinary(t *testing.T, binary, root string, imposed ...string) *daemonProcess {
	t.Helper()
	must(t, os.MkdirAll(root, 0o755))
	// The daemon's words are its own bounded log, <root>/daemon.log (cl-096) — the
	// product surface the arms read. stderr, the startup-refusal pipe, is kept beside it.
	log, err := os.Create(filepath.Join(root, "daemon-test.log"))
	must(t, err)
	cmd := exec.Command(binary)
	cmd.Args[0] = "cozy-daemon"
	cmd.Env = childEnv(t, root, imposed...)
	cmd.Stdout, cmd.Stderr = log, log
	setProcessGroup(cmd)
	must(t, cmd.Start())

	exited := make(chan int, 1)
	go func() {
		_ = cmd.Wait()
		exited <- cmd.ProcessState.ExitCode()
	}()
	s := &daemonProcess{root: root, cmd: cmd, exited: exited}
	t.Cleanup(func() {
		_ = killGroup(cmd.Process.Pid)
		log.Close()
	})
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		lock, lockErr := os.ReadFile(filepath.Join(root, "daemon.lock"))
		if lockErr == nil {
			for _, line := range strings.Split(string(lock), "\n") {
				if value, ok := strings.CutPrefix(line, "addr="); ok {
					s.addr = strings.TrimSpace(value)
				}
				if value, ok := strings.CutPrefix(line, "token="); ok {
					s.token = strings.TrimSpace(value)
				}
			}
			if s.addr != "" && s.token != "" {
				return s
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("the Cozy daemon did not publish its address in 30s\n%s", tail(filepath.Join(root, "daemon.log")))
	return nil
}

type reply struct {
	Status int
	Header http.Header
	Body   []byte
}

func (r reply) code() string {
	var doc map[string]any
	_ = json.Unmarshal(r.Body, &doc)
	if e, ok := doc["error"].(map[string]any); ok {
		code, _ := e["code"].(string)
		return code
	}
	return ""
}

func (r reply) brief() string {
	body := strings.TrimSpace(string(r.Body))
	if len(body) > 140 {
		body = body[:140] + "…"
	}
	return fmt.Sprintf("%d %s", r.Status, body)
}

// call issues one authenticated request. Headers may be overridden per call, which is how
// the refusal matrix presents a foreign Host or a foreign Origin.
func (s *daemonProcess) call(t *testing.T, method, path string, body any, headers ...string) reply {
	t.Helper()
	var data []byte
	var contentType string
	if body != nil {
		var err error
		data, err = json.Marshal(body)
		must(t, err)
		contentType = "application/json"
	}
	return s.callBytes(t, method, path, data, contentType, headers...)
}

// callBytes drives byte-oriented routes without smuggling a host path or JSON encoding
// into the request. Authentication and hostile-header overrides remain identical to call.
func (s *daemonProcess) callBytes(t *testing.T, method, path string, body []byte, contentType string, headers ...string) reply {
	t.Helper()
	var reader io.Reader
	if body != nil {
		reader = strings.NewReader(string(body))
	}
	req, err := http.NewRequest(method, "http://"+s.addr+path, reader)
	must(t, err)
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	req.Header.Set("Authorization", "Bearer "+s.token)
	for i := 0; i+1 < len(headers); i += 2 {
		switch {
		case strings.EqualFold(headers[i], "Host"):
			req.Host = headers[i+1]
		case headers[i+1] == "":
			req.Header.Del(headers[i])
		default:
			req.Header.Set(headers[i], headers[i+1])
		}
	}
	res, err := (&http.Client{Timeout: time.Minute}).Do(req)
	if err != nil {
		return reply{Body: []byte(err.Error())}
	}
	defer res.Body.Close()
	data, _ := io.ReadAll(res.Body)
	return reply{Status: res.StatusCode, Header: res.Header, Body: data}
}

// ----------------------------------------------------------------- the product binary

// runCozy runs the product binary as a user would type it, against one daemon root. The
// child's environment comes through the PRODUCT's own allowlist: the suite has no business
// inventing a second child-env mechanism, and the env fence says there is one reader.
func runCozy(t *testing.T, root string, args ...string) (int, string) {
	t.Helper()
	return runCozyWith(t, root, nil, args...)
}

// runCozyWith is runCozy with extra child environment, such as another HOME.
func runCozyWith(t *testing.T, root string, imposed []string, args ...string) (int, string) {
	t.Helper()
	prepareExplicitInstall(t, root, args)
	// Machine results are one stdout document; awaited JSONL progress is stderr.
	// Dedicated stream proofs validate that channel independently.
	for _, arg := range args {
		if arg == "--" {
			break
		}
		if arg == "--json" {
			code, stdout, _ := runCozyStreamsWith(t, root, imposed, args...)
			return code, stdout
		}
	}
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root, imposed...)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	skipWithoutMachine(t, code, string(data))
	return code, string(data)
}

func runCozyStreams(t *testing.T, root string, args ...string) (int, string, string) {
	t.Helper()
	return runCozyStreamsWith(t, root, nil, args...)
}

func runCozyStreamsWith(t *testing.T, root string, imposed []string, args ...string) (int, string, string) {
	t.Helper()
	prepareExplicitInstall(t, root, args)
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root, imposed...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	_ = cmd.Run()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	skipWithoutMachine(t, code, stdout.String()+stderr.String())
	return code, stdout.String(), stderr.String()
}

// childEnv is the ONE chokepoint every daemon-capable child process passes: a Cozy daemon
// can only ever be started by a process that got its COZY_HOME from here. Registering the
// root here is therefore the same thing as registering every daemon that can exist, and a
// test cannot forget to do it. See reap_test.go for what the registration arms.
// testDefaultHub is the suite's default Tensorhub: loopback port 1, which only root can bind,
// so it always refuses. It is never production and never the owner's local Hub on 8819.
const testDefaultHub = "http://127.0.0.1:1"

func childEnv(t *testing.T, root string, imposed ...string) []string {
	t.Helper()
	trackDaemonRoot(t, root)
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	// Both homes are isolated: a product test must never reach the user's ~/.tensorfs.
	base := []string{"COZY_HOME=" + root, "TENSORFS_HOME=" + filepath.Join(root, "tensorfs")}
	// Nor production Tensorhub, the product default: a root that selects no hub gets an
	// unanswered loopback one.
	if raw, _ := os.ReadFile(filepath.Join(root, config.FileName)); !strings.Contains(string(raw), "tensorhub_url") {
		base = append(base, "TENSORHUB_URL="+testDefaultHub)
	}
	provisionMachine(t, root)
	return cfg.Child(append(base, imposed...)...)
}

// --------------------------------------------------------------------------- the small stuff

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func fatal(t *testing.T, e *exit.Error) {
	t.Helper()
	if e != nil {
		t.Fatal(e.Message)
	}
}

func briefly(e *exit.Error) string {
	if e == nil {
		return "accepted"
	}
	return e.Message
}

func tail(path string) string {
	data, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	lines := strings.Split(strings.TrimRight(string(data), "\n"), "\n")
	if len(lines) > 25 {
		lines = lines[len(lines)-25:]
	}
	return "    | " + strings.Join(lines, "\n    | ")
}

func trimLog(line string) string {
	if i := strings.Index(line, "] "); i >= 0 {
		line = line[i+2:]
	}
	if len(line) > 150 {
		line = line[:150] + "…"
	}
	return line
}

func sha256Of(data []byte) []byte {
	sum := sha256.Sum256(data)
	return sum[:]
}

func stat(path string) (int64, error) {
	info, err := os.Stat(path)
	if err != nil {
		return 0, err
	}
	return info.Size(), nil
}

// listedRentalQuote answers POST /v1/rental-quotes on a stand-in Hub the way Tensorhub does
// when the request's disk changes nothing: the listed rate of the SKU and width asked for,
// read from the stand-in's own GET /v1/rental-skus.
func listedRentalQuote(listing http.Handler) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var request struct {
			SKU   string `json:"sku"`
			Count int    `json:"accelerator_count"`
			Disk  int    `json:"container_disk_gb"`
		}
		if json.NewDecoder(r.Body).Decode(&request) != nil {
			w.WriteHeader(http.StatusBadRequest)
			return
		}
		recorder := httptest.NewRecorder()
		listing.ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/v1/rental-skus", nil))
		var rows []struct {
			Name   string `json:"name"`
			Widths []struct {
				Count   int   `json:"accelerator_count"`
				Price   int64 `json:"price_usd_micros_per_hour"`
				Storage int64 `json:"storage_usd_micros_per_hour"`
			} `json:"widths"`
		}
		_ = json.Unmarshal(recorder.Body.Bytes(), &rows)
		for _, row := range rows {
			for _, width := range row.Widths {
				if row.Name == request.SKU && (request.Count == 0 || request.Count == width.Count) {
					w.Header().Set("Content-Type", "application/json")
					_ = json.NewEncoder(w).Encode(map[string]any{"name": row.Name, "accelerator_count": width.Count,
						"price_usd_micros_per_hour": width.Price, "storage_usd_micros_per_hour": width.Storage,
						"maximum_total_hourly_rate_usd_micros": width.Price + width.Storage,
						"container_disk_gb":                    request.Disk})
					return
				}
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusNotFound)
		_, _ = w.Write([]byte(`{"error":{"code":"rental.sku_not_found","message":"no such rental SKU"}}`))
	}
}

// Explicit installation exercises real bootstrap into this root. Auto-provisioning
// the shared template first would turn it into an update of every test's interpreter.
func prepareExplicitInstall(t *testing.T, root string, args []string) {
	t.Helper()
	if len(args) >= 2 && args[0] == "machine" && args[1] == "install" {
		must(t, os.MkdirAll(filepath.Join(root, "machine"), 0700))
	}
}
