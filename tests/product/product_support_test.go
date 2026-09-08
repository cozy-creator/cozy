package producttest

import (
	"bytes"
	"crypto/sha256"
	"database/sql"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The two binaries the suite drives as real processes, built once by TestMain.
var cozyBin, fakeWorkerBin string

func TestMain(m *testing.M) {
	if len(os.Args) == 2 && strings.HasPrefix(os.Args[1], "--cozy-test-daemon-parent=") {
		daemonPath := strings.TrimPrefix(os.Args[1], "--cozy-test-daemon-parent=")
		cmd := exec.Command(daemonPath)
		cmd.Args[0] = "cozy-daemon"
		cmd.Stdout = io.Discard
		cmd.Stderr = os.Stderr
		if err := cmd.Start(); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		os.Exit(0)
	}
	dir, err := os.MkdirTemp("", "cozy-product-test-bin")
	if err != nil {
		panic(err)
	}
	// The suite must never reach the user's ~/.tensorfs: config freezes on its first
	// in-process Load, so the isolated TensorFS home is pinned before any test runs.
	if err := os.Setenv("TENSORFS_HOME", filepath.Join(dir, "tensorfs")); err != nil {
		panic(err)
	}
	cozyBin = filepath.Join(dir, "cozy")
	fakeWorkerBin = filepath.Join(dir, "cozy-fakeworker")
	for _, b := range [][2]string{{cozyBin, "."}, {fakeWorkerBin, "./tests/support/fakeworker"}} {
		build := exec.Command("go", "build", "-o", b[0], b[1])
		build.Dir = "../.."
		if out, err := build.CombinedOutput(); err != nil {
			fmt.Fprintf(os.Stderr, "building %s: %v\n%s", b[1], err, out)
			os.Exit(1)
		}
	}
	code := m.Run()
	_ = os.RemoveAll(dir)
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
	root := filepath.Join(os.TempDir(), "cozy-product-test", name)
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
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
		Cfg: cfg, Layout: l, Store: st, Yield: "smart", Log: log,
		MaxOutputMiB: 8,
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

// close releases the root so a later `cozy run list` on the same root is the only owner of it.
func (o *owner) close() { o.once.Do(o.closer) }

// fakeSpec is a worker slot whose process is tests/support/fakeworker speaking raw protocol
// bytes: a real process dialing the real socket over the committed contract. `device` is
// the granted envelope, comma-separated for a multi-device one.
func fakeSpec(name, device string, args ...string) orchestrator.WorkerLaunchSpec {
	ref := func(label string) *pb.Ref {
		digest := sha256.Sum256([]byte(label))
		return &pb.Ref{Digest: digest[:], Length: uint64(len(label))}
	}
	spelled := func(raw []byte) string { value, _ := canonical.Spell(raw); return value }
	environment := map[string]canonical.Value{
		"local_wheels": []canonical.Value{},
	}
	environmentIdentity := map[string]canonical.Value{
		"format":       "cozy.worker.v1.Environment/1",
		"local_wheels": environment["local_wheels"],
	}
	environmentBytes, _ := canonical.Write(environmentIdentity)
	entrypointIdentity := map[string]canonical.Value{
		"name": "fake", "slots": []canonical.Value{},
	}
	entrypointBytes, _ := canonical.Write(entrypointIdentity)
	entrypointDigest := canonical.Digest(entrypointBytes)
	entrypoints := []canonical.Value{map[string]canonical.Value{
		"entrypoint_binding_digest": spelled(entrypointDigest),
		"name":                      "fake", "slots": []canonical.Value{},
	}}
	bindingsBytes, _ := canonical.Write(map[string]canonical.Value{
		"entrypoints": entrypoints, "models": []canonical.Value{},
	})
	setBytes, setDigest, _ := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: "plc-fake-" + name,
		PackageMode: &pb.Placement_Package{Package: &pb.PackageSelection{
			Package: "fake/" + name, Release: "1.0.0"}},
		EnvironmentDigest: canonical.Digest(environmentBytes),
		PackageInterface:  ref("fake-interface"), BindingsDigest: canonical.Digest(bindingsBytes),
		Entrypoints: []*pb.Entrypoint{{Name: "fake", EntrypointBindingDigest: entrypointDigest}},
		Environment: &pb.Environment{},
	}}})
	setID := spelled(setDigest)
	placement, problem := orchestrator.PlacementFromExact("fake/"+name, "", setID,
		setBytes, map[string][]string{"fake": []string{"image"}})
	if problem != nil {
		panic(problem.Message)
	}
	return orchestrator.WorkerLaunchSpec{
		Python:    fakeWorkerBin,
		Args:      args,
		Devices:   strings.Split(device, ","),
		Placement: placement,
	}
}

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

// waitEvent watches the orchestrator's OWN event log for a line — the owner's words, not
// the suite's inference about them.
func waitEvent(o *owner, substr string, timeout time.Duration) (string, bool) {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		for _, line := range o.c.Events() {
			if strings.Contains(line, substr) {
				return line, true
			}
		}
		time.Sleep(25 * time.Millisecond)
	}
	return "", false
}

func countEvents(o *owner, substr string) int {
	n := 0
	for _, line := range o.c.Events() {
		if strings.Contains(line, substr) {
			n++
		}
	}
	return n
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

func startDaemonProcess(t *testing.T, root string) *daemonProcess {
	t.Helper()
	must(t, os.MkdirAll(root, 0o755))
	// The daemon's words are its own bounded log, <root>/daemon.log (cl-096) — the
	// product surface the arms read. stderr, the startup-refusal pipe, is kept beside it.
	log, err := os.Create(filepath.Join(root, "daemon-test.log"))
	must(t, err)
	cmd := exec.Command(cozyBin)
	cmd.Args[0] = "cozy-daemon"
	cmd.Env = childEnv(t, root)
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
	// Machine results are one stdout document; awaited JSONL progress is stderr.
	// Dedicated stream proofs validate that channel independently.
	for _, arg := range args {
		if arg == "--" {
			break
		}
		if arg == "--json" {
			code, stdout, _ := runCozyStreams(t, root, args...)
			return code, stdout
		}
	}
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, string(data)
}

func runCozyStreams(t *testing.T, root string, args ...string) (int, string, string) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	_ = cmd.Run()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, stdout.String(), stderr.String()
}

func childEnv(t *testing.T, root string, imposed ...string) []string {
	t.Helper()
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	// Both homes are isolated: a product test must never reach the user's ~/.tensorfs.
	return cfg.Child(append([]string{"COZY_HOME=" + root,
		"TENSORFS_HOME=" + filepath.Join(root, "tensorfs")}, imposed...)...)
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

// revertRentalsBeforeWidth puts the rentals table back into its pre-schema-33 shape, for a
// fixture that builds an OLD database by mutating a current one. Schema 33 added the
// rental's width (cl-179), and `records.Open` compares a claimed prior schema's sqlite_master
// character for character — so a fixture stamping 27, 28 or 31 must carry the rentals table
// those schemas had, not this one. The DDL is read back from the database rather than
// restated here: a second copy of the released shape is exactly what the comparison exists
// to catch.
func revertRentalsBeforeWidth(t *testing.T, db *sql.DB) {
	t.Helper()
	var ddl string
	must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE name='rentals'`).Scan(&ddl))
	prior := strings.Replace(ddl, "  accelerator_count INTEGER NOT NULL DEFAULT 0,\n", "", 1)
	if prior == ddl {
		t.Fatalf("the rentals table carries no width column to remove:\n%s", ddl)
	}
	columns := "id,machine_name,sku,accelerator_model,hourly_rate_usd_micros,managed_request_id," +
		"address,cert_path,state,hub,rented_at,media_address,expected_worker_id," +
		"expected_worker_boot_id,ready_at,failure_code,failure_image_digest,failure_provider," +
		"failure_provider_resource_id,failure_provider_host_id,failure_provider_state," +
		"failure_container_state"
	var index string
	must(t, db.QueryRow(`SELECT sql FROM sqlite_master WHERE name='rentals_machine_name'`).Scan(&index))
	for _, statement := range []string{
		`DROP INDEX rentals_machine_name`,
		`ALTER TABLE rentals RENAME TO rentals_current`,
		prior,
		`INSERT INTO rentals(` + columns + `) SELECT ` + columns + ` FROM rentals_current`,
		`DROP TABLE rentals_current`,
		index,
	} {
		if _, err := db.Exec(statement); err != nil {
			t.Fatalf("cannot revert the rentals table: %v (%s)", err, statement)
		}
	}
}
