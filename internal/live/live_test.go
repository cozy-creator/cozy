package live

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/config"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
)

// The two binaries the suite drives as real processes, built once by TestMain.
var cozyBin, fakeWorkerBin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "cozy-live-bin")
	if err != nil {
		panic(err)
	}
	cozyBin = filepath.Join(dir, "cozy")
	fakeWorkerBin = filepath.Join(dir, "cozy-fakeworker")
	for _, b := range [][2]string{{cozyBin, "."}, {fakeWorkerBin, "./internal/live/fakeworker"}} {
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

// owner is the REAL orchestrator and record owner on a fresh root — everything `cozy up`
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

func hostOwner(t *testing.T, name string) *owner {
	t.Helper()
	root := filepath.Join(os.TempDir(), "cozy-live", name)
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	must(t, os.Setenv("COZY_HOME", root))

	cfg, e := config.Load()
	fatal(t, e)
	cfg.Home = root // Load freezes on first call; a suite that reuses the process re-derives
	l, e := home.Open(cfg.Home)
	fatal(t, e)
	st, e := records.Open(l.DB)
	fatal(t, e)
	log, err := os.Create(filepath.Join(root, "orchestrator.log"))
	must(t, err)
	c, e := orchestrator.Open(orchestrator.Options{
		Cfg: cfg, Layout: l, Store: st, Yield: "smart", Log: log,
		EnvironmentSpecDigest: "sha256:" + strings.Repeat("11", 32),
		ConfigDigest:          "sha256:" + strings.Repeat("22", 32),
		MaxOutputMiB:          8,
	})
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

// close releases the root so a later `cozy up` on the same root is the only owner of it.
func (o *owner) close() { o.once.Do(o.closer) }

// fakeSpec is a worker slot whose process is internal/live/fakeworker speaking raw protocol
// bytes: a real process dialing the real socket over the committed contract.
func fakeSpec(name, device string, args ...string) orchestrator.WorkerLaunchSpec {
	return orchestrator.WorkerLaunchSpec{
		Python:  fakeWorkerBin,
		Args:    args,
		Devices: []string{device},
		Placement: orchestrator.DesiredPlacement{
			Endpoint:  "fake/" + name,
			ReleaseID: fakeRelease,
			Bindings: []*orchestrator.Binding{{
				Entrypoint: "fake",
				Record:     map[string]any{"entrypoint": "fake", "slot": name},
			}},
		},
	}
}

const fakeRelease = "cozy/fake@live"

func planIDOf(t *testing.T, spec orchestrator.WorkerLaunchSpec) string {
	t.Helper()
	id, e := spec.Placement.Bindings[0].PlanID()
	fatal(t, e)
	return id
}

func submission(planID, endpoint, idem string, body map[string]any) orchestrator.Submission {
	data, _ := json.Marshal(body)
	return orchestrator.Submission{
		IdemKey: idem, Endpoint: endpoint, Entrypoint: "fake", PlanID: planID,
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

// ------------------------------------------------------------- a real `cozy up` process

// service is one detached `cozy up` this suite owns. The separate process is the point:
// the suite is a CLIENT, which is the seat a real client occupies.
type service struct {
	root, addr, token string
	cmd               *exec.Cmd
}

func startService(t *testing.T, root string) *service {
	t.Helper()
	must(t, os.MkdirAll(root, 0o755))
	port := freePort(t)
	log, err := os.Create(filepath.Join(root, "service.log"))
	must(t, err)
	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "up", "--port", fmt.Sprint(port))
	cmd.Env = childEnv(t, root)
	cmd.Stdout, cmd.Stderr = log, log
	setProcessGroup(cmd)
	must(t, cmd.Start())

	s := &service{root: root, addr: fmt.Sprintf("127.0.0.1:%d", port), cmd: cmd}
	t.Cleanup(func() {
		_ = killGroup(cmd.Process.Pid)
		go func() { _ = cmd.Wait() }()
		log.Close()
	})
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if s.alive() {
			if data, err := os.ReadFile(filepath.Join(root, "client.cred")); err == nil {
				s.token = strings.TrimSpace(string(data))
				return s
			}
		}
		time.Sleep(50 * time.Millisecond)
	}
	t.Fatalf("cozy up did not answer on %s in 30s\n%s", s.addr, tail(filepath.Join(root, "service.log")))
	return nil
}

func (s *service) alive() bool {
	res, err := (&http.Client{Timeout: time.Second}).Get("http://" + s.addr + "/healthz")
	if err != nil {
		return false
	}
	defer res.Body.Close()
	_, _ = io.Copy(io.Discard, res.Body)
	return res.StatusCode == http.StatusOK
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
func (s *service) call(t *testing.T, method, path string, body any, headers ...string) reply {
	t.Helper()
	var reader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		must(t, err)
		reader = strings.NewReader(string(data))
	}
	req, err := http.NewRequest(method, "http://"+s.addr+path, reader)
	must(t, err)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
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

// runCozy runs the product binary as a user would type it, against one service root. The
// child's environment comes through the PRODUCT's own allowlist: the suite has no business
// inventing a second child-env mechanism, and the env fence says there is one reader.
func runCozy(t *testing.T, root string, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, string(data)
}

func childEnv(t *testing.T, root string, imposed ...string) []string {
	t.Helper()
	must(t, os.Setenv("COZY_HOME", root))
	cfg, e := config.Load()
	fatal(t, e)
	return cfg.Child(append([]string{"COZY_HOME=" + root}, imposed...)...)
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

func freePort(t *testing.T) int {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow the SUITE probes for a free port; the product binds through internal/api
	must(t, err)
	defer ln.Close()
	return ln.Addr().(*net.TCPAddr).Port
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

func (r reply) json(t *testing.T) map[string]any {
	t.Helper()
	var doc map[string]any
	_ = json.Unmarshal(r.Body, &doc)
	return doc
}
