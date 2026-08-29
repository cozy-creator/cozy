package live

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/records"
)

const weightlessRef = "cozy/weightless"

func TestControllerWebLifecycle(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-live", "controller-web")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, help := runCozy(t, root)
	for _, want := range []string{
		"Usage: cozy", "package install", "model download", "invoke run",
		"rental new", "up", "down", "unload",
	} {
		if code != 0 || !strings.Contains(help, want) {
			t.Fatalf("bare cozy omitted %q [exit %d]\n%s", want, code, help)
		}
	}
	for _, retired := range []string{" exit ", "workflow", "video", "job submit"} {
		if strings.Contains(help, retired) {
			t.Fatalf("bare cozy retained %q\n%s", retired, help)
		}
	}
	if code, out := runCozy(t, root, "exit"); code != 2 || !strings.Contains(out, "cli.usage") {
		t.Fatalf("retired exit command did not refuse [exit %d]\n%s", code, out)
	}

	env := childEnv(t, root)
	results := make(chan cozyResult, 2)
	for range 2 {
		go func() { results <- runCozyEnv(env, "up", "--json", "--full") }()
	}
	first, second := <-results, <-results
	if first.code != 0 || second.code != 0 {
		t.Fatalf("concurrent up did not converge: first=[%d] %s second=[%d] %s",
			first.code, first.output, second.code, second.output)
	}
	up := first.output
	var upDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	var secondDocument struct {
		URL     string `json:"url"`
		PID     int    `json:"pid"`
		Changed bool   `json:"changed"`
	}
	if err := json.Unmarshal([]byte(up), &upDocument); err != nil ||
		upDocument.URL == "" || upDocument.PID == 0 {
		t.Fatalf("up did not start one controller: %v\n%s", err, up)
	}
	if err := json.Unmarshal([]byte(second.output), &secondDocument); err != nil ||
		upDocument.PID != secondDocument.PID || upDocument.URL != secondDocument.URL {
		t.Fatalf("concurrent up returned different controller generations: %v\n%s\n%s",
			err, first.output, second.output)
	}
	if upDocument.Changed == secondDocument.Changed {
		t.Fatalf("concurrent up did not report one winner: first changed=%v second changed=%v\n%s\n%s",
			upDocument.Changed, secondDocument.Changed, first.output, second.output)
	}
	if strings.Contains(strings.ToLower(first.output+second.output), "lock") {
		t.Fatalf("up exposed its internal singleton mechanism\n%s\n%s", first.output, second.output)
	}
	url := upDocument.URL
	client := &http.Client{Timeout: 5 * time.Second}
	response, err := client.Get(url)
	if err != nil {
		t.Fatalf("up returned an unreachable web UI %q: %v", url, err)
	}
	page, readErr := io.ReadAll(response.Body)
	response.Body.Close()
	if readErr != nil || response.StatusCode != http.StatusOK ||
		!strings.Contains(string(page), "local generative workspace is running") {
		t.Fatalf("web stub is not ready at up return: status=%d err=%v\n%s",
			response.StatusCode, readErr, page)
	}
	if _, err := os.Stat(filepath.Join(root, "controller.log")); !os.IsNotExist(err) {
		t.Fatalf("up created a persistent controller log: %v", err)
	}
	if code, out := runCozy(t, root, "up", "--json", "--full"); code != 0 {
		t.Fatalf("repeated up failed [exit %d]\n%s", code, out)
	} else {
		var repeated struct {
			URL     string `json:"url"`
			PID     int    `json:"pid"`
			Changed bool   `json:"changed"`
		}
		if err := json.Unmarshal([]byte(out), &repeated); err != nil ||
			repeated.Changed || repeated.URL != url || repeated.PID != upDocument.PID {
			t.Fatalf("repeated up did not return the same healthy controller with changed=false: %v\n%s",
				err, out)
		}
	}
	if code, out := runCozy(t, root, "down"); code != 0 || !strings.Contains(out, "controller: stopped") {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "invoke", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"invocations":[]`) {
		t.Fatalf("stateful command did not auto-start the controller [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "workers") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "up"); code != 0 || !strings.Contains(out, "changed: false") {
		t.Fatalf("unload stopped the controller [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 || !strings.Contains(out, "controller: stopped") {
		t.Fatalf("final down [exit %d]\n%s", code, out)
	}
}

func TestControllerStartupDiagnostic(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-live", "controller-startup-diagnostic")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	held, err := net.Listen("tcp4", "127.0.0.1:0") //cozy:allow live proof occupies one loopback port to exercise the real startup refusal
	must(t, err)
	port := held.Addr().(*net.TCPAddr).Port
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"),
		[]byte(fmt.Sprintf("port: %d\n", port)), 0o600))

	began := time.Now()
	code, out := runCozy(t, root, "up", "--json")
	if code != 1 {
		t.Fatalf("failed controller startup exited %d, want operational 1\n%s", code, out)
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("failed controller startup waited %s instead of relaying the exited child", took)
	}
	var document struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &document); err != nil ||
		document.Error.Code != "controller_startup_failed" ||
		!strings.Contains(document.Error.Message, fmt.Sprintf("127.0.0.1:%d", port)) ||
		!strings.Contains(document.Error.Message, "held by another process") {
		t.Fatalf("startup failure did not relay its child diagnostic: %v\n%s", err, out)
	}
	if strings.Contains(strings.ToLower(out), "lock") {
		t.Fatalf("startup failure exposed its internal singleton mechanism\n%s", out)
	}
	if len(out) > maxControllerDiagnosticOutput {
		t.Fatalf("startup diagnostic is unbounded: %d bytes", len(out))
	}
	if _, err := os.Stat(filepath.Join(root, "controller.log")); !os.IsNotExist(err) {
		t.Fatalf("failed startup created a persistent controller log: %v", err)
	}

	must(t, held.Close())
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("root did not recover after the failed child [exit %d]\n%s", code, out)
	}
}

const maxControllerDiagnosticOutput = 18 << 10

// TestProductPath drives only the public Kong surface: install a real weightless
// release, auto-start the controller on invoke, cross Runtime and the worker wire,
// accept an output, release idle residency, then stop cleanly.
func TestProductPath(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-live", "product")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	archive := weightlessRelease(t)
	data, err := os.ReadFile(archive)
	must(t, err)
	digest := sha256.Sum256(data)
	code, out := runCozy(t, root, "package", "install", weightlessRef,
		"--from", archive, "--digest", "sha256:"+hex.EncodeToString(digest[:]))
	if code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "package", "list", "--json"); code != 0 ||
		!strings.Contains(out, weightlessRef) {
		t.Fatalf("package list omitted the install [exit %d]\n%s", code, out)
	}

	outDir := filepath.Join(root, "out")
	code, out = runCozy(t, root, "invoke", "run", weightlessRef+"/v1/tile",
		"size=32", "seed=7", "--out", outDir)
	if code != 0 {
		t.Fatalf("invoke run [exit %d]\n%s", code, out)
	}
	saved := filepath.Join(outDir, "image.png")
	image, err := os.ReadFile(saved)
	if err != nil || len(image) < 8 || string(image[1:4]) != "PNG" {
		t.Fatalf("invoke did not publish its declared PNG at %s: %v", saved, err)
	}
	// A second invocation reuses the same warm serving worker rather than spawning
	// another process onto the device envelope.
	secondOut := filepath.Join(root, "out-2")
	if code, out := runCozy(t, root, "invoke", "run", weightlessRef+"/v1/tile",
		"size=16", "seed=8", "--out", secondOut); code != 0 {
		t.Fatalf("warm invoke [exit %d]\n%s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "records.db"))
	fatal(t, problem)
	workers, problem := store.LiveWorkers()
	fatal(t, problem)
	store.Close()
	localWorkers := 0
	for _, worker := range workers {
		if worker.WorkerID != "remote" {
			localWorkers++
		}
	}
	if localWorkers != 1 {
		t.Fatalf("warm reuse left %d local workers; wanted exactly one", localWorkers)
	}

	code, listed := runCozy(t, root, "invoke", "list", "--json")
	if code != 0 {
		t.Fatalf("invoke list [exit %d]\n%s", code, listed)
	}
	var document map[string]any
	err = json.Unmarshal([]byte(listed), &document)
	invocations, ok := document["invocations"].([]any)
	if err != nil || !ok || len(invocations) != 2 {
		t.Fatalf("invoke list is not one successful JSON document: %v\n%s", err, listed)
	}
	if code, out := runCozy(t, root, "invoke", "run", weightlessRef+"/v1/nosuch"); code != 1 || !strings.Contains(out, "not_found") {
		t.Fatalf("operational refusal was not shell exit 1 with detail [exit %d]\n%s", code, out)
	}

	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "workers") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 || !strings.Contains(out, "controller: stopped") {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
}

type cozyResult struct {
	code   int
	output string
}

func runCozyEnv(env []string, args ...string) cozyResult {
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = env
	data, _ := cmd.CombinedOutput()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return cozyResult{code: code, output: string(data)}
}

func weightlessRelease(t *testing.T) string {
	t.Helper()
	home, err := os.UserHomeDir()
	must(t, err)
	repo := filepath.Join(home, "cozy_v2", "cozy-runtime") //cozy:allow peer source; the fixture builds it and never executes a host runtime
	if _, err := os.Stat(filepath.Join(repo, "pyproject.toml")); err != nil {
		t.Skipf("no cozy-runtime peer at %s: %v", repo, err)
	}
	dir := t.TempDir()
	build := exec.Command("/usr/bin/nice", "-n", "19", "python3",
		"internal/live/testdata/build-weightless.py", "--out", dir)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return filepath.Join(dir, "weightless-1.0.0.tar.gz")
}
