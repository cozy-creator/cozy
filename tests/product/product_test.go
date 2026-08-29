package producttest

import (
	"crypto/sha256"
	"encoding/hex"
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
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

const weightlessRef = "cozy/weightless"

func TestPackagePublishMetadataGrammar(t *testing.T) {
	root := t.TempDir()
	if code, help := runCozy(t, root, "package", "publish", "--help"); code != 0 ||
		strings.Contains(help, "--release") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "<package>") {
		t.Fatalf("package publish retained caller-authored identity [exit %d]\n%s", code, help)
	}
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name = "proof-package"
version = "1.0.0"
`), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte(
		"[application]\nobject = \"proof_package:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version = 1\n"), 0o644))
	code, out := runCozyDir(t, root, project, []string{"TENSORHUB_TOKEN=proof-token"},
		"package", "publish")
	if code != 1 || !strings.Contains(out, "must declare [tool.cozy] organization") {
		t.Fatalf("missing [tool.cozy] organization was not refused before build [exit %d]\n%s", code, out)
	}
}

func TestDaemonWebLifecycle(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-web")
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
	if code, out := runCozy(t, root, "exit"); code != 2 || !strings.Contains(out, "unexpected argument exit") {
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
		t.Fatalf("up did not start one daemon: %v\n%s", err, up)
	}
	if err := json.Unmarshal([]byte(second.output), &secondDocument); err != nil ||
		upDocument.PID != secondDocument.PID || upDocument.URL != secondDocument.URL {
		t.Fatalf("concurrent up returned different daemon generations: %v\n%s\n%s",
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
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("up created a persistent daemon log: %v", err)
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
			t.Fatalf("repeated up did not return the same healthy daemon with changed=false: %v\n%s",
				err, out)
		}
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "invoke", "list", "--json"); code != 0 ||
		!strings.Contains(out, `"invocations":[]`) {
		t.Fatalf("stateful command did not auto-start the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, "No workers found.") {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "up"); code != 0 || !strings.Contains(out, "changed: false") {
		t.Fatalf("unload stopped the daemon [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
		t.Fatalf("final down [exit %d]\n%s", code, out)
	}
}

func TestUpReportsLocalGPUCompatibility(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "up-gpu-compatibility")
	must(t, os.RemoveAll(root))
	toolDir := t.TempDir()
	nvidiaSMI := filepath.Join(toolDir, "nvidia-smi")
	must(t, os.WriteFile(nvidiaSMI, []byte(`#!/bin/sh
case "$1" in
  --query-gpu=*) printf '0,"NVIDIA, Test GPU",12288,24576,550.54,8.9\n' ;;
  *) printf '| NVIDIA-SMI 550.54 Driver Version: 550.54 CUDA Version: 12.6 |\n' ;;
esac
`), 0o755))
	env := childEnv(t, root, "PATH="+toolDir)
	t.Cleanup(func() { _ = runCozyEnv(env, "down", "--all") })

	result := runCozyEnv(env, "up", "--json", "--full")
	if result.code != 0 {
		t.Fatalf("up with NVIDIA GPU failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs       []string `json:"gpus"`
		GPUCount   int      `json:"gpu_count"`
		GPUDetails []struct {
			Model             string `json:"model"`
			VRAMFreeBytes     int64  `json:"vram_free_bytes"`
			VRAMTotalBytes    int64  `json:"vram_total_bytes"`
			DriverVersion     string `json:"driver_version"`
			DriverCUDAVersion string `json:"driver_cuda_version"`
			ComputeCapability string `json:"compute_capability"`
			SM                string `json:"sm"`
		} `json:"gpu_details"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil ||
		document.GPUCount != 1 || len(document.GPUs) != 1 || len(document.GPUDetails) != 1 {
		t.Fatalf("up did not return one typed GPU: %v\n%s", err, result.output)
	}
	gpu := document.GPUDetails[0]
	if gpu.Model != "NVIDIA, Test GPU" || gpu.VRAMFreeBytes != 12<<30 ||
		gpu.VRAMTotalBytes != 24<<30 || gpu.DriverVersion != "550.54" ||
		gpu.DriverCUDAVersion != "12.6" || gpu.ComputeCapability != "8.9" || gpu.SM != "sm_89" {
		t.Fatalf("up GPU compatibility facts drifted: %+v", gpu)
	}
	for _, want := range []string{"NVIDIA, Test GPU", "12.0 / 24.0 GiB free", "driver 550.54", "driver CUDA 12.6", "sm_89"} {
		if !strings.Contains(document.GPUs[0], want) {
			t.Fatalf("up GPU summary omitted %q: %s", want, document.GPUs[0])
		}
	}
}

func TestRentalGPUCatalog(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet || r.URL.Path != "/v1/rental-skus" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `[{"name":"h200","accelerator_model":"NVIDIA H200","vram_gb":141,"price_usd_micros_per_hour":6000000},{"name":"rtx-4090","accelerator_model":"NVIDIA GeForce RTX 4090","vram_gb":24,"price_usd_micros_per_hour":1250000}]`)
	}))
	defer server.Close()
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-gpu-catalog")
	must(t, os.RemoveAll(root))
	env := childEnv(t, root, "TENSORHUB_URL="+server.URL)

	result := runCozyEnv(env, "rental", "new", "--json")
	if result.code != 0 {
		t.Fatalf("rental catalog failed [exit %d]\n%s", result.code, result.output)
	}
	var document struct {
		GPUs []map[string]string `json:"gpus"`
	}
	if err := json.Unmarshal([]byte(result.output), &document); err != nil || len(document.GPUs) != 2 {
		t.Fatalf("rental catalog was not a two-row GPU list: %v\n%s", err, result.output)
	}
	if document.GPUs[0]["name"] != "h200" || document.GPUs[0]["model"] != "NVIDIA H200" ||
		document.GPUs[0]["vram"] != "141 GB" || document.GPUs[0]["price"] != "$6/hr" ||
		document.GPUs[1]["price"] != "$1.25/hr" {
		t.Fatalf("rental catalog values drifted: %#v", document.GPUs)
	}
	if result := runCozyEnv(env, "rental", "new", "h200"); result.code != 2 || !strings.Contains(result.output, "org/package/vN/function") {
		t.Fatalf("selected GPU without package did not explain the remaining argument [exit %d]\n%s", result.code, result.output)
	}
}

func TestDefaultWebPortPreferenceAndFallback(t *testing.T) {
	held, err := net.Listen("tcp4", "127.0.0.1:8818") //cozy:allow product proof occupies the preferred loopback port to exercise fallback
	if err != nil {
		t.Skipf("localhost:8818 is already occupied outside this product test: %v", err)
	}

	fallbackRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-fallback")
	must(t, os.RemoveAll(fallbackRoot))
	code, out := runCozy(t, fallbackRoot, "up", "--json", "--full")
	if code != 0 {
		held.Close()
		t.Fatalf("up with occupied preferred port failed [exit %d]\n%s", code, out)
	}
	var fallback struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &fallback); err != nil || fallback.URL == "" ||
		strings.Contains(fallback.URL, ":8818") {
		held.Close()
		t.Fatalf("occupied 8818 did not select a fallback: %v\n%s", err, out)
	}
	if code, out := runCozy(t, fallbackRoot, "down"); code != 0 {
		held.Close()
		t.Fatalf("fallback daemon down [exit %d]\n%s", code, out)
	}
	must(t, held.Close())

	preferredRoot := filepath.Join(os.TempDir(), "cozy-product-test", "default-port-preferred")
	must(t, os.RemoveAll(preferredRoot))
	t.Cleanup(func() { _, _ = runCozy(t, preferredRoot, "down", "--all") })
	code, out = runCozy(t, preferredRoot, "up", "--json", "--full")
	if code != 0 {
		t.Fatalf("up on available preferred port failed [exit %d]\n%s", code, out)
	}
	var preferred struct {
		URL string `json:"url"`
	}
	if err := json.Unmarshal([]byte(out), &preferred); err != nil ||
		preferred.URL != "http://127.0.0.1:8818/" {
		t.Fatalf("available default did not bind 8818: %v\n%s", err, out)
	}
}

func TestDaemonStartupDiagnostic(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-startup-diagnostic")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	held, err := net.Listen("tcp4", "127.0.0.1:0") //cozy:allow product proof occupies one loopback port to exercise the real startup refusal
	must(t, err)
	port := held.Addr().(*net.TCPAddr).Port
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"),
		[]byte(fmt.Sprintf("port: %d\n", port)), 0o600))

	began := time.Now()
	code, out := runCozy(t, root, "up", "--json")
	if code != 1 {
		t.Fatalf("failed daemon startup exited %d, want operational 1\n%s", code, out)
	}
	if took := time.Since(began); took > 5*time.Second {
		t.Fatalf("failed daemon startup waited %s instead of relaying the exited child", took)
	}
	var document struct {
		Error struct {
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal([]byte(out), &document); err != nil ||
		document.Error.Code != "daemon_startup_failed" ||
		!strings.Contains(document.Error.Message, fmt.Sprintf("127.0.0.1:%d", port)) ||
		!strings.Contains(document.Error.Message, "held by another process") {
		t.Fatalf("startup failure did not relay its child diagnostic: %v\n%s", err, out)
	}
	if strings.Contains(strings.ToLower(out), "lock") {
		t.Fatalf("startup failure exposed its internal singleton mechanism\n%s", out)
	}
	if len(out) > maxDaemonDiagnosticOutput {
		t.Fatalf("startup diagnostic is unbounded: %d bytes", len(out))
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.log")); !os.IsNotExist(err) {
		t.Fatalf("failed startup created a persistent daemon log: %v", err)
	}

	must(t, held.Close())
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("root did not recover after the failed child [exit %d]\n%s", code, out)
	}
}

const maxDaemonDiagnosticOutput = 18 << 10

// TestProductPath drives only the public Kong surface: install a real weightless
// release, auto-start the daemon on invoke, cross Runtime and the worker wire,
// accept an output, release idle residency, then stop cleanly.
func TestProductPath(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "product")
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
	if code, out := runCozy(t, root, "invoke", "run", weightlessRef+"/v1/nosuch"); code != 1 ||
		!strings.Contains(out, "registers no function") {
		t.Fatalf("operational refusal was not shell exit 1 with detail [exit %d]\n%s", code, out)
	}

	if code, out := runCozy(t, root, "unload"); code != 0 || !strings.Contains(out, weightlessRef) {
		t.Fatalf("unload [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 ||
		!strings.Contains(out, "daemon:") || !strings.Contains(out, "stopped") {
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
		"tests/product/testdata/build-weightless.py", "--out", dir)
	build.Dir = "../.."
	build.Env = childEnv(t, repo, "RUNTIME_REPO="+repo)
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("building the weightless release: %v\n%s", err, out)
	}
	return filepath.Join(dir, "weightless-1.0.0.tar.gz")
}
