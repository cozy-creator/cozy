package producttest

import (
	"context"
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

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const weightlessRef = "cozy/weightless"

func TestPackagePublishMetadataGrammar(t *testing.T) {
	root := t.TempDir()
	if code, help := runCozy(t, root, "package", "publish", "--help"); code != 0 ||
		strings.Contains(help, "--release") || strings.Contains(help, "--dir") ||
		strings.Contains(help, "<package>") {
		t.Fatalf("package publish retained caller-authored identity [exit %d]\n%s", code, help)
	}
	if code, help := runCozy(t, root, "package", "install", "--help"); code != 0 ||
		strings.Contains(help, "--profile") || strings.Contains(help, "--major") {
		t.Fatalf("package install retained the unusable managed-local lane [exit %d]\n%s", code, help)
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

func TestPackagePublishRefusesSilentlyOmittedPrivateFiles(t *testing.T) {
	project := t.TempDir()
	writePublishProject(t, project, "private-package", "1.0.0", nil, "", true)
	must(t, os.WriteFile(filepath.Join(project, ".env"), []byte("TOKEN=secret\n"), 0o600))
	pack, problem := packagepublish.PrepareFrom(project)
	if pack != nil {
		pack.Close()
	}
	if problem == nil || problem.Name != "package_source_file_refused" {
		t.Fatalf("private source file was silently skipped: %v", problem)
	}
}

func TestPackagePublishCommittedReplaySkipsBuildAndStaysCompact(t *testing.T) {
	project := t.TempDir()
	must(t, os.Mkdir(filepath.Join(project, "replay_package"), 0o755))
	must(t, os.WriteFile(filepath.Join(project, "replay_package", "__init__.py"), []byte("VALUE = 1\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[build-system]
requires = []
build-backend = "backend.that.does.not.exist"

[project]
name = "replay-package"
version = "1.0.0"

[tool.cozy]
organization = "proof"
`), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = \"replay_package:app\"\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(project, "uv.lock"), []byte("version = 1\n"), 0o644))

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.Method {
		case http.MethodPost:
			_, _ = io.WriteString(w, `{"project_wheel_upload":{"already_uploaded":true,"required_headers":{},"url":""},"state":"committed"}`)
		case http.MethodPut:
			_, _ = io.WriteString(w, `{"compatible_profiles":[],"package_executions":[],"profiles":[],"qualification_state":"future-compatible-state","requirements":[],"requires_python":""}`)
		default:
			http.Error(w, "unexpected method", http.StatusMethodNotAllowed)
		}
	}))
	defer server.Close()
	code, out := runCozyDir(t, t.TempDir(), project,
		[]string{"TENSORHUB_URL=" + server.URL, "TENSORHUB_TOKEN=proof-token"}, "package", "publish")
	if code != 0 || !strings.Contains(out, "status:  already published") ||
		strings.Contains(out, "qualification:") || strings.Contains(out, "changed:") {
		t.Fatalf("committed replay built the project or emitted verbose state [exit %d]\n%s", code, out)
	}
}

func TestPackagePublishBuildsBoundedLocalDependencyClosure(t *testing.T) {
	workspace := t.TempDir()
	projects := filepath.Join(workspace, "projects")
	must(t, os.MkdirAll(projects, 0o755))
	must(t, os.WriteFile(filepath.Join(workspace, "pyproject.toml"), []byte(
		"[tool.uv.workspace]\nmembers = [\"projects/*\"]\n"), 0o644))

	a := filepath.Join(projects, "local-a")
	b := filepath.Join(projects, "local-b")
	c := filepath.Join(projects, "local-c")
	platformCandidate := filepath.Join(projects, "platform-candidate")
	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b>=2,<3", "local-b[images]>=2,<3", "cozy-runtime>=0.0.3"},
		"local-b = [{ workspace = true, marker = \"sys_platform == 'linux'\" }, { index = \"pypi\", marker = \"sys_platform != 'linux'\" }]\ncozy-runtime = { workspace = true, editable = true }\n", true)
	writePublishProject(t, b, "local-b", "2.1.0", nil,
		"local-c = { path = \"../local-c\", editable = true }\nabsent-local = { path = \"../absent-local\" }\n", false)
	appendProjectTOML(t, b, "\n[project.optional-dependencies]\nimages = [\"local-c==3.0.0\"]\nunused = [\"absent-local==1\"]\n")
	writePublishProject(t, c, "local-c", "3.0.0", nil, "", false)
	writePublishProject(t, platformCandidate, "cozy-runtime", "0.0.3", nil, "", false) //cozy:allow distribution fixture, not executable access

	pack, problem := preparePublishPackage(a)
	fatal(t, problem)
	defer pack.Close()
	wheels := map[string]string{}
	for _, dependency := range pack.DependencyWheels {
		identity, problem := wheel.InspectIdentity(dependency.Path)
		fatal(t, problem)
		wheels[identity.Distribution] = identity.Version
	}
	if len(pack.DependencyWheels) != 3 || wheels["local-b"] != "2.1.0" ||
		wheels["local-c"] != "3.0.0" || wheels["cozy-runtime"] != "0.0.3" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("local dependency closure did not include the requested extra and base-name candidate: %+v", pack.DependencyWheels)
	}
	for _, dependency := range pack.DependencyWheels {
		if !strings.HasSuffix(dependency.Filename, ".whl") {
			t.Fatalf("dependency did not retain a wheel basename: %+v", dependency)
		}
		if info, err := os.Stat(dependency.Path); err != nil || !info.Mode().IsRegular() {
			t.Fatalf("dependency wheel is not a staged regular file: %+v err=%v", dependency, err)
		}
	}

	// Creator does not guess base ownership. It uploads Runtime as candidate
	// custody; Tensorhub's exact profile inventory must select the base copy and
	// omit this wheel from the eventual overlay.
	if wheels["cozy-runtime"] != "0.0.3" { //cozy:allow distribution assertion, not executable access
		t.Fatalf("local Runtime candidate was silently discarded: %+v", pack.DependencyWheels)
	}

	writePublishProject(t, a, "local-a", "1.0.0", []string{"local-b>=3"},
		"local-b = { workspace = true }\n", true)
	if incompatible, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_version_incompatible" {
		if incompatible != nil {
			incompatible.Close()
		}
		t.Fatalf("local wheel outside the parent PEP 440 requirement did not refuse: %v", problem)
	}

	writePublishProject(t, a, "local-a", "1.0.0",
		[]string{"local-b[images]>=2,<3"},
		"local-b = { workspace = true }\n", true)
	writePublishProject(t, c, "local-c", "3.0.0", []string{"local-a==1.0.0"},
		"local-a = { path = \"../local-a\" }\n", false)
	if cycle, problem := preparePublishPackage(a); problem == nil || problem.Name != "local_dependency_cycle" {
		if cycle != nil {
			cycle.Close()
		}
		t.Fatalf("A->B->C->A did not refuse as a local dependency cycle: %v", problem)
	}

	conflictRoot := filepath.Join(t.TempDir(), "root")
	x1 := filepath.Join(filepath.Dir(conflictRoot), "x1")
	x2 := filepath.Join(filepath.Dir(conflictRoot), "x2")
	left := filepath.Join(filepath.Dir(conflictRoot), "left")
	right := filepath.Join(filepath.Dir(conflictRoot), "right")
	writePublishProject(t, conflictRoot, "conflict-root", "1.0.0",
		[]string{"left==1", "right==1"},
		"left = { path = \"../left\" }\nright = { path = \"../right\" }\n", true)
	writePublishProject(t, left, "left", "1", []string{"shared==1"},
		"shared = { path = \"../x1\" }\n", false)
	writePublishProject(t, right, "right", "1", []string{"shared==2"},
		"shared = { path = \"../x2\" }\n", false)
	writePublishProject(t, x1, "shared", "1", nil, "", false)
	writePublishProject(t, x2, "shared", "2", nil, "", false)
	if conflict, problem := preparePublishPackage(conflictRoot); problem == nil || problem.Name != "local_dependency_duplicate" {
		if conflict != nil {
			conflict.Close()
		}
		t.Fatalf("different sources for one normalized name did not refuse: %v", problem)
	}

	countRoot := filepath.Join(t.TempDir(), "root")
	writePublishProject(t, countRoot, "count-root", "1", []string{"count-01==1"},
		"count-01 = { path = \"../count-01\" }\n", true)
	countParent := filepath.Dir(countRoot)
	for i := 1; i <= packagepublish.MaxDependencyWheels+1; i++ {
		name := fmt.Sprintf("count-%02d", i)
		var dependencies []string
		var sources string
		if i <= packagepublish.MaxDependencyWheels {
			next := fmt.Sprintf("count-%02d", i+1)
			dependencies = []string{next + "==1"}
			sources = fmt.Sprintf("%s = { path = \"../%s\" }\n", next, next)
		}
		writePublishProject(t, filepath.Join(countParent, name), name, "1", dependencies, sources, false)
	}
	if counted, problem := preparePublishPackage(countRoot); problem == nil || problem.Name != "local_dependency_count_exceeded" {
		if counted != nil {
			counted.Close()
		}
		t.Fatalf("dependency closure above %d wheels was not refused: %v",
			packagepublish.MaxDependencyWheels, problem)
	}

	directRoot := filepath.Join(t.TempDir(), "direct")
	writePublishProject(t, directRoot, "direct-root", "1", []string{"foreign @ https://example.invalid/foreign.whl"}, "", true)
	if direct, problem := preparePublishPackage(directRoot); problem == nil || problem.Name != "project_dependency_direct_url_unsupported" {
		if direct != nil {
			direct.Close()
		}
		t.Fatalf("direct URL dependency did not refuse: %v", problem)
	}

	gitRoot := filepath.Join(t.TempDir(), "git")
	writePublishProject(t, gitRoot, "git-root", "1", []string{"foreign==1"},
		"foreign = { git = \"https://example.invalid/foreign.git\" }\n", true)
	if git, problem := preparePublishPackage(gitRoot); problem == nil || problem.Name != "project_dependency_source_unsupported" {
		if git != nil {
			git.Close()
		}
		t.Fatalf("VCS dependency source did not refuse: %v", problem)
	}
}

func preparePublishPackage(root string) (*packagepublish.Package, *exit.Error) {
	pack, problem := packagepublish.PrepareFrom(root)
	if problem != nil {
		return nil, problem
	}
	if problem := pack.Build(context.Background()); problem != nil {
		pack.Close()
		return pack, problem
	}
	return pack, nil
}

func writePublishProject(t *testing.T, root, name, version string, dependencies []string, sources string, publishable bool) {
	t.Helper()
	if dependencies == nil {
		dependencies = []string{}
	}
	must(t, os.MkdirAll(filepath.Join(root, strings.ReplaceAll(name, "-", "_")), 0o755))
	must(t, os.WriteFile(filepath.Join(root, strings.ReplaceAll(name, "-", "_"), "__init__.py"),
		[]byte("VALUE = 1\n"), 0o644))
	dependencyJSON, err := json.Marshal(dependencies)
	must(t, err)
	document := fmt.Sprintf(`[build-system]
requires = ["uv_build>=0.12.7,<0.13"]
build-backend = "uv_build"

[project]
name = %q
version = %q
dependencies = %s

[tool.uv.build-backend]
module-root = ""
`, name, version, dependencyJSON)
	if sources != "" {
		document += "\n[tool.uv.sources]\n" + sources
	}
	if publishable {
		document += "\n[tool.cozy]\norganization = \"proof\"\n"
		must(t, os.WriteFile(filepath.Join(root, "package.toml"), []byte(
			"[application]\nobject = \""+strings.ReplaceAll(name, "-", "_")+":app\"\n"), 0o644))
		must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte("version = 1\n"), 0o644))
	}
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(document), 0o644))
}

func appendProjectTOML(t *testing.T, root, document string) {
	t.Helper()
	file, err := os.OpenFile(filepath.Join(root, "pyproject.toml"), os.O_APPEND|os.O_WRONLY, 0)
	must(t, err)
	_, err = file.WriteString(document)
	must(t, err)
	must(t, file.Close())
}

func TestDaemonWebLifecycle(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "daemon-web")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down", "--all") })

	code, help := runCozy(t, root)
	for _, want := range []string{
		"Usage: cozy", "package install", "model download", "auth login", "invoke run",
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
