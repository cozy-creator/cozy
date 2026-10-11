package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/records"
)

// Authored package directories retain package-relative modules and dependencies;
// the CLI's static description must not execute their source or modify their lock.
func TestRunExplicitPackageDirectoryCapturesItsTypedInterface(t *testing.T) {
	root := t.TempDir()
	project, metadata, source := directoryProofProject(t)
	run := func(target string, extra ...string) (int, string) {
		t.Helper()
		args := append([]string{"run", target, "--describe"}, extra...)
		cmd := exec.Command(cozyBin, args...)
		cmd.Dir, cmd.Env = filepath.Dir(project), childEnv(t, root)
		var out bytes.Buffer
		cmd.Stdout, cmd.Stderr = &out, &out
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), out.String()
	}
	for i, target := range []string{"./project", "./project/verify-value", project + "/verify_value", project} {
		code, out := run(target)
		if code != 0 || !strings.Contains(out, "local/directory-proof/verify_value") || !strings.Contains(out, "value") {
			t.Fatalf("directory %s [exit %d]: %s", target, code, out)
		}
		// The first run reads the tree; an unchanged tree then reuses that reading.
		if captured := strings.Contains(out, "Reading local package"); captured != (i == 0) {
			t.Fatalf("directory %s captured=%t: %s", target, captured, out)
		}
	}
	code, out := run("./project/hidden-job", "--json")
	if code == 0 || !strings.Contains(out, `"code":"callable_internal"`) {
		t.Fatalf("internal function: %d %s", code, out)
	}
	code, out = run("./missing/verify", "--json")
	if code == 0 || !strings.Contains(out, "local_package_not_found") {
		t.Fatalf("missing directory: %d %s", code, out)
	}
	must(t, os.WriteFile(filepath.Join(project, "directory_proof", "__init__.py"), []byte(strings.Replace(source, "@app.job(internal=True)", "@app.job", 1)), 0600))
	code, out = run("./project")
	if code != 0 || !strings.Contains(out, "hidden_job") || !strings.Contains(out, "verify_value") || !strings.Contains(out, "Reading local package") {
		t.Fatalf("multiple callable directory, read again after its edit: %d %s", code, out)
	}
	if _, err := os.Stat(filepath.Join(project, "uv.lock")); !os.IsNotExist(err) {
		t.Fatalf("description mutated author lock: %v", err)
	}
	actual, err := os.ReadFile(filepath.Join(project, "pyproject.toml"))
	must(t, err)
	if string(actual) != metadata {
		t.Fatal("description mutated project metadata")
	}
}

func TestRunNamedPackageNeverUsesAnExistingDirectory(t *testing.T) {
	var reads atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		reads.Add(1)
		http.NotFound(w, r)
	}))
	defer server.Close()
	root, project := t.TempDir(), t.TempDir()
	must(t, os.MkdirAll(filepath.Join(project, "org", "package"), 0700))
	must(t, os.WriteFile(filepath.Join(project, "org", "package", "package.toml"), []byte("invalid local package must never be read"), 0600))
	cmd := exec.Command(cozyBin, "run", "org/package/verify", "--describe", "--rental-only", "--tensorhub="+server.URL, "--json")
	cmd.Dir, cmd.Env = project, childEnv(t, root)
	out, err := cmd.CombinedOutput()
	if err == nil || reads.Load() == 0 || strings.Contains(string(out), "local_package") {
		t.Fatalf("named package escaped selected Hub: %v %s, reads=%d", err, out, reads.Load())
	}
}

func directoryProofProject(t *testing.T) (project, metadata, source string) {
	t.Helper()
	project = filepath.Join(t.TempDir(), "project")
	must(t, os.MkdirAll(filepath.Join(project, "directory_proof"), 0700))
	metadata = fmt.Sprintf(`[project]
name="directory-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="directory_proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
packages=["directory_proof"]
`, hostruntime.PackageFloor)
	if *privateScriptRuntimeWheel != "" {
		metadata += fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\n", *privateScriptRuntimeWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='directory_proof:app'\n"), 0600))
	source = `import msgspec
from cozy_runtime.author import App, Context
app = App()
class Request(msgspec.Struct):
    value: int = 7
class Result(msgspec.Struct):
    value: int
@app.job
def verify_value(payload: Request, ctx: Context) -> Result:
    from .helper import increment
    return Result(increment(payload.value))
@app.job(internal=True)
def hidden_job(payload: Request, ctx: Context) -> Result:
    return Result(payload.value)
raise RuntimeError("directory source must not execute during description")
`
	must(t, os.WriteFile(filepath.Join(project, "directory_proof", "__init__.py"), []byte(source), 0600))
	must(t, os.WriteFile(filepath.Join(project, "directory_proof", "helper.py"), []byte("def increment(value): return value + 1\n"), 0600))
	return
}

// The ordinary endpoint command uploads the captured directory and executes its
// relative helper on a real CPU machine; no prior editable install is needed.
func TestRunDirectoryExecutesOnExplicitEndpoint(t *testing.T) {
	runDirectoryOnExplicitEndpoint(t, false)
}

func TestRunDirectoryEntrypointUsesItsCaptureWithoutAnActivePin(t *testing.T) {
	runDirectoryOnExplicitEndpoint(t, true)
}

func runDirectoryOnExplicitEndpoint(t *testing.T, serving bool) {
	t.Helper()
	if *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires a real machine and paired Runtime fixture wheel")
	}
	root, err := os.MkdirTemp("", "czdir-")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Log("endpoint directory evidence retained", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start: %d %s", code, out)
	}
	var agent struct {
		WorkerID   string `json:"worker_id"`
		WorkerPort int    `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	leaf, err := os.ReadFile(filepath.Join(root, "machine", "leaf.pem"))
	must(t, err)
	endpoint := filepath.Join(root, "endpoint.json")
	document, _ := json.Marshal(map[string]string{"format": "cozy.machine.endpoint/1", "address": fmt.Sprintf("127.0.0.1:%d", agent.WorkerPort), "worker_id": agent.WorkerID, "worker_boot_id": "boot", "tls_certificate_pem": string(leaf), "execution_workspace_id": "workspace"})
	must(t, os.WriteFile(endpoint, document, 0600))
	project, _, source := directoryProofProject(t)
	source = strings.Replace(source, "raise RuntimeError(\"directory source must not execute during description\")\n", "", 1)
	if serving {
		source = strings.Replace(source, "@app.job\ndef verify_value", "@app.entrypoint\ndef verify_value", 1)
	}
	must(t, os.WriteFile(filepath.Join(project, "directory_proof", "__init__.py"), []byte(source), 0600))
	code, out := runCozy(t, root, "run", project+"/verify-value", "--machine-endpoint-file", endpoint, "--await", "--json")
	if code != 0 || !strings.Contains(out, `"value":8`) {
		t.Fatalf("captured endpoint job: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByReference("1")
	fatal(t, problem)
	if request == nil || request.LocalInstallationID == "" || request.InstallID == "" || request.State != "succeeded" {
		t.Fatalf("source capture was not durably executed: %+v", request)
	}
	pins, problem := store.Pins("local/directory-proof")
	fatal(t, problem)
	if len(pins) != 0 {
		t.Fatal("explicit source execution silently installed an active package pin")
	}
	if _, err := os.Stat(filepath.Join(project, "uv.lock")); !os.IsNotExist(err) {
		t.Fatalf("run wrote an author lock: %v", err)
	}
}
