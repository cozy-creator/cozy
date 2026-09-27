package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"flag"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

var (
	machineHostBinary    = flag.String("machine-host", "", "pod-supervisor binary every test root's local machine runs")
	requireMachineHost   = flag.Bool("require-machine-host", false, "fail, never skip, a local execution the run cannot host (CI)")
	machineRuntimeWheel  = flag.String("machine-runtime-wheel", "", "Runtime wheel the test machines run; default: the published Runtime")
	machineTensorFSWheel = flag.String("machine-tensorfs-wheel", "", "TensorFS wheel paired with -machine-runtime-wheel")
)

// One installed machine layout for the whole run; each root's machine links its executables,
// as a pod's image is shared and its / is its own.
var machineTemplate struct {
	once    sync.Once
	dir     string
	problem *exit.Error
}

func machineTemplateDir(t *testing.T) string {
	t.Helper()
	machineTemplate.once.Do(func() {
		machineTemplate.dir = filepath.Join(scratchBase, "machine-template")
		uv, err := exec.LookPath("uv")
		if err != nil {
			machineTemplate.problem = exit.New(exit.NotFound, "uv lays out the test machines: %s", err)
			return
		}
		source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
		_, machineTemplate.problem = machines.NewHost(machineTemplate.dir, nil).Install(context.Background(), source, uv)
	})
	fatal(t, machineTemplate.problem)
	return machineTemplate.dir
}

// provisionMachine gives a test root this computer's machine when the run names a Host. A
// root on the suite's unanswered default hub is also registered, as `cozy auth login` plus a
// first run would register it, with the suite's worker doors as its hub; a root naming its
// own stand-in hub registers through that hub.
func provisionMachine(t *testing.T, root string) {
	t.Helper()
	if *machineHostBinary == "" {
		return
	}
	dir := filepath.Join(root, "machine")
	if _, err := os.Stat(filepath.Join(dir, "installed.json")); err == nil {
		return
	}
	template := machineTemplateDir(t)
	for _, link := range []string{"usr/local/bin/pod-supervisor", "usr/local/bin/tfs", "usr/local/bin/uv", "opt/cozy/bin/cozy-runtime-worker", "opt/cozy/python"} {
		target, err := filepath.EvalSymlinks(filepath.Join(template, "root", link))
		must(t, err)
		path := filepath.Join(dir, "root", link)
		must(t, os.MkdirAll(filepath.Dir(path), 0o755))
		if strings.HasSuffix(link, "pod-supervisor") {
			// Its own path, so the running Host is found and stopped by this root's teardown.
			if os.Link(target, path) != nil {
				raw, err := os.ReadFile(target)
				must(t, err)
				must(t, os.WriteFile(path, raw, 0o755))
			}
			continue
		}
		must(t, os.Symlink(target, path))
	}
	if raw, _ := os.ReadFile(filepath.Join(root, config.FileName)); !strings.Contains(string(raw), "tensorhub_url") {
		registration, _ := json.Marshal(map[string]string{"hub": testDefaultHub, "id": "om-" + randomToken(t)[:22], "worker_token": randomToken(t)})
		environment, _ := json.Marshal(suiteWorkerDoors(t))
		must(t, os.WriteFile(filepath.Join(dir, "registration.json"), registration, 0o600))
		must(t, os.WriteFile(filepath.Join(dir, "environment.json"), environment, 0o600))
	}
	installed, err := os.ReadFile(filepath.Join(template, "installed.json"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), installed, 0o600))
}

// suiteWorkerDoors is the hub a pre-registered test machine's Host calls: its idle release
// and cache reports, accepted.
var workerDoors struct {
	once        sync.Once
	environment map[string]string
}

func suiteWorkerDoors(t *testing.T) map[string]string {
	t.Helper()
	workerDoors.once.Do(func() {
		server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/v1/worker/rental/release", "/v1/worker/rental/cache-observations":
				w.WriteHeader(http.StatusNoContent)
			default:
				http.NotFound(w, r)
			}
		}))
		workerDoors.environment = map[string]string{"TENSORHUB_ORIGIN": server.URL, "TENSORHUB_PUBLIC_ORIGIN": server.URL,
			"TENSORHUB_CA_DER_B64URL": base64.RawURLEncoding.EncodeToString(server.Certificate().Raw)}
	})
	return workerDoors.environment
}

// skipWithoutMachine turns a local execution this run cannot host into a skip: the suite
// gives each root a machine only when it is told which Host binary to run. A run that
// requires the Host (CI) fails instead.
func skipWithoutMachine(t *testing.T, code int, output string) {
	t.Helper()
	if code == 0 || *machineHostBinary != "" || !strings.Contains(output, "machine.not_installed") {
		return
	}
	if *requireMachineHost {
		t.Fatalf("local execution needs this computer's machine and the run has no -machine-host:\n%s", output)
	}
	t.Skip("local execution runs on this computer's machine; pass -machine-host=<pod-supervisor>")
}

// A run that requires the machine Host proves it has one before any test relies on it.
func TestMachineHostIsPresentWhenRequired(t *testing.T) {
	if !*requireMachineHost {
		t.Skip("the run does not require the machine Host")
	}
	if *machineHostBinary == "" {
		t.Fatal("-require-machine-host without -machine-host: local execution tests cannot run")
	}
	if info, err := os.Stat(*machineHostBinary); err != nil || info.Mode()&0o111 == 0 {
		t.Fatalf("-machine-host %s is not an executable: %v", *machineHostBinary, err)
	}
	machineTemplateDir(t)
}

// stubMachine gives root an installed, registered machine whose Host is script: a sentinel
// that proves a launch was attempted without running one.
func stubMachine(t *testing.T, root, script string) {
	t.Helper()
	dir := filepath.Join(root, "machine")
	binary := filepath.Join(dir, "root", "usr", "local", "bin", "pod-supervisor")
	must(t, os.MkdirAll(filepath.Dir(binary), 0o755))
	must(t, os.WriteFile(binary, []byte(script), 0o700)) //cozy:allow sentinel Host proves whether a launch was attempted
	for name, body := range map[string]string{
		"installed.json":    `{"host":{"name":"pod-supervisor"}}`,
		"registration.json": `{"hub":"` + testDefaultHub + `","id":"om-stub","worker_token":"` + strings.Repeat("A", 43) + `"}`,
		"environment.json":  `{"TENSORHUB_ORIGIN":"https://hub.invalid"}`,
	} {
		must(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600))
	}
}
