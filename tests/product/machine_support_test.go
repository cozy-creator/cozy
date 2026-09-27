package producttest

import (
	"context"
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
)

var (
	machineHostBinary    = flag.String("machine-host", "", "pod-supervisor binary every test root's local machine runs")
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

// provisionMachine gives a test root this computer's machine when the run names a Host.
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
		must(t, os.Symlink(target, path))
	}
	installed, err := os.ReadFile(filepath.Join(template, "installed.json"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "installed.json"), installed, 0o600))
}

// skipWithoutMachine turns a local execution this run cannot host into a skip: the suite
// gives each root a machine only when it is told which Host binary to run.
func skipWithoutMachine(t *testing.T, code int, output string) {
	t.Helper()
	if code != 0 && *machineHostBinary == "" && strings.Contains(output, "machine.not_installed") {
		t.Skip("local execution runs on this computer's machine; pass -machine-host=<pod-supervisor>")
	}
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
