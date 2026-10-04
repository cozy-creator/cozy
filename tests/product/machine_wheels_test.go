package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// `cozy machine install` makes this computer's machine the Rust machine a worker image runs:
// the named executable at usr/local/bin/cozy-machine, exactly the Runtime and TensorFS wheels
// at opt/cozy/machine/wheels, uv beside it, and no Python worker. It starts as a persistent
// machine, proves readiness through Status, installs and runs a local package, and installing
// again updates it in place with Run kind: update.
func TestMachineKeepsTheWheelsItInstalled(t *testing.T) {
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires -machine-host and a -machine-runtime-wheel/-machine-tensorfs-wheel pair")
	}
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the machine root")
	}
	runtime, tensorfs := *machineRuntimeWheel, *machineTensorFSWheel
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czw")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("evidence retained at %s\nlocal machine log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	machineRoot := filepath.Join(root, "machine", "root")
	version := func(wheel string) string { return strings.SplitN(filepath.Base(wheel), "-", 3)[1] }
	show := func() map[string]any {
		t.Helper()
		code, out := runCozy(t, root, "machine", "show", "--json")
		var shown map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil {
			t.Fatalf("machine show [exit %d]\n%s", code, out)
		}
		return shown
	}

	first := localBuild(t, runtime, "test1")
	if code, out := runCozy(t, root, "machine", "install", "--host", *machineHostBinary, "--runtime-wheel", first, "--tensorfs-wheel", tensorfs); code != 0 {
		t.Fatalf("machine install [exit %d]\n%s", code, out)
	}
	entries, err := os.ReadDir(filepath.Join(machineRoot, "opt/cozy/machine/wheels"))
	must(t, err)
	var held []string
	for _, entry := range entries {
		held = append(held, entry.Name())
	}
	want := []string{filepath.Base(first), filepath.Base(tensorfs)}
	slices.Sort(want)
	if !slices.Equal(held, want) {
		t.Fatalf("the machine holds %v, want exactly %v", held, want)
	}
	if fileSHA(t, filepath.Join(machineRoot, "usr/local/bin/cozy-machine")) != fileSHA(t, *machineHostBinary) {
		t.Fatal("the installed machine differs from the named executable")
	}
	if _, err := os.Stat(filepath.Join(machineRoot, "usr/local/bin/uv")); err != nil {
		t.Fatal("the machine root has no uv")
	}
	for _, retired := range []string{"opt/cozy/python", "opt/cozy/bin/cozy-runtime-worker"} {
		if _, err := os.Lstat(filepath.Join(machineRoot, retired)); err == nil {
			t.Fatalf("the install laid out the retired %s", retired)
		}
	}
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start [exit %d]\n%s", code, out)
	}
	if shown := show(); shown["phase"] != "ready" || shown["runtime_version"] != version(first) {
		t.Fatalf("the started machine is not ready on %s: %v", version(first), shown)
	}

	if code, out := runCozy(t, root, "package", "install", sdkProbe(t, "wheels-probe"), "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	// A local package installs through the helper the machine makes from its embedded client,
	// and runs the machine's own Runtime, not the one its lock captured.
	code, out := runCozy(t, root, "run", "local/wheels-probe/sdk", "value=1", "--await", "--json")
	if code != 0 || !strings.Contains(out, `"status":"completed"`) || !strings.Contains(out, `"runtime":"`+version(first)+`"`) {
		t.Fatalf("the local package did not run on the machine's Runtime %s [exit %d]\n%s", version(first), code, out)
	}

	second := localBuild(t, runtime, "test2")
	if code, out := runCozy(t, root, "machine", "install", "--host", *machineHostBinary, "--runtime-wheel", second, "--tensorfs-wheel", tensorfs); code != 0 {
		t.Fatalf("machine update [exit %d]\n%s", code, out)
	}
	if shown := show(); shown["runtime_version"] != version(second) {
		t.Fatalf("installing again did not update the machine to %s: %v", version(second), shown)
	}
}

func fileSHA(t *testing.T, path string) string {
	t.Helper()
	raw, err := os.ReadFile(path)
	must(t, err)
	return sha256Hex(raw)
}

// localBuild is wheel under the local version <public version>+<label>, which no index
// serves: the same bytes a developer's own Runtime build carries.
func localBuild(t *testing.T, wheel, label string) string {
	t.Helper()
	parts := strings.SplitN(filepath.Base(wheel), "-", 3)
	public, _, _ := strings.Cut(parts[1], "+")
	version := public + "+" + label
	from, to := parts[0]+"-"+parts[1]+".dist-info/", parts[0]+"-"+version+".dist-info/"
	archive, err := zip.OpenReader(wheel)
	must(t, err)
	defer archive.Close()
	path := filepath.Join(t.TempDir(), parts[0]+"-"+version+"-"+parts[2])
	file, err := os.Create(path)
	must(t, err)
	out := zip.NewWriter(file)
	var record strings.Builder
	write := func(header zip.FileHeader, body []byte) {
		w, err := out.CreateHeader(&header)
		must(t, err)
		_, err = w.Write(body)
		must(t, err)
	}
	for _, entry := range archive.File {
		name := strings.Replace(entry.Name, from, to, 1)
		name = strings.Replace(name, parts[0]+"-"+parts[1]+".data/", parts[0]+"-"+version+".data/", 1)
		if name == to+"RECORD" {
			continue
		}
		reader, err := entry.Open()
		must(t, err)
		body, err := io.ReadAll(reader)
		reader.Close()
		must(t, err)
		if name == to+"METADATA" {
			relabeled := bytes.Replace(body, []byte("\nVersion: "+parts[1]+"\n"), []byte("\nVersion: "+version+"\n"), 1)
			if bytes.Equal(relabeled, body) {
				t.Fatalf("%s names no version %s", entry.Name, parts[1])
			}
			body = relabeled
		}
		header := entry.FileHeader
		header.Name = name
		write(header, body)
		sum := sha256.Sum256(body)
		fmt.Fprintf(&record, "%s,sha256=%s,%d\n", name, base64.RawURLEncoding.EncodeToString(sum[:]), len(body))
	}
	write(zip.FileHeader{Name: to + "RECORD", Method: zip.Deflate}, []byte(record.String()+to+"RECORD,,\n"))
	must(t, out.Close())
	must(t, file.Close())
	return path
}

// sdkProbe is an editable package whose one entrypoint reports the Runtime its environment holds.
func sdkProbe(t *testing.T, name string) string {
	t.Helper()
	module := strings.ReplaceAll(name, "-", "_")
	project := filepath.Join(t.TempDir(), name)
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="`+name+`"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+runtimeFloor+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="`+module+`:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["`+module+`.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\""+module+":app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, module+".py"), []byte(`import importlib.metadata

import msgspec
from cozy_runtime.author import App


class Ask(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class SDK(msgspec.Struct):
    runtime: str


app = App()


@app.entrypoint
def sdk(payload: Ask) -> SDK:
    return SDK(runtime=importlib.metadata.version("`+hostruntime.Distribution+`"))
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--refresh-package", hostruntime.Distribution, "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking %s: %v\n%s", name, err, out)
	}
	return project
}
