package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
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
	"github.com/cozy-creator/cozy/internal/machines"
)

// `cozy machine install` keeps exactly the Runtime and TensorFS wheels it installed in
// <machine root>/opt/cozy/wheels, replacing an earlier build's, and none after a published
// install. With -machine-host and a -machine-runtime-wheel that installs packages from that
// directory, a package on the machine runs the machine's local build, and without the build's
// wheel the machine refuses rather than put PyPI's Runtime in its place.
func TestMachineKeepsTheWheelsItInstalled(t *testing.T) {
	if _, err := exec.LookPath("uv"); err != nil {
		t.Skip("uv lays out the machine root")
	}
	runtime, tensorfs := *machineRuntimeWheel, *machineTensorFSWheel
	if runtime == "" {
		runtime, tensorfs = publishedWheel(t, hostruntime.Distribution, oldestRuntime), publishedWheel(t, "tensorfs", "0.3.74")
	}
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
			t.Logf("evidence retained at %s\nlocal Host log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	dir := filepath.Join(root, "machine", "root", "opt", "cozy", "wheels")
	install := func(wheels ...string) {
		t.Helper()
		// The Host is this cozy; without -machine-host it is installed and never launched.
		args := []string{"machine", "install"}
		if len(wheels) > 0 {
			args = append(args, "--runtime-wheel", wheels[0], "--tensorfs-wheel", wheels[1])
		}
		if code, out := runCozy(t, root, args...); code != 0 {
			t.Fatalf("machine install [exit %d]\n%s", code, out)
		}
		entries, err := os.ReadDir(dir)
		must(t, err)
		var held, want []string
		for _, entry := range entries {
			held = append(held, entry.Name())
		}
		for _, wheel := range wheels {
			want = append(want, filepath.Base(wheel))
			if fileSHA(t, wheel) != fileSHA(t, filepath.Join(dir, filepath.Base(wheel))) {
				t.Fatalf("%s is not the installed %s", dir, filepath.Base(wheel))
			}
		}
		slices.Sort(want)
		if !slices.Equal(held, want) {
			t.Fatalf("%s holds %v, want exactly %v", dir, held, want)
		}
		host := filepath.Join(root, "machine", "root", "usr", "local", "bin")
		if link, _ := os.Readlink(filepath.Join(host, "pod-supervisor")); link != "cozy" || fileSHA(t, filepath.Join(host, "cozy")) != fileSHA(t, cozyBin) {
			t.Fatalf("the machine's Host is not this cozy: pod-supervisor -> %q", link)
		}
	}
	install(localBuild(t, runtime, "test1"), tensorfs)
	build := localBuild(t, runtime, "test2")
	install(build, tensorfs)

	if *machineHostBinary != "" && *machineRuntimeWheel != "" {
		virtualInventory(t, filepath.Join(root, "machine", "root"))
		version := strings.SplitN(filepath.Base(build), "-", 3)[1]
		if code, out := runCozy(t, root, "package", "install", sdkProbe(t, "wheels-probe"), "--editable"); code != 0 {
			t.Fatalf("editable install [exit %d]\n%s", code, out)
		}
		code, out := runCozy(t, root, "run", "local/wheels-probe/sdk", "value=1", "--await", "--json")
		if code != 0 || !strings.Contains(out, `"runtime":"`+version+`"`) {
			t.Fatalf("the package did not run the machine's %s [exit %d]\n%s", version, code, out)
		}
		must(t, os.Remove(filepath.Join(dir, filepath.Base(build))))
		code, out = runCozy(t, root, "package", "install", sdkProbe(t, "wheels-probe-b"), "--editable")
		if code == 0 {
			code, out = runCozy(t, root, "run", "local/wheels-probe-b/sdk", "value=1", "--await", "--json")
		}
		if code == 0 || !strings.Contains(out, "package_runtime_unavailable") {
			t.Fatalf("a machine lacking its build's wheel did not refuse [exit %d]\n%s", code, out)
		}
	}

	install()
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

// sdkProbe is an editable package whose one job reports the Runtime its environment holds.
func sdkProbe(t *testing.T, name string) string {
	t.Helper()
	module := strings.ReplaceAll(name, "-", "_")
	project := filepath.Join(t.TempDir(), name)
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="`+name+`"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+machines.RuntimeFloor+`", "msgspec>=0.19"]
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


@app.job
def sdk(payload: Ask) -> SDK:
    return SDK(runtime=importlib.metadata.version("`+hostruntime.Distribution+`"))
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking %s: %v\n%s", name, err, out)
	}
	return project
}
