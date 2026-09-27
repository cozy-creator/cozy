package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// Retain a real interpreter outside uv's normal discovery and require every
// resolving uv operation to use the executable returned by the Runtime seam.
// The wrapper delegates all real resolution, wheel builds and metadata parsing.
func pythonExportToolchain(t *testing.T) (string, string, string) {
	t.Helper()
	uv, err := exec.LookPath("uv")
	must(t, err)
	runtime, err := exec.LookPath("cozy-runtime") //cozy:allow preserve the actual Runtime behind the selection-only test wrapper
	must(t, err)
	output, err := exec.Command(uv, "python", "find", "--no-project", "--no-config", "--no-python-downloads", "3.12").Output()
	must(t, err)
	source := strings.TrimSpace(string(output))
	output, err = exec.Command(source, "-I", "-c", "import platform; print(platform.python_version())").Output()
	must(t, err)
	version := strings.TrimSpace(string(output))
	root := t.TempDir()
	python := filepath.Join(root, "owned-python")
	must(t, os.Symlink(source, python))
	selected, err := json.Marshal(hostruntime.PythonInterpreter{Executable: python, Version: version, ABI: "cp312"})
	must(t, err)
	runtimeScript := fmt.Sprintf(`#!/bin/sh
if [ "$2" = version ]; then
 printf '%%s\n' '{"distribution":"%s","wire_protocol":"cozy.worker.v1+minor.54"}'
 exit 0
fi
if [ "$2" = python-ensure ]; then
 printf '%%s\n' '%s'
 exit 0
fi
exec %q "$@"
`, hostruntime.ToolFloor, selected, runtime)
	must(t, os.WriteFile(filepath.Join(root, "cozy-runtime"), []byte(runtimeScript), 0700)) //cozy:allow stand-in Runtime selection delegates every other command to the real tool
	log := filepath.Join(root, "uv-operations")
	wrapper := fmt.Sprintf(`#!/bin/sh
case "$1" in
 run|lock|export|tree|build)
  selected=''; previous=''; disabled=''
  for value do
   if [ "$previous" = --python ]; then selected="$value"; fi
   if [ "$value" = --no-python-downloads ]; then disabled=yes; fi
   previous="$value"
  done
  # A frozen export reads only the lock and may omit --python; everything else resolves with it.
  if { [ "$1" != export ] || [ -n "$selected" ]; } && [ "$selected" != %q ] || [ "$disabled" != yes ]; then
   echo "uv must receive the exact Runtime executable and disabled downloads" >&2
   exit 93
  fi
  printf '%%s\n' "$1" >> %q
  ;;
esac
exec %q "$@"
`, python, log, uv)
	must(t, os.WriteFile(filepath.Join(root, "uv"), []byte(wrapper), 0700)) //cozy:allow command-contract wrapper delegates to real uv
	// Test subprocesses need only these explicitly discovered tool directories.
	t.Setenv("PATH", strings.Join([]string{root, filepath.Dir(uv), filepath.Dir(runtime), filepath.Dir(source), "/usr/bin", "/bin"}, string(os.PathListSeparator)))
	return python, version, log
}

func TestPythonExportCaptureAndMetadataUseRuntimeExecutable(t *testing.T) {
	python, version, log := pythonExportToolchain(t)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte(`[project]
name = "python-export-proof"
version = "1.0.0"
requires-python = ">=3.12"
dependencies = ["packaging>=26.2,<27"]
[project.entry-points."cozy.application"]
app = "operation:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["operation.py"]
`), 0600))
	must(t, os.WriteFile(filepath.Join(root, "package.toml"), []byte("[application]\nobject = 'operation:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(root, ".python-version"), []byte(version+"\n"), 0600))
	must(t, os.WriteFile(filepath.Join(root, "operation.py"), []byte("from cozy_runtime.author import App\napp = App()\n"), 0600))
	pack, problem := packagepublish.PrepareUnpublishedFrom(context.Background(), root)
	fatal(t, problem)
	defer pack.Close()
	fatal(t, pack.Build(context.Background()))
	if pack.PythonVersion != version || len(pack.Registry) != 1 || pack.Registry[0].Name != "packaging" {
		t.Fatalf("wrong captured closure: python=%s registry=%+v", pack.PythonVersion, pack.Registry)
	}
	graph, problem := packagepublish.WheelClosures(context.Background(), pack.Tree, python,
		"python-export-proof==1.0.0\npackaging=="+pack.Registry[0].Version, "python-export-proof", "")
	fatal(t, problem)
	if graph["packaging"]["packaging"] != pack.Registry[0].Version {
		t.Fatalf("wrong native closure graph: %+v", graph)
	}
	wheelBytes, err := os.ReadFile(pack.Wheel)
	must(t, err)
	wheelHash := sha256.Sum256(wheelBytes)
	index := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/python-export-proof/":
			fmt.Fprintf(w, `<a href="/%s">wheel</a>`, filepath.Base(pack.Wheel))
		case "/" + filepath.Base(pack.Wheel):
			_, _ = w.Write(wheelBytes)
		default:
			http.NotFound(w, r)
		}
	}))
	defer index.Close()
	receipt, problem := install.MaterializePublishedEnvironment(pack.Tree, filepath.Join(t.TempDir(), "venv"), &install.PublishedSource{
		IndexURL: index.URL, ProjectWheel: install.PublishedWheel{
			Distribution: "python-export-proof", Version: "1.0.0", Filename: filepath.Base(pack.Wheel),
			Digest: "sha256:" + hex.EncodeToString(wheelHash[:]), Length: int64(len(wheelBytes)),
		},
	})
	fatal(t, problem)
	if receipt.Python != version {
		t.Fatalf("published environment changed Python: %+v", receipt)
	}
	raw, err := os.ReadFile(log)
	must(t, err)
	if strings.Count(string(raw), "export\n") != 2 {
		t.Fatalf("both registry and published export must run: %s", raw)
	}
	for _, operation := range []string{"lock\n", "build\n", "export\n", "tree\n"} {
		if !strings.Contains(string(raw), operation) {
			t.Fatalf("did not exercise %s: %s", operation, raw)
		}
	}
}

func TestPythonScriptLockUsesRuntimeExecutable(t *testing.T) {
	_, version, log := pythonExportToolchain(t)
	path := filepath.Join(t.TempDir(), "operation.py")
	must(t, os.WriteFile(path, []byte("# /// script\n# requires-python = '>=3.12'\n# dependencies = []\n# ///\nprint('not executed during capture')\n"), 0600))
	pack, problem := packagepublish.PrepareScript(context.Background(), path)
	fatal(t, problem)
	defer pack.Close()
	raw, err := os.ReadFile(filepath.Join(pack.Tree, ".python-version"))
	must(t, err)
	if want := strings.Join(strings.SplitN(version, ".", 3)[:2], "."); strings.TrimSpace(string(raw)) != want {
		t.Fatalf("script snapshot pinned %s, want the executor's minor %s", raw, want)
	}
	raw, err = os.ReadFile(log)
	must(t, err)
	if !strings.Contains(string(raw), "lock\n") {
		t.Fatalf("script lock not exercised: %s", raw)
	}
}
