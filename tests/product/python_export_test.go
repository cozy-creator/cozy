package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hostruntime"
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
`, hostruntime.Floor, selected, runtime)
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
  if [ "$selected" != %q ] || [ "$disabled" != yes ]; then
   echo "uv must receive the exact Runtime executable and disabled downloads" >&2
   exit 93
  fi
  printf '%%s\n' "$1" >> %q
  ;;
esac
exec %q "$@"
`, python, log, uv)
	must(t, os.WriteFile(filepath.Join(root, "uv"), []byte(wrapper), 0700)) //cozy:allow command-contract wrapper delegates to real uv
	t.Setenv("PATH", root+string(os.PathListSeparator)+os.Getenv("PATH"))   //cozy:allow retain the original test PATH behind tool guards
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
	active, problem := packagepublish.EvaluateRequirements(context.Background(), []string{
		"selected>=1; python_version == '3.13'", "inactive>=1; python_version == '3.12'",
	}, "3.13.10")
	fatal(t, problem)
	if len(active) != 1 || !strings.HasPrefix(active[0], "selected") {
		t.Fatalf("target markers changed: %v", active)
	}
	raw, err := os.ReadFile(log)
	must(t, err)
	for _, operation := range []string{"lock\n", "build\n", "export\n", "tree\n", "run\n"} {
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
	if strings.TrimSpace(string(raw)) != version {
		t.Fatalf("script lost exact patch: %s", raw)
	}
	raw, err = os.ReadFile(log)
	must(t, err)
	if !strings.Contains(string(raw), "lock\n") {
		t.Fatalf("script lock not exercised: %s", raw)
	}
}
