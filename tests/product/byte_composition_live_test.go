package producttest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// Ordinary scripts receive large native files/trees and forward them without
// inlining bytes, publishing code, or requiring a prior run identifier.
func TestOrdinaryScriptByteResultsAndMemoReuse(t *testing.T) {
	integration(t)
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires an exact Runtime wheel with native child custody")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("candidate environment: %v\n%s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-byte-result-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("byte artifact evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := t.TempDir()
	tools := filepath.Join(project, "byte_tools")
	must(t, os.MkdirAll(tools, 0700))
	module, err := os.ReadFile(filepath.Join("testdata", "byte_composition", "byte_tools", "byte_tools.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(tools, "byte_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="byte-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[project.entry-points."cozy.application"]
default="byte_tools:app"
[tool.uv.sources]
cozy-runtime={path=%s}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["byte_tools.py"]
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(tools, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(tools, "package.toml"), []byte("[application]\nobject=\"byte_tools:app\"\n"), 0600))
	body, err := os.ReadFile(filepath.Join("testdata", "byte_composition", "prepare.py"))
	must(t, err)
	body = []byte(strings.NewReplacer("__VERSION__", version, "__WHEEL__", strconv.Quote(wheel)).Replace(string(body)))
	script := filepath.Join(project, "prepare.py")
	must(t, os.WriteFile(script, body, 0600))
	for run := range 2 {
		if run == 1 {
			must(t, os.WriteFile(script, append(body, []byte("\n# Independent edited script; reuse compatible operations.\n")...), 0600))
		}
		code, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
		if code != 0 {
			t.Fatalf("plain byte script %d [exit %d]\n%s\n%s", run, code, out, productWorkerLogs(root))
		}
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	first, second := machineChildren(t, root, store, "1"), machineChildren(t, root, store, "2")
	if len(first) != 2 || len(second) != 2 || first[0].Executions != 1 || first[1].Executions != 1 || second[0].Executions != 0 || second[1].Executions != 1 || first[0].Computation != second[0].Computation {
		t.Fatalf("Runtime did not reuse native production and execute both recipient reads: first=%+v second=%+v", first, second)
	}
}
