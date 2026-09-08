package producttest

import (
	"database/sql"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// Ordinary scripts receive large native files/trees and forward them without
// inlining bytes, publishing code, or requiring a prior run identifier.
func TestOrdinaryScriptNativeModelServing(t *testing.T) {
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires the exact wire43 Runtime candidate wheel")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := strings.Split(filepath.Base(wheel), "-")[1]
	control := filepath.Join(t.TempDir(), "control")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"},
		{"pip", "install", "--python", filepath.Join(control, "bin", "python"), wheel}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("candidate environment: %v\n%s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-native-serving-")
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
			_ = os.RemoveAll(root)
		}
	})
	defer tracePrivateChildWait(t, root)()
	project := t.TempDir()
	tools := filepath.Join(project, "model_tools")
	must(t, os.MkdirAll(tools, 0700))
	module, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "model_tools.py"))
	must(t, err)
	must(t, os.WriteFile(filepath.Join(tools, "model_tools.py"), module, 0600))
	metadata := fmt.Sprintf(`[project]
name="model-tools"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s", "torch>=2.13,<3"]
[project.entry-points."cozy.application"]
default="model_tools:app"
[tool.uv.sources]
cozy-runtime={path=%s}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["model_tools.py"]
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(tools, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(tools, "package.toml"), []byte("[application]\nobject=\"model_tools:app\"\n"), 0600))
	body, err := os.ReadFile(filepath.Join("testdata", "managed_serving", "prepare.py"))
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
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite"))
	must(t, err)
	defer db.Close()
	var produced, reused, verified int
	must(t, db.QueryRow("SELECT count(*),sum(CASE WHEN reused_from<>'' THEN 1 ELSE 0 END) FROM requests WHERE parent_request_id<>'' AND entrypoint='produce' AND state='succeeded'").Scan(&produced, &reused))
	must(t, db.QueryRow("SELECT count(*) FROM requests WHERE parent_request_id<>'' AND entrypoint='generate' AND state='succeeded' AND reused_from=''").Scan(&verified))
	if produced != 2 || reused != 1 || verified != 4 {
		t.Fatalf("independent scripts did not reuse production and execute both recipient reads: produce=%d reused=%d verify=%d", produced, reused, verified)
	}
}
