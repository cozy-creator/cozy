package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

func TestPythonInventoryCarriesRuntimeOwnedRoot(t *testing.T) {
	for _, root := range []string{filepath.Join(t.TempDir(), "runtime-selected-location"), "relative/root", ""} {
		t.Run(root, func(t *testing.T) {
			toolDir := t.TempDir()
			body, err := json.Marshal(hostruntime.PythonInventory{
				Format: "cozy.python-interpreters/1", ManagedRoot: root, SupportedMinors: []string{"3.12", "3.13", "3.14"},
			})
			must(t, err)
			script := fmt.Sprintf(`#!/bin/sh
if [ "$2" = version ]; then
 printf '%%s\n' '{"distribution":"%s","wire_protocol":"cozy.worker.v1+minor.54"}'
 exit 0
fi
if [ "$2" != python-interpreters ]; then exit 91; fi
printf '%%s\n' '%s'
`, hostruntime.ToolFloor, body)
			must(t, os.WriteFile(filepath.Join(toolDir, "cozy-runtime"), []byte(script), 0700)) //cozy:allow read-only Runtime inventory boundary fixture
			t.Setenv("PATH", toolDir)
			inventory, problem := hostruntime.PythonExecutors(context.Background())
			if !filepath.IsAbs(root) {
				if problem == nil || problem.Name != "python_window_unavailable" {
					t.Fatalf("unowned root accepted: %+v %v", inventory, problem)
				}
				return
			}
			fatal(t, problem)
			if inventory.ManagedRoot != root {
				t.Fatalf("Runtime root changed: %+v", inventory)
			}
			if _, err := os.Stat(root); !os.IsNotExist(err) {
				t.Fatalf("inventory created managed root: %v", err)
			}
		})
	}
}

// A newer Runtime's inventory report keeps the members this host reads.
func TestPythonInventoryAcceptsANewerReportRevision(t *testing.T) {
	root := filepath.Join(t.TempDir(), "runtime-selected-location")
	newer := 2
	body := fmt.Sprintf(`{"format":"cozy.python-interpreters/%d","managed_root":%q,"free_threaded":[],"supported_minors":[]}`, newer, root)
	toolDir := t.TempDir()
	script := fmt.Sprintf(`#!/bin/sh
if [ "$2" = version ]; then
 printf '%%s\n' '{"distribution":"%s","wire_protocol":"cozy.worker.v1+minor.54"}'
 exit 0
fi
printf '%%s\n' '%s'
`, hostruntime.ToolFloor, body)
	must(t, os.WriteFile(filepath.Join(toolDir, "cozy-runtime"), []byte(script), 0700)) //cozy:allow read-only Runtime inventory boundary fixture
	t.Setenv("PATH", toolDir)
	inventory, problem := hostruntime.PythonExecutors(context.Background())
	fatal(t, problem)
	if inventory.ManagedRoot != root {
		t.Fatalf("Runtime root changed: %+v", inventory)
	}
}

func TestPythonLocalServeUsesReportedManagedRoot(t *testing.T) {
	integration(t)
	if runtime.GOOS != "linux" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires Linux process inspection and the exact Runtime wheel")
	}
	root, err := os.MkdirTemp("", "cz-pyroot-")
	must(t, err)
	t.Cleanup(func() {
		if t.Failed() {
			t.Log("local root proof retained", root)
			return
		}
		must(t, removeAllForce(root))
	})
	inventoryCommand := exec.Command("cozy-runtime", "--json", "python-interpreters") //cozy:allow actual read-only CLI ownership report under the fixture home
	inventoryCommand.Env = childEnv(t, root)
	raw, err := inventoryCommand.Output()
	must(t, err)
	var inventory hostruntime.PythonInventory
	must(t, json.Unmarshal(raw, &inventory))
	if !filepath.IsAbs(inventory.ManagedRoot) {
		t.Fatal("Runtime omitted its managed root")
	}
	script := filepath.Join(t.TempDir(), "root_probe.py")
	source := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[model-execution]"]
# [tool.uv.sources]
# cozy-runtime = {path = %q}
# ///
def main() -> str:
    return "shared-root-ready"
`, *privateScriptRuntimeWheel)
	must(t, os.WriteFile(script, []byte(source), 0600))
	code, out := runCozy(t, root, "run", script, "--await", "--json")
	if code != 0 || !strings.Contains(out, "shared-root-ready") {
		t.Fatalf("shared-root local run [%d]: %s", code, out)
	}
	raw, err = os.ReadFile(filepath.Join(root, "runtime", "process.json"))
	must(t, err)
	var process struct {
		PID int `json:"pid"`
	}
	must(t, json.Unmarshal(raw, &process))
	raw, err = os.ReadFile(filepath.Join("/proc", strconv.Itoa(process.PID), "environ"))
	must(t, err)
	values := map[string]string{}
	for _, value := range bytes.Split(raw, []byte{0}) {
		name, body, _ := strings.Cut(string(value), "=")
		values[name] = body
	}
	if values["COZY_HOME"] != filepath.Join(root, "runtime") || values["COZY_PYTHON_ROOT"] != inventory.ManagedRoot {
		t.Fatalf("serve changed interpreter ownership: home=%q python=%q reported=%q", values["COZY_HOME"], values["COZY_PYTHON_ROOT"], inventory.ManagedRoot)
	}
}
