package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
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
