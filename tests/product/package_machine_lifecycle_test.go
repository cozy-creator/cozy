package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// A directory installs on a machine (here this computer's, as --rental=local), installs again
// with nothing to do, is listed, removed with its environment, and a second removal names it
// not installed. A real machine and installer; no Hub.
func TestPackageDirectoryInstallsAndRemovesOnAMachine(t *testing.T) {
	if *machineHostBinary == "" || *privateScriptRuntimeWheel == "" {
		t.Skip("requires a real machine and paired Runtime fixture wheel")
	}
	root, err := os.MkdirTemp("", "czpkg-")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Log("package lifecycle evidence retained", root)
		} else {
			_ = removeAllForce(root)
		}
	})
	if code, out := runCozy(t, root, "machine", "start"); code != 0 {
		t.Fatalf("machine start: %d %s", code, out)
	}
	project, _, source := directoryProofProject(t)
	source = strings.Replace(source, "raise RuntimeError(\"directory source must not execute during description\")\n", "", 1)
	must(t, os.WriteFile(filepath.Join(project, "directory_proof", "__init__.py"), []byte(source), 0600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v %s", err, out)
	}
	install := func() string {
		t.Helper()
		code, out := runCozy(t, root, "package", "install", project, "--rental=local", "--json")
		if code != 0 || !strings.Contains(out, `"status":"succeeded"`) || !strings.Contains(out, `"target":"local/directory-proof@0.0.1"`) {
			t.Fatalf("install on the machine: %d %s", code, out)
		}
		return out
	}
	listed := func() []map[string]string {
		t.Helper()
		code, out := runCozy(t, root, "package", "list", "--rental=local", "--json", "--full")
		var doc struct{ Packages []map[string]string }
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
			t.Fatalf("list: %d %s", code, out)
		}
		return doc.Packages
	}
	install()
	if rows := listed(); len(rows) != 1 || rows[0]["package"] != "local/directory-proof" || rows[0]["hub"] != "" {
		t.Fatalf("listed after install: %+v", rows)
	}
	// Unchanged: no new environment here or there.
	install()
	if rows := listed(); len(rows) != 1 {
		t.Fatalf("an unchanged install made another installation: %+v", rows)
	}
	code, out := runCozy(t, root, "package", "remove", "local/directory-proof", "--rental=local", "--json")
	if code != 0 || !strings.Contains(out, `"package":"local/directory-proof"`) || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("remove: %d %s", code, out)
	}
	if rows := listed(); len(rows) != 0 {
		t.Fatalf("listed after remove: %+v", rows)
	}
	code, out = runCozy(t, root, "package", "remove", "local/directory-proof", "--rental=local", "--json")
	if code == 0 || !strings.Contains(out, "package.not_installed") {
		t.Fatalf("a second remove: %d %s", code, out)
	}
	// It installs again from the same capture.
	install()
	if rows := listed(); len(rows) != 1 {
		t.Fatalf("listed after reinstall: %+v", rows)
	}
}
