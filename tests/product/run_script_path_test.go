package producttest

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hostruntime"
)

// `cozy run <path>.py` runs the script however the path is spelled: relative with or
// without ./, or absolute. A package ref never ends in .py except org/name.py, and there
// an existing file wins, said in one line.
func TestRunTakesAScriptPathHoweverItIsSpelled(t *testing.T) {
	root, err := os.MkdirTemp(scratchBase, "script-path-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	project := t.TempDir()
	script := []byte("# /// script\n# requires-python=\">=3.12,<3.13\"\n# dependencies=[\"cozy-runtime>=" +
		hostruntime.PackageFloor + "\"]\n# ///\ndef main(): pass\n")
	for _, path := range []string{"examples/client-scripts/h3_lanes.py", "lanes/h3.py"} {
		must(t, os.MkdirAll(filepath.Join(project, filepath.Dir(path)), 0o755))
		must(t, os.WriteFile(filepath.Join(project, path), script, 0o600))
	}
	run := func(target string) (int, string, string) {
		t.Helper()
		cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run", target, "--describe", "--json")
		cmd.Dir, cmd.Env = project, childEnv(t, root)
		var stdout, stderr bytes.Buffer
		cmd.Stdout, cmd.Stderr = &stdout, &stderr
		_ = cmd.Run()
		return cmd.ProcessState.ExitCode(), stdout.String(), stderr.String()
	}
	for _, target := range []string{"examples/client-scripts/h3_lanes.py", "./examples/client-scripts/h3_lanes.py",
		filepath.Join(project, "examples/client-scripts/h3_lanes.py"), "lanes/h3.py"} {
		code, out, errs := run(target)
		var described struct {
			Fields []json.RawMessage `json:"fields"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &described) != nil || described.Fields == nil {
			t.Fatalf("cozy run %s was not described as the script [exit %d]: %s\n%s", target, code, out, errs)
		}
		ambiguous := strings.Contains(errs, "running the local file "+target)
		if ambiguous != (target == "lanes/h3.py") {
			t.Fatalf("cozy run %s: the org/name.py note was %v: %s", target, ambiguous, errs)
		}
	}
	if code, out, errs := run("examples/client-scripts/missing.py"); code == 0 || strings.Contains(out+errs, "package.not_found") {
		t.Fatalf("a missing script was read as a package [exit %d]: %s\n%s", code, out, errs)
	}
}

// --describe says where a job runs: its own accelerator declaration, in both outputs.
func TestDescribeShowsAJobsAcceleratorDeclaration(t *testing.T) {
	root, err := os.MkdirTemp(scratchBase, "describe-accelerator-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	for _, test := range []struct{ declared, human, json string }{
		{"", "runs on: CPU", `"accelerator":false`}, {"# [tool.cozy]\n# accelerator = true\n", "runs on: GPU", `"accelerator":true`},
	} {
		script := filepath.Join(t.TempDir(), "job.py")
		must(t, os.WriteFile(script, []byte("# /// script\n# requires-python=\">=3.12,<3.13\"\n# dependencies=[\"cozy-runtime>="+
			hostruntime.PackageFloor+"\"]\n"+test.declared+"# ///\ndef main(): pass\n"), 0o600))
		if code, out := runCozy(t, root, "run", script, "--describe"); code != 0 || !strings.Contains(out, test.human) {
			t.Fatalf("human --describe omitted %q [exit %d]: %s", test.human, code, out)
		}
		if code, out := runCozy(t, root, "run", script, "--describe", "--json"); code != 0 || !strings.Contains(out, test.json) {
			t.Fatalf("--describe --json omitted %s [exit %d]: %s", test.json, code, out)
		}
	}
}
