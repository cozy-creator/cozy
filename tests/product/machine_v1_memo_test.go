package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// A job's memoized call that ran on a machine is held by that machine: the package's next job
// has the same call answered without running, and the new one runs. This computer carries no
// results; the machine answers where its results are.
func TestMemoizedCallsAreAnsweredFromTheMachinesMemo(t *testing.T) {
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<cozy-machine> and -cpu-longform=<cozy-machine>/tests/fixtures/cpu_longform")
	}
	root, err := os.MkdirTemp(os.TempDir(), "czm")
	must(t, err)
	provisionMachine(t, root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			t.Logf("evidence retained at %s\nmachine log tail:\n%s", root, tail(filepath.Join(root, "machine", "host.log")))
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "cpu_memo")
	must(t, os.CopyFS(project, os.DirFS(filepath.Join(filepath.Dir(*cpuLongform), "cpu_memo"))))
	if out, err := exec.Command("uv", "lock", "--directory", project).CombinedOutput(); err != nil {
		t.Fatalf("uv lock: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	counter := filepath.Join(root, "measured")
	survey := func(values ...int) []int {
		t.Helper()
		in := filepath.Join(t.TempDir(), "req.json")
		raw, _ := json.Marshal(map[string]any{"values": values, "counter": counter})
		must(t, os.WriteFile(in, raw, 0o600))
		code, out := runCozy(t, root, "run", "local/cozy-machine-cpu-memo/survey", "--input", in, "--await", "--json")
		var run struct {
			Result struct {
				Squares []int `json:"squares"`
			} `json:"result"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &run) != nil {
			t.Fatalf("survey %v [exit %d]\n%s", values, code, out)
		}
		return run.Result.Squares
	}
	measured := func() string {
		raw, err := os.ReadFile(counter)
		must(t, err)
		return string(raw)
	}
	if got := survey(3, 4); fmt.Sprint(got) != "[9 16]" || measured() != "2" {
		t.Fatalf("first survey answered %v after %s calls ran", got, measured())
	}
	if got := survey(3, 5); fmt.Sprint(got) != "[9 25]" || measured() != "3" {
		t.Fatalf("second survey answered %v after %s calls ran; the held call for 3 ran again", got, measured())
	}
}
