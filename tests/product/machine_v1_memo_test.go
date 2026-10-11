package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
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
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	counter := filepath.Join(root, "measured")
	type shownCall struct {
		Function    string `json:"function"`
		Status      string `json:"status"`
		Memoized    bool   `json:"memoized"`
		Computation string `json:"computation"`
	}
	// survey runs one job and returns its squares and its calls as `run show --json` shows them.
	survey := func(values ...int) ([]int, []shownCall) {
		t.Helper()
		in := filepath.Join(t.TempDir(), "req.json")
		raw, _ := json.Marshal(map[string]any{"values": values, "counter": counter})
		must(t, os.WriteFile(in, raw, 0o600))
		code, out := runCozy(t, root, "run", "local/cozy-machine-cpu-memo/survey", "--input", in, "--await", "--json")
		var run struct {
			Job    string `json:"job"`
			Result struct {
				Squares []int `json:"squares"`
			} `json:"result"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &run) != nil || run.Job == "" {
			t.Fatalf("survey %v [exit %d]\n%s", values, code, out)
		}
		code, shown := runCozy(t, root, "run", "show", run.Job, "--json")
		var report struct {
			Calls []shownCall `json:"calls"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(shown)), &report) != nil {
			t.Fatalf("run show %s [exit %d]\n%s", run.Job, code, shown)
		}
		var calls []shownCall
		for _, call := range report.Calls {
			if call.Function == "measure" {
				calls = append(calls, call)
			}
		}
		return run.Result.Squares, calls
	}
	measured := func() string {
		raw, err := os.ReadFile(counter)
		must(t, err)
		return string(raw)
	}
	got, first := survey(3, 4)
	if fmt.Sprint(got) != "[9 16]" || measured() != "2" {
		t.Fatalf("first survey answered %v after %s calls ran", got, measured())
	}
	got, second := survey(3, 5)
	if fmt.Sprint(got) != "[9 25]" || measured() != "3" {
		t.Fatalf("second survey answered %v after %s calls ran; the held call for 3 ran again", got, measured())
	}
	// The owner sees which call a held result answered, and that it is the same computation.
	if len(first) != 2 || len(second) != 2 || first[0].Memoized || first[1].Memoized ||
		!second[0].Memoized || second[0].Computation != first[0].Computation || second[0].Computation == "" ||
		second[1].Memoized || second[1].Computation == first[1].Computation {
		t.Fatalf("run show does not say how each call was answered:\nfirst %+v\nsecond %+v", first, second)
	}
	code, human := runCozy(t, root, "run", "show", "2")
	if code != 0 || !strings.Contains(human, "reused (memo)") {
		t.Fatalf("run show does not mark the reused call [%d]:\n%s", code, human)
	}
}
