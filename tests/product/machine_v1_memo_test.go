package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// A memoized call that ran on a machine is held by that machine: the package's next run has
// the same call answered without running, and the new one runs. This computer carries no
// results; the machine answers where its results are.
func TestMemoizedCallsAreAnsweredFromTheMachinesMemo(t *testing.T) {
	root, counter := memoMachine(t)
	type shownCall struct {
		Function    string `json:"function"`
		Status      string `json:"status"`
		Memoized    bool   `json:"memoized"`
		Computation string `json:"computation"`
	}
	// survey runs once and returns its squares and its calls as `run show --json` shows them.
	survey := func(values ...int) ([]int, []shownCall) {
		t.Helper()
		in := filepath.Join(t.TempDir(), "req.json")
		raw, _ := json.Marshal(map[string]any{"values": values, "counter": counter})
		must(t, os.WriteFile(in, raw, 0o600))
		code, out := runCozy(t, root, "run", "local/tensord-cpu-memo/survey", "--input", in, "--await", "--json", "--fields=id,status,result")
		var run struct {
			ID     string `json:"id"`
			Result struct {
				Squares []int `json:"squares"`
			} `json:"result"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &run) != nil || run.ID == "" {
			t.Fatalf("survey %v [exit %d]\n%s", values, code, out)
		}
		code, shown := runCozy(t, root, "run", "show", run.ID, "--json")
		var report struct {
			Calls []shownCall `json:"calls"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(shown)), &report) != nil {
			t.Fatalf("run show %s [exit %d]\n%s", run.ID, code, shown)
		}
		var calls []shownCall
		for _, call := range report.Calls {
			if call.Function == "measure" {
				calls = append(calls, call)
			}
		}
		return run.Result.Squares, calls
	}
	measured := func() string { return measuredCount(t, counter) }
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

// A memoizable function run directly is held by its machine too: the same request again is
// answered from the machine's memo as a completed run that ran nothing, and `run show` says
// so; a different request runs.
func TestMemoizedRunIsAnsweredFromTheMachinesMemo(t *testing.T) {
	root, counter := memoMachine(t)
	type memoRun struct {
		ID          string          `json:"request_id"`
		Memoized    bool            `json:"memoized"`
		Computation string          `json:"computation"`
		Result      json.RawMessage `json:"result"`
	}
	// measure runs `measure` once and returns its run as `run show --json` shows it.
	measure := func(value int) memoRun {
		t.Helper()
		in := filepath.Join(t.TempDir(), "req.json")
		raw, _ := json.Marshal(map[string]any{"payload": map[string]any{"value": value, "counter": counter}})
		must(t, os.WriteFile(in, raw, 0o600))
		code, out := runCozy(t, root, "run", "local/tensord-cpu-memo/measure", "--input", in, "--await", "--json", "--fields=id,status,result")
		var run struct {
			ID     string `json:"id"`
			Status string `json:"status"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &run) != nil || run.ID == "" || run.Status != "completed" {
			t.Fatalf("measure %d [exit %d]\n%s", value, code, out)
		}
		code, shown := runCozy(t, root, "run", "show", run.ID, "--json")
		var report memoRun
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(shown)), &report) != nil {
			t.Fatalf("run show %s [exit %d]\n%s", run.ID, code, shown)
		}
		return report
	}
	first := measure(3)
	if measuredCount(t, counter) != "1" || first.Memoized || first.Computation == "" || string(first.Result) != `{"square":9}` {
		t.Fatalf("the first run did not run and hold its computation (%s ran): %+v", measuredCount(t, counter), first)
	}
	again := measure(3)
	if measuredCount(t, counter) != "1" || !again.Memoized || again.Computation != first.Computation || string(again.Result) != string(first.Result) {
		t.Fatalf("the same request was not answered from the memo (%s ran): %+v", measuredCount(t, counter), again)
	}
	other := measure(4)
	if measuredCount(t, counter) != "2" || other.Memoized || other.Computation == first.Computation || string(other.Result) != `{"square":16}` {
		t.Fatalf("a different request was not run (%s ran): %+v", measuredCount(t, counter), other)
	}
	code, human := runCozy(t, root, "run", "show", again.ID)
	if code != 0 || !strings.Contains(human, "memo: served from this machine's memo (nothing ran) · "+first.Computation) {
		t.Fatalf("run show does not say the run was served from the memo [%d]:\n%s", code, human)
	}
}

// memoMachine is a fresh root whose own machine (the -machine-host build) has the cpu_memo
// fixture installed, and the file its `measure` counts real executions in.
func memoMachine(t *testing.T) (root, counter string) {
	t.Helper()
	if *machineHostBinary == "" || *cpuLongform == "" {
		t.Skip("requires -machine-host=<tensord> and -cpu-longform=<tensord>/tests/fixtures/cpu_longform")
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
	project := tensordFixture(t, filepath.Join(filepath.Dir(*cpuLongform), "cpu_memo"))
	if code, out := runCozy(t, root, "package", "install", project); code != 0 {
		t.Fatalf("package install [exit %d]\n%s", code, out)
	}
	return root, filepath.Join(root, "measured")
}

// measuredCount is how many times `measure` truly ran.
func measuredCount(t *testing.T, counter string) string {
	t.Helper()
	raw, err := os.ReadFile(counter)
	must(t, err)
	return string(raw)
}
