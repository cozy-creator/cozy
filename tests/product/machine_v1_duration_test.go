package producttest

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// A cozy.machine.v1 run's execution ticks while it runs on its machine, alike in run list, run
// show and their JSON. At the end the machine's measurement replaces it: the callable alone,
// not the wait for the machine to prepare the package.
func TestAV1RunsExecutionTicksAndExcludesItsWait(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", restartProject(t)); code != 0 {
		t.Fatalf("installing the restart package [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/restart-proof/paced", "seconds=8", "--json", "--idempotency-key", "timed"); code != 0 {
		t.Fatalf("submitting the run [exit %d]\n%s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("timed")
	fatal(t, problem)
	reaches := func(what, state string) {
		eventually(t, root, what, func() bool {
			row, problem := store.RequestRow(request.ID)
			return problem == nil && row != nil && row.State == state
		})
	}
	execution := func() int64 {
		t.Helper()
		listed := listedMachineTiming(t, root, request.ID)
		if !listed.ExecutionKnown || listed.ExecutionMS == nil {
			t.Fatalf("a run that has run shows no execution: %+v", listed)
		}
		return *listed.ExecutionMS
	}
	reaches("the run running on this computer's machine", "dispatching")
	first := execution()
	time.Sleep(time.Second)
	if second := execution(); second < first+900 {
		t.Fatalf("the running run's execution did not tick with the clock: %d then %d", first, second)
	}

	if code, out := runCozy(t, root, "run", "watch", request.ID, "--json"); code != 0 {
		t.Fatalf("the run did not complete [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var shown struct {
		ExecutionMS *int64 `json:"execution_ms"`
		WallMS      int64  `json:"wall_ms"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || shown.ExecutionMS == nil {
		t.Fatalf("the finished run has no measured execution [exit %d]\n%s", code, out)
	}
	cell := fmt.Sprintf("%.1fs", float64(*shown.ExecutionMS)/1000)
	for _, args := range [][]string{{"run", "list"}, {"run", "show", request.ID}} {
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, cell) {
			t.Fatalf("cozy %s does not print the execution %s [exit %d]\n%s", strings.Join(args, " "), cell, code, out)
		}
	}
	t.Logf("execution %d ms of wall %d ms", *shown.ExecutionMS, shown.WallMS)
	if *shown.ExecutionMS < 7500 || shown.WallMS < *shown.ExecutionMS+500 {
		t.Fatalf("execution_ms %d is not the callable alone (8 s) inside wall %d ms", *shown.ExecutionMS, shown.WallMS)
	}
}
