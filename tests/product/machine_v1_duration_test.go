package producttest

import (
	"encoding/json"
	"testing"
	"time"
)

// A cozy.machine.v1 run's wall time advances while it runs on its machine, its execution stays
// unknown until the machine measured it, and the measured execution_ms is the callable's own
// time: not the wait for the machine to prepare the package before it.
func TestAV1RunsExecutionExcludesItsWaitAndItsWallAdvances(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", restartProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the restart package [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/restart-proof/slow", "seconds=4", "--json", "--idempotency-key", "timed"); code != 0 {
		t.Fatalf("submitting the job [exit %d]\n%s", code, out)
	}
	request, problem := store.RequestByIdempotencyKey("timed")
	fatal(t, problem)
	eventually(t, root, "the job running on this computer's machine", func() bool {
		row, problem := store.RequestRow(request.ID)
		return problem == nil && row != nil && row.State == "dispatching"
	})
	first := listedMachineTiming(t, root, request.ID)
	time.Sleep(time.Second)
	second := listedMachineTiming(t, root, request.ID)
	if first.AttemptWallMS <= 0 || second.AttemptWallMS < first.AttemptWallMS+900 {
		t.Fatalf("the running job's wall did not advance with the clock: %d then %d", first.AttemptWallMS, second.AttemptWallMS)
	}
	if second.ExecutionKnown || second.ExecutionMS != nil {
		t.Fatalf("execution was known before the machine measured it: %+v", second)
	}
	if code, out := runCozy(t, root, "run", "watch", request.ID, "--json"); code != 0 {
		t.Fatalf("the job did not complete [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var shown struct {
		ExecutionMS    *int64 `json:"execution_ms"`
		ExecutionKnown bool   `json:"execution_known"`
		WallMS         int64  `json:"wall_ms"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || !shown.ExecutionKnown || shown.ExecutionMS == nil {
		t.Fatalf("the finished job has no measured execution [exit %d]\n%s", code, out)
	}
	t.Logf("execution %d ms of wall %d ms", *shown.ExecutionMS, shown.WallMS)
	if *shown.ExecutionMS < 4000 || *shown.ExecutionMS > 5000 || shown.WallMS < *shown.ExecutionMS+500 {
		t.Fatalf("execution_ms %d is not the 4 s callable alone inside wall %d ms", *shown.ExecutionMS, shown.WallMS)
	}
}
