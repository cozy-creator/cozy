package producttest

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"
)

// A cozy.machine.v1 run's execution ticks while it runs on its machine and holds while it rests
// paused, alike in run list, run show and their JSON; a resume does not count the pause. At the
// end the machine's measurement replaces it: both attempts of the callable alone, not the wait
// for the machine to prepare the package.
func TestAV1RunsExecutionTicksHoldsPausedAndExcludesItsWait(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", restartProject(t)); code != 0 {
		t.Fatalf("installing the restart package [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/restart-proof/paced", "seconds=8", "--json", "--idempotency-key", "timed"); code != 0 {
		t.Fatalf("submitting the job [exit %d]\n%s", code, out)
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
	reaches("the job running on this computer's machine", "dispatching")
	first := execution()
	time.Sleep(time.Second)
	if second := execution(); second < first+900 {
		t.Fatalf("the running job's execution did not tick with the clock: %d then %d", first, second)
	}

	if code, out := runCozy(t, root, "run", "pause", request.ID, "--json"); code != 0 {
		t.Fatalf("run pause [exit %d]\n%s", code, out)
	}
	reaches("the job to rest paused", "paused")
	held := execution()
	time.Sleep(time.Second)
	if still := execution(); still != held {
		t.Fatalf("the paused job's execution moved: %d then %d", held, still)
	}
	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var shown struct {
		ExecutionMS *int64 `json:"execution_ms"`
		WallMS      int64  `json:"wall_ms"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || shown.ExecutionMS == nil || *shown.ExecutionMS != held {
		t.Fatalf("run show disagrees with run list's held execution %d ms [exit %d]\n%s", held, code, out)
	}
	cell := fmt.Sprintf("%.1fs", float64(held)/1000)
	for _, args := range [][]string{{"run", "list"}, {"run", "show", request.ID}} {
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, cell) {
			t.Fatalf("cozy %s does not print the held execution %s [exit %d]\n%s", strings.Join(args, " "), cell, code, out)
		}
	}

	resumed := time.Now()
	if code, out := runCozy(t, root, "run", "resume", request.ID, "--json"); code != 0 {
		t.Fatalf("run resume [exit %d]\n%s", code, out)
	}
	reaches("the resumed job running again", "dispatching")
	if again, since := execution(), time.Since(resumed).Milliseconds(); again <= held || again > held+since+100 {
		t.Fatalf("the resumed job's execution %d ms is not its held %d ms plus at most the %d ms since its resume", again, held, since)
	}

	if code, out := runCozy(t, root, "run", "watch", request.ID, "--json"); code != 0 {
		t.Fatalf("the job did not complete [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "show", request.ID, "--json")
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || shown.ExecutionMS == nil {
		t.Fatalf("the finished job has no measured execution [exit %d]\n%s", code, out)
	}
	t.Logf("execution %d ms (held %d ms at the pause) of wall %d ms", *shown.ExecutionMS, held, shown.WallMS)
	if *shown.ExecutionMS < held+7500 || shown.WallMS < *shown.ExecutionMS+500 {
		t.Fatalf("execution_ms %d is not both attempts of the callable alone (held %d, then 8 s) inside wall %d ms", *shown.ExecutionMS, held, shown.WallMS)
	}
}
