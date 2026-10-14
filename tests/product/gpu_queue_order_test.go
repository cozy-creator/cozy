package producttest

import (
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// A GPU wait names its blocker in the machine's acceptance order, which its GPU queue follows,
// whatever the local run numbers say (runs 5519/5520 raced and the machine accepted 5520 first).
func TestAGPUWaitNamesItsBlockerInMachineAcceptanceOrder(t *testing.T) {
	wait := func(number int64, blocker orchestrator.WaitingRun) string {
		return cli.PhaseCell(api.Lifecycle{Number: number, Status: "in_progress", Phase: orchestrator.PhaseGPUWait,
			PhaseDetail: "needs 4", WaitingFor: &blocker})
	}
	for _, c := range []struct{ got, want string }{
		{wait(5521, orchestrator.WaitingRun{Number: 5520}), "waiting for GPU (needs 4, behind run 5520)"},
		{wait(5519, orchestrator.WaitingRun{Number: 5520}), "waiting for GPU (needs 4, behind run 5520, which reached the machine first)"},
		{wait(5519, orchestrator.WaitingRun{Number: 5520, Borrowing: true}), "waiting for GPU (needs 4, run 5520 hands them over after its current step)"},
	} {
		if c.got != c.want {
			t.Errorf("got %q, want %q", c.got, c.want)
		}
	}
}
