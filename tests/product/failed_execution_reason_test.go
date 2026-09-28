package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// On nemuri (run 1474) a crashed lane left the worker FAILED with the root open and no
// reason shown. Runtime ends a FAILED worker's executions failed, with the worker's reason;
// the run fails with that cause and message, and `cozy run show` says so.
func TestAFailedExecutionShowsItsReason(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &runtimeMachine{triage: bundle}
	root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, nil)
	startDaemonProcess(t, root)
	const key = "failed-worker-execution"
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	eventually(t, root, "the machine accepting the run", func() bool { return machine.submitted() != nil })
	const reason = "worker FAILED: watchdog lane crashed finishing a child result: KeyError: 'child-3'"
	machine.mu.Lock()
	machine.failure, machine.code = reason, pb.CauseCode_CAUSE_CODE_LOCAL_SAFETY
	machine.mu.Unlock()
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var row *records.Request
	eventually(t, root, "the run failing", func() bool {
		row, problem = store.RequestByIdempotencyKey(key)
		fatal(t, problem)
		link, problem := store.MachineExecution(row.ID)
		fatal(t, problem)
		return row.State == "failed" && link.Collected
	})
	if _, show := cozyWithin(t, root, time.Minute, "run", "show", row.ID); !strings.Contains(show, "LOCAL_SAFETY") || !strings.Contains(show, reason) {
		t.Fatalf("run show does not give the failure's cause and message:\n%s", show)
	}
}
