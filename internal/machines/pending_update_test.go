package machines

import (
	"strings"
	"testing"
)

func TestPendingInstallIsAConflictForANewRequest(t *testing.T) {
	state := &RuntimeState{Update: &RuntimeUpdateState{
		Operation: "op-1", State: "waiting_activation",
		To: RuntimePair{Runtime: "0.18.89", TensorFS: "0.3.78"},
	}}
	problem := pendingUpdateConflict(state.Update)
	if problem == nil || problem.ErrName() != "machine.update_in_progress" || !strings.Contains(problem.Message, "op-1") {
		t.Fatalf("new request was not refused behind pending candidate: %v", problem)
	}
}

func TestPendingInstalledDoesNotClaimCandidateActive(t *testing.T) {
	state := &RuntimeState{Runtime: "0.18.88", TensorFS: "0.3.78", Update: &RuntimeUpdateState{
		Operation: "op-1", State: "waiting_activation",
		From: RuntimePair{Runtime: "0.18.88", TensorFS: "0.3.78"},
		To:   RuntimePair{Runtime: "0.18.89", TensorFS: "0.3.78"},
	}}
	installed := pendingInstalled(state)
	if installed.Pending == nil || installed.Runtime.Name != "cozy-runtime 0.18.88" || installed.Pending.To.Runtime != "0.18.89" {
		t.Fatalf("pending install confused active and candidate pairs: %+v", installed)
	}
}
