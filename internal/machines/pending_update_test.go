package machines

import "testing"

func TestPendingSourceMatchesCandidateVersions(t *testing.T) {
	state := &RuntimeState{Update: &RuntimeUpdateState{
		Operation: "op-1", State: "waiting_activation",
		To: RuntimePair{Runtime: "0.18.89", TensorFS: "0.3.78"},
	}}
	if problem := pendingSourceMatches(state, Source{
		RuntimeWheel:  "cozy_runtime-0.18.89-cp312-abi3-manylinux_2_28_x86_64.whl",
		TensorFSWheel: "tensorfs-0.3.78-cp312-abi3-manylinux_2_17_x86_64.whl",
	}); problem != nil {
		t.Fatalf("matching pending candidate refused: %v", problem)
	}
	if problem := pendingSourceMatches(state, Source{RuntimeWheel: "cozy_runtime-0.18.88-cp312-abi3-manylinux_2_28_x86_64.whl"}); problem == nil || problem.ErrName() != "machine.update_in_progress" {
		t.Fatalf("different pending candidate was accepted: %v", problem)
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
