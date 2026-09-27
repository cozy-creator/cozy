package producttest

import (
	"testing"
	"time"
)

// TestADeletedCallEndsTheConversationInsteadOfRedialingForever is the money arm.
//
// A pod that has DELETED a lane answers codes.Unimplemented, and returning any status
// from the WorkerControl bidi handler tears down the whole session. Before this change
// the owner treated only codes.FailedPrecondition as a worker's final word, and even that
// only while a desired revision was still pending — so an Unimplemented refusal fell
// through to the redial at the bottom of attach: dial, claim, mint a control epoch, get
// refused, sleep 200 ms, repeat. Forever, on a rented pod at roughly a dollar an hour.
// That is not a hypothetical: tensorhub th-124 deleted the model-source carrier lane and
// left exactly this status behind a caller the daemon still drives, and a sibling session
// measured the same shape from a different cause at 2,434 revisions in 100 minutes.
//
// This runs the real orchestrator against a real second implementation of the worker
// protocol over pinned TLS. The measurement is the pod's own claim count: the owner's
// redial rate seen from the other end. Nothing is stubbed inside the code under test.
func TestADeletedCallEndsTheConversationInsteadOfRedialingForever(t *testing.T) {
	pod := &standInPod{unimplemented: make(chan struct{})}
	o, _ := attachStandInRental(t, "unimplemented-terminal", pod)

	// The lane is deleted from here on — after the claim, after the snapshot, after the
	// desired state converged. That ordering is the defect's home: the model-source
	// frames that produced it are sent long after a pod has accepted its desired state,
	// so a terminal rule that only answers a PENDING revision never fires.
	settled := pod.claims()
	close(pod.unimplemented)

	// The owner's verdict first, then five redial windows: a loop would have claimed
	// roughly that many more times.
	if _, ok := waitEvent(o, "worker.call_unimplemented", 10*time.Second); !ok {
		t.Fatal("the owner never refused the worker that deleted a call")
	}
	time.Sleep(time.Second)

	if grew := pod.claims() - settled; grew > 1 {
		t.Fatalf("the owner re-claimed the worker %d more time(s) after a deleted call "+
			"refused it; an Unimplemented refusal must end the conversation, not pace it",
			grew)
	}
}
