package producttest

import (
	"strings"
	"testing"
	"time"
)

// A worker that refuses one desired revision with FailedPrecondition refuses that desire
// only. The refusal is what a waiter on it gets; the control stream is redialed as usual,
// the refused desire is not restated, and the next desire is sent and accepted.
func TestRefusedDesiredRevisionKeepsTheRentalConversation(t *testing.T) {
	pod := &standInPod{refuseDesired: "the placement set names a device this pod does not have"}
	o, instance := attachStandInRental(t, "desired-revision-refused", pod)
	claimed := pod.claims()

	fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))
	if _, ok := waitEvent(o, "REFUSED before it was applied", 30*time.Second); !ok {
		t.Fatalf("the refused revision was not recorded:\n%s", pod.report())
	}
	problem := o.c.EnsurePlacementReady(instance, "sha256:"+strings.Repeat("0", 64), "")
	if problem == nil || problem.ErrName() != "worker.desired_state_refused" ||
		!strings.Contains(problem.Message, "does not have") {
		t.Fatalf("a waiter on the refused revision got %v", problem)
	}
	deadline := time.Now().Add(30 * time.Second)
	for pod.claims() == claimed && time.Now().Before(deadline) {
		time.Sleep(25 * time.Millisecond)
	}
	// Five redial windows (200 ms each): a loop restating the refused desire would have
	// re-claimed and re-sent it several times over.
	time.Sleep(time.Second)
	if grew := pod.claims() - claimed; grew != 1 {
		t.Fatalf("the owner re-claimed the pod %d time(s) after the refusal, want exactly one redial", grew)
	}
	if n := pod.desires(); n != 1 {
		t.Fatalf("the refused desire was sent %d time(s), want once", n)
	}

	// The next desire is this owner's own and is sent on the redialed stream.
	fatal(t, o.c.ConvergePackageSet(instance, delegatedPackages(), nil))
	deadline = time.Now().Add(30 * time.Second)
	for pod.desires() < 2 && time.Now().Before(deadline) {
		time.Sleep(25 * time.Millisecond)
	}
	if n := pod.desires(); n != 2 {
		t.Fatalf("the next desire was not sent: %d desire(s)", n)
	}
	if n := countEvents(o, "REFUSING the claimed worker"); n != 0 {
		t.Fatal("a refused desire ended the conversation with the rented worker")
	}
}
