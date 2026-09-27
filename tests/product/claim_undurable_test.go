package producttest

import (
	"sync/atomic"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A worker that answers a claim CLAIM_REJECTION_UNDURABLE, as a Runtime still committing its
// readiness barrier does, is claimed again rather than refused: a rented run reaches it.
func TestUndurableClaimIsClaimedAgain(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &runtimeMachine{blocker: "none"}
	var claims atomic.Int32
	pod := &fakePod{machine: machine, deviceCount: 4,
		onFrame: func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
			claim := frame.GetClaim()
			if claim == nil || claims.Add(1) > 1 {
				return false, nil
			}
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
				RecordOwnerEpoch: claim.RecordOwnerEpoch, WorkerBootId: podBootID, WireMinor: pb.WireMinor,
				Rejection: pb.ClaimRejection_CLAIM_REJECTION_UNDURABLE}}})
		}}
	root, _ := rentedLadderMachine(t, h, pod, nil)
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("the run on a worker that was not yet durable was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the run to reach the re-claimed worker", func() bool { return machine.submitted() != nil })
	if n := claims.Load(); n < 2 {
		t.Fatalf("the worker was claimed %d time(s); the undurable refusal was not followed by a claim", n)
	}
}
