package producttest

import (
	"context"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type missingClosureRouteMachine struct{ *runtimeMachine }

func (*missingClosureRouteMachine) CloseMachineSubmission(context.Context, *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	return nil, status.Error(codes.Unimplemented, "this machine does not serve CloseMachineSubmission")
}

// The ordinary CLI reaches a current Runtime capability through an old Host. It
// fails before freezing an offer, instead of creating an acceptance-unknown queue.
func TestNewRunOnHostMissingClosureFailsBeforeTransmission(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &missingClosureRouteMachine{runtimeMachine: &runtimeMachine{}}
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4}, nil)
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "closure-admission"); code != 0 && !strings.Contains(out, "machine.submission_closure_required") {
		t.Fatalf("submit local request: %d %s", code, out)
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("closure-admission")
	fatal(t, problem)
	waitFor(t, root, "unsupported machine to fail before sending", func() bool { row, _ := store.RequestRow(request.ID); return row != nil && row.State == "failed" })
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if len(link.Submission) != 0 || len(link.Receipt) != 0 || machine.submitted() != nil {
		t.Fatal("unsupported Host received or froze real work")
	}
	if code, out := runCozy(t, root, "run", "show", request.ID, "--json"); code != 0 || !strings.Contains(out, "no submission was sent") || !strings.Contains(out, "CloseMachineSubmission") {
		t.Fatalf("missing actionable refusal: %d %s", code, out)
	}
}
