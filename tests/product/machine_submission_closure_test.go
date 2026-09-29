package producttest

import (
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestSubmissionClosureSettlesOnlyMatchingAbsentAcceptance(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	_, problem := store.CancelMachineBeforeAcceptance(request.ID)
	fatal(t, problem)
	closed := &pb.MachineSubmissionClosure{RequestId: request.ID, SubmissionId: request.IdemKey, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
	for _, arm := range []string{"workspace", "submission", "receipt"} {
		wrong := proto.Clone(closed).(*pb.MachineSubmissionClosure)
		switch arm {
		case "workspace":
			wrong.ExecutionWorkspaceId = "replacement"
		case "submission":
			wrong.SubmissionId = "other"
		case "receipt":
			wrong.Receipt = receipt
		}
		if problem := store.CompleteMachineSubmissionClosure(request.ID, wrong); problem == nil {
			t.Fatalf("accepted wrong %s closure", arm)
		}
	}
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if !owed {
		t.Fatal("unknown acceptance lost its reconciliation obligation")
	}
	fatal(t, store.CompleteMachineSubmissionClosure(request.ID, closed))
	fatal(t, store.CompleteMachineSubmissionClosure(request.ID, closed))
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	owed, problem = store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if row.State != "canceled" || !link.SubmissionClosed || len(link.Submission) == 0 || len(link.Receipt) > 0 || owed {
		t.Fatal("closed cancellation did not preserve its proof and original submission")
	}
}
