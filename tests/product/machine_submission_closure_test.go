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

func TestSubmissionRefusalRequiresDurableNonacceptance(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	closed := &pb.MachineSubmissionClosure{RequestId: request.ID, SubmissionId: request.IdemKey, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId}
	for _, arm := range []string{"missing", "workspace", "submission", "accepted"} {
		wrong := proto.Clone(closed).(*pb.MachineSubmissionClosure)
		switch arm {
		case "missing":
			wrong = nil
		case "workspace":
			wrong.ExecutionWorkspaceId = "replacement"
		case "submission":
			wrong.SubmissionId = "another"
		case "accepted":
			wrong.Receipt = receipt
		}
		if problem := store.RefuseMachineSubmission(request.ID, "machine.refused", "not accepted", wrong); problem == nil {
			t.Fatalf("ended frozen work with %s nonacceptance proof", arm)
		}
		row, problem := store.RequestRow(request.ID)
		fatal(t, problem)
		if row.State != request.State {
			t.Fatalf("invalid proof changed the request to %s", row.State)
		}
	}
	fatal(t, store.RefuseMachineSubmission(request.ID, "machine.refused", "not accepted", closed))
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if !link.SubmissionClosed || len(link.Submission) == 0 || owed {
		t.Fatal("refusal lost its durable closure or left work owed")
	}
}
