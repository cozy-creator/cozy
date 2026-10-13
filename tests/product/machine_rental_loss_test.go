package producttest

import (
	"bytes"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestEndedRentalSettlesMachineControlWithoutInventingOutcome(t *testing.T) {
	for _, accepted := range []bool{false, true} {
		t.Run(map[bool]string{false: "ambiguous-submission", true: "accepted"}[accepted], func(t *testing.T) {
			store, request, receipt := machineObserverFixture(t)
			fatal(t, store.RecordRental(records.Rental{ID: "pr-owned-machine", MachineName: "ayanojou", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "ready", Hub: "https://hub.example"}))
			if accepted {
				fatal(t, store.AcceptMachineExecution(request.ID, receipt))
				fatal(t, store.RecordMachineControl(request.ID, &pb.MachineExecutionControl{
					Execution: &pb.MachineExecutionQuery{RequestId: request.ID, ExpectedExecutionWorkspaceId: receipt.ExecutionWorkspaceId},
					CommandId: "cancel-before-loss", ExpectedGeneration: 1, Action: pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL,
				}))
			} else {
				_, problem := store.CancelMachineBeforeAcceptance(request.ID)
				fatal(t, problem)
			}
			before, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			// An absent unrelated pod and a pending release cannot discharge accepted work.
			// A frozen offer with unknown acceptance still needs authoritative closure.
			_, problem = store.ForgetRental("pr-unrelated")
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if !owed {
				t.Fatalf("before its machine is known gone the run owes it work: %v, want %v", owed, true)
			}
			// This is the same records boundary used after the Hub confirms destruction.
			_, problem = store.ForgetRental("pr-owned-machine")
			fatal(t, problem)
			owed, problem = store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if owed {
				t.Fatal("destroyed machine still owes an unreachable execution acknowledgement")
			}
			held, problem := store.Obligations()
			fatal(t, problem)
			if len(held) != 0 {
				t.Fatalf("ended machine still holds the daemon: %+v", held)
			}
			after, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !bytes.Equal(before.Receipt, after.Receipt) || !bytes.Equal(before.Outcome, after.Outcome) ||
				!bytes.Equal(before.Submission, after.Submission) || after.Collected {
				t.Fatal("machine loss fabricated acceptance, outcome, collection or changed history")
			}
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			if row.State != "canceled" {
				t.Fatalf("explicit cancellation after destruction remains %s", row.State)
			}
			lost, problem := store.MachineExecutionLost(request.ID)
			fatal(t, problem)
			if !lost {
				t.Fatalf("destroyed context remained observable for acceptance=%v", accepted)
			}
			if accepted {
				late := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId,
					WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId,
					Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
				if problem := store.ObserveMachineExecution(request.ID, late, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1}); problem == nil {
					t.Fatal("late machine observation resurrected execution after confirmed destruction")
				}
			}
		})
	}
}

func TestRuntimeEventCannotClaimRentalDestruction(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId,
		WorkerBootId: receipt.WorkerBootId, ExecutionWorkspaceId: receipt.ExecutionWorkspaceId,
		Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{
		NextAfter: 1, HeadSequence: 1, Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1,
			AtMs: 1001, Kind: "state_lost", BodyCanonicalBytes: []byte(`{"machine_id":"pr-owned-machine"}`)}},
	}))
	lost, problem := store.MachineExecutionLost(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if lost || !owed {
		t.Fatal("a Runtime observation impersonated the client's confirmed rental destruction")
	}
}
