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
			fatal(t, store.RecordRental(records.Rental{ID: "pr-owned-machine", State: "ready", Hub: "https://hub.example"}))
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
			// An absent unrelated pod and a pending release cannot discharge this work.
			_, problem = store.ForgetRental("pr-unrelated")
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if !owed {
				t.Fatal("unconfirmed machine loss waived execution obligations")
			}
			// This is the same records boundary used after the Hub confirms destruction.
			_, problem = store.ForgetRental("pr-owned-machine")
			fatal(t, problem)
			owed, problem = store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if owed {
				t.Fatal("destroyed machine still owes an unreachable execution acknowledgement")
			}
			blocked, problem := store.ClientShutdownObligations()
			fatal(t, problem)
			if len(blocked) != 0 {
				t.Fatalf("ended machine still blocks observer shutdown: %+v", blocked)
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
		})
	}
}
