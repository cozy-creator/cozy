package producttest

import (
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// This peer returns a permanent refusal without the historical "unaccepted" trailer.
// Its closure supplies the authoritative answer, including acceptance whose reply was lost.
type permanentSubmitRefusal struct {
	*terminalMachines
	accepted bool
	closed   atomic.Int32
}

func (m *permanentSubmitRefusal) SubmitMachineExecution(ctx context.Context, request *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	if m.accepted {
		if _, err := m.terminalMachines.SubmitMachineExecution(ctx, request); err != nil {
			return nil, err
		}
	}
	return nil, status.Error(codes.FailedPrecondition, "hub_access_absent: this machine has no execution access for the requested Hub")
}
func (m *permanentSubmitRefusal) CloseMachineSubmission(ctx context.Context, request *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if !strings.HasPrefix(request.RequestId, "closure-probe-") {
		m.closed.Add(1)
	}
	return m.terminalMachines.CloseMachineSubmission(ctx, request)
}

func TestPermanentSubmissionRefusalUsesClosureAndPreservesAcceptedWork(t *testing.T) {
	for _, accepted := range []bool{false, true} {
		name := "absent"
		if accepted {
			name = "accepted reply lost"
		}
		t.Run(name, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &permanentSubmitRefusal{accepted: accepted, terminalMachines: newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
				return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
			})}
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, nil)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			row, _ := rentedRun(t, root, store, "permanent-refusal", "generate", "steps=1")
			link, problem := store.MachineExecution(row.ID)
			fatal(t, problem)
			if machine.closed.Load() != 1 {
				t.Fatalf("permanent refusal made %d closure calls", machine.closed.Load())
			}
			if accepted {
				if row.State == "refused" || len(link.Receipt) == 0 || !link.Collected || link.SubmissionClosed {
					t.Fatal("closure lost or failed an already accepted execution")
				}
			} else {
				if row.State != "refused" || len(link.Receipt) != 0 || !link.SubmissionClosed {
					t.Fatalf("absent submission did not retain durable refusal: %s", row.State)
				}
				_, shown := runCozy(t, root, "run", "show", row.ID, "--json")
				if !strings.Contains(shown, "hub_access_absent") {
					t.Fatalf("run lost the refusal reason: %s", shown)
				}
			}
		})
	}
}
