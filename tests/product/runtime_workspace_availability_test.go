package producttest

import (
	"bytes"
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type unavailableWorkspaceMachine struct {
	*closureAdmissionMachine
	available             atomic.Bool
	probeUnavailable      atomic.Bool
	probeWorkspaceChanged atomic.Bool
	reads                 atomic.Int32
	phase                 string
	trailer               bool
	workspaceError        bool
}

func (m *unavailableWorkspaceMachine) unavailable(ctx context.Context) {
	values := metadata.Pairs("cozy-snapshot-source", "journal", "cozy-runtime-available", "false", "cozy-runtime-state", m.phase)
	if m.trailer {
		_ = grpc.SetTrailer(ctx, values)
	} else {
		_ = grpc.SetHeader(ctx, values)
	}
}
func (m *unavailableWorkspaceMachine) SubmissionClosureAvailable() bool { return m.available.Load() }
func (m *unavailableWorkspaceMachine) GetMachineExecutionWorkspace(ctx context.Context, q *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	if !m.available.Load() {
		m.unavailable(ctx)
		if m.workspaceError {
			return nil, status.Error(codes.FailedPrecondition, pb.CapabilityUnavailableCode+": Runtime is unavailable")
		}
	}
	return m.closureAdmissionMachine.GetMachineExecutionWorkspace(ctx, q)
}
func (m *unavailableWorkspaceMachine) CloseMachineSubmission(ctx context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if m.probeUnavailable.Load() {
		if m.probeWorkspaceChanged.Load() {
			m.unavailable(ctx)
			_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_workspace_changed"))
			return nil, status.Error(codes.FailedPrecondition, "execution_workspace_changed: Runtime is replacing the workspace")
		}
		m.unavailable(ctx)
		return nil, status.Error(codes.Unimplemented, "closure unavailable while replacing Runtime")
	}
	return m.closureAdmissionMachine.CloseMachineSubmission(ctx, q)
}
func (m *unavailableWorkspaceMachine) GetMachineExecution(ctx context.Context, q *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.reads.Add(1)
	return m.closureAdmissionMachine.GetMachineExecution(ctx, q)
}
func (m *unavailableWorkspaceMachine) AcknowledgeMachineExecutionCollection(ctx context.Context, q *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	if !m.available.Load() {
		return nil, status.Error(codes.Unavailable, "Runtime unavailable for collection acknowledgement")
	}
	return m.closureAdmissionMachine.AcknowledgeMachineExecutionCollection(ctx, q)
}

func TestRuntimeFallbackWaitsBeforeNewSubmission(t *testing.T) {
	for _, arm := range []struct {
		name                                             string
		trailer, workspaceError, probe, workspaceChanged bool
	}{
		{name: "booting"}, {name: "restarting", trailer: true}, {name: "updating"},
		{name: "failed_workspace", workspaceError: true}, {name: "probe_restart", probe: true},
		{name: "probe_restart_workspace_changed", probe: true, workspaceChanged: true},
	} {
		t.Run(arm.name, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &unavailableWorkspaceMachine{closureAdmissionMachine: newClosureAdmissionMachine("supported"), phase: arm.name, trailer: arm.trailer, workspaceError: arm.workspaceError}
			machine.available.Store(arm.probe)
			machine.probeUnavailable.Store(arm.probe)
			machine.probeWorkspaceChanged.Store(arm.workspaceChanged)
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4}, nil)
			code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "transient-runtime")
			if code != 0 {
				t.Fatalf("local submission failed: %s", out)
			}
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			request, problem := store.RequestByIdempotencyKey("transient-runtime")
			fatal(t, problem)
			waitFor(t, root, "temporary Runtime wait surfaced", func() bool {
				_, out := runCozy(t, root, "run", "show", request.ID, "--json")
				return strings.Contains(out, "temporarily unavailable")
			})
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if records.Settled(row.State) || len(link.Submission) != 0 || machine.sent.Load() != 0 {
				t.Fatal("journal fallback failed or transmitted new work")
			}
			if machine.probes.Load() != 0 {
				t.Fatal("journal fallback was treated as a live closure proof")
			}
			machine.available.Store(true)
			machine.probeUnavailable.Store(false)
			waitFor(t, root, "same request resumed when Runtime became live", func() bool { row, _ := store.RequestRow(request.ID); return row != nil && row.State == "succeeded" })
			if machine.sent.Load() != 1 {
				t.Fatalf("submission count=%d", machine.sent.Load())
			}
		})
	}
}

func TestRuntimeFallbackPreservesFrozenAcceptedObservation(t *testing.T) {
	for _, knownReceipt := range []bool{false, true} {
		t.Run(map[bool]string{false: "acceptance_unknown", true: "receipt_known"}[knownReceipt], func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &unavailableWorkspaceMachine{closureAdmissionMachine: newClosureAdmissionMachine("supported"), phase: "updating"}
			var request records.Request
			var frozen []byte
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, func(_ home.Layout, store *records.Store) {
				request, frozen = frozenSubmissionRecord(t, store, false)
				var message pb.MachineExecutionSubmit
				must(t, proto.Unmarshal(frozen, &message))
				message.Claim = &pb.Claim{WorkerId: podWorkerID, WorkerBootId: podBootID, RecordOwnerEpoch: 1}
				receipt, err := machine.terminalMachines.SubmitMachineExecution(context.Background(), &message)
				must(t, err)
				if knownReceipt {
					fatal(t, store.AcceptMachineExecution(request.ID, receipt))
				}
			})
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			if knownReceipt {
				waitFor(t, root, "accepted observer reads journal while Runtime unavailable", func() bool { return machine.reads.Load() > 0 })
			} else {
				waitFor(t, root, "unknown acceptance waits without stopping observer", func() bool {
					_, out := runCozy(t, root, "run", "show", request.ID, "--json")
					return strings.Contains(out, "temporarily unavailable")
				})
			}
			before, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !bytes.Equal(before.Submission, frozen) || before.SubmissionClosed || machine.sent.Load() != 0 {
				t.Fatal("temporary Runtime loss changed frozen acceptance or retransmitted work")
			}
			machine.available.Store(true)
			waitFor(t, root, "observer recovers retained accepted outcome", func() bool { link, _ := store.MachineExecution(request.ID); return link != nil && link.Collected })
			after, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if !bytes.Equal(after.Submission, frozen) || len(after.Receipt) == 0 {
				t.Fatal("recovery replaced frozen identity or lost acceptance")
			}
			if knownReceipt && machine.sent.Load() != 0 {
				t.Fatal("known accepted work was resubmitted")
			}
		})
	}
}
