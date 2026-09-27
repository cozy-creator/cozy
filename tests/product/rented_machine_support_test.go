package producttest

import (
	"context"
	"encoding/json"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// terminalMachines is Runtime on a rented pod: it accepts every submission and has already
// run it to the terminal `finish` names for its payload.
type terminalMachines struct {
	mu     sync.Mutex
	finish func(payload map[string]any) *pb.AttemptOutcomeBody
	runs   map[string]*terminalRun
	order  []string
}

type terminalRun struct {
	submit *pb.MachineExecutionSubmit
	state  *pb.MachineExecutionState
	body   *pb.AttemptOutcomeBody
}

func newTerminalMachines(finish func(map[string]any) *pb.AttemptOutcomeBody) *terminalMachines {
	return &terminalMachines{finish: finish, runs: map[string]*terminalRun{}}
}

func (m *terminalMachines) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *terminalMachines) SubmitMachineExecution(_ context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := submit.Offer.RequestId
	if m.runs[id] == nil {
		var payload map[string]any
		if err := json.Unmarshal(submit.PayloadCanonicalBytes, &payload); err != nil {
			return nil, status.Error(codes.InvalidArgument, "payload")
		}
		body := m.finish(payload)
		if body.RequestId == "" {
			body.RequestId = id
		}
		if body.InvocationSpecDigest == "" {
			body.InvocationSpecDigest, _ = canonical.Spell(submit.Offer.InvocationSpecDigest)
		}
		body.AttemptOrdinal = 1
		state := "succeeded"
		if body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
			state = "failed"
		}
		m.runs[id] = &terminalRun{submit: proto.Clone(submit).(*pb.MachineExecutionSubmit), body: body,
			state: &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
				ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: state}}
		m.order = append(m.order, id)
	}
	return &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: submit.CaptureDigest,
		InvocationSpecDigest: submit.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *terminalMachines) run(id string) (*terminalRun, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if run := m.runs[id]; run != nil {
		return run, nil
	}
	return nil, status.Error(codes.NotFound, "no such execution")
}

func (m *terminalMachines) GetMachineExecution(_ context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	run, err := m.run(query.RequestId)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(run.state).(*pb.MachineExecutionState), nil
}

func (m *terminalMachines) ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return &pb.MachineExecutionEventPage{}, nil
}

func (m *terminalMachines) CollectMachineExecution(_ context.Context, collect *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	run, err := m.run(collect.Execution.RequestId)
	if err != nil {
		return nil, err
	}
	body, digest, err := canonical.Identity(run.body)
	if err != nil {
		return nil, err
	}
	return &pb.AttemptOutcome{RequestId: collect.Execution.RequestId, AttemptOrdinal: 1,
		InvocationSpecDigest: run.submit.Offer.InvocationSpecDigest, OutcomeId: "outcome-" + collect.Execution.RequestId,
		OutcomeDigest: digest, OutcomeCanonicalBytes: body}, nil
}

func (m *terminalMachines) AcknowledgeMachineExecutionCollection(_ context.Context, ack *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	run, err := m.run(ack.Execution.RequestId)
	if err != nil {
		return nil, err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	run.state.Collected = true
	return proto.Clone(run.state).(*pb.MachineExecutionState), nil
}

func (m *terminalMachines) submitted() []*pb.MachineExecutionSubmit {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]*pb.MachineExecutionSubmit, 0, len(m.order))
	for _, id := range m.order {
		out = append(out, m.runs[id].submit)
	}
	return out
}

// outcome is a Runtime terminal: its status, its diagnosis and what it measured.
func outcome(status pb.OutcomeStatus, message string, metrics *pb.AttemptMetrics) *pb.AttemptOutcomeBody {
	return &pb.AttemptOutcomeBody{Status: status, SafeMessage: message, Metrics: metrics,
		Result: &pb.ResultEnvelope{InlineResult: []byte(`{}`)}}
}

// rentedRun is `cozy run proof/h3/<function> args... --rental=tessa --json` under `key`: the
// settled run, collected when Runtime accepted it, and what the command printed.
func rentedRun(t *testing.T, root string, store *records.Store, key, function string, args ...string) (*records.Request, string) {
	t.Helper()
	argv := append([]string{"run", ladderPackage + "/" + function}, args...)
	_, out := runCozy(t, root, append(argv, "--rental=tessa", "--json", "--idempotency-key", key)...)
	waitFor(t, root, "run "+key+" to settle ("+out+")", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		if problem != nil || row == nil {
			return false
		}
		link, _ := store.MachineExecution(row.ID)
		return records.Settled(row.State) && (link == nil || len(link.Receipt) == 0 || link.Collected)
	})
	row, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	return row, out
}
