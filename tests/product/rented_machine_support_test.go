package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// terminalMachines is Runtime on a rented pod: it accepts every submission and has already
// run it to the terminal `finish` names for its payload.
type terminalMachines struct {
	modelOverrides bool
	// unavailable is how many submissions the transport loses before one arrives.
	unavailable atomic.Int32
	attempts    atomic.Int32
	// refuse is the refusal this Runtime answers a release root of a package with.
	refuse map[string]string
	// older is a Runtime from before release roots; a Runtime update can change it.
	older  atomic.Bool
	mu     sync.Mutex
	finish func(payload map[string]any) *pb.AttemptOutcomeBody
	runs   map[string]*terminalRun
	closed map[string]bool
	order  []string
}

type terminalRun struct {
	submit *pb.MachineExecutionSubmit
	state  *pb.MachineExecutionState
	body   *pb.AttemptOutcomeBody
}

func newTerminalMachines(finish func(map[string]any) *pb.AttemptOutcomeBody) *terminalMachines {
	return &terminalMachines{finish: finish, runs: map[string]*terminalRun{}, closed: map[string]bool{}}
}

func (m *terminalMachines) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace", RunOutputLog: true, ModelOverrides: m.modelOverrides}, nil
}

// minted is what a Runtime names a submission's capture and invocation by: the submitted
// ones, or for a root by its release the ones it minted itself.
func minted(submit *pb.MachineExecutionSubmit) ([]byte, []byte) {
	if submit.ReleaseRoot == nil {
		return submit.CaptureDigest, submit.Offer.InvocationSpecDigest
	}
	digest := sha256.Sum256([]byte(submit.SubmissionId))
	return digest[:], digest[:]
}

func (m *terminalMachines) SubmitMachineExecution(ctx context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.attempts.Add(1)
	if m.unavailable.Add(-1) >= 0 {
		return nil, status.Error(codes.Unavailable, "Tensorhub is restarting")
	}
	if refusal := m.refuse[submit.GetReleaseRoot().GetPackage()]; refusal != "" {
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_submission_refused"))
		return nil, status.Error(codes.FailedPrecondition, refusal)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	id := submit.Offer.RequestId
	if m.closed[submit.SubmissionId] {
		return nil, status.Error(codes.FailedPrecondition, "submission closed")
	}
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
			_, invocation := minted(submit)
			body.InvocationSpecDigest, _ = canonical.Spell(invocation)
		}
		body.AttemptOrdinal = 1
		state := "succeeded"
		if body.Status != pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED {
			state = "failed"
		}
		m.runs[id] = &terminalRun{submit: proto.Clone(submit).(*pb.MachineExecutionSubmit), body: body,
			state: &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
				ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: state, Sequence: 1}}
		m.order = append(m.order, id)
	}
	capture, invocation := minted(submit)
	return &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: capture,
		InvocationSpecDigest: invocation, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *terminalMachines) CloseMachineSubmission(_ context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := &pb.MachineSubmissionClosure{RequestId: q.RequestId, SubmissionId: q.SubmissionId, ExecutionWorkspaceId: q.ExpectedExecutionWorkspaceId}
	if run := m.runs[q.RequestId]; run != nil {
		capture, invocation := minted(run.submit)
		out.Receipt = &pb.MachineExecutionReceipt{RequestId: q.RequestId, SubmissionId: run.submit.SubmissionId, ExecutionWorkspaceId: "rented-workspace", WorkerId: run.submit.Claim.WorkerId, WorkerBootId: run.submit.Claim.WorkerBootId, CaptureDigest: capture, InvocationSpecDigest: invocation, AcceptedAtMs: 100}
	} else {
		m.closed[q.SubmissionId] = true
	}
	return out, nil
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

// ListMachineExecutionEvents is the run's log: it has already ended, in its terminal entry.
func (m *terminalMachines) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	id := query.GetExecution().GetRequestId()
	run, err := m.run(id)
	if err != nil {
		return nil, err
	}
	body, digest, err := canonical.Identity(run.body)
	if err != nil {
		return nil, err
	}
	_, invocation := minted(run.submit)
	outcome := &pb.AttemptOutcome{RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: invocation,
		OutcomeId: "outcome-" + id, OutcomeDigest: digest, OutcomeCanonicalBytes: body}
	return outcomePage(query.After, 1, run.state.State, outcome), nil
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
