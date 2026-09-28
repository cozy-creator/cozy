package producttest

import (
	"context"
	"crypto/sha256"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// flakyFirstSubmit is a fresh pod whose first submission fails inside Runtime before
// anything is journaled. Runtime proves the absence (execution_submission_refused) and
// answers UNAVAILABLE, as `checked_submit` does for an unexpected exception (run 1416).
// Every later submission is accepted and runs until the test ends.
type flakyFirstSubmit struct {
	*runtimeMachine
	failed   atomic.Bool
	accepted sync.Map // request id -> *pb.MachineExecutionState
}

func (m *flakyFirstSubmit) SubmitMachineExecution(ctx context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	if m.failed.CompareAndSwap(false, true) {
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_submission_refused"))
		return nil, status.Error(codes.Unavailable, "machine execution operation failed")
	}
	id := submit.Offer.RequestId
	m.accepted.LoadOrStore(id, &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
		ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: "running"})
	digest := sha256.Sum256([]byte(submit.SubmissionId))
	return &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: digest[:],
		InvocationSpecDigest: digest[:], AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *flakyFirstSubmit) GetMachineExecution(_ context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	state, ok := m.accepted.Load(query.RequestId)
	if !ok {
		return nil, status.Error(codes.NotFound, "execution is not held by this owner")
	}
	return proto.Clone(state.(*pb.MachineExecutionState)).(*pb.MachineExecutionState), nil
}

func (m *flakyFirstSubmit) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return &pb.MachineExecutionEventPage{NextAfter: query.After, HeadSequence: query.After}, nil
}

// A submission that failed in transit before acceptance, and that Runtime proved it never
// accepted, is sent again while the machine lives; it does not fail the run. Two runs
// arrive together on a fresh pod, and the first submission's transport fails (run 1416).
func TestAPreAcceptanceTransportFailureIsRetriedNotFailed(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &flakyFirstSubmit{runtimeMachine: &runtimeMachine{}}
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, nil)
	keys := []string{"arrives-first", "arrives-together"}
	var wg sync.WaitGroup
	for _, key := range keys {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_, _ = cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key)
		}()
	}
	wg.Wait()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	eventually(t, root, "both runs accepted by the machine", func() bool {
		accepted := 0
		for _, key := range keys {
			row, problem := store.RequestByIdempotencyKey(key)
			if problem == nil && row != nil && (row.State == "refused" || row.State == "failed") {
				t.Fatalf("run %s failed on a transient pre-acceptance transport failure: %s", key, row.State)
			}
			if problem == nil && row != nil {
				if _, ok := machine.accepted.Load(row.ID); ok {
					accepted++
				}
			}
		}
		return accepted == len(keys)
	})
	if !machine.failed.Load() {
		t.Fatal("the first submission's transport never failed")
	}
}
