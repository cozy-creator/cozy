package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (p *fakePod) RecordOperationResult(_ context.Context, call *pb.RecordOperationResultCall) (*pb.RecordOperationResultResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if p.recordOperation == nil {
		return nil, status.Error(codes.Unimplemented, "no operation cache configured")
	}
	return p.recordOperation(call)
}

func (p *fakePod) NumericalEnvironment(_ context.Context, call *pb.NumericalEnvironmentCall) (*pb.NumericalEnvironmentResult, error) {
	if err := p.verifyClaim(call.GetClaim(), false); err != nil {
		return nil, err
	}
	if len(p.numericalDigest) != 32 {
		return nil, status.Error(codes.Unimplemented, "no numerical environment configured")
	}
	return &pb.NumericalEnvironmentResult{Digest: p.numericalDigest}, nil
}

// Optional cache admission is distinct from durably accepting the result. A
// transport failure retries; a conclusive uncached answer permits the original ACK.
func TestOperationCacheDeclineAcknowledgesSuccessfulResult(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	numerical, _ := canonical.Raw(childDigest("4"))
	pod := &fakePod{controlKey: public, numericalDigest: numerical}
	var calls, acknowledgements atomic.Int64
	replay := make(chan func() error, 1)
	const id = "req-cache-admission-declined"
	invocation, _ := canonical.Raw(childDigest("1"))
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: id, AttemptOrdinal: 1,
		InvocationSpecDigest: childDigest("1"), Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}})
	must(t, err)
	outcomeDigest, _ := canonical.Spell(digest)
	pod.snapshotHeld = []*pb.HeldAttempt{{RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: invocation, State: pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK}}
	pod.recordOperation = func(call *pb.RecordOperationResultCall) (*pb.RecordOperationResultResult, error) {
		if call.RequestId != id || call.OutcomeId != "out-cache-declined" || !bytes.Equal(call.OutcomeDigest, digest) {
			t.Error("cache record changed original result provenance")
		}
		count := calls.Add(1)
		if count == 1 {
			return nil, status.Error(codes.Unavailable, "controlled journal outage")
		}
		if count > 2 {
			return nil, status.Error(codes.FailedPrecondition, "settled source already compacted")
		}
		return &pb.RecordOperationResultResult{ComputationDigest: call.ComputationDigest, Recorded: false}, nil
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if ack := frame.GetSnapshotAck(); ack != nil {
			outcome := &pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
				RecordOwnerEpoch: ack.RecordOwnerEpoch, ControlStreamEpoch: ack.ControlStreamEpoch, WorkerBootId: ack.WorkerBootId,
				RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: invocation, OutcomeId: "out-cache-declined",
				OutcomeDigest: digest, OutcomeCanonicalBytes: body}}}
			select {
			case replay <- func() error { return send(outcome) }:
			default:
			}
			return false, send(outcome)
		}
		if ack := frame.GetOutcomeAck(); ack != nil && ack.RequestId == id {
			if calls.Load() < 2 || ack.OutcomeId != "out-cache-declined" || !bytes.Equal(ack.OutcomeDigest, digest) {
				t.Error("result was acknowledged before journal recovery or with changed identity")
			}
			acknowledgements.Add(1)
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "operation-cache-decline-"+records.NewID("proof"), rentalWiring(connection, private))
	instance := (orchestrator.WorkerLaunchSpec{Connection: connection}).InstanceID()
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: instance, WorkerID: podWorkerID}))
	request, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("2"),
		Package: "local/operation", Entrypoint: "compute", Kind: "job", RetainWork: true, Payload: []byte(`{}`),
		Worker: podRental, ParentRequestID: "retained-parent", ParentCallIndex: 0, ChildReusable: true, ChildTargetDigest: childDigest("3")})
	fatal(t, problem)
	_, problem = o.store.BindOperationContext(id, childDigest("4"))
	fatal(t, problem)
	_, problem = o.store.Dispatch(records.Attempt{RequestID: id, InstanceID: instance, SessionID: podBootID,
		InvocationDigest: childDigest("1"), InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, 1, podBootID))
	_, problem = o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: 1, SessionID: podBootID,
		InvocationDigest: childDigest("1"), TerminalID: "out-cache-declined", TerminalDigest: outcomeDigest,
		Status: "SUCCEEDED", RequestState: "succeeded", Body: body})
	fatal(t, problem)
	_, _, _, problem = o.c.EnsureRental(podRental)
	fatal(t, problem)
	waitUntil(t, "uncached success receives its original ACK", func() bool { return acknowledgements.Load() > 0 })
	waitUntil(t, "the original outcome closure is durable", func() bool {
		row, problem := o.store.AttemptRow(id, 1)
		fatal(t, problem)
		return row.State == "closed"
	})
	fatal(t, o.store.Recover(id, 1, podBootID))
	// A delayed duplicate may outlive the workspace's acknowledged source row.
	// Creator already completed cache admission, so only its exact ACK is replayed.
	must(t, (<-replay)())
	waitUntil(t, "historical terminal replays only its ACK", func() bool { return acknowledgements.Load() == 2 })
	current, problem := o.store.RequestRow(request.ID)
	fatal(t, problem)
	if current.State != "succeeded" || current.Ordinal != 1 || calls.Load() != 2 {
		t.Fatalf("cache refusal changed success or kept retrying: state=%s ordinal=%d calls=%d", current.State, current.Ordinal, calls.Load())
	}
}
