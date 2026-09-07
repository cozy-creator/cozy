package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Drop the release ACK before the Host applies it, then wait until Creator has
// already settled cancellation before allowing snapshot/terminal replay.
func TestCanceledRetainedAttemptReplaysLostReleaseAck(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var mu sync.Mutex
	var terminal *pb.AttemptOutcome
	held, dropped, allowReplay, released := false, false, false, false
	offers := 0
	pod.hostHeld = func() []*pb.HeldAttempt {
		mu.Lock()
		defer mu.Unlock()
		if !held || terminal == nil {
			return nil
		}
		return []*pb.HeldAttempt{{RequestId: terminal.RequestId, AttemptOrdinal: terminal.AttemptOrdinal, InvocationSpecDigest: terminal.InvocationSpecDigest, State: pb.AttemptState_ATTEMPT_STATE_OUTCOME_PENDING_ACK}}
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		mu.Lock()
		defer mu.Unlock()
		if frame.GetClaim() != nil && dropped && !allowReplay {
			return true, status.Error(codes.Unavailable, "release ACK intentionally lost")
		}
		if offer := frame.GetAttemptOffer(); offer != nil {
			offers++
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest}}}); err != nil {
				return true, err
			}
			terminal = privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_FAILED, pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR).GetAttemptOutcome()
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: terminal}})
		}
		if ack := frame.GetOutcomeAck(); ack != nil {
			if terminal == nil || ack.OutcomeId != terminal.OutcomeId || !bytes.Equal(ack.OutcomeDigest, terminal.OutcomeDigest) {
				t.Error("release ACK changed original terminal")
			}
			if ack.RetainWork {
				held = true
				return true, nil
			}
			if !dropped {
				dropped = true
				return true, status.Error(codes.Unavailable, "drop before applying release ACK")
			}
			held, released = false, true
			return true, nil
		}
		if ack := frame.GetSnapshotAck(); ack != nil && held && allowReplay {
			replay := proto.Clone(terminal).(*pb.AttemptOutcome)
			replay.RecordOwnerEpoch, replay.ControlStreamEpoch, replay.WorkerBootId = ack.RecordOwnerEpoch, ack.ControlStreamEpoch, ack.WorkerBootId
			return false, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: replay}})
		}
		return false, nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "retained-ack-replay", rentalWiring(connection, private))
	fatal(t, o.store.RecordRental(records.Rental{ID: podRental, MachineName: "otter", State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, Address: connection.Addr, CertPath: connection.CACert, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "ack-loss", Package: "cozy/h3-package", Entrypoint: "prepare", PlanID: childDigest("1"), Release: "1.0.7", Kind: "job", Payload: []byte(`{}`), Worker: podRental, Rental: true, RentalRequired: true, RetainWork: true})
	fatal(t, problem)
	waitUntil(t, "original retained terminal", func() bool { mu.Lock(); defer mu.Unlock(); return held })
	original, problem := o.store.AttemptRow(id, 1)
	fatal(t, problem)
	fatal(t, o.c.CancelRetainedRequest(id, "abandon after completion"))
	waitUntil(t, "cancellation settled despite lost forward", func() bool {
		row, problem := o.store.RequestRow(id)
		fatal(t, problem)
		mu.Lock()
		defer mu.Unlock()
		return row.State == "canceled" && dropped
	})
	mu.Lock()
	allowReplay = true
	mu.Unlock()
	waitUntil(t, "snapshot replay receives exact nonretaining ACK", func() bool { mu.Lock(); defer mu.Unlock(); return released })
	after, problem := o.store.RequestRow(id)
	fatal(t, problem)
	last, problem := o.store.AttemptRow(id, 1)
	fatal(t, problem)
	mu.Lock()
	defer mu.Unlock()
	if after.State != "canceled" || offers != 1 || last.TerminalID != original.TerminalID || last.TerminalDigest != original.TerminalDigest || last.TerminalStatus != original.TerminalStatus || last.ClosedAt != original.ClosedAt {
		t.Fatal("lost-ACK recovery rewrote canceled execution or ran another attempt")
	}
}
