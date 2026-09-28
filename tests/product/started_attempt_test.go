package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// Started work is never run again (owner ruling 2026-09-28): an attempt that began
// executing and then failed or was abandoned ends its request with the attempt's own cause
// and message. Only an attempt that never began is offered again.
func TestAStartedAttemptEndsTheRequestWithItsReason(t *testing.T) {
	for _, c := range []struct {
		name     string
		status   pb.OutcomeStatus
		cause    pb.CauseCode
		started  bool
		state    string
		attempts int
	}{
		{"failed after starting", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, pb.CauseCode_CAUSE_CODE_EXECUTOR_FAULT, true, "failed", 1},
		{"abandoned after starting", pb.OutcomeStatus_OUTCOME_STATUS_ABANDONED, pb.CauseCode_CAUSE_CODE_EXECUTOR_INVALIDATED, true, "abandoned", 1},
		{"lost before starting", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, pb.CauseCode_CAUSE_CODE_EXECUTOR_FAULT, false, "succeeded", 2},
	} {
		t.Run(c.name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			const reason = "watchdog lane crashed finishing a child result: KeyError: 'child-3'"
			var offers atomic.Int64
			var ready *pb.WorkerFrame
			pod := &fakePod{controlKey: public, serve: true, jobReady: true}
			pod.preparedPlacement = func(raw []byte, name, version string) *pb.Placement {
				return podPlacement(raw, name, version, "h3-package")
			}
			pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
				frame.GetObservedState().ConvergedRevision = 0
				ready = proto.Clone(frame).(*pb.WorkerFrame)
				return send(frame)
			}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
				if frame.GetOutcomeAck() != nil && ready != nil {
					return true, send(proto.Clone(ready).(*pb.WorkerFrame))
				}
				return false, nil
			}
			pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
				n := offers.Add(1)
				identity, err := canonical.Spell(offer.InvocationSpecDigest)
				if err != nil {
					return nil, err
				}
				result := &pb.AttemptOutcomeBody{RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
					InvocationSpecDigest: identity, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true,
					Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}}
				if n == 1 {
					result.Status, result.ExecutionStarted, result.SafeMessage = c.status, c.started, reason
					result.Cause = &pb.OutcomeCause{Code: c.cause, Origin: pb.CauseOrigin_CAUSE_ORIGIN_WORKER}
				}
				body, digest, err := canonical.Identity(result)
				return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
					WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
					InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: fmt.Sprintf("out-started-%d", n),
					OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "started-attempt", rentalWiring(connection, private), func(options *orchestrator.Options) {
				options.RentalFleet = func(records.Request) (string, *exit.Error) { return "one existing rental", nil }
			})
			id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "started-attempt", Package: "cozy/h3-package",
				Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + sixtyFour("5"), Kind: "job", Org: "paul",
				Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true})
			fatal(t, problem)
			waitUntil(t, "the request settles "+c.state, func() bool {
				row, problem := o.store.RequestRow(id)
				fatal(t, problem)
				return row != nil && row.State == c.state
			})
			attempts, problem := o.store.Attempts(id)
			fatal(t, problem)
			row, problem := o.store.RequestRow(id)
			fatal(t, problem)
			if len(attempts) != c.attempts || offers.Load() != int64(c.attempts) || row.Requeues != int64(c.attempts-1) {
				t.Fatalf("%d attempt(s), %d offer(s), %d requeue(s); want %d attempt(s)", len(attempts), offers.Load(), row.Requeues, c.attempts)
			}
			if !c.started {
				return
			}
			errorType, _, message, problem := o.store.SettledFailure(id)
			fatal(t, problem)
			if want := trimCause(c.cause); errorType != want || message != reason {
				t.Fatalf("the request ended %q: %q, want %q: %q", errorType, message, want, reason)
			}
		})
	}
}

// An attempt that may have started on a rented machine that was lost fails naming the
// loss; it is never offered again, and nothing is bought to run it.
func TestALostRentalFailsItsStartedAttempt(t *testing.T) {
	var purchases atomic.Int32
	o := hostOwner(t, "lost-started-attempt", func(opt *orchestrator.Options) {
		opt.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			purchases.Add(1)
			return orchestrator.PlacementDecision{}, "", nil
		}
	})
	const id = "req-lost-started"
	_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + sixtyFour("1"), Package: "proof/model",
		Entrypoint: "run", Payload: []byte("{}"), Rental: true, RentalRequired: true, Worker: "gone"})
	fatal(t, problem)
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "lost", Package: "proof/model", WorkerID: "lost", Devices: []string{"cpu"}}))
	ordinal, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: "session", InstanceID: "lost",
		InvocationDigest: "sha256:" + sixtyFour("2"), InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, ordinal, "session"))
	fatal(t, o.store.Accepted(id, ordinal, "session"))
	o.c.RecoverLostWork()
	row, problem := o.store.RequestRow(id)
	fatal(t, problem)
	attempts, problem := o.store.Attempts(id)
	fatal(t, problem)
	errorType, _, message, problem := o.store.SettledFailure(id)
	fatal(t, problem)
	if row.State != "failed" || row.Requeues != 0 || len(attempts) != 1 || purchases.Load() != 0 ||
		errorType != "rental.lost" || !strings.Contains(message, "the rented machine gone was lost") {
		t.Fatalf("a lost started attempt was not failed with its loss: %s, %d requeue(s), %d attempt(s), %d purchase(s), %s: %s",
			row.State, row.Requeues, len(attempts), purchases.Load(), errorType, message)
	}
}

func trimCause(code pb.CauseCode) string {
	return code.String()[len("CAUSE_CODE_"):]
}
