package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// This is a protocol peer, not a simulated storage implementation. It withholds
// the first attempt's terminal until the test releases its writer, then verifies
// that Creator acknowledges retained custody and issues a new immutable attempt
// under the same request. Native output replay is proved by the actual Host test.
func TestPrivateTransactionPauseFencesAttemptBeforeResume(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	peer := &privateAttemptPeer{release: make(chan struct{}), canceled: make(chan struct{})}
	t.Cleanup(func() { peer.releaseOnce.Do(func() { close(peer.release) }) })
	pod.onFrame = peer.frame
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		peer.mu.Lock()
		peer.ready = frame
		peer.mu.Unlock()
		return send(frame)
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "private-transaction-attempts", rentalWiring(connection, private))
	fatal(t, o.store.RecordRental(records.Rental{ID: podRental, MachineName: "otter", State: "ready",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	sub := orchestrator.Submission{
		IdemKey: "private-attempt-proof", Package: "cozy/h3-package", Entrypoint: "prepare",
		PlanID: "sha256:" + fmt.Sprintf("%064x", 17), Release: "1.0.7", Kind: "job", Org: "local",
		Payload: []byte(`{"source_revision":"fixed","seed":17}`),
		Worker:  podRental, Rental: true, RentalRequired: true, RetainWork: true,
	}
	requestID, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	waitUntil(t, "first private attempt accepted", func() bool {
		attempt, problem := o.store.AttemptRow(requestID, 1)
		fatal(t, problem)
		return attempt != nil && attempt.State == "accepted"
	})
	before, problem := o.store.RequestByReference(requestID)
	fatal(t, problem)
	fatal(t, o.c.PauseRequest(requestID, "product proof"))
	select {
	case <-peer.canceled:
	case <-time.After(10 * time.Second):
		t.Fatal("pause did not reach the actual attempt")
	}
	row, problem := o.store.RequestByReference(requestID)
	fatal(t, problem)
	if row.State != "pausing" {
		t.Fatalf("pause acknowledged before the active writer stopped: %s", row.State)
	}
	if problem := o.c.ResumeRequest(requestID, "too soon"); problem == nil {
		t.Fatal("resume accepted while the original writer was still active")
	}
	peer.mu.Lock()
	if len(peer.offers) != 1 || len(peer.acks) != 0 {
		t.Errorf("pause minted computation or acknowledged an active writer: offers=%d acks=%d",
			len(peer.offers), len(peer.acks))
	}
	peer.mu.Unlock()
	peer.releaseOnce.Do(func() { close(peer.release) })
	waitUntil(t, "paused request after attempt closure", func() bool {
		row, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		return row.State == "paused"
	})
	first, problem := o.store.AttemptRow(requestID, 1)
	fatal(t, problem)
	if first == nil || first.State != "closed" || first.TerminalStatus != "CANCELED" || first.TerminalID == "" {
		t.Fatalf("pause did not preserve the immutable canceled attempt: %+v", first)
	}
	assertPrivateTransactionIdentity(t, o.store, *before, "paused")
	waitUntil(t, "retained acknowledgment reaches peer", func() bool {
		peer.mu.Lock()
		defer peer.mu.Unlock()
		return len(peer.acks) == 1
	})
	peer.mu.Lock()
	retained := peer.acks[0].RetainWork
	peer.mu.Unlock()
	if !retained {
		t.Fatal("pause acknowledged attempt with permission to dispose retained Host state")
	}

	fatal(t, o.c.ResumeRequest(requestID, "product proof"))
	waitUntil(t, "resumed attempt deterministically blocked", func() bool {
		row, problem := o.store.RequestRow(requestID)
		fatal(t, problem)
		return row.State == "blocked"
	})
	after, problem := o.store.RequestByReference(requestID)
	fatal(t, problem)
	if after.ID != before.ID || after.Number != before.Number || after.BodyDigest != before.BodyDigest ||
		after.Ordinal != 2 || after.Requeues != before.Requeues || !bytes.Equal(after.Payload, before.Payload) {
		t.Fatalf("resume replaced the request or consumed automatic retry budget: before=%+v after=%+v", before, after)
	}
	firstAfter, problem := o.store.AttemptRow(requestID, 1)
	fatal(t, problem)
	if firstAfter.TerminalID != first.TerminalID || firstAfter.TerminalDigest != first.TerminalDigest ||
		firstAfter.TerminalStatus != first.TerminalStatus || firstAfter.ClosedAt != first.ClosedAt {
		t.Fatal("resume rewrote the prior attempt's outcome")
	}
	if problem := o.c.ResumeRequest(requestID, "repeat deterministic failure"); problem == nil {
		t.Fatal("unchanged author exception became a metered retry loop")
	}
	waitUntil(t, "blocked failure retains custody at the peer", func() bool {
		peer.mu.Lock()
		defer peer.mu.Unlock()
		return len(peer.acks) == 2
	})
	peer.mu.Lock()
	defer peer.mu.Unlock()
	if !peer.acks[1].RetainWork {
		t.Fatal("blocked failure discarded its retained intermediate work")
	}
	if len(peer.offers) != 2 || peer.offers[0].RequestId != peer.offers[1].RequestId ||
		peer.offers[0].AttemptOrdinal != 1 || peer.offers[1].AttemptOrdinal != 2 ||
		!bytes.Equal(peer.offers[0].InvocationSpecDigest, peer.offers[1].InvocationSpecDigest) ||
		!bytes.Equal(peer.offers[0].InvocationSpecCanonicalBytes, peer.offers[1].InvocationSpecCanonicalBytes) {
		t.Fatal("resumption did not offer exactly the original executable intent as attempt two")
	}
}

type privateAttemptPeer struct {
	mu          sync.Mutex
	offers      []*pb.AttemptOffer
	acks        []*pb.AttemptOutcomeAck
	ready       *pb.WorkerFrame
	release     chan struct{}
	canceled    chan struct{}
	releaseOnce sync.Once
	cancelOnce  sync.Once
}

func (p *privateAttemptPeer) frame(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
	if offer := frame.GetAttemptOffer(); offer != nil {
		p.mu.Lock()
		p.offers = append(p.offers, offer)
		p.mu.Unlock()
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
			RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
			WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, PlacementId: offer.PlacementId,
		}}}); err != nil {
			return true, err
		}
		if offer.AttemptOrdinal == 2 {
			return true, send(privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_FAILED,
				pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR))
		}
		return true, nil
	}
	if cancel := frame.GetCancelAttempt(); cancel != nil {
		p.mu.Lock()
		if len(p.offers) != 1 || cancel.RequestId != p.offers[0].RequestId || cancel.AttemptOrdinal != 1 {
			p.mu.Unlock()
			return true, fmt.Errorf("pause addressed another attempt")
		}
		offer := p.offers[0]
		p.mu.Unlock()
		p.cancelOnce.Do(func() {
			close(p.canceled)
			go func() {
				<-p.release
				_ = send(privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_CANCELED,
					pb.CauseCode_CAUSE_CODE_DRAIN_CANCEL, pb.CauseOrigin_CAUSE_ORIGIN_RECORD_OWNER))
			}()
		})
		return true, nil
	}
	if ack := frame.GetOutcomeAck(); ack != nil {
		p.mu.Lock()
		p.acks = append(p.acks, ack)
		ready := p.ready
		p.mu.Unlock()
		return true, send(ready)
	}
	return false, nil
}

func privateAttemptOutcome(offer *pb.AttemptOffer, status pb.OutcomeStatus, code pb.CauseCode, origin pb.CauseOrigin) *pb.WorkerFrame {
	digest, _ := canonical.Spell(offer.InvocationSpecDigest)
	body, hash, err := canonical.Identity(&pb.AttemptOutcomeBody{
		RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: digest,
		Status: status, ExecutionStarted: true, SafeMessage: "controlled private transaction interruption",
		Cause: &pb.OutcomeCause{Code: code, Origin: origin},
	})
	if err != nil {
		panic(err)
	}
	return &pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
		RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
		WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: offer.InvocationSpecDigest, PlacementId: offer.PlacementId,
		OutcomeId: fmt.Sprintf("private-outcome-%d", offer.AttemptOrdinal), OutcomeDigest: hash, OutcomeCanonicalBytes: body,
	}}}
}
