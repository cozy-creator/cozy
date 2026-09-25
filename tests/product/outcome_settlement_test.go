package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A blocked byte-plane response must not hide a newer capacity report. Neither
// that report nor a duplicate terminal permits an ack before verified custody.
// After ack, blocked cleanup must not delay the next outcome's acknowledgment.
func TestOutcomeSettlementKeepsControlAndAcknowledgmentsMoving(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	type offered struct {
		message *pb.AttemptOffer
		send    func(*pb.WorkerFrame) error
	}
	offers := make(chan offered, 2)
	acks := make(chan *pb.AttemptOutcomeAck, 3)
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if offer := frame.GetAttemptOffer(); offer != nil {
			offers <- offered{offer, send}
			return true, nil
		}
		if ack := frame.GetOutcomeAck(); ack != nil {
			acks <- ack
			return true, nil
		}
		return false, nil
	}
	mirroring, mirrorGate := make(chan struct{}), make(chan struct{})
	cleaning, cleanupGate := make(chan struct{}), make(chan struct{})
	var mirrorOnce, cleanupOnce, releaseMirror, releaseCleanup sync.Once
	defer releaseMirror.Do(func() { close(mirrorGate) })
	defer releaseCleanup.Do(func() { close(cleanupGate) })
	media := []byte("one verified output")
	pod.mediaRequest = func(w http.ResponseWriter, r *http.Request) bool {
		switch {
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			mirrorOnce.Do(func() { close(mirroring) })
			<-mirrorGate
			_, _ = w.Write(media)
			return true
		case r.Method == http.MethodDelete:
			cleanupOnce.Do(func() { close(cleaning) })
			<-cleanupGate
			_, _ = w.Write([]byte(`{}`))
			return true
		}
		return false
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "outcome-settlement-control", rentalWiring(connection, private))
	const pkg = "paul/outcome-settlement"
	submit := func(id string, image bool) string {
		s := orchestrator.Submission{IdemKey: id, Package: pkg, Release: "1.0.0", Entrypoint: "tile",
			PlanID: podPlanID(pkg), Payload: []byte(`{}`), Worker: podRental, Rental: true, RentalRequired: true}
		if image {
			s.Outputs = []string{"image"}
		}
		request, _, problem := o.c.Submit(s)
		fatal(t, problem)
		return request
	}
	finish := func(item offered, image bool) *pb.WorkerFrame {
		offer := item.message
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		must(t, err)
		body := &pb.AttemptOutcomeBody{RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: identity, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
			ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}}
		if image {
			body.OutputManifest = &pb.OutputManifest{Outputs: []*pb.OutputEntry{{OutputId: "image",
				Digest: canonical.Digest(media), Length: uint64(len(media)), MimeType: "image/png"}}}
		}
		raw, digest, err := canonical.Identity(body)
		must(t, err)
		frame := &pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
			RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
			WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + offer.RequestId,
			OutcomeDigest: digest, OutcomeCanonicalBytes: raw}}}
		must(t, item.send(frame))
		return frame
	}
	first := submit("first", true)
	waitUntil(t, "first offer", func() bool { return len(offers) == 1 })
	a := <-offers
	terminal := finish(a, true)
	waitUntil(t, "blocked output mirror", func() bool {
		select {
		case <-mirroring:
			return true
		default:
			return false
		}
	})
	pod.mu.Lock()
	desired := pod.desired[len(pod.desired)-1]
	pod.noSeats = true
	pod.mu.Unlock()
	state := pod.served(desired, 1)
	state.GetObservedState().AdmissionEpoch = 23
	must(t, a.send(state))
	attempt, problem := o.store.AttemptRow(first, 1)
	fatal(t, problem)
	waitUntil(t, "capacity report while output mirror is blocked", func() bool {
		facts := o.c.Worker(attempt.InstanceID)
		return facts != nil && facts.AdmissionEpoch == 23 && facts.AvailableSlots == 0
	})
	if len(acks) != 0 {
		t.Fatal("outcome acknowledged before its output bytes arrived")
	}
	must(t, a.send(terminal)) // replay queues behind the first immutable terminal
	releaseMirror.Do(func() { close(mirrorGate) })
	waitUntil(t, "both exact outcome acknowledgments", func() bool { return len(acks) == 2 })
	for range 2 {
		ack := <-acks
		if ack.RequestId != first {
			t.Fatalf("unexpected acknowledgment: %s", ack.RequestId)
		}
		row, problem := o.store.AttemptRow(first, 1)
		fatal(t, problem)
		if row.State != "terminal" && row.State != "closed" {
			t.Fatal("ack preceded durable terminal")
		}
	}
	outputs, problem := o.store.VisibleOutputs(first)
	fatal(t, problem)
	if len(outputs) != 1 {
		t.Fatalf("duplicate outcome published %d outputs", len(outputs))
	}
	got, err := os.ReadFile(outputs[0].Path)
	must(t, err)
	if string(got) != string(media) {
		t.Fatal("ack preceded verified output custody")
	}
	waitUntil(t, "blocked post-ack cleanup", func() bool {
		select {
		case <-cleaning:
			return true
		default:
			return false
		}
	})
	pod.mu.Lock()
	pod.noSeats = false
	pod.mu.Unlock()
	state = pod.served(desired, 1)
	state.GetObservedState().AdmissionEpoch = 24
	must(t, a.send(state))
	second := submit("second", false)
	waitUntil(t, "second offer", func() bool { return len(offers) == 1 })
	finish(<-offers, false)
	waitUntil(t, "next ack while prior cleanup remains blocked", func() bool { return len(acks) == 1 })
	if ack := <-acks; ack.RequestId != second {
		t.Fatalf("unexpected acknowledgment: %s", ack.RequestId)
	}
	releaseCleanup.Do(func() { close(cleanupGate) })
	// The durable export row is settled independently and remains replayable.
	if export, problem := o.store.OutputExportOf(first); problem != nil || (export != nil && export.State != "published") {
		t.Fatalf("output export not independently settled: %v, %v", export, problem)
	}
}
