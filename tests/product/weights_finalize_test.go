package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A Runtime which has lost its attempt history needs the exact original declaration
// even when the producer refused before starting and no writer exists to look up.
func TestWeightsFinalizationCarriesStoredInvocation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true,
		answerOffer: func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
			specDigest, err := canonical.Spell(offer.InvocationSpecDigest)
			if err != nil {
				return nil, err
			}
			body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
				RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
				InvocationSpecDigest: specDigest,
				Status:               pb.OutcomeStatus_OUTCOME_STATUS_REFUSED,
				Cause: &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_PROTOCOL,
					Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME},
				SafeMessage: "producer refused before creating writers",
			})
			return &pb.AttemptOutcome{
				RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
				WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId,
				AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest,
				OutcomeId: "outcome-before-writers", OutcomeDigest: digest, OutcomeCanonicalBytes: body,
			}, err
		}}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "finalize-invocation", rentalWiring(connection, private))
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "finalize-invocation", Package: "cozy/h3-package", Entrypoint: "four-lane",
		PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7",
		Kind: "job", Org: "paul", Payload: []byte(`{"steps":4}`), Outputs: []string{"model"},
		WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "model",
			MimeType: orchestrator.WeightsManifestMime, MaxBytes: 1 << 30}},
		Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "weights finalization for refused producer", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.finalizations) > 0
	})
	pod.mu.Lock()
	offer, finalize := pod.offers[0], pod.finalizations[0]
	pod.mu.Unlock()
	stored, problem := o.store.AttemptRow(requestID, int64(offer.AttemptOrdinal))
	fatal(t, problem)
	if stored == nil || len(stored.InvocationCanonical) == 0 {
		t.Fatal("the dispatched invocation was not durably stored")
	}
	if !bytes.Equal(finalize.InvocationSpecCanonicalBytes, stored.InvocationCanonical) ||
		!bytes.Equal(finalize.InvocationSpecCanonicalBytes, offer.InvocationSpecCanonicalBytes) {
		t.Fatalf("finalization carried %d invocation bytes, stored and offered %d",
			len(finalize.InvocationSpecCanonicalBytes), len(stored.InvocationCanonical))
	}
	if !bytes.Equal(finalize.InvocationSpecDigest, canonical.Digest(finalize.InvocationSpecCanonicalBytes)) {
		t.Fatal("finalization's declaration does not hash to its bound invocation digest")
	}
	if finalize.RequestId != requestID || finalize.OutputSlot != "model" ||
		finalize.Disposition != pb.WeightsFinalizeDisposition_WEIGHTS_FINALIZE_DISPOSITION_ABANDON_UNCOMMITTED {
		t.Fatalf("wrong absent-writer finalization: %v", finalize)
	}
	rows, problem := o.store.PendingWeightsFinalizations(requestID, int64(offer.AttemptOrdinal))
	fatal(t, problem)
	if len(rows) != 1 {
		t.Fatalf("unanswered finalization lost its durable pending decision: %d rows", len(rows))
	}
}
