package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRemoteScalarResultDoesNotCreateOutputDirectory(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return nil, err
		}
		body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: identity, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
			ExecutionStarted: true, SafeMessage: "Polo",
			Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME},
		})
		return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch,
			ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + offer.RequestId,
			OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "remote-scalar-no-directory", rentalWiring(connection, private))
	const pkg = "paul/marco-polo"
	request, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "scalar-no-directory", Package: pkg, Release: "1.0.0", Entrypoint: "tile",
		PlanID: podPlanID(pkg), Payload: []byte(`{}`), Worker: podRental,
		Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "scalar remote completion", func() bool {
		row, problem := o.store.RequestRow(request)
		fatal(t, problem)
		return row != nil && row.State == "succeeded"
	})
	if _, err := os.Stat(o.l.PackageOutputs(pkg)); !os.IsNotExist(err) {
		t.Fatalf("scalar-only remote run created package output directory: %v", err)
	}
	export, problem := o.store.OutputExportOf(request)
	fatal(t, problem)
	if export != nil {
		t.Fatal("scalar-only remote run recorded a file export")
	}
}
