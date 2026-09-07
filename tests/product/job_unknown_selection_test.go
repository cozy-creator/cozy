package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// An unobserved model selection is the same state shape older Creator left on
// remote jobs. It cannot prove readiness for a newly requested checkpoint.
func TestRemoteJobUnknownSelectionRequiresCheckpointPreparation(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	const pkg = "cozy/h3-package"
	const plan = "sha256:3535353535353535353535353535353535353535353535353535353535353535"
	checkpoint := "sha256:" + strings.Repeat("b", 64)
	entered, release := make(chan struct{}), make(chan struct{})
	var releaseOnce sync.Once
	defer releaseOnce.Do(func() { close(release) })
	var prepared atomic.Bool
	var fleetSelections atomic.Int64
	var latestReady *pb.WorkerFrame
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	pod.preparedPlacement = func(raw []byte, name, version string) *pb.Placement {
		doc, err := canonical.Read(raw, &pb.DownloadDelegation{})
		must(t, err)
		models := doc.List("models")
		if len(models) != 0 {
			if len(models) != 1 || models[0].Str("manifest") != checkpoint {
				t.Error("preparation changed the requested checkpoint")
			}
			close(entered)
			<-release
			prepared.Store(true)
		}
		return podPlacement(raw, name, version, "h3-package")
	}
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		frame.GetObservedState().ConvergedRevision = 0
		latestReady = proto.Clone(frame).(*pb.WorkerFrame)
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil && latestReady != nil {
			return true, send(proto.Clone(latestReady).(*pb.WorkerFrame))
		}
		return false, nil
	}
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return nil, err
		}
		body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: identity, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
			ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME},
		})
		return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
			WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-" + offer.RequestId,
			OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "job-unknown-selection", rentalWiring(connection, private), func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "one existing rental", nil }
		options.AcquireManagedRental = func(req records.Request) (orchestrator.RentalDecision, string, *exit.Error) {
			fleetSelections.Add(1)
			_, problem := options.Store.PinRental(req.ID, podRental)
			return orchestrator.RentalDecision{RentalID: podRental}, "", problem
		}
	})
	first, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "unknown-selection", Package: pkg,
		Release: "1.0.7", Entrypoint: "four-lane", PlanID: plan, Kind: "job", Org: "paul",
		Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "job with unknown model selection closes", func() bool {
		rows, p := o.store.Attempts(first)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	pod.mu.Lock()
	priorPrepares := len(pod.prepares)
	pod.mu.Unlock()
	second, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "checkpoint-after-unknown", Package: pkg,
		Release: "1.0.7", Entrypoint: "four-lane", PlanID: plan, Kind: "job", Org: "paul",
		Payload: []byte("{}"), Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{{Package: pkg, Slot: "source", BindingPath: "four-lane.models.source",
			Model: "proof/model", Release: "1.0.0", Lane: "native", Manifest: checkpoint, ManifestLength: 164, Bytes: 4096}},
	})
	fatal(t, problem)
	waitUntil(t, "unknown selection cannot satisfy a checkpoint input", func() bool {
		select {
		case <-entered:
			return true
		default:
			return false
		}
	})
	attempts, problem := o.store.Attempts(second)
	fatal(t, problem)
	if len(attempts) != 0 || prepared.Load() {
		t.Fatal("checkpoint job crossed held source preparation")
	}
	releaseOnce.Do(func() { close(release) })
	waitUntil(t, "checkpoint job closes after its own preparation", func() bool {
		rows, p := o.store.Attempts(second)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	if !prepared.Load() || fleetSelections.Load() != 1 {
		t.Fatal("checkpoint did not use exactly one existing-rental preparation")
	}
	pod.mu.Lock()
	prepares, offers := len(pod.prepares), append([]*pb.AttemptOffer(nil), pod.offers...)
	pod.mu.Unlock()
	if prepares != priorPrepares+1 || len(offers) != 2 {
		t.Fatalf("prior prepares=%d final prepares=%d offers=%d", priorPrepares, prepares, len(offers))
	}
	spec, err := canonical.Read(offers[1].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	matched := false
	for _, input := range spec.List("inputs") {
		if input.Str("input_id") == "model:source" {
			matched = input.Str("digest") == checkpoint
		}
	}
	if !matched || offers[1].AttemptOrdinal != 1 {
		t.Fatal("second request lost its exact checkpoint or reused the old attempt")
	}
}
