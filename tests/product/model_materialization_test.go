package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// A machine that finds a selected model's bytes gone refuses the job before admission
// (model_materialization_required). That typed refusal ends the run at once with its reason
// and one attempt; the next run of the same model prepares it again instead of borrowing
// the revoked readiness.
func TestJobModelCacheMissFailsAndTheNextRunReprepares(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	const pkg = "cozy/h3-package"
	const plan = "sha256:3535353535353535353535353535353535353535353535353535353535353535"
	const refusal = "model_materialization_required: cache was evicted"
	manifest := childDigest("8")
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	var prepares, offers atomic.Int64
	var latest *pb.WorkerFrame
	pod.preparedPlacement = func(raw []byte, name, version string) *pb.Placement {
		prepares.Add(1)
		doc, err := canonical.Read(raw, &pb.DownloadDelegation{})
		must(t, err)
		if len(doc.List("models")) != 1 || doc.List("models")[0].Str("manifest") != manifest {
			t.Error("preparation changed exact model selection")
		}
		return podPlacement(raw, name, version, "h3-package")
	}
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		frame.GetObservedState().ConvergedRevision = 0
		latest = proto.Clone(frame).(*pb.WorkerFrame)
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil && latest != nil {
			return true, send(proto.Clone(latest).(*pb.WorkerFrame))
		}
		return false, nil
	}
	firstPreparation := int64(0)
	pod.answerOffer = func(offer *pb.AttemptOffer) (*pb.AttemptOutcome, error) {
		n := offers.Add(1)
		if n == 1 {
			firstPreparation = prepares.Load()
		} else if prepares.Load() <= firstPreparation {
			t.Error("the next run reused the revoked preparation instead of ensuring model bytes")
		}
		identity, err := canonical.Spell(offer.InvocationSpecDigest)
		if err != nil {
			return nil, err
		}
		result := &pb.AttemptOutcomeBody{RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: identity,
			Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME}}
		if n == 1 {
			result.Status, result.ExecutionStarted, result.SafeMessage = pb.OutcomeStatus_OUTCOME_STATUS_REFUSED, false, refusal
			result.Cause = &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_PLACEMENT_NOT_DISPATCHABLE, Origin: pb.CauseOrigin_CAUSE_ORIGIN_WORKER}
		}
		body, digest, err := canonical.Identity(result)
		return &pb.AttemptOutcome{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
			InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: fmt.Sprintf("model-miss-%d", n), OutcomeDigest: digest, OutcomeCanonicalBytes: body}, err
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "job-model-reensure", rentalWiring(connection, private))
	submit := func(key string) string {
		id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: key, Package: pkg, Release: "1.0.7", Entrypoint: "four-lane", PlanID: plan, Kind: "job", Org: "paul", Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
			Models: []orchestrator.ModelRef{{Package: pkg, Slot: "source", BindingPath: "four-lane.models.source", Model: "proof/model", Release: "1.0.0", Lane: "native", Manifest: manifest, ManifestLength: 164, Bytes: 4096}}})
		fatal(t, problem)
		return id
	}
	settles := func(id, state string) {
		waitUntil(t, id+" settles "+state, func() bool {
			row, e := o.store.RequestRow(id)
			fatal(t, e)
			return row != nil && row.State == state
		})
	}
	first := submit("job-model-miss")
	settles(first, "refused")
	rows, problem := o.store.Attempts(first)
	fatal(t, problem)
	_, _, message, problem := o.store.SettledFailure(first)
	fatal(t, problem)
	if len(rows) != 1 || offers.Load() != 1 || message != refusal {
		t.Fatalf("the refused run made %d attempt(s) and %d offer(s), ending %q", len(rows), offers.Load(), message)
	}
	settles(submit("job-model-after-miss"), "succeeded")
}
