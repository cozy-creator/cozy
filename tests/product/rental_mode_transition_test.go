package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
	"strings"
	"testing"
)

func TestIdleRentalJobCanBecomeServing(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	var latestReady *pb.WorkerFrame
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	checkpoint := "sha256:" + strings.Repeat("b", 64)
	pod.preparedPlacement = func(raw []byte, name, release string) *pb.Placement {
		if name == "acme/tile" {
			doc, err := canonical.Read(raw, &pb.DownloadDelegation{})
			must(t, err)
			models := doc.List("models")
			if len(models) != 1 || models[0].Str("manifest") != checkpoint {
				t.Error("serving preparation lost the requested checkpoint")
			}
		}
		return podPlacement(raw, name, release, name)
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
	o := hostOwner(t, "idle-job-serving", rentalWiring(connection, private))
	first, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "finished-job", Package: "cozy/h3-package",
		Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul",
		Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "job closes", func() bool {
		rows, p := o.store.Attempts(first)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	waitUntil(t, "settled rental becomes serving capacity", func() bool { reason, held := o.c.RentalStanding(podRental, false); return reason == "" && held == 0 })
	second, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "serving-after-job", Package: "acme/tile", Release: "1.0.0",
		Entrypoint: "tile", PlanID: podPlanID("acme/tile"), Org: "paul", Payload: []byte(`{"size":16}`), Worker: podRental,
		Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{{Package: "acme/tile", Slot: "model", BindingPath: "tile.models.model",
			Model: "proof/model", Release: "1.0.0", Lane: "native", Manifest: checkpoint, ManifestLength: 164, Bytes: 4096}},
	})
	fatal(t, problem)
	waitUntil(t, "serving dispatch after completed job", func() bool {
		rows, p := o.store.Attempts(second)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	waitUntil(t, "idle serving becomes job capacity", func() bool { reason, held := o.c.RentalStanding(podRental, true); return reason == "" && held == 0 })
	pod.mu.Lock()
	preparedBeforeNextJob := len(pod.prepares)
	pod.mu.Unlock()
	if preparedBeforeNextJob != 2 {
		t.Fatalf("prepared old producer again during serving transition: %d prepares", preparedBeforeNextJob)
	}
	third, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "job-after-serving", Package: "cozy/h3-package",
		Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul",
		Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "job dispatch after completed serving", func() bool {
		rows, p := o.store.Attempts(third)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 3 || pod.offers[0].WorkerBootId != pod.offers[1].WorkerBootId || pod.offers[0].WorkerBootId != pod.offers[2].WorkerBootId {
		t.Fatal("reused rental changed worker or duplicated offer")
	}
	spec, err := canonical.Read(pod.offers[1].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	if _, ok := spec["serving"]; !ok {
		t.Fatal("second invocation remained a job")
	}
	spec, err = canonical.Read(pod.offers[2].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	if _, ok := spec["job"]; !ok {
		t.Fatal("third invocation remained serving")
	}
}

func TestActiveRentalJobCannotBeReplacedByServing(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "active-job-serving", rentalWiring(connection, private))
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "active-job", Package: "cozy/h3-package", Release: "1.0.7",
		Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul", Payload: []byte("{}"),
		Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "job offer crosses", func() bool { pod.mu.Lock(); defer pod.mu.Unlock(); return len(pod.offers) == 1 })
	reason, _ := o.c.RentalStanding(podRental, false)
	if reason != orchestrator.ExcludedModeConflict {
		t.Fatalf("offered job was replaceable: %s", reason)
	}
	rows, problem := o.store.Attempts(id)
	fatal(t, problem)
	if len(rows) != 1 || rows[0].State == "closed" {
		t.Fatal("active job disappeared")
	}
	instance := rows[0].InstanceID
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "acme/tile", Release: "1.0.0"}}, nil))
	problem = o.c.EnsurePlacementReady(instance, "sha256:"+strings.Repeat("35", 32), "")
	if problem == nil || !strings.Contains(problem.Message, "finishing its current job") {
		t.Fatalf("direct convergence replaced an offered job: %v", problem)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.desired) == 0 || pod.desired[len(pod.desired)-1].GetJob() == nil {
		t.Fatal("serving desired state replaced the active job")
	}
}
