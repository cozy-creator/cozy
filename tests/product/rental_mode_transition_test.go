package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
	"strings"
	"sync"
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

// Paul's contract: `cozy run ... rental=ABC` pins the job to ABC. A pinned job that is
// still queued — no attempt, no offer yet — is already a claim on that rental's job mode,
// so serving work must not flip the rental away from under it.
// The pin claims the mode whether the scheduler assigned it (worker) or the caller
// selected the rental with `cozy run --rental` and it is still unassigned.
func TestQueuedPinnedJobKeepsRentalJobMode(t *testing.T) {
	t.Run("assigned", func(t *testing.T) { queuedPinnedJobKeepsRentalJobMode(t, false) })
	t.Run("requested", func(t *testing.T) { queuedPinnedJobKeepsRentalJobMode(t, true) })
}

func queuedPinnedJobKeepsRentalJobMode(t *testing.T, requested bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	release := make(chan struct{})
	var withheld sync.Once
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		ready := proto.Clone(frame).(*pb.WorkerFrame)
		// Prepared but occupied: the job is ready yet cannot be offered.
		frame.GetObservedState().JobCapacity = &pb.JobCapacity{JobsInFlight: 1}
		first := false
		withheld.Do(func() { first = true })
		if !first {
			return send(ready)
		}
		go func() { <-release; _ = send(ready) }()
		return send(frame)
	}
	// A settled job reports its free seat again, as the pod's runtime does.
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() == nil {
			return false, nil
		}
		pod.mu.Lock()
		last := pod.desired[len(pod.desired)-1]
		pod.mu.Unlock()
		return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: &pb.ObservedWorkerState{
			RecordOwnerEpoch: last.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: podBootID,
			AcceptedDesiredStateRevision: last.Revision, ConvergedRevision: last.Revision,
			WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AppliedWireMinor: pb.WireMinor,
			AdmissionState: pb.AdmissionState_ADMISSION_STATE_OPEN, AdmissionEpoch: 7,
			AvailableAttemptSlots: 1, JobCapacity: &pb.JobCapacity{JobsAvailable: 1},
		}}})
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
	// The managed fleet is present, as in the daemon; the selected rental already holds
	// the job's placement, so it is never asked to buy or pin.
	fleet := func(options *orchestrator.Options) {
		options.RentalFleet = func() (string, *exit.Error) { return "", nil }
		options.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			return orchestrator.PlacementDecision{}, "", exit.Internalf("fleet asked to place work its rental already holds")
		}
	}
	o := hostOwner(t, "queued-pinned-job", rentalWiring(connection, private), fleet)
	job, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "pinned-job", Package: "cozy/h3-package",
		Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul",
		Payload: []byte("{}"), Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	sum := sha256.Sum256([]byte("rental/" + podRental))
	instance := "ins-" + hex.EncodeToString(sum[:12])
	waitUntil(t, "owner observes the prepared job without a free seat", func() bool {
		pod.mu.Lock()
		var revision uint64
		if n := len(pod.desired); n > 0 && pod.desired[n-1].GetJob() != nil {
			revision = pod.desired[n-1].Revision
		}
		pod.mu.Unlock()
		facts := o.c.Worker(instance)
		return revision != 0 && facts != nil && facts.ConvergedRevision == revision
	})
	rows, problem := o.store.Attempts(job)
	fatal(t, problem)
	if len(rows) != 0 {
		t.Fatalf("pinned job left the queue before its capacity: %d attempts", len(rows))
	}
	if requested {
		// `cozy run --rental`: the rental already holds the job's placement, so the run
		// waits for a seat with only requested_rental set, as live. The assigned job then
		// leaves, and the unassigned one is the rental's only job-mode work.
		selected, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "requested-job", Package: "cozy/h3-package",
			Release: "1.0.7", Entrypoint: "four-lane", PlanID: "sha256:" + strings.Repeat("35", 32), Kind: "job", Org: "paul",
			Payload: []byte("{}"), RequestedRental: podRental, Rental: true, RentalRequired: true,
		})
		fatal(t, problem)
		fatal(t, o.c.CancelQueued(job, "test client"))
		job = selected
		row, problem := o.store.RequestRow(job)
		fatal(t, problem)
		if row == nil || row.Worker != "" || row.RequestedRental != podRental || (row.State != "submitted" && row.State != "queued") {
			t.Fatalf("requested job is not queued unassigned: %+v", row)
		}
	}
	if reason, _ := o.c.RentalStanding(podRental, false); reason != orchestrator.ExcludedModeConflict {
		t.Fatalf("queued pinned job did not claim its rental's mode: %q", reason)
	}
	serving, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "serving-behind-job", Package: "acme/tile",
		Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID("acme/tile"), Org: "paul", Payload: []byte(`{"size":16}`),
		Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	close(release)
	waitUntil(t, "pinned job runs on its rental", func() bool {
		rows, p := o.store.Attempts(job)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	waitUntil(t, "serving runs after the job", func() bool {
		rows, p := o.store.Attempts(serving)
		fatal(t, p)
		return len(rows) == 1 && rows[0].State == "closed"
	})
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 2 || pod.offers[0].RequestId != job || pod.offers[1].RequestId != serving {
		t.Fatalf("pinned job did not run first on its rental: %d offers", len(pod.offers))
	}
	spec, err := canonical.Read(pod.offers[0].InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
	must(t, err)
	if _, ok := spec["job"]; !ok {
		t.Fatal("pinned job was offered as serving work")
	}
}
