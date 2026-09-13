package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestFailedRentalPackagePreservesItsRuntimeDiagnosis(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	detail := "ordinal 0 has 84460830720 B measured headroom for 103012383680 B of declared simultaneous weights"
	pod := &fakePod{controlKey: public}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		desired := frame.GetDesiredState()
		if desired == nil || desired.GetPlacementSet() == nil {
			return false, nil
		}
		answer := pod.served(desired, 1)
		observed := answer.GetObservedState()
		observed.ConvergedRevision = 0
		observed.AvailableAttemptSlots = 0
		placement := observed.Placements[0]
		placement.Materialization = pb.MaterializationState_MATERIALIZATION_STATE_FAILED
		placement.Serving = pb.ServingState_SERVING_STATE_OFFLINE
		placement.DispatchableBindingDigests = nil
		placement.Faults = []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_CONFIG_REFUSED,
			Subject: placement.PlacementId, Reason: "device_group_infeasible", Detail: detail}}
		// An unrelated global diagnosis must not replace this placement's failure.
		observed.Faults = []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_CONFIG_REFUSED,
			Subject: "other-placement", Reason: "unrelated", Detail: "other request's failure"}}
		return true, send(answer)
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "failed-package-diagnosis", rentalWiring(connection, private))
	id, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "failed-package-diagnosis", Package: "acme/weightless", Entrypoint: "tile",
		PlanID: podPlanID("acme/weightless"), Release: "1.0.0", Payload: []byte(`{"size":16}`),
		Outputs: []string{"image"}, Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	failed := awaitDurable(t, o, id, "request.failed")
	message, _ := failed.Payload["error"].(string)
	if failed.Payload["error_type"] != "worker.placement_refused" ||
		!strings.Contains(message, "device_group_infeasible: "+detail) || strings.Contains(message, "unrelated") {
		t.Fatalf("Runtime diagnosis was lost or came from another placement: %+v", failed.Payload)
	}
}

func TestInitialPackageRefusalNeedsNoOwnedPlacementSet(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "initial-package-refusal", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection, ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			facts := testPrepareFacts(ref.Package, ref.Release)
			facts.ImageInventory = nil // actual host REFUSED event before any set exists
			return facts, nil
		}
	})
	id, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "initial-package-refusal", Package: "acme/weightless", Entrypoint: "tile",
		PlanID: podPlanID("acme/weightless"), Release: "1.0.0", Payload: []byte(`{"size":16}`),
		Outputs: []string{"image"}, Worker: podRental, Rental: true, RentalRequired: true,
	})
	fatal(t, problem)
	waitUntil(t, "initial preparation refusal settles request", func() bool { row, _ := o.store.RequestRow(id); return row != nil && row.State == "failed" })
	if _, ok := waitEvent(o, "package_prepare_image_inventory_missing", time.Second); !ok {
		t.Fatal("initial host refusal lost its typed diagnosis")
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.desired) != 0 || len(pod.preparedSet) != 0 || len(pod.offers) != 0 {
		t.Fatal("the failed initial package unexpectedly owned a set or attempt")
	}
}

// H3's corrected placement was ACTIVATING while Runtime repeated a historical
// binding fault. Reports are independent wire messages, not successful model
// byte verdicts; the real owner must preserve the queued request while preparing.
func TestHistoricalFaultCannotRefuseCurrentPlacementActivation(t *testing.T) {
	proveCurrentPlacementFault(t, false)
}

func TestPendingReplacementFaultNamesExactIncomingPlacement(t *testing.T) {
	proveCurrentPlacementFault(t, true)
}

func proveCurrentPlacementFault(t *testing.T, pending bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	type observedPeer struct {
		desired *pb.DesiredWorkerState
		send    func(*pb.WorkerFrame) error
	}
	desires := make(chan observedPeer, 1)
	pod := &fakePod{controlKey: public, latch: &pb.Fault{
		Kind:   pb.FaultKind_FAULT_KIND_BINDING_UNAVAILABLE,
		Reason: "unhandled_exception", Detail: "TraversalOrderMismatch: old header audio_proj_in.bias",
	}}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if desired := frame.GetDesiredState(); desired != nil && desired.GetPlacementSet() != nil {
			desires <- observedPeer{desired, send}
			return true, nil
		}
		return false, nil
	}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "current-placement-fault", rentalWiring(connection, private), func(opt *orchestrator.Options) { opt.Packages = localLauncher{revision: revision} })
	id := submitPrivateRental(t, o, revision, "current-placement-fault")
	var peer observedPeer
	select {
	case peer = <-desires:
	case <-time.After(5 * time.Second):
		t.Fatal("no desired placement")
	}
	instance, _, _, problem := o.c.EnsureRental(podRental)
	fatal(t, problem)
	sequence := uint64(100)
	report := func(change func(*pb.ObservedWorkerState), count int) {
		t.Helper()
		for i := 0; i < count; i++ {
			frame := pod.report(peer.desired, 1)
			r := frame.GetObservedState()
			r.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_STAGED
			r.Placements[0].ExecutorEpoch = 2
			change(r)
			sequence++
			r.AdmissionEpoch = sequence
			must(t, peer.send(frame))
			waitUntil(t, "owner consumed exact fault report", func() bool { w := o.c.Worker(instance); return w != nil && w.AdmissionEpoch == sequence })
		}
	}
	queued := func(label string) {
		t.Helper()
		time.Sleep(40 * time.Millisecond) // let the actual readiness waiter consume its verdict
		row, problem := o.store.RequestRow(id)
		fatal(t, problem)
		if row.State != "submitted" {
			t.Fatalf("%s failed current preparation: %s", label, row.State)
		}
		attempts, problem := o.store.Attempts(id)
		fatal(t, problem)
		if len(attempts) != 0 {
			t.Fatal("preparation fault invented an attempt")
		}
	}
	for _, state := range []pb.ServingState{pb.ServingState_SERVING_STATE_ACTIVATING, pb.ServingState_SERVING_STATE_DRAINING} {
		report(func(r *pb.ObservedWorkerState) { r.Placements[0].Serving = state }, orchestrator.StillFactor+2)
		queued(fmt.Sprint(state))
	}
	report(func(r *pb.ObservedWorkerState) {
		r.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_MATERIALIZING
	}, orchestrator.StillFactor+2)
	queued("materializing")
	report(func(r *pb.ObservedWorkerState) { r.AcceptedPlacementSetDigest = bytes.Repeat([]byte{0x71}, 32) }, orchestrator.StillFactor+2)
	queued("old accepted set")
	report(func(r *pb.ObservedWorkerState) { r.Placements[0].PlacementSetDigest = bytes.Repeat([]byte{0x72}, 32) }, orchestrator.StillFactor+2)
	queued("old placement set")
	report(func(r *pb.ObservedWorkerState) {
		r.Placements[0].PlacementSetDigest = bytes.Repeat([]byte{0x72}, 32)
		r.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_FAILED
	}, orchestrator.StillFactor+2)
	queued("failed axis from old placement set")
	report(func(r *pb.ObservedWorkerState) {
		r.AcceptedDesiredStateRevision--
		r.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_FAILED
	}, orchestrator.StillFactor+2)
	queued("failed axis from old desired revision")
	report(func(r *pb.ObservedWorkerState) {
		r.Placements[0].PlacementId = "old-placement"
		r.Faults[0].Subject = "old-placement"
	}, orchestrator.StillFactor+2)
	queued("old placement")
	report(func(r *pb.ObservedWorkerState) { r.Placements = nil; r.Faults[0].Subject = "historical-binding" }, orchestrator.StillFactor+2)
	queued("unassociated global fault")
	if pending {
		report(func(r *pb.ObservedWorkerState) {
			r.Placements[0].PlacementId = "outgoing-fallback"
			r.Placements[0].PlacementSetDigest = bytes.Repeat([]byte{0x73}, 32)
			r.Placements[0].Serving = pb.ServingState_SERVING_STATE_DISPATCHABLE
			// The global fault still names the exact current incoming ID.
		}, orchestrator.StillFactor)
	} else {
		// A real current OFFLINE failure still settles, but a replacement executor
		// cannot inherit the predecessor's partially accumulated report count.
		report(func(*pb.ObservedWorkerState) {}, orchestrator.StillFactor-2)
		queued("six current executor reports")
		report(func(r *pb.ObservedWorkerState) { r.Placements[0].ExecutorEpoch = 3 }, orchestrator.StillFactor-2)
		queued("new executor restarted the count")
		report(func(r *pb.ObservedWorkerState) { r.Placements[0].ExecutorEpoch = 3 }, 2)
	}
	waitUntil(t, "current offline failure settles", func() bool { row, _ := o.store.RequestRow(id); return row != nil && row.State == "failed" })
}
