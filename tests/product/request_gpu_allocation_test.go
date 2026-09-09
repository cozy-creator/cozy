package producttest

import (
	"fmt"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// This independent TLS peer enforces the physical contract rather than trusting
// Creator's lane arithmetic: changed busy pins or any overlap refuse the stream.
type gpuPeer struct {
	mu            sync.Mutex
	pod           *fakePod
	desired       *pb.DesiredWorkerState
	pins          map[string][]uint32
	active        map[string]*pb.AttemptOffer
	started       map[string][]uint32
	send          func(*pb.WorkerFrame) error
	coldFirst     bool
	snapshotFault string
}

func newGPUPeer(t *testing.T, width int) *gpuPeer {
	prepared := modelBearingPlacement(t)
	p := &gpuPeer{pod: &fakePod{deviceCount: uint32(width), preparedPlacement: func(raw []byte, pkg, release string) *pb.Placement {
		row := prepared(raw, pkg, release)
		_, row.EnvironmentDigest, _ = canonical.Identity(row.Environment)
		sealPodBindings(row)
		return row
	}},
		pins: map[string][]uint32{}, active: map[string]*pb.AttemptOffer{}, started: map[string][]uint32{}}
	p.pod.onFrame = p.frame
	return p
}

func (p *gpuPeer) frame(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.send = send
	if claim := frame.GetClaim(); claim != nil && p.desired != nil {
		if err := p.pod.verifyClaim(claim, true); err != nil {
			return true, err
		}
		p.desired.RecordOwnerEpoch = claim.RecordOwnerEpoch
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
			RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: podBootID,
			Accepted: true, WireMinor: pb.WireMinor, WorkerId: podWorkerID, WorkerInstanceId: "inst-pod-1",
			Resources: &pb.WorkerResources{Backend: "cuda", DeviceName: "fake-4090", DeviceCount: p.pod.deviceCount, DeviceMemoryTotalBytes: 24 << 30}}}}); err != nil {
			return true, err
		}
		observed := p.observed().GetObservedState()
		snapshot := &pb.WorkerSnapshotBody{
			WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AdmissionEpoch: 7,
			AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED, AvailableAttemptSlots: observed.AvailableAttemptSlots,
			AcceptedDesiredStateRevision: p.desired.Revision, ConvergedRevision: p.desired.Revision,
			AcceptedPlacementSetDigest: p.desired.GetPlacementSet().PlacementSetDigest,
			Placements:                 observed.Placements, Lanes: observed.Lanes, HeldAttempts: observed.HeldAttempts}
		if p.snapshotFault == "unpaired" {
			snapshot.AcceptedPlacementSetDigest = nil
		}
		body, digest, err := canonical.Identity(snapshot)
		if err != nil {
			return true, err
		}
		return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{
			RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: 1, WorkerBootId: podBootID,
			SnapshotId: "recovered-gpus", SnapshotDigest: digest, SnapshotCanonicalBytes: body,
			AcceptedPlacementSetCanonicalBytes: p.desired.GetPlacementSet().PlacementSetCanonicalBytes}}})
	}
	if frame.GetSnapshotAck() != nil && p.desired != nil {
		return true, send(p.observed())
	}
	if desired := frame.GetDesiredState(); desired != nil && desired.GetPlacementSet() != nil {
		pins := map[string][]uint32{}
		occupied := map[uint32]bool{}
		for _, pin := range desired.GetPlacementSet().DevicePins {
			for _, ordinal := range pin.DeviceOrdinals {
				if occupied[ordinal] || ordinal >= p.pod.deviceCount {
					return true, fmt.Errorf("overlapping or invalid physical ordinal %d", ordinal)
				}
				occupied[ordinal] = true
			}
			pins[pin.PlacementId] = append([]uint32(nil), pin.DeviceOrdinals...)
		}
		for _, offer := range p.active {
			if !reflect.DeepEqual(p.pins[offer.PlacementId], pins[offer.PlacementId]) {
				return true, fmt.Errorf("active placement %s was removed or repinned", offer.PlacementId)
			}
		}
		p.pins, p.desired = pins, desired
		return true, send(p.observed())
	}
	if offer := frame.GetAttemptOffer(); offer != nil {
		pins := p.pins[offer.PlacementId]
		if len(pins) == 0 {
			return true, fmt.Errorf("offer has no physical placement")
		}
		for _, held := range p.active {
			for _, a := range p.pins[held.PlacementId] {
				for _, b := range pins {
					if a == b {
						return true, fmt.Errorf("offer overlapped active GPU %d", a)
					}
				}
			}
		}
		p.active[offer.RequestId], p.started[offer.RequestId] = offer, pins
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
			RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest,
			PlacementId: offer.PlacementId, LaneId: "lane-" + offer.PlacementId}}}); err != nil {
			return true, err
		}
		return true, send(p.observed())
	}
	if ack := frame.GetOutcomeAck(); ack != nil {
		delete(p.active, ack.RequestId)
		return true, send(p.observed())
	}
	return false, nil
}

func TestGPUAllocationRecoversActivePinsFromWorkerSnapshot(t *testing.T) {
	for _, fault := range []string{"valid", "empty", "outside", "unpaired"} {
		t.Run(fault, func(t *testing.T) { proveGPUAllocationRecovery(t, fault) })
	}
}

func proveGPUAllocationRecovery(t *testing.T, fault string) {
	peer := newGPUPeer(t, 2)
	connection, cert := startFakePod(t, t.TempDir(), peer.pod)
	wiring := rentalWidthWiring(t, peer.pod, connection, cert, 2)
	o := hostOwner(t, "gpu-allocation-recovery", wiring)
	submission := orchestrator.Submission{IdemKey: "first", Package: "cozy/h3-package", Entrypoint: "tile",
		Release: "1.1.2", Payload: []byte(`{}`), Worker: podRental, RequestedRental: podRental,
		Rental: true, RentalRequired: true, NeedsAccelerator: true, RequestedGPUs: 1,
		Models: []records.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model", Model: "source/h3",
			Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}}
	first, _, problem := o.c.Submit(submission)
	fatal(t, problem)
	waitUntil(t, "first GPU attempt active", func() bool {
		peer.mu.Lock()
		defer peer.mu.Unlock()
		return peer.active[first] != nil
	})
	o.c.Close(0)
	peer.mu.Lock()
	peer.snapshotFault = fault
	peer.mu.Unlock()
	options := orchestrator.Options{Cfg: o.cfg, Layout: o.l, Store: o.store, Yield: "smart", MaxOutputMiB: 8}
	wiring(&options)
	next, problem := orchestrator.Open(options)
	fatal(t, problem)
	go func() { _ = next.Serve() }()
	defer next.Close(0)
	submission.IdemKey = "second"
	second, _, problem := next.Submit(submission)
	fatal(t, problem)
	if fault != "valid" {
		waitUntil(t, "invalid recovery occupancy refuses allocation", func() bool {
			for _, line := range next.Events() {
				if strings.Contains(line, "valid physical GPU occupancy") || strings.Contains(line, "NOT acknowledged") {
					return true
				}
			}
			return false
		})
		peer.mu.Lock()
		if peer.started[second] != nil {
			t.Error("invalid snapshot allocated another request")
		}
		if fault == "unpaired" {
			peer.mu.Unlock()
			return
		}
		peer.snapshotFault = "valid"
		must(t, peer.send(peer.observed()))
		peer.mu.Unlock()
	}
	waitUntil(t, "second request uses free GPU after recovery", func() bool {
		peer.mu.Lock()
		defer peer.mu.Unlock()
		return len(peer.active) == 2
	})
	peer.mu.Lock()
	if reflect.DeepEqual(peer.started[first], peer.started[second]) {
		t.Fatal("recovered request lost its exclusive GPU")
	}
	peer.mu.Unlock()
	peer.pod.mu.Lock()
	if len(peer.pod.prepares) != 1 {
		t.Errorf("snapshot recovery re-prepared unchanged package %d times", len(peer.pod.prepares))
	}
	peer.pod.mu.Unlock()
	peer.finish(t, first)
	peer.finish(t, second)
}

func (p *gpuPeer) observed() *pb.WorkerFrame {
	frame := p.pod.served(p.desired, 1)
	state := frame.GetObservedState()
	state.AvailableAttemptSlots = 0
	for _, placement := range state.Placements {
		id := placement.PlacementId
		placement.DeviceLaneId = "lane-" + id
		if p.coldFirst && len(p.pins[id]) > 0 && p.pins[id][0] == 0 {
			placement.Serving = pb.ServingState_SERVING_STATE_OFFLINE
			placement.MaterializableBindingDigests = placement.DispatchableBindingDigests
			placement.DispatchableBindingDigests = nil
		}
		lane := &pb.DeviceLane{LaneId: placement.DeviceLaneId, DeviceOrdinals: p.pins[id],
			PlacementIds: []string{id}, ResidentPlacementIds: []string{id}, AvailableAttemptSlots: 1}
		for _, offer := range p.active {
			if offer.PlacementId == id {
				lane.AvailableAttemptSlots = 0
				if p.snapshotFault == "empty" {
					lane.DeviceOrdinals = nil
				}
				if p.snapshotFault == "outside" {
					lane.DeviceOrdinals = []uint32{p.pod.deviceCount}
				}
				state.HeldAttempts = append(state.HeldAttempts, &pb.HeldAttempt{RequestId: offer.RequestId,
					AttemptOrdinal: offer.AttemptOrdinal, PlacementId: id, LaneId: lane.LaneId,
					InvocationSpecDigest: offer.InvocationSpecDigest, Kind: pb.AttemptKind_ATTEMPT_KIND_SERVING,
					State: pb.AttemptState_ATTEMPT_STATE_RUNNING, ExecutorEpoch: 1})
			}
		}
		state.AvailableAttemptSlots += lane.AvailableAttemptSlots
		state.Lanes = append(state.Lanes, lane)
	}
	return frame
}

func TestWarmingGPUDoesNotBlockAnIndependentReadyGPU(t *testing.T) {
	peer := newGPUPeer(t, 2)
	peer.coldFirst = true
	connection, cert := startFakePod(t, t.TempDir(), peer.pod)
	o := hostOwner(t, "gpu-independent-warmup", rentalWidthWiring(t, peer.pod, connection, cert, 2))
	sub := orchestrator.Submission{IdemKey: "cold", Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2",
		Payload: []byte(`{}`), Worker: podRental, RequestedRental: podRental, Rental: true, RentalRequired: true, NeedsAccelerator: true,
		Models: []records.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model", Model: "source/h3",
			Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}}
	first, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	sub.IdemKey = "ready"
	second, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	waitUntil(t, "independent ready GPU runs while first warms", func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return peer.active[second] != nil })
	peer.mu.Lock()
	if peer.active[first] != nil {
		t.Error("cold placement was dispatched before it became ready")
	}
	peer.coldFirst = false
	must(t, peer.send(peer.observed()))
	peer.mu.Unlock()
	waitUntil(t, "first GPU finishes warming", func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return len(peer.active) == 2 })
	peer.finish(t, first)
	peer.finish(t, second)
}

func (p *gpuPeer) finish(t *testing.T, id string) {
	t.Helper()
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.active[id] == nil {
		t.Fatalf("request %s is not active", id)
	}
	must(t, p.send(privateAttemptOutcome(p.active[id], pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME)))
}

func TestRentalAllocatesDisjointRequestGPUs(t *testing.T) {
	for _, widths := range [][]int{{0, 0}, {1, 1}, {4}, {2, 2}, {2, 1, 1}} {
		t.Run(fmt.Sprint(widths), func(t *testing.T) {
			width := 0
			for _, n := range widths {
				width += max(1, n)
			}
			peer := newGPUPeer(t, width)
			connection, cert := startFakePod(t, t.TempDir(), peer.pod)
			o := hostOwner(t, fmt.Sprint("request-gpus-", widths), rentalWidthWiring(t, peer.pod, connection, cert, width))
			var ids []string
			for index, n := range widths {
				id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: fmt.Sprint(index),
					Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2", Payload: []byte(`{}`),
					Worker: podRental, RequestedRental: podRental, Rental: true, RentalRequired: true, NeedsAccelerator: true,
					RequestedGPUs: n, Models: []records.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model",
						Model: "source/h3", Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}})
				fatal(t, problem)
				ids = append(ids, id)
			}
			deadline := time.Now().Add(10 * time.Second)
			for {
				peer.mu.Lock()
				active := len(peer.active)
				peer.mu.Unlock()
				if active == len(widths) {
					break
				}
				if time.Now().After(deadline) {
					t.Fatalf("only %d/%d simultaneous attempts: %v", active, len(widths), o.c.Events())
				}
				time.Sleep(20 * time.Millisecond)
			}
			peer.mu.Lock()
			for i, id := range ids {
				if len(peer.started[id]) != max(1, widths[i]) {
					t.Errorf("request %s got %v, wanted %d GPUs", id, peer.started[id], widths[i])
				}
			}
			peer.mu.Unlock()
			for _, id := range ids {
				peer.finish(t, id)
			}
		})
	}
}

func TestEarlierWideRequestReservesFreedGPUsWithoutPreemption(t *testing.T) {
	peer := newGPUPeer(t, 2)
	connection, cert := startFakePod(t, t.TempDir(), peer.pod)
	o := hostOwner(t, "request-gpu-fairness", rentalWidthWiring(t, peer.pod, connection, cert, 2))
	submit := func(key string, width int) string {
		t.Helper()
		id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: key, Package: "cozy/h3-package",
			Entrypoint: "tile", Release: "1.1.2", Payload: []byte(`{}`), Worker: podRental, RequestedRental: podRental,
			Rental: true, RentalRequired: true, NeedsAccelerator: true, RequestedGPUs: width,
			Models: []records.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model", Model: "source/h3",
				Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}})
		fatal(t, problem)
		return id
	}
	started := func(id string) bool {
		peer.mu.Lock()
		defer peer.mu.Unlock()
		return peer.started[id] != nil
	}
	first := submit("first", 1)
	waitUntil(t, "first singleton active", func() bool { return started(first) })
	wide := submit("wide", 2)
	late := submit("later-singleton", 1)
	waitUntil(t, "wide waits for complete group", func() bool {
		for _, line := range o.c.Events() {
			if strings.Contains(line, wide) && strings.Contains(line, "waiting for 2 GPUs to be free together") {
				return true
			}
		}
		return false
	})
	if started(wide) || started(late) {
		t.Fatal("a queued gang overlapped active work or was overtaken by a later singleton")
	}
	peer.finish(t, first)
	waitUntil(t, "wide gets both GPUs", func() bool { return started(wide) })
	if started(late) {
		t.Fatal("later singleton ran ahead of the waiting gang")
	}
	peer.mu.Lock()
	if len(peer.active) != 1 || len(peer.started[wide]) != 2 {
		t.Errorf("wide group is not exclusive: active=%d pins=%v", len(peer.active), peer.started[wide])
	}
	peer.mu.Unlock()
	peer.finish(t, wide)
	waitUntil(t, "group splits for later singleton", func() bool { return started(late) })
	peer.finish(t, late)
}

func TestAutomaticGPURequestUsesSpareCardWithoutAnotherPurchase(t *testing.T) {
	peer := newGPUPeer(t, 2)
	connection, cert := startFakePod(t, t.TempDir(), peer.pod)
	o := hostOwner(t, "gpu-spare-card", rentalWidthWiring(t, peer.pod, connection, cert, 2), func(opt *orchestrator.Options) {
		opt.RentalFleet = func() (string, *exit.Error) { return "one existing two-GPU rental", nil }
		opt.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			t.Error("spare existing GPU triggered acquisition")
			return orchestrator.PlacementDecision{}, "", exit.Internalf("unexpected acquisition")
		}
	})
	sub := orchestrator.Submission{IdemKey: "first", Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2",
		Payload: []byte(`{}`), Worker: podRental, RequestedRental: podRental, Rental: true, RentalRequired: true, NeedsAccelerator: true,
		Models: []records.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model", Model: "source/h3",
			Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}}
	first, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	waitUntil(t, "existing GPU is busy", func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return len(peer.active) == 1 })
	sub.IdemKey, sub.Worker, sub.RequestedRental = "automatic", "", ""
	second, _, problem := o.c.Submit(sub)
	fatal(t, problem)
	waitUntil(t, "automatic request uses spare GPU", func() bool { peer.mu.Lock(); defer peer.mu.Unlock(); return len(peer.active) == 2 })
	row, problem := o.store.RequestRow(second)
	fatal(t, problem)
	if row.Worker != podRental || row.PlanID == "" {
		t.Fatalf("automatic allocation did not freeze its selected rental and binding: %+v", row)
	}
	peer.finish(t, first)
	peer.finish(t, second)
}
