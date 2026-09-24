package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// Detaching and re-attaching a rental keeps its running attempt: same boot, same
// ordinal, no cancellation, no new offer, and progress resumes on the new watch.
func TestRentalDetachReattachPreservesRunningAttempt(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true}
	const id = "reattach-active-ordinal-one"
	var claims, watches, unsafe atomic.Int64
	var heldMu sync.Mutex
	var held []*pb.HeldAttempt
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetCancelAttempt() != nil {
			unsafe.Add(1)
		}
		if frame.GetDesiredState() != nil && claims.Load() > 1 {
			unsafe.Add(1)
		}
		claim := frame.GetClaim()
		if claim == nil {
			return false, nil
		}
		if err := pod.verifyClaim(claim, true); err != nil {
			return true, err
		}
		epoch := uint64(claims.Add(1))
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: epoch, WorkerBootId: podBootID, Accepted: true, WireMinor: pb.WireMinor, WorkerId: podWorkerID, WorkerInstanceId: "inst-pod-1", Resources: &pb.WorkerResources{Backend: "cuda", DeviceName: "fake-4090", DeviceCount: 1, DeviceMemoryTotalBytes: 24 << 30}}}}); err != nil {
			return true, err
		}
		heldMu.Lock()
		snapshotHeld := append([]*pb.HeldAttempt(nil), held...)
		heldMu.Unlock()
		snapshot := &pb.WorkerSnapshotBody{WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE, AdmissionEpoch: 1, AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED, HeldAttempts: snapshotHeld}
		var accepted []byte
		pod.mu.Lock()
		if len(pod.desired) > 0 {
			d := pod.desired[len(pod.desired)-1]
			snapshot.AcceptedDesiredStateRevision = d.Revision
			snapshot.ConvergedRevision = d.Revision
			snapshot.AcceptedPlacementSetDigest = d.GetPlacementSet().GetPlacementSetDigest()
			accepted = d.GetPlacementSet().GetPlacementSetCanonicalBytes()
		}
		pod.mu.Unlock()
		body, digest, err := canonical.Identity(snapshot)
		if err != nil {
			return true, err
		}
		return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: epoch, WorkerBootId: podBootID, SnapshotId: "reattach-snapshot", SnapshotDigest: digest, SnapshotCanonicalBytes: body, AcceptedPlacementSetCanonicalBytes: accepted}}})
	}
	pod.watchProgress = func(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
		watches.Add(1)
		if open.ControlStreamEpoch == 1 {
			return status.Error(codes.Unavailable, "old watch disconnected while durable control remains healthy")
		}
		data, _ := json.Marshal(map[string]any{"type": "progress", "payload": map[string]any{"stage": "denoise", "position": 2, "total": 30, "step_ms": 2000}})
		if err := stream.Send(&pb.AttemptProgress{RecordOwnerEpoch: open.RecordOwnerEpoch, ControlStreamEpoch: open.ControlStreamEpoch, WorkerBootId: open.WorkerBootId, RequestId: id, AttemptOrdinal: 1, Seq: 2, Data: data}); err != nil {
			return err
		}
		<-stream.Context().Done()
		return nil
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "reattach-held-proof", rentalWiring(connection, private))
	instance, _, _, problem := o.c.EnsureRental(podRental)
	fatal(t, problem)
	fatal(t, o.c.ConvergePackageSet(instance, []*pb.DownloadPackageRef{{Package: "cozy/h3-package", Release: "1.0.7"}}, nil))
	waitUntil(t, "the initial selection to reach the worker", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.desired) == 1
	})
	waitUntil(t, "the first progress watch to fail", func() bool { return watches.Load() >= 1 })
	_, _, problem = o.store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("2"), Package: "cozy/h3-package", Entrypoint: "denoise", Kind: "job", Payload: []byte(`{}`), Worker: podRental, Rental: true})
	fatal(t, problem)
	ordinal, problem := o.store.Dispatch(records.Attempt{RequestID: id, InstanceID: instance, SessionID: podBootID, InvocationDigest: childDigest("1"), InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	if ordinal != 1 {
		t.Fatalf("initial ordinal=%d", ordinal)
	}
	fatal(t, o.store.OfferDispatch(id, 1, podBootID))
	fatal(t, o.store.Accepted(id, 1, podBootID))
	invocation, _ := canonical.Raw(childDigest("1"))
	heldMu.Lock()
	held = []*pb.HeldAttempt{{RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: invocation, State: pb.AttemptState_ATTEMPT_STATE_RUNNING, Kind: pb.AttemptKind_ATTEMPT_KIND_JOB}}
	heldMu.Unlock()
	if !o.c.DetachRental(podRental) {
		t.Fatal("remote control was not detached")
	}
	before, problem := o.store.AttemptRow(id, 1)
	fatal(t, problem)
	if before.State != "accepted" || before.SessionID != podBootID {
		t.Fatalf("detach changed active obligation: %+v", before)
	}
	_, _, _, problem = o.c.EnsureRental(podRental)
	fatal(t, problem)
	waitUntil(t, "progress on the new watch of the same held attempt", func() bool { p, ok := o.c.LatestProgress(id, 1); return ok && p.Position != nil && *p.Position == 2 })
	after, problem := o.store.Attempts(id)
	fatal(t, problem)
	if len(after) != 1 || after[0].Attempt != 1 || after[0].SessionID != podBootID || after[0].TerminalID != "" || after[0].State != "recovered_open" {
		t.Fatalf("reattach changed active identity: %+v", after)
	}
	if _, problem := o.store.NextOrdinal(id); problem == nil {
		t.Fatal("reattach opened a new ordinal")
	}
	if claims.Load() != 2 || watches.Load() < 2 || unsafe.Load() != 0 {
		t.Fatalf("claims=%d watches=%d unsafe=%d", claims.Load(), watches.Load(), unsafe.Load())
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.offers) != 0 || len(pod.desired) != 1 {
		t.Fatalf("reattach offered new work or changed selection: offers=%d desired=%d", len(pod.offers), len(pod.desired))
	}
}
