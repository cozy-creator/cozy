package live

import (
	"bytes"
	"context"
	"io"
	"path/filepath"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/home"
	"github.com/cozy-creator/cozy-creator/internal/media"
	"github.com/cozy-creator/cozy-creator/internal/mediawire"
	"github.com/cozy-creator/cozy-creator/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator/internal/records"
	"github.com/cozy-creator/cozy-creator/internal/secret"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

func TestRemotePlacementRevisionReusesClaimAndPersistsAcquisition(t *testing.T) {
	root := t.TempDir()
	certPath, keyPath := podTLS(t, root)
	tokenText := "revision-owner-token"
	token := secret.New(tokenText)
	mediaAddr := serveMediaPeer(t, certPath, keyPath, tokenText, mediawire.ContractRev)

	a := makeRevisionPlacement(t, "a", 1)
	b := makeRevisionPlacement(t, "b", 2)
	c := makeRevisionPlacement(t, "c", 3)
	peer := &revisionPeer{frames: make(chan string, 16), scenarios: make(chan revisionScenario, 3)}
	peer.scenarios <- revisionScenario{actualSpec: a.specDigest, planID: a.planID, converged: 1,
		acquisition: &pb.PlacementAcquisitionObservation{
			Endpoint: &pb.AcquisitionLegObservation{StartedMonotonicNs: 10, EndedMonotonicNs: 40, DownloadedBytes: 100},
			Model:    &pb.AcquisitionLegObservation{StartedMonotonicNs: 20, EndedMonotonicNs: 50, DownloadedBytes: 200},
		}}
	peer.scenarios <- revisionScenario{actualSpec: b.specDigest, planID: b.planID, converged: 2,
		acquisition: &pb.PlacementAcquisitionObservation{
			Endpoint: &pb.AcquisitionLegObservation{StartedMonotonicNs: 60, EndedMonotonicNs: 70, ReusedBytes: 100},
			Model:    &pb.AcquisitionLegObservation{StartedMonotonicNs: 61, EndedMonotonicNs: 71, ReusedBytes: 200},
		}}
	peer.scenarios <- revisionScenario{actualSpec: b.specDigest, fallbackSpec: b.specDigest,
		planID: b.planID, converged: 2}
	controlAddr, stop := serveWorkerPeer(t, certPath, keyPath, peer)
	defer stop()

	layout, problem := home.Open(filepath.Join(root, "cozy-home"))
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	var stateMu sync.Mutex
	active, placementRevision, grantRevision := a, uint64(1), uint64(0)
	owner, problem := orchestrator.Open(orchestrator.Options{
		Layout: layout, Store: store, Log: io.Discard, MaxOutputMiB: 1,
		ObserveRental: func(orchestrator.RentalObservation) *exit.Error { return nil },
		RelayRentalSession: func(context.Context, *orchestrator.WorkerConnection,
			orchestrator.RentalSessionEvidence) *exit.Error {
			return nil
		},
		ArtifactGrants: func(context.Context, *orchestrator.WorkerConnection) (
			uint64, *pb.ArtifactGrant, *exit.Error) {
			stateMu.Lock()
			defer stateMu.Unlock()
			grantRevision++
			return grantRevision, &pb.ArtifactGrant{
				GrantId: "grant", Subjects: cloneSubjects(active.subjects),
				ExpiresAtUnix: uint64(time.Now().Add(time.Hour).Unix()),
			}, nil
		},
		PlacementRevisions: func(_ context.Context, _ *orchestrator.WorkerConnection,
			endpointRef, _, _ string) (orchestrator.DesiredPlacement, uint64, *exit.Error) {
			stateMu.Lock()
			defer stateMu.Unlock()
			switch endpointRef {
			case b.placement.Endpoint:
				active = b
			case c.placement.Endpoint:
				active = c
			default:
				return orchestrator.DesiredPlacement{}, 0, exit.New(exit.NotFound, "unknown test endpoint")
			}
			placementRevision++
			return active.placement, placementRevision, nil
		},
	})
	fatal(t, problem)
	defer owner.Close(time.Second)

	connection := &orchestrator.WorkerConnection{
		RentalID: "rental-1", Addr: controlAddr, Token: token, CACert: certPath,
		Media: &media.Spec{Addr: mediaAddr, Token: token, CACert: certPath},
	}
	initial := a.placement
	initial.Endpoint += "@rental-1"
	instanceID, _, problem := owner.EnsureWorker(orchestrator.WorkerLaunchSpec{
		Placement: initial, Connection: connection,
	})
	fatal(t, problem)
	wantFrames(t, peer.frames, "snapshot_ack", "grant", "desired")
	first := waitWorkerRevision(t, owner, instanceID, 1)
	if first.BootID != "boot-revision" || first.PID != 0 ||
		first.AcceptedPlacementSetDigest != a.placement.ExactPlacementSetDigest ||
		len(first.GrantSubjects) != 2 {
		t.Fatalf("initial worker identity = boot %q pid %d", first.BootID, first.PID)
	}
	cold := waitPlacementAcquisition(t, store, instanceID, first.BootID, a.specSpelling)
	if cold == nil || max64(cold.Endpoint.StartedNS, cold.Model.StartedNS) >=
		min64(cold.Endpoint.EndedNS, cold.Model.EndedNS) {
		t.Fatalf("cold endpoint/model intervals did not overlap: %#v", cold)
	}

	second, revision, problem := owner.ReviseRental(context.Background(), "rental-1",
		b.placement.Endpoint, "revise-b", "test warm reuse")
	fatal(t, problem)
	if revision != 2 || second.InstanceID != first.InstanceID || second.BootID != first.BootID || second.PID != first.PID {
		t.Fatalf("revision restarted worker: first=%#v second=%#v revision=%d", first, second, revision)
	}
	wantFrames(t, peer.frames, "grant", "desired")
	warmFacts := waitWorkerRevision(t, owner, instanceID, 2)
	warm := waitPlacementAcquisition(t, store, instanceID, warmFacts.BootID, b.specSpelling)
	if warm == nil || warm.Endpoint.DownloadedBytes != 0 || warm.Model.DownloadedBytes != 0 ||
		warm.Endpoint.ReusedBytes == 0 || warm.Model.ReusedBytes == 0 {
		t.Fatalf("warm revision did not prove zero-download reuse: %#v", warm)
	}

	third, revision, problem := owner.ReviseRental(context.Background(), "rental-1",
		c.placement.Endpoint, "revise-c", "test fallback")
	fatal(t, problem)
	if revision != 3 || third.InstanceID != first.InstanceID || third.BootID != first.BootID || third.PID != first.PID {
		t.Fatalf("fallback revision restarted worker: first=%#v third=%#v", first, third)
	}
	wantFrames(t, peer.frames, "grant", "desired")
	fallback := waitWorkerRevision(t, owner, instanceID, 3)
	if fallback.ConvergedRevision != 2 || fallback.Serving != "DISPATCHABLE" ||
		fallback.PlacementSpecDigest != b.specSpelling ||
		fallback.RetainedFallbackSpecDigest != b.specSpelling {
		t.Fatalf("failed activation did not retain dispatchable fallback: %#v", fallback)
	}
}

type revisionPlacement struct {
	placement            orchestrator.DesiredPlacement
	subjects             []*pb.ArtifactSubject
	planID, specSpelling string
	specDigest           []byte
}

func makeRevisionPlacement(t *testing.T, name string, revision uint64) revisionPlacement {
	t.Helper()
	planData, modelData := []byte("plan-"+name), []byte("model-"+name)
	planID, _ := canonical.Spell(canonical.Digest(planData))
	modelID, _ := canonical.Spell(canonical.Digest(modelData))
	planSubject := &pb.ArtifactSubject{Digest: canonical.Digest(planData), SubjectId: planID,
		Kind: "plan", Length: uint64(len(planData))}
	modelSubject := &pb.ArtifactSubject{Digest: canonical.Digest(modelData), SubjectId: modelID,
		Kind: "model_object_set", Length: uint64(len(modelData))}
	spec := &pb.PlacementSpec{EndpointReleaseId: "release-" + name,
		BindingPlans: []*pb.ArtifactSubject{planSubject}, ModelObjectSet: modelSubject}
	_, specDigest, err := canonical.Identity(spec)
	must(t, err)
	specSpelling, _ := canonical.Spell(specDigest)
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: "acquisition-1", Spec: spec,
	}}})
	must(t, err)
	setSpelling, _ := canonical.Spell(setDigest)
	subjects := []*pb.ArtifactSubject{planSubject, modelSubject}
	sort.Slice(subjects, func(i, j int) bool { return bytes.Compare(subjects[i].Digest, subjects[j].Digest) < 0 })
	return revisionPlacement{
		placement: orchestrator.DesiredPlacement{
			PlacementRevision: revision,
			Endpoint:          "cozy/endpoint-" + name, ReleaseID: "release-" + name,
			PlacementIDValue: "acquisition-1", ExactPlacementSetDigest: setSpelling,
			ExactPlacementSetBytes: setBytes, ModelObjectSetDigest: modelID,
			ModelObjectSetLength: uint64(len(modelData)),
			Bindings: []*orchestrator.Binding{{Entrypoint: "generate",
				RuntimePlan: &orchestrator.BindingPlanSubject{
					SubjectID: planID, Kind: "plan", Digest: planID, Length: uint64(len(planData)),
				}}},
		},
		subjects: subjects, planID: planID, specDigest: specDigest, specSpelling: specSpelling,
	}
}

type revisionScenario struct {
	actualSpec, fallbackSpec []byte
	planID                   string
	converged                uint64
	acquisition              *pb.PlacementAcquisitionObservation
}

type revisionPeer struct {
	pb.UnimplementedWorkerControlServer
	frames    chan string
	scenarios chan revisionScenario
}

func (p *revisionPeer) WatchProgress(_ *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}

func (p *revisionPeer) Control(stream pb.WorkerControl_ControlServer) error {
	first, err := stream.Recv()
	if err != nil || first.GetClaim() == nil {
		return err
	}
	claim := first.GetClaim()
	schemaDigest, _ := canonical.Raw(pb.SchemaDigest)
	const bootID = "boot-revision"
	const generation = 1
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
		RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
		WorkerBootId: bootID, Accepted: true, WireMinor: pb.WireMinor,
		WorkerId: "worker-revision", WorkerInstanceId: "worker-instance-revision",
		WorkerReleaseId: "release-a", WireSchemaDigest: schemaDigest,
		Resources: &pb.WorkerResources{Backend: "cuda", DeviceName: "NVIDIA H100", DeviceCount: 1},
	}}}); err != nil {
		return err
	}
	emptyBytes, emptyDigest, err := canonical.Identity(&pb.PlacementSet{})
	if err != nil {
		return err
	}
	bodyBytes, bodyDigest, err := canonical.Identity(&pb.WorkerSnapshotBody{
		WorkerPhase:         pb.WorkerPhase_WORKER_PHASE_ONLINE,
		AdmissionState:      pb.AdmissionState_ADMISSION_STATE_CLOSED,
		AdmissionGeneration: 1, AcceptedPlacementSetDigest: emptyDigest,
	})
	if err != nil {
		return err
	}
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: &pb.WorkerSnapshot{
		RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
		WorkerBootId: bootID, SnapshotId: "snapshot-revision", SnapshotDigest: bodyDigest,
		SnapshotCanonicalBytes: bodyBytes, AcceptedPlacementSetCanonicalBytes: emptyBytes,
	}}}); err != nil {
		return err
	}
	grantRevision, grantID := uint64(0), ""
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil
		}
		switch {
		case frame.GetSnapshotAck() != nil:
			p.frames <- "snapshot_ack"
		case frame.GetArtifactGrantUpdate() != nil:
			grantRevision = frame.GetArtifactGrantUpdate().GrantRevision
			grantID = frame.GetArtifactGrantUpdate().Grant.GrantId
			p.frames <- "grant"
		case frame.GetDesiredState() != nil:
			desired := frame.GetDesiredState()
			scenario := <-p.scenarios
			set := desired.GetPlacementSet()
			observed := &pb.ObservedWorkerState{
				RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
				WorkerBootId: bootID, AppliedWireMinor: pb.WireMinor,
				WorkerPhase:         pb.WorkerPhase_WORKER_PHASE_ONLINE,
				AdmissionState:      pb.AdmissionState_ADMISSION_STATE_OPEN,
				AdmissionGeneration: 1, AvailableAttemptSlots: 1,
				AcceptedDesiredStateRevision: desired.Revision,
				AcceptedPlacementSetDigest:   set.PlacementSetDigest,
				ConvergedRevision:            scenario.converged,
				AppliedGrantRevision:         grantRevision, AppliedArtifactGrantId: grantID,
				Placements: []*pb.PlacementStatus{{
					PlacementId: "acquisition-1", ExecutorGeneration: 1,
					DispatchablePlanIds: []string{scenario.planID}, PlacementSpecDigest: scenario.actualSpec,
					RetainedFallbackSpecDigest: scenario.fallbackSpec,
					Materialization:            pb.MaterializationState_MATERIALIZATION_STATE_STAGED,
					Serving:                    pb.ServingState_SERVING_STATE_DISPATCHABLE,
					Acquisition:                scenario.acquisition,
				}},
			}
			if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: observed}}); err != nil {
				return err
			}
			p.frames <- "desired"
		}
	}
}

func cloneSubjects(in []*pb.ArtifactSubject) []*pb.ArtifactSubject {
	out := make([]*pb.ArtifactSubject, 0, len(in))
	for _, subject := range in {
		out = append(out, &pb.ArtifactSubject{Digest: append([]byte(nil), subject.Digest...),
			SubjectId: subject.SubjectId, Kind: subject.Kind, Length: subject.Length})
	}
	return out
}

func wantFrames(t *testing.T, frames <-chan string, expected ...string) {
	t.Helper()
	for _, want := range expected {
		select {
		case got := <-frames:
			if got != want {
				t.Fatalf("frame = %s, want %s", got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("no %s frame", want)
		}
	}
}

func waitWorkerRevision(t *testing.T, owner *orchestrator.Orchestrator,
	instanceID string, revision uint64) orchestrator.WorkerFacts {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if facts := owner.Worker(instanceID); facts != nil && facts.AcceptedRevision == revision {
			return *facts
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("worker %s did not accept revision %d", instanceID, revision)
	return orchestrator.WorkerFacts{}
}

func waitPlacementAcquisition(t *testing.T, store *records.Store, instanceID,
	bootID, specDigest string) *records.PlacementAcquisition {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		observed, problem := store.PlacementAcquisition(instanceID, bootID,
			"acquisition-1", specDigest)
		fatal(t, problem)
		if observed != nil {
			return observed
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("placement acquisition %s was not persisted", specDigest)
	return nil
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}
func max64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}
