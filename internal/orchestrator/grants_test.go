package orchestrator

import (
	"bytes"
	"context"
	"io"
	"sort"
	"testing"
	"time"

	"github.com/cozy-creator/cozy-creator/internal/canonical"
	"github.com/cozy-creator/cozy-creator/internal/exit"
	pb "github.com/cozy-creator/cozy-creator/protocol/cozy/worker/v1"
)

func TestRemoteGrantPrecedesPlacementAndRefreshDoesNotReviseDesiredState(t *testing.T) {
	planBytes := []byte("plan-bytes")
	modelBytes := []byte("model-object-set")
	planDigest, _ := canonical.Spell(canonical.Digest(planBytes))
	modelDigest, _ := canonical.Spell(canonical.Digest(modelBytes))
	planSubject := &pb.ArtifactSubject{
		Digest: canonical.Digest(planBytes), SubjectId: planDigest, Kind: "plan", Length: uint64(len(planBytes)),
	}
	modelSubject := &pb.ArtifactSubject{
		Digest: canonical.Digest(modelBytes), SubjectId: modelDigest, Kind: "model_object_set", Length: uint64(len(modelBytes)),
	}
	set := &pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: "acq-1",
		Spec: &pb.PlacementSpec{
			EndpointReleaseId: "release-1", BindingPlans: []*pb.ArtifactSubject{planSubject},
			ModelObjectSet: modelSubject,
		},
	}}}
	setBytes, setDigest, err := canonical.Identity(set)
	if err != nil {
		t.Fatal(err)
	}
	setSpelling, _ := canonical.Spell(setDigest)
	placement := DesiredPlacement{
		Endpoint: "cozy/endpoint", ReleaseID: "release-1", PlacementIDValue: "acq-1",
		ExactPlacementSetBytes: setBytes, ExactPlacementSetDigest: setSpelling,
		ModelObjectSetDigest: modelDigest, ModelObjectSetLength: uint64(len(modelBytes)),
		Bindings: []*Binding{{Entrypoint: "generate", RuntimePlan: &BindingPlanSubject{
			SubjectID: planDigest, Kind: "plan", Digest: planDigest, Length: uint64(len(planBytes)),
		}}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	s := &session{ctx: ctx, bootID: "boot-1", generation: 7, instanceID: "worker-1",
		out: make(chan *pb.RecordOwnerFrame, 8)}
	w := newWorker("worker-1", WorkerLaunchSpec{
		Placement: placement, Connection: &WorkerConnection{RentalID: "rental-1"},
	})
	revision := uint64(0)
	c := &Orchestrator{
		opt: Options{Log: io.Discard, ArtifactGrants: func(context.Context, *WorkerConnection) (
			uint64, *pb.ArtifactGrant, *exit.Error) {
			revision++
			subjects := []*pb.ArtifactSubject{cloneSubject(planSubject), cloneSubject(modelSubject)}
			sort.Slice(subjects, func(i, j int) bool {
				return bytes.Compare(subjects[i].Digest, subjects[j].Digest) < 0
			})
			return revision, &pb.ArtifactGrant{
				GrantId: "grant", Subjects: subjects,
				ExpiresAtUnix: uint64(time.Now().Add(2 * time.Second).Unix()),
			}, nil
		}},
		sessions: map[string]*session{"boot-1": s}, workers: map[string]*worker{"worker-1": w},
	}
	go c.runRemoteGrantLoop(s, w)

	first := receiveFrame(t, s.out, 2*time.Second)
	if got := first.GetArtifactGrantUpdate(); got == nil || got.GrantRevision != 1 {
		t.Fatalf("first post-snapshot frame = %T, want ArtifactGrantUpdate revision 1", first.Msg)
	}
	second := receiveFrame(t, s.out, 2*time.Second)
	desired := second.GetDesiredState()
	if desired == nil || desired.Revision == 0 {
		t.Fatalf("second post-snapshot frame = %T, want DesiredWorkerState", second.Msg)
	}
	desiredRevision := desired.Revision

	third := receiveFrame(t, s.out, 3*time.Second)
	if got := third.GetArtifactGrantUpdate(); got == nil || got.GrantRevision != 2 {
		t.Fatalf("refresh frame = %T, want ArtifactGrantUpdate revision 2", third.Msg)
	}
	select {
	case frame := <-s.out:
		if got := frame.GetDesiredState(); got != nil && got.Revision != desiredRevision {
			t.Fatalf("grant refresh manufactured desired revision %d after %d", got.Revision, desiredRevision)
		}
	case <-time.After(150 * time.Millisecond):
	}
}

func TestRemoteGrantMustCoverDesiredPlanAndModelSubjects(t *testing.T) {
	planDigest := "sha256:" + string(bytes.Repeat([]byte{'1'}, 64))
	modelDigest := "sha256:" + string(bytes.Repeat([]byte{'2'}, 64))
	rawPlan, _ := canonical.Raw(planDigest)
	rawModel, _ := canonical.Raw(modelDigest)
	placement := DesiredPlacement{
		Bindings: []*Binding{{Entrypoint: "generate", RuntimePlan: &BindingPlanSubject{
			SubjectID: planDigest, Digest: planDigest, Kind: "plan", Length: 10,
		}}},
		ModelObjectSetDigest: modelDigest, ModelObjectSetLength: 20,
	}
	grant := &pb.ArtifactGrant{
		GrantId: "grant", ExpiresAtUnix: uint64(time.Now().Add(time.Minute).Unix()),
		Subjects: []*pb.ArtifactSubject{
			{Digest: rawPlan, SubjectId: planDigest, Kind: "plan", Length: 10},
			{Digest: rawModel, SubjectId: modelDigest, Kind: "model_object_set", Length: 20},
		},
	}
	if problem := grantCovers(placement, grant); problem != nil {
		t.Fatal(problem)
	}
	grant.Subjects[0].Kind = "entrypoint_binding_plan"
	if problem := grantCovers(placement, grant); problem == nil ||
		problem.ErrName() != "rental.artifact_grant_closure_mismatch" {
		t.Fatalf("wrong-kind plan subject was not refused: %v", problem)
	}
	grant.Subjects[0].Kind = "plan"
	grant.Subjects = grant.Subjects[:1]
	if problem := grantCovers(placement, grant); problem == nil ||
		problem.ErrName() != "rental.artifact_grant_closure_mismatch" {
		t.Fatalf("missing model-object-set subject was not refused: %v", problem)
	}
}

func receiveFrame(t *testing.T, frames <-chan *pb.RecordOwnerFrame, within time.Duration) *pb.RecordOwnerFrame {
	t.Helper()
	select {
	case frame := <-frames:
		return frame
	case <-time.After(within):
		t.Fatalf("no frame within %s", within)
		return nil
	}
}

func cloneSubject(subject *pb.ArtifactSubject) *pb.ArtifactSubject {
	return &pb.ArtifactSubject{
		Digest: append([]byte(nil), subject.Digest...), SubjectId: subject.SubjectId,
		Kind: subject.Kind, Length: subject.Length,
	}
}
