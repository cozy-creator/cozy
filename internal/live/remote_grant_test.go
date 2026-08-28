package live

import (
	"bytes"
	"context"
	"crypto/tls"
	"io"
	"net"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/protobuf/proto"

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

// TestRemotePlacementReceivesGrantBeforeDesired drives the real RecordOwner over a pinned
// TLS gRPC stream and the real pod media health leg. It proves the endpoint-distribution
// hardcut at the process boundary: after SnapshotAck the next frame is ArtifactGrantUpdate,
// then the exact PlacementSet/2, with no PutPlan request available on the media server.
func TestRemotePlacementReceivesGrantBeforeDesired(t *testing.T) {
	root := t.TempDir()
	certPath, keyPath := podTLS(t, root)
	tokenText := "remote-owner-token"
	token := secret.New(tokenText)
	mediaAddr := serveMediaPeer(t, certPath, keyPath, tokenText, mediawire.ContractRev)

	peer := &grantPeer{frames: make(chan string, 8), allowReady: make(chan struct{})}
	controlAddr, stopControl := serveWorkerPeer(t, certPath, keyPath, peer)
	defer stopControl()

	planData, modelData := []byte("exact-plan"), []byte("exact-model-object-set")
	planDigest, _ := canonical.Spell(canonical.Digest(planData))
	modelDigest, _ := canonical.Spell(canonical.Digest(modelData))
	planSubject := &pb.ArtifactSubject{Digest: canonical.Digest(planData), SubjectId: planDigest,
		Kind: "plan", Length: uint64(len(planData))}
	modelSubject := &pb.ArtifactSubject{Digest: canonical.Digest(modelData), SubjectId: modelDigest,
		Kind: "model_object_set", Length: uint64(len(modelData))}
	setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{Placements: []*pb.Placement{{
		PlacementId: "acquisition-1", Spec: &pb.PlacementSpec{
			EndpointReleaseId: "release-1", BindingPlans: []*pb.ArtifactSubject{planSubject},
			ModelObjectSet: modelSubject,
		},
	}}})
	must(t, err)
	setSpelling, _ := canonical.Spell(setDigest)
	placement := orchestrator.DesiredPlacement{
		PlacementRevision: 1,
		Endpoint:          "cozy/endpoint@rental-1", ReleaseID: "release-1",
		PlacementIDValue: "acquisition-1", ExactPlacementSetDigest: setSpelling,
		ExactPlacementSetBytes: setBytes, ModelObjectSetDigest: modelDigest,
		ModelObjectSetLength: uint64(len(modelData)),
		Bindings: []*orchestrator.Binding{{Entrypoint: "generate",
			RuntimePlan: &orchestrator.BindingPlanSubject{
				SubjectID: planDigest, Kind: "plan", Digest: planDigest, Length: uint64(len(planData)),
			}}},
	}

	layout, problem := home.Open(filepath.Join(root, "cozy-home"))
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	subjects := []*pb.ArtifactSubject{planSubject, modelSubject}
	sort.Slice(subjects, func(i, j int) bool { return bytes.Compare(subjects[i].Digest, subjects[j].Digest) < 0 })
	relayed := make(chan orchestrator.RentalSessionEvidence, 2)
	owner, problem := orchestrator.Open(orchestrator.Options{
		Layout: layout, Store: store, Log: io.Discard, MaxOutputMiB: 1,
		ObserveRental: func(orchestrator.RentalObservation) *exit.Error { return nil },
		ArtifactGrants: func(context.Context, *orchestrator.WorkerConnection) (
			uint64, *pb.ArtifactGrant, *exit.Error) {
			return 1, &pb.ArtifactGrant{
				GrantId: "acquisition-1-grant-1", Subjects: subjects,
				ExpiresAtUnix: uint64(time.Now().Add(time.Hour).Unix()),
			}, nil
		},
		RelayRentalSession: func(_ context.Context, connection *orchestrator.WorkerConnection,
			evidence orchestrator.RentalSessionEvidence) *exit.Error {
			if connection.RentalID != "rental-1" {
				return exit.Internalf("relayed rental %s", connection.RentalID)
			}
			relayed <- evidence
			return nil
		},
	})
	fatal(t, problem)
	defer owner.Close(time.Second)
	_, change, problem := owner.EnsureWorker(orchestrator.WorkerLaunchSpec{
		Placement: placement,
		Connection: &orchestrator.WorkerConnection{
			RentalID: "rental-1", Addr: controlAddr, Token: token, CACert: certPath,
			Media: &media.Spec{Addr: mediaAddr, Token: token, CACert: certPath},
		},
	})
	fatal(t, problem)
	if change != orchestrator.ChangeWorkerStarted {
		t.Fatalf("worker change = %s", change)
	}
	for i, want := range []string{"snapshot_ack", "artifact_grant", "desired_state"} {
		select {
		case got := <-peer.frames:
			if got != want {
				t.Fatalf("post-claim frame %d = %s, want %s", i+1, got, want)
			}
		case <-time.After(5 * time.Second):
			t.Fatalf("no %s frame", want)
		}
	}
	fault := receiveRelayedEvidence(t, relayed, placement.PlacementRevision)
	if observed := fault.GetObservedState(); observed.GetAdmissionState() !=
		pb.AdmissionState_ADMISSION_STATE_CLOSED || len(observed.GetPlacements()) != 1 ||
		len(observed.GetPlacements()[0].GetFaults()) != 1 || observed.GetConvergedRevision() != 0 {
		t.Fatalf("typed convergence fault was not relayed exactly: %v", observed)
	}
	close(peer.allowReady)
	ready := receiveRelayedEvidence(t, relayed, placement.PlacementRevision)
	if observed := ready.GetObservedState(); observed.GetAdmissionState() !=
		pb.AdmissionState_ADMISSION_STATE_OPEN || observed.GetConvergedRevision() != placement.PlacementRevision {
		t.Fatalf("ready convergence was not relayed exactly: %v", observed)
	}
}

func receiveRelayedEvidence(t *testing.T, relayed <-chan orchestrator.RentalSessionEvidence,
	revision uint64) *pb.WorkerFrame {
	t.Helper()
	select {
	case evidence := <-relayed:
		if evidence.DesiredRevision != revision || len(evidence.BootFailure) != 0 {
			t.Fatalf("relay revision/alternative = %#v", evidence)
		}
		var observed *pb.WorkerFrame
		for name, raw := range map[string][]byte{
			"claim": evidence.ClaimAck, "snapshot": evidence.Snapshot, "observed": evidence.ObservedState,
		} {
			var frame pb.WorkerFrame
			if err := proto.Unmarshal(raw, &frame); err != nil {
				t.Fatalf("%s relay frame: %v", name, err)
			}
			reencoded, err := (proto.MarshalOptions{Deterministic: true}).Marshal(&frame)
			if err != nil || !bytes.Equal(raw, reencoded) {
				t.Fatalf("%s relay frame is not deterministic: %v", name, err)
			}
			if name == "claim" && frame.GetClaimAck().GetControlRuntimeDigest() == "" {
				t.Fatal("relayed ClaimAck omitted control_runtime_digest")
			}
			if name == "observed" {
				if frame.GetObservedState().GetAppliedWireMinor() != pb.WireMinor {
					t.Fatalf("relayed ObservedWorkerState applied minor = %d, want %d",
						frame.GetObservedState().GetAppliedWireMinor(), pb.WireMinor)
				}
				observed = &frame
			}
		}
		return observed
	case <-time.After(5 * time.Second):
		t.Fatal("no private-rental convergence relay")
		return nil
	}
}

type grantPeer struct {
	pb.UnimplementedWorkerControlServer
	frames     chan string
	allowReady chan struct{}
}

func (p *grantPeer) WatchProgress(_ *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
	<-stream.Context().Done()
	return nil
}

func (p *grantPeer) Control(stream pb.WorkerControl_ControlServer) error {
	first, err := stream.Recv()
	if err != nil || first.GetClaim() == nil {
		return err
	}
	claim := first.GetClaim()
	bootID, generation := "boot-remote", uint64(1)
	schemaDigest, _ := canonical.Raw(pb.SchemaDigest)
	if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
		RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
		WorkerBootId: bootID, Accepted: true, WireMinor: pb.WireMinor,
		WorkerId: "worker-remote", WorkerInstanceId: "worker-instance-remote",
		WorkerReleaseId: "release-1", WireSchemaDigest: schemaDigest,
		ControlRuntimeDigest: "sha256:1111111111111111111111111111111111111111111111111111111111111111",
		Resources:            &pb.WorkerResources{Backend: "cuda", DeviceName: "NVIDIA H100", DeviceCount: 1},
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
		WorkerBootId: bootID, SnapshotId: "snapshot-1", SnapshotDigest: bodyDigest,
		SnapshotCanonicalBytes: bodyBytes, AcceptedPlacementSetCanonicalBytes: emptyBytes,
	}}}); err != nil {
		return err
	}
	for {
		frame, err := stream.Recv()
		if err != nil {
			return nil
		}
		switch {
		case frame.GetSnapshotAck() != nil:
			p.frames <- "snapshot_ack"
		case frame.GetArtifactGrantUpdate() != nil:
			update := frame.GetArtifactGrantUpdate()
			if update.Grant == nil || len(update.Grant.Subjects) < 2 {
				p.frames <- "invalid_artifact_grant"
				continue
			}
			p.frames <- "artifact_grant"
		case frame.GetDesiredState() != nil:
			desiredFrame := frame.GetDesiredState()
			desired := desiredFrame.GetPlacementSet()
			if desired == nil {
				p.frames <- "invalid_desired_state"
				continue
			}
			doc, err := canonical.Read(desired.PlacementSetCanonicalBytes, &pb.PlacementSet{})
			if err != nil || len(doc.List("placements")) != 1 {
				p.frames <- "invalid_desired_state"
				continue
			}
			model := doc.List("placements")[0].Sub("spec").Sub("model_object_set")
			if model.Str("kind") != "model_object_set" || model.Str("subject_id") != model.Str("digest") ||
				model.Int("length") <= 0 {
				p.frames <- "invalid_desired_state"
				continue
			}
			p.frames <- "desired_state"
			placement := doc.List("placements")[0]
			planID := placement.Sub("spec").List("binding_plans")[0].Str("subject_id")
			if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{
				ObservedState: &pb.ObservedWorkerState{
					RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
					WorkerBootId: bootID, WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE,
					AppliedWireMinor: pb.WireMinor,
					AdmissionState:   pb.AdmissionState_ADMISSION_STATE_CLOSED, AdmissionGeneration: 2,
					AcceptedDesiredStateRevision: desiredFrame.Revision,
					AcceptedPlacementSetDigest:   desired.PlacementSetDigest,
					Placements: []*pb.PlacementStatus{{PlacementId: placement.Str("placement_id"),
						Materialization: pb.MaterializationState_MATERIALIZATION_STATE_FAILED,
						Serving:         pb.ServingState_SERVING_STATE_OFFLINE,
						Faults: []*pb.Fault{{Kind: pb.FaultKind_FAULT_KIND_ARTIFACT_FETCH_FAILED,
							Subject: placement.Str("placement_id"), Reason: "artifact_fetch_failed",
							Detail: "typed test fault"}}}},
				},
			}}); err != nil {
				return err
			}
			select {
			case <-p.allowReady:
			case <-stream.Context().Done():
				return nil
			}
			if err := stream.Send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{
				ObservedState: &pb.ObservedWorkerState{
					RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamGeneration: generation,
					WorkerBootId: bootID, WorkerPhase: pb.WorkerPhase_WORKER_PHASE_ONLINE,
					AppliedWireMinor: pb.WireMinor,
					AdmissionState:   pb.AdmissionState_ADMISSION_STATE_OPEN, AdmissionGeneration: 2,
					AvailableAttemptSlots: 1, AcceptedDesiredStateRevision: desiredFrame.Revision,
					AcceptedPlacementSetDigest: desired.PlacementSetDigest,
					ConvergedRevision:          desiredFrame.Revision,
					Placements: []*pb.PlacementStatus{{PlacementId: placement.Str("placement_id"),
						Materialization:     pb.MaterializationState_MATERIALIZATION_STATE_STAGED,
						Serving:             pb.ServingState_SERVING_STATE_DISPATCHABLE,
						DispatchablePlanIds: []string{planID}}},
				},
			}}); err != nil {
				return err
			}
		}
	}
}

func serveWorkerPeer(t *testing.T, certPath, keyPath string,
	peer pb.WorkerControlServer) (string, func()) {
	t.Helper()
	certificate, err := tls.LoadX509KeyPair(certPath, keyPath)
	must(t, err)
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow live proof's TLS worker peer
	must(t, err)
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{
		MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{certificate},
	})))
	pb.RegisterWorkerControlServer(server, peer)
	go func() { _ = server.Serve(listener) }()
	return listener.Addr().String(), func() { server.Stop(); _ = listener.Close() }
}
