package producttest

import (
	"bytes"
	"crypto/ed25519"
	"crypto/rand"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// These are exact outputs of Runtime's real tiny-wheel/native-checkpoint proof.
// The protocol peer reports code staging without pretending it can dispatch it.
func TestStagedUnpublishedCodeAdvancesToModelPreparation(t *testing.T) {
	for _, condition := range []string{"staged", "wrong_revision", "wrong_placement", "wrong_set", "materializing", "failed"} {
		t.Run(condition, func(t *testing.T) {
			root := filepath.Join("testdata", "unpublished_preparation")
			readSet := func(name string) *pb.DesiredPlacementSet {
				raw, err := os.ReadFile(filepath.Join(root, name))
				must(t, err)
				return &pb.DesiredPlacementSet{PlacementSetCanonicalBytes: raw, PlacementSetDigest: canonical.Digest(raw)}
			}
			code, modeled := readSet("prepared-code.json"), readSet("prepared-model.json")
			doc, err := canonical.Read(code.PlacementSetCanonicalBytes, &pb.PlacementSet{})
			must(t, err)
			row := doc.List("placements")[0]
			development, wheel := row.Sub("development"), row.Sub("development").Sub("project_wheel")
			wheelPath, err := filepath.Abs(filepath.Join(root, wheel.Str("filename")))
			must(t, err)
			revision := localpackage.Installation{
				Package: development.Str("package"), Release: development.Str("release"),
				SourceDigest: development.Str("source_digest"), Digest: development.Str("local_revision_digest"),
				PackageInterfaceDigest: row.Sub("package_interface").Str("digest"),
				PackageInterfaceLength: row.Sub("package_interface").Int("length"),
				Files: []localpackage.File{{Filename: wheel.Str("filename"), Path: wheelPath,
					Digest: wheel.Sub("ref").Str("digest"), Length: wheel.Sub("ref").Int("length"), Kind: "project"}},
			}
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			pod := &fakePod{controlKey: public}
			pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
				if err := pod.verifyClaim(call.Claim, false); err != nil {
					return err
				}
				if spell, _ := canonical.Spell(call.LocalPackageSet.Package.InstallationId); spell != revision.ID {
					t.Errorf("preparation changed captured revision: %s", spell)
				}
				return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, PlacementSet: code})
			}
			entered := make(chan *pb.PreparePrivatePlacementCall, 1)
			pod.privatePrepare = func(call *pb.PreparePrivatePlacementCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
				if err := pod.verifyClaim(call.Claim, false); err != nil {
					return err
				}
				entered <- call
				return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, PlacementSet: modeled})
			}
			type controlSend struct {
				desired *pb.DesiredWorkerState
				send    func(*pb.WorkerFrame) error
			}
			desired := make(chan controlSend, 4)
			pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
				if state := frame.GetDesiredState(); state != nil && state.GetPlacementSet() != nil {
					desired <- controlSend{state, send}
					return true, nil
				}
				return false, nil
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "staged-unpublished-"+condition, rentalWiring(connection, private))
			instance, _, _, problem := o.c.EnsureRental(podRental)
			fatal(t, problem)
			fatal(t, o.c.ConvergeLocalPackage(instance, "code-only", revision, "", nil))
			var stage controlSend
			select {
			case stage = <-desired:
			case <-time.After(5 * time.Second):
				t.Fatal("code desire was not sent")
			}
			if !bytes.Equal(stage.desired.GetPlacementSet().PlacementSetCanonicalBytes, code.PlacementSetCanonicalBytes) {
				t.Fatal("Creator changed prepared code bytes")
			}
			wanted := revision.ID
			if condition == "wrong_revision" {
				wanted = childDigest("a")
			}
			finished := make(chan *exit.Error, 1)
			go func() {
				finished <- o.c.ConvergeUnpublishedPlacement(instance, "model-only", wanted,
					[]*pb.DownloadModelRef{{Slot: "generate.models.model", Model: "proof/native", Manifest: childDigest("b")}})
			}()
			// Actual installed Runtime451 emitted this after preparing the fixture wheel,
			// applying DesiredWorkerState and completing convergence. Only stream identity
			// is rebound here; serving/materialization/admission facts remain its bytes.
			observed, err := os.ReadFile(filepath.Join(root, "observed.pb"))
			must(t, err)
			state := &pb.ObservedWorkerState{}
			must(t, proto.Unmarshal(observed, state))
			state.RecordOwnerEpoch = stage.desired.RecordOwnerEpoch
			state.ControlStreamEpoch = stage.desired.ControlStreamEpoch
			state.WorkerBootId = pod.bootID()
			state.AcceptedDesiredStateRevision = stage.desired.Revision

			switch condition {
			case "wrong_placement":
				state.Placements[0].PlacementId = "package-other"
			case "wrong_set":
				state.Placements[0].PlacementSetDigest = bytes.Repeat([]byte{0xaa}, 32)
			case "materializing":
				state.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_MATERIALIZING
			case "failed":
				state.Placements[0].Materialization = pb.MaterializationState_MATERIALIZATION_STATE_FAILED
			}
			must(t, stage.send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ObservedState{ObservedState: proto.Clone(state).(*pb.ObservedWorkerState)}}))
			waitUntil(t, "the actual observation applied by Creator", func() bool { return o.c.Worker(instance).AdmissionEpoch == state.AdmissionEpoch })
			facts := o.c.Worker(instance)
			if facts.ConvergedRevision != 0 || facts.AvailableSlots != 0 || len(facts.Dispatchable) != 0 || facts.Admission != "CLOSED" {
				t.Fatalf("code staging became executable: %+v", facts)
			}
			if condition != "staged" {
				select {
				case <-entered:
					t.Fatal("model preparation started without exact staged revision")
				case <-time.After(100 * time.Millisecond):
				}
				return
			}
			select {
			case call := <-entered:
				if spell, _ := canonical.Spell(call.PrivatePlacementSet.InstallationId); spell != revision.ID {
					t.Fatal("model preparation changed revision")
				}
			case <-time.After(3 * time.Second):
				t.Fatal("STAGED/OFFLINE code never advanced to private model preparation")
			}
			select {
			case problem := <-finished:
				fatal(t, problem)
			case <-time.After(time.Second):
				t.Fatal("model prepare call did not finish")
			}
			select {
			case next := <-desired:
				if !bytes.Equal(next.desired.GetPlacementSet().PlacementSetCanonicalBytes, modeled.PlacementSetCanonicalBytes) {
					t.Fatal("Creator changed actual prepared model bytes")
				}
			case <-time.After(3 * time.Second):
				t.Fatal("prepared modeled set did not reach WorkerControl")
			}
		})
	}
}
