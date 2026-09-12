package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (p *fakePod) PreparePrivatePlacement(call *pb.PreparePrivatePlacementCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
	if p.privatePrepare == nil {
		return status.Error(codes.Unimplemented, "private model preparation was not configured")
	}
	return p.privatePrepare(call, stream)
}

func TestUnpublishedJobDownloadsInputsBeforeDispatchWithoutServing(t *testing.T) {
	for _, versioned := range []bool{false, true} {
		t.Run(map[bool]string{false: "checkpoint", true: "release_lane"}[versioned], func(t *testing.T) {
			privateJobDownload(t, versioned)
		})
	}
}

func privateJobDownload(t *testing.T, versioned bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, jobReady: true, localJobOnly: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "private-job-download-"+t.Name(), rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = localLauncher{revision: revision}
		opt.RentalPackageSet = func(packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
			if len(packages) != 0 || len(models) != 1 {
				t.Errorf("job input preparation included code or native inputs: packages=%d models=%d", len(packages), len(models))
			}
			raw, _, err := canonical.Identity(&pb.DownloadDelegation{Models: models})
			if err != nil {
				return nil, exit.Internalf("encode test selection: %s", err)
			}
			return raw, nil
		}
	})
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	install := records.PackageInstall{ID: "private-job-download", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"), SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", "job"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	model := records.ModelRef{Package: revision.Package, Slot: "source", BindingPath: "prepare.models.source", Model: "proof/input",
		Manifest: childDigest("8"), ManifestLength: 161, Bytes: 4096, HubCheckpoint: !versioned}
	native := records.ModelRef{Package: revision.Package, Slot: "prior", BindingPath: "prepare.models.prior", Model: "job-source/weights",
		Manifest: childDigest("9"), ManifestLength: 161}
	if versioned {
		model.Release, model.Lane = "1.0.0", "bf16"
	}
	entered, release, offered := make(chan struct{}, 1), make(chan struct{}), make(chan string, 1)
	defer func() {
		select {
		case <-release:
		default:
			close(release)
		}
	}()
	pod.privatePrepare = func(call *pb.PreparePrivatePlacementCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		selected := call.PrivatePlacementSet
		doc, err := canonical.Read(selected.DownloadDelegation, &pb.DownloadDelegation{})
		if err != nil {
			return err
		}
		rows := doc.List("models")
		if len(rows) != 1 || rows[0].Str("manifest") != model.Manifest || rows[0].Str("slot") != model.BindingPath || rows[0].Str("release") != model.Release || rows[0].Str("lane") != model.Lane {
			t.Errorf("exact job input selection changed: %s", selected.DownloadDelegation)
		}
		entered <- struct{}{}
		select {
		case <-release:
		case <-stream.Context().Done():
			return stream.Context().Err()
		}
		pod.mu.Lock()
		prepared := &pb.DesiredPlacementSet{PlacementSetDigest: append([]byte(nil), pod.preparedDig...), PlacementSetCanonicalBytes: append([]byte(nil), pod.preparedSet...)}
		pod.mu.Unlock()
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, PlacementSet: prepared})
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, _ func(*pb.WorkerFrame) error) (bool, error) {
		if offer := frame.GetAttemptOffer(); offer != nil {
			offered <- offer.RequestId
		}
		return false, nil
	}
	id, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "private-job-download", Package: revision.Package, Entrypoint: "prepare", PlanID: childDigest("4"),
		Release: revision.Release, LocalPackageDigest: revision.Digest, Payload: []byte(`{}`),
		Worker: podRental, InstallID: install.ID, Rental: true, RentalRequired: true, Kind: "job", RetainWork: true,
		ProducerParams: []string{"source", "prior"}, Models: []records.ModelRef{model, native},
	})
	fatal(t, problem)
	select {
	case <-entered:
	case early := <-offered:
		t.Fatalf("attempt %s dispatched before model download", early)
	case <-time.After(10 * time.Second):
		t.Fatal("job never prepared its input models")
	}
	select {
	case early := <-offered:
		t.Fatalf("attempt %s ran while model preparation was blocked", early)
	default:
	}
	close(release)
	select {
	case got := <-offered:
		if got != id {
			t.Fatalf("offered %s, wanted %s", got, id)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("prepared job did not dispatch")
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	for _, desired := range pod.desired {
		if desired.GetPlacementSet() != nil {
			t.Fatal("job model preparation activated serving")
		}
	}
}
