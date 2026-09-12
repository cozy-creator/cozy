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
)

type libraryChildLauncher struct{ localLauncher }

func (l libraryChildLauncher) ResolveUnpublishedChild(parent records.Request, _, _, _ string, payload []byte) (orchestrator.Submission, string, *exit.Error) {
	return orchestrator.Submission{Kind: "job", Package: parent.Package, Entrypoint: "compute", Release: parent.Release,
		InstallID: parent.InstallID, LocalPackageDigest: parent.LocalPackageDigest, PlanID: childDigest("5"), Payload: payload,
		RetainWork: true, Worker: parent.Worker, Rental: true, RentalRequired: true}, childDigest("6"), nil
}

func TestLibraryConsumerRunsButCannotQueueChildBehindItself(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, jobReady: true, localJobOnly: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "library-consumer-slot", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = libraryChildLauncher{localLauncher{revision: revision}}
	})
	fatal(t, o.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	install := records.PackageInstall{ID: "library-job", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"), SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", "job"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	fatal(t, o.store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: install.ID, ChildInstallID: install.ID,
		InterfaceDigest: childDigest("b"), Module: "helper", Export: "compute", Entrypoint: "compute"}}))
	intent, err := canonical.NormalizeJCS([]byte(`{"interface_digest":"` + childDigest("b") + `","module":"helper","export":"compute","request":{}}`))
	must(t, err)
	result := make(chan *pb.ChildCallResult, 1)
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if reply := frame.GetChildCallResult(); reply != nil {
			result <- reply
			return true, nil
		}
		if offer := frame.GetAttemptOffer(); offer != nil {
			iface, _ := canonical.Raw(childDigest("b"))
			return false, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ChildCallRequest{ChildCallRequest: &pb.ChildCallRequest{
				RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
				ParentRequestId: offer.RequestId, ParentAttemptOrdinal: offer.AttemptOrdinal, ParentInvocationSpecDigest: offer.InvocationSpecDigest,
				InterfaceDigest: iface, IntentDigest: canonical.Digest(intent), Module: "helper", Export: "compute", RequestCanonicalBytes: []byte(`{}`),
			}}})
		}
		return false, nil
	}
	requestID, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "library-job", Package: revision.Package, Entrypoint: "prepare", PlanID: childDigest("4"),
		Release: revision.Release, LocalPackageDigest: revision.Digest, Payload: []byte(`{}`),
		Worker: podRental, InstallID: install.ID, Rental: true, RentalRequired: true, Kind: "job", RetainWork: true,
		Outputs: []string{"weights"}, WeightsOutputs: []orchestrator.WeightsOutput{{OutputID: "weights", MimeType: orchestrator.WeightsManifestMime, MaxBytes: 4096}},
	})
	fatal(t, problem)
	select {
	case reply := <-result:
		if reply.State != pb.ChildCallState_CHILD_CALL_STATE_REFUSED || reply.SafeCode != "child.parent_slot_unavailable" {
			t.Fatalf("actual unsupported nested call did not refuse promptly: %+v", reply)
		}
	case <-time.After(15 * time.Second):
		request, _ := o.store.RequestRow(requestID)
		t.Fatalf("library request failed to execute or nested call deadlocked: %+v", request)
	}
	children, problem := o.store.Children(requestID)
	fatal(t, problem)
	if len(children) != 0 {
		t.Fatalf("ordinary slot holder enqueued a child behind itself: %+v", children)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.jobDirectives) != 1 || pod.jobDirectives[0].Orchestration {
		t.Fatalf("ordinary library weights job lost its slot/resources: %+v", pod.jobDirectives)
	}
}
