package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func TestActiveParentChildRetryPrecedesUnrelatedRentalRoot(t *testing.T) {
	for _, pinned := range []bool{true, false} {
		t.Run(map[bool]string{true: "pinned", false: "unpinned"}[pinned], func(t *testing.T) {
			activeParentChildRetryPrecedesUnrelatedRentalRoot(t, pinned)
		})
	}
}

func activeParentChildRetryPrecedesUnrelatedRentalRoot(t *testing.T, pinned bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, jobReady: true, localJobOnly: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	owner := hostOwner(t, "parent-priority", rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = libraryChildLauncher{localLauncher{revision: revision}}
		opt.RentalFleet = func() (string, *exit.Error) { return "one busy rental", nil }
		opt.AcquireManagedRental = func(records.Request) (orchestrator.PlacementDecision, string, *exit.Error) {
			return orchestrator.PlacementDecision{}, "", nil
		}
	})
	fatal(t, owner.store.RecordRental(records.Rental{AcceleratorCount: 1, ID: podRental, MachineName: "otter", State: "ready", SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100000,
		Address: connection.Addr, CertPath: connection.CACert, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	install := records.PackageInstall{ID: "parent-priority", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(owner.root, "checkout"),
		Dir: filepath.Join(owner.root, "installs", "parent"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := owner.store.Activate(install)
	fatal(t, problem)
	fatal(t, owner.store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: install.ID, ChildInstallID: install.ID, Module: "helper", Export: "compute", Entrypoint: "compute"}}))
	intent, err := canonical.NormalizeJCS([]byte(`{"interface_digest":"` + childDigest("b") + `","module":"helper","export":"compute","request":{}}`))
	must(t, err)
	var offers atomic.Int32
	childOffers := make(chan *pb.AttemptOffer, 3)
	peerSend := make(chan func(*pb.WorkerFrame) error, 1)
	var latestReady *pb.WorkerFrame
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		frame.GetObservedState().JobCapacity.OrchestrationAvailable = 1
		if offers.Load() > 0 {
			frame.GetObservedState().JobCapacity.OrchestrationAvailable = 0
			frame.GetObservedState().JobCapacity.OrchestrationInFlight = 1
		}
		latestReady = proto.Clone(frame).(*pb.WorkerFrame)
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil && latestReady != nil {
			return false, send(proto.Clone(latestReady).(*pb.WorkerFrame))
		}
		offer := frame.GetAttemptOffer()
		if offer == nil {
			return false, nil
		}
		if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
			RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
			RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest}}}); err != nil {
			return false, err
		}
		if offers.Add(1) == 1 {
			return false, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ChildCallRequest{ChildCallRequest: &pb.ChildCallRequest{
				RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
				ParentRequestId: offer.RequestId, ParentAttemptOrdinal: offer.AttemptOrdinal, ParentInvocationSpecDigest: offer.InvocationSpecDigest,
				IntentDigest: canonical.Digest(intent), Module: "helper", Export: "compute", RequestCanonicalBytes: []byte(`{}`)}}})
		}
		childOffers <- proto.Clone(offer).(*pb.AttemptOffer)
		select {
		case peerSend <- send:
		default:
		}
		return false, nil
	}
	submit := func(id string, pinned bool) string {
		worker := ""
		if pinned {
			worker = podRental
		}
		t.Helper()
		request, _, problem := owner.c.Submit(orchestrator.Submission{IdemKey: id, Package: revision.Package, Entrypoint: "prepare", PlanID: childDigest("4"),
			Release: revision.Release, LocalInstallationID: revision.ID, Payload: []byte(`{}`), Worker: worker, RequestedRental: worker,
			InstallID: install.ID, Rental: true, RentalRequired: true, Kind: "job", RetainWork: true})
		fatal(t, problem)
		return request
	}
	parent := submit("active-parent", true)
	var first *pb.AttemptOffer
	select {
	case first = <-childOffers:
	case <-time.After(10 * time.Second):
		t.Fatal("parent did not dispatch its child")
	}
	before := offers.Load()
	baseline := submit("unrelated-root", pinned)
	waitUntil(t, "unrelated root queued", func() bool { return owner.c.QueuePosition(baseline) > 0 })
	outcomeIdentity, _ := canonical.Spell(first.InvocationSpecDigest)
	raw, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: first.RequestId, AttemptOrdinal: first.AttemptOrdinal, InvocationSpecDigest: outcomeIdentity,
		Status: pb.OutcomeStatus_OUTCOME_STATUS_ABANDONED, ExecutionStarted: true, Cause: &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_EXECUTOR_INVALIDATED, Origin: pb.CauseOrigin_CAUSE_ORIGIN_EXECUTOR}})
	must(t, err)
	send := <-peerSend
	must(t, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
		RecordOwnerEpoch: first.RecordOwnerEpoch, ControlStreamEpoch: first.ControlStreamEpoch, WorkerBootId: first.WorkerBootId, RequestId: first.RequestId, AttemptOrdinal: first.AttemptOrdinal,
		InvocationSpecDigest: first.InvocationSpecDigest, OutcomeId: "out-private-child-retry", OutcomeDigest: digest, OutcomeCanonicalBytes: raw}}}))
	select {
	case next := <-childOffers:
		if next.RequestId != first.RequestId || next.AttemptOrdinal != 2 {
			t.Fatalf("unrelated root overtook active parent's retry: %+v", next)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("active parent's child remained behind unrelated root")
	}
	if offers.Load() != before+1 {
		t.Fatal("unexpected additional execution")
	}
	original, problem := owner.store.RequestRow(parent)
	fatal(t, problem)
	other, problem := owner.store.Attempts(baseline)
	fatal(t, problem)
	if original == nil || original.State != "dispatching" || len(other) != 0 {
		t.Fatal("priority changed the live parent or dispatched another root")
	}
}
