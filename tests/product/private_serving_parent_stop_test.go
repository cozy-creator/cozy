package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/protobuf/proto"
)

// The first child already put the worker in serving mode. A later model join
// cannot publish its finished placement after the CPU parent stops awaiting it.
func TestStoppedParentPreventsSecondServingPlacementAfterModelPreparation(t *testing.T) {
	for _, stop := range []string{"pause", "cancel"} {
		for _, mixed := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/mixed=%t", stop, mixed), func(t *testing.T) { stoppedServingParent(t, stop, mixed) })
		}
	}
}

func stoppedServingParent(t *testing.T, stop string, mixed bool) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	root := t.TempDir()
	connection, _ := startFakePod(t, root, pod)
	revision := stageLocalRevision(t, root)
	o := hostOwner(t, "stopped-serving-parent-"+stop, rentalWiring(connection, private), func(opt *orchestrator.Options) {
		opt.Packages = libraryChildLauncher{localLauncher{revision: revision}}
		opt.RentalPackageSet = func(packages []*pb.DownloadPackageRef, models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
			raw, _, err := canonical.Identity(&pb.DownloadDelegation{Packages: packages, Models: models})
			if err != nil {
				return nil, exit.Internalf("encode fixture download set: %s", err)
			}
			return raw, nil
		}
	})
	fatal(t, o.store.RecordRental(records.Rental{ID: podRental, MachineName: "otter", State: "ready", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000,
		Address: connection.Addr, CertPath: connection.CACert, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}))
	install := records.PackageInstall{ID: "stopped-parent", Package: revision.Package, Major: 1, Version: revision.Release,
		SourceKind: "local", SourceRef: filepath.Join(o.root, "checkout"), SourceDigest: revision.SourceDigest,
		Dir: filepath.Join(o.root, "installs", "parent"), Python: "/usr/bin/python3", Platform: "linux-x86"}
	_, problem := o.store.Activate(install)
	fatal(t, problem)
	fatal(t, o.store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: install.ID, ChildInstallID: install.ID, InterfaceDigest: childDigest("b"), Module: "helper", Export: "compute", Entrypoint: "compute"}}))
	var offers atomic.Int32
	parentOffered := make(chan *pb.AttemptOffer, 1)
	pod.onJobReady = func(frame *pb.WorkerFrame, send func(*pb.WorkerFrame) error) error {
		frame.GetObservedState().JobCapacity.OrchestrationAvailable = 1
		return send(frame)
	}
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if frame.GetOutcomeAck() != nil {
			pod.mu.Lock()
			var desired *pb.DesiredWorkerState
			for _, d := range pod.desired {
				if d.GetPlacementSet() != nil {
					desired = proto.Clone(d).(*pb.DesiredWorkerState)
				}
			}
			pod.mu.Unlock()
			if desired != nil {
				return true, send(pod.served(desired, 1))
			}
		}
		if offer := frame.GetAttemptOffer(); offer != nil {
			accepted := &pb.AttemptAccepted{RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch, WorkerBootId: offer.WorkerBootId,
				RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal, InvocationSpecDigest: offer.InvocationSpecDigest}
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: accepted}}); err != nil {
				return true, err
			}
			if offers.Add(1) == 1 {
				parentOffered <- proto.Clone(offer).(*pb.AttemptOffer)
				return true, nil
			}
			return true, send(privateAttemptOutcome(offer, pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_UNSPECIFIED))
		}
		return false, nil
	}
	parent, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "parent", Package: revision.Package, Entrypoint: "prepare", PlanID: childDigest("4"),
		Release: revision.Release, LocalPackageDigest: revision.Digest, Payload: []byte(`{}`), Worker: podRental, RequestedRental: podRental,
		InstallID: install.ID, Rental: true, RentalRequired: true, Kind: "job", RetainWork: true})
	fatal(t, problem)
	select {
	case <-parentOffered:
	case <-time.After(10 * time.Second):
		t.Fatal("CPU parent was not admitted")
	}
	parentRow, problem := o.store.RequestRow(parent)
	fatal(t, problem)
	parentAttempt, problem := o.store.AttemptRow(parent, parentRow.Ordinal)
	fatal(t, problem)
	child := func(id string, index int64, models []records.ModelRef, arguments []byte) records.Request {
		t.Helper()
		plan := podPlanID(revision.Package)
		if len(models) != 0 {
			// A modeled binding has a different slot signature from the first
			// code-only callable and must be prepared before it can be selected.
			plan = childDigest("d")
		}
		r, _, problem := o.store.SubmitChild(records.Request{ID: id, IdemKey: id, Kind: "serving", Package: revision.Package, Entrypoint: "tile", Release: revision.Release,
			InstallID: install.ID, LocalPackageDigest: revision.Digest, PlanID: plan, Payload: []byte(`{"size":48}`), BodyDigest: childDigest("1"),
			ParentRequestID: parent, ParentCallIndex: index, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"), Models: models},
			parentAttempt.Attempt, parentAttempt.InvocationDigest, parentAttempt.SessionID, arguments)
		fatal(t, problem)
		_, problem = o.store.RequestPause(r.ID, "fixture parks before ordinary resume")
		fatal(t, problem)
		_, problem = o.store.CompleteRequestPause(r.ID)
		fatal(t, problem)
		return r
	}
	first := child("first-serving", 0, nil, []byte(`{"models":{},"payload":{"size":48}}`))
	fatal(t, o.c.ResumeRequest(first.ID, "first serving child"))
	waitUntil(t, "first serving child completes", func() bool {
		r, problem := o.store.RequestRow(first.ID)
		fatal(t, problem)
		return r.State == "succeeded"
	})
	// Reuse the existing verified-record fixture seam: no native API or numerical
	// result is simulated here; this independent peer tests the owner protocol.
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	producer, _, problem := o.store.SubmitChild(records.Request{ID: "verified-producer", IdemKey: "verified-producer", Kind: "job", Package: "local/source", Entrypoint: "produce",
		Payload: []byte(`{}`), BodyDigest: childDigest("1"), ParentRequestID: parent, ParentCallIndex: 1, ChildIntentDigest: childDigest("2"), ChildTargetDigest: childDigest("3"),
		ChildArtifacts: true, WeightsOutputs: `[{"output_id":"weights"}]`}, parentAttempt.Attempt, parentAttempt.InvocationDigest, parentAttempt.SessionID, nil)
	fatal(t, problem)
	producer = offerChildParent(t, o.store, producer)
	nativeReceipt := []byte(`{}`)
	nativeDigest, _ := canonical.Spell(canonical.Digest(nativeReceipt))
	receipt, _, err := canonical.Identity(&pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: producer.ID, InvocationSpecDigest: childDigest("1"), OutputSlot: "weights",
		WeightsTransactionId: childDigest("5"), TensorfsReceiptDigest: nativeDigest, TensorfsReceiptCanonicalBytes: nativeReceipt})
	must(t, err)
	_, err = canonical.Read(receipt, &pb.WeightsReceipt{})
	must(t, err)
	receiptDigest, _ := canonical.Spell(canonical.Digest(receipt))
	weights := records.ModelTransferWeights{RequestID: producer.ID, Attempt: 1, OutputSlot: "weights", ManifestID: childDigest("6"), ManifestLength: 161,
		InvocationDigest: childDigest("1"), TransactionID: childDigest("5"), ReceiptDigest: receiptDigest, Receipt: receipt}
	fatal(t, o.store.RecordModelTransferWeights(weights))
	closeChild(t, o.store, producer, "SUCCEEDED", "succeeded")
	artifact := records.ModelArtifact{ProducerRequestID: producer.ID, OutputSlot: "weights", Manifest: records.ArtifactObjectRef{Digest: weights.ManifestID, Length: 161}, TensorFSReceiptDigest: nativeDigest}
	modelArguments := map[string]any{"model": artifact}
	if mixed {
		modelArguments["adapter"] = nil
	} // The caller leaves selection to the callee's default.
	arguments, err := json.Marshal(map[string]any{"models": modelArguments, "payload": map[string]any{"size": 48}})
	must(t, err)
	arguments, err = canonical.NormalizeJCS(arguments)
	must(t, err)
	models := []records.ModelRef{{Package: revision.Package, Slot: "model", BindingPath: "tile.models.model", Model: "fixture/model", Manifest: weights.ManifestID, ManifestLength: 161}}
	if mixed {
		models = append(models, records.ModelRef{Package: revision.Package, Slot: "adapter", BindingPath: "tile.models.adapter", Model: "fixture/base", Release: "1.0.0", Lane: "bf16", Manifest: childDigest("7"), ManifestLength: 161})
	}
	second := child("second-serving", 2, models, arguments)
	retentions, problem := o.store.WeightsRetentions(second.ID)
	fatal(t, problem)
	if len(retentions) != 1 {
		t.Fatalf("serving input was not admitted with exact custody: %+v", retentions)
	}
	fatal(t, o.store.ConfirmWeightsRetention(retentions[0].RetentionID, parentAttempt.InstanceID, podBootID))
	entered, release := make(chan struct{}, 1), make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	pod.privatePrepare = func(call *pb.PreparePrivatePlacementCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		if call.PrivatePlacementSet.OperationId != second.ID || len(call.PrivatePlacementSet.NativeModels) != 1 {
			t.Error("final preparation changed its native child selection")
		}
		if mixed {
			download, err := canonical.Read(call.PrivatePlacementSet.DownloadDelegation, &pb.DownloadDelegation{})
			if err != nil {
				t.Error(err)
			} else {
				rows := download.List("models")
				if len(rows) != 1 || rows[0].Str("slot") != "tile.models.adapter" || rows[0].Str("manifest") != childDigest("7") {
					t.Error("mixed child changed or lost its exact downloaded input")
				}
			}
		} else if len(call.PrivatePlacementSet.DownloadDelegation) != 0 {
			t.Error("native input was sent to the registry")
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
	fatal(t, o.c.ResumeRequest(second.ID, "second serving child"))
	select {
	case <-entered:
	case <-time.After(10 * time.Second):
		t.Fatal("second child never reached final native model preparation")
	}
	servingSets := func() int {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		count := 0
		for _, desired := range pod.desired {
			if desired.GetPlacementSet() != nil {
				count++
			}
		}
		return count
	}
	before := servingSets()
	if before != 1 {
		t.Fatalf("first child did not establish exactly one serving set: %d", before)
	}
	if stop == "pause" {
		fatal(t, o.c.PauseRequest(parent, "stop during final native prepare"))
	} else {
		fatal(t, o.c.CancelRetainedRequest(parent, "stop during final native prepare"))
	}
	unblock()
	waitUntil(t, "final preparation is refused after parent stop", func() bool {
		after := servingSets()
		if after != before {
			t.Fatalf("stopped parent gained a new serving placement: %d -> %d", before, after)
		}
		return strings.Contains(tail(filepath.Join(o.root, "orchestrator.log")), "private preparation lost its current execution authority")
	})
	row, problem := o.store.RequestRow(second.ID)
	fatal(t, problem)
	if row.Ordinal != 0 || offers.Load() != 2 {
		t.Fatal("stopped parent's second child was offered for execution")
	}
}
