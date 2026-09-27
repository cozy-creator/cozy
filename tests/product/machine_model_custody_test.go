package producttest

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Model outputs stand on their own custody records. A machine job declares two model
// outputs and returns a custody record for `refiner` that names a different native receipt:
// that output fails alone with a warning, and `base` is still collected into recipient
// custody and the run completes.
func TestModelOutputWithBadCustodyFailsAlone(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "model-custody")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	surface := `{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[` +
		`{"name":"base","type":{"input":"model"},"wire":"required"},{"name":"refiner","type":{"input":"model"},"wire":"required"}]}}]}`
	installed := &pb.InstalledPackage{InstallationId: "model-custody-install", Package: "local/example", Release: "1.0.0", PackageInterface: []byte(surface)}
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-model-custody", IdemKey: "model-custody", Package: installed.Package,
		Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	spec := []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: captureDigest,
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))

	manifests := map[string]*pb.Ref{}
	model := func(slot, manifest, native, transaction, retention string) (json.RawMessage, *pb.RetainedModelResult) {
		artifact := records.ModelArtifact{ProducerRequestID: request.ID, OutputSlot: slot,
			Manifest: records.ArtifactObjectRef{Digest: childDigest(manifest), Length: 123}, TensorFSReceiptDigest: childDigest(native)}
		raw, err := json.Marshal(artifact)
		must(t, err)
		raw, err = canonical.NormalizeJCS(raw)
		must(t, err)
		nativeReceipt, err := canonical.Raw(artifact.TensorFSReceiptDigest)
		must(t, err)
		manifestDigest, err := canonical.Raw(artifact.Manifest.Digest)
		must(t, err)
		manifests[string(nativeReceipt)] = &pb.Ref{Digest: manifestDigest, Length: 123}
		return raw, &pb.RetainedModelResult{ResultPointer: "/" + slot, ModelArtifactCanonicalBytes: raw,
			Retention: &pb.DerivedRetentionRequest{WeightsTransactionId: childDigest(transaction), TensorfsReceiptDigest: nativeReceipt, RetentionId: childDigest(retention)}}
	}
	base, baseCustody := model("base", "a", "b", "c", "d")
	refiner, refinerCustody := model("refiner", "e", "f", "0", "2")
	refinerCustody.Retention.TensorfsReceiptDigest = bytes.Repeat([]byte{3}, 32)
	inline, err := json.Marshal(map[string]json.RawMessage{"base": base, "refiner": refiner})
	must(t, err)
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		Result: &pb.ResultEnvelope{InlineResult: inline, RetainedModels: []*pb.RetainedModelResult{baseCustody, refinerCustody}}})
	must(t, err)
	machine := &finishedMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "model-custody-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}
	pod := &fakePod{controlKey: public, machine: machine,
		derivedRetain: func(_ context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
			held := call.Request
			return &pb.DerivedRetentionResult{WeightsTransactionId: held.WeightsTransactionId, TensorfsReceiptDigest: held.TensorfsReceiptDigest,
				RetentionId: held.RetentionId, Manifest: manifests[string(held.TensorfsReceiptDigest)]}, nil
		}}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "collector")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	row := records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)

	code, out := runCozy(t, root, "run", "watch", request.ID)
	if code != 0 {
		t.Fatalf("one bad model custody record failed the run [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
	}
	const want = `warning: model output "refiner" failed: its custody record names a different native receipt`
	if strings.Count(out, want) != 1 {
		t.Fatalf("watch did not show %q exactly once:\n%s", want, out)
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || !link.Collected {
		t.Fatal("the result was not collected")
	}
	holds, problem := store.MachineModelRetentions(request.ID)
	fatal(t, problem)
	if len(holds) != 1 || holds[0].ResultPointer != "/base" || holds[0].State != "held" {
		t.Fatalf("recipient model custody = %+v, want only base held", holds)
	}
}
