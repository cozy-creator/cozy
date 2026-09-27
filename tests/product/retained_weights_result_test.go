package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// retainedWeights is a finished rented job whose declared weights output no collected
// model result carries, on a fake pod beside a stand-in Hub: the upload bench's shape.
type retainedWeights struct {
	root     string
	store    *records.Store
	request  records.Request
	hub      *fakeRentalHub
	pod      *fakePod
	held     *atomic.Int32
	objects  map[string][]byte // the output's native closure, by object id, manifest included
	manifest string
	written  int64 // what the write added: its objects, header and manifest
}

func objectID(raw []byte) string {
	sum := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(sum[:])
}

// newRetainedWeights records the finished job and its pod; startDaemonProcess starts it.
func newRetainedWeights(t *testing.T) *retainedWeights {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	identity, problem := rental.PendingCreatorIdentity(layout, "retained-weights")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	surface := `{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,` +
		`"weights_outputs":[{"output_id":"model","max_bytes":1073741824,"mime_type":"application/vnd.cozy.model-manifest"}],"request":{"fields":[]},"result":{"fields":[]}}]}`
	installed := &pb.InstalledPackage{InstallationId: "retained-weights-install", Package: "local/example", Release: "1.0.0", PackageInterface: []byte(surface)}
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-retained-weights", IdemKey: "retained-weights", Package: installed.Package,
		Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		Rental: true, RentalRequired: true, RequestedRental: podRental, RetainWork: true, MachineExecutionObserver: true,
		LocalInstallationID: installed.InstallationId,
		WeightsOutputs:      `[{"output_id":"model","max_bytes":1073741824,"mime_type":"application/vnd.cozy.model-manifest"}]`})
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
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)

	// The job wrote its declared weights output and returned a plain dict. Its native
	// receipt names what the write added: two tensor objects, a header and the manifest.
	f := &retainedWeights{root: root, store: store, request: request, held: &atomic.Int32{}, objects: map[string][]byte{}}
	blobs := [][]byte{[]byte(strings.Repeat("tensor-a ", 900)), []byte(strings.Repeat("tensor-b ", 1400))}
	manifestBytes := []byte(`{"format":"cozytensors/1","objects":2}`)
	f.manifest = objectID(manifestBytes)
	f.objects[f.manifest] = manifestBytes
	var added []string
	for _, blob := range blobs {
		f.objects[objectID(blob)] = blob
		added = append(added, `{"length":`+strconv.Itoa(len(blob))+`,"sha256":"`+strings.TrimPrefix(objectID(blob), "sha256:")+`"}`)
		f.written += int64(len(blob))
	}
	sort.Strings(added)
	f.written += 2048 + int64(len(manifestBytes))
	native := []byte(`{"added_objects":[` + strings.Join(added, ",") + `],"header":{"length":2048,"sha256":"` + strings.Repeat("c", 64) +
		`"},"manifest":{"length":` + strconv.Itoa(len(manifestBytes)) + `,"sha256":"` + strings.TrimPrefix(f.manifest, "sha256:") + `"}}`)
	nativeDigest, err := canonical.Spell(canonical.Digest(native))
	must(t, err)
	weights, weightsDigest, err := canonical.Identity(&pb.WeightsReceipt{OwnerAuthorityScope: "owner", RequestId: request.ID,
		InvocationSpecDigest: specDigest, OutputSlot: "model", WeightsTransactionId: childDigest("7"),
		TensorfsReceiptDigest: nativeDigest, TensorfsReceiptCanonicalBytes: native})
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		Result:          &pb.ResultEnvelope{InlineResult: []byte(`{}`)},
		WeightsReceipts: []*pb.WeightsReceiptRef{{WeightsReceiptDigest: weightsDigest, WeightsReceiptCanonicalBytes: weights}}})
	must(t, err)
	machine := &finishedMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "retained-weights-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}
	manifest, err := canonical.Raw(f.manifest)
	must(t, err)
	f.pod = &fakePod{controlKey: public, machine: machine,
		derivedRetain: func(_ context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
			f.held.Add(1)
			ask := call.Request
			return &pb.DerivedRetentionResult{WeightsTransactionId: ask.WeightsTransactionId, TensorfsReceiptDigest: ask.TensorfsReceiptDigest,
				RetentionId: ask.RetentionId, Manifest: &pb.Ref{Digest: manifest, Length: uint64(len(manifestBytes))}}, nil
		}}
	connection, certPath := startFakePod(t, root, f.pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	f.hub = newFakeRentalHub(t, 0)
	f.hub.publishListing()
	f.hub.add(podRental, "collector")
	f.hub.set(podRental, "requested_accelerator_model", "fake-4090")
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu",
		AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: f.hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID},
		string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+f.hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	return f
}

// settle follows the finished run until it settles, naming its retained output.
func (f *retainedWeights) settle(t *testing.T) {
	t.Helper()
	code, out, errs := runCozyWithin(t, f.root, "run", "watch", f.request.ID, "--json")
	var settled struct {
		Status   string   `json:"status"`
		Retained []string `json:"retained"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &settled) != nil || settled.Status != "completed" ||
		len(settled.Retained) != 1 || settled.Retained[0] != "model retained on collector" {
		t.Fatalf("the finished run did not settle naming its retained weights [exit %d]: %s%s", code, out, errs)
	}
}

// A rented job that writes its declared weights output and returns a plain result, run
// without --upload-to (upload bench, runs 1280/1283): its weights have no recipient here.
// The weights are held on the machine in this host's custody, the run settles with them
// named as retained on the rental, and the rental is free for maintenance. Before, the
// collection waited for a custody nothing drove: the follower never settled and each
// retry held the rental in use, so `cozy rental update` was refused.
func TestRentedWeightsOutputSettlesRetainedOnTheRental(t *testing.T) {
	f := newRetainedWeights(t)
	startDaemonProcess(t, f.root)
	f.settle(t)
	holds, problem := f.store.MachineModelRetentions(f.request.ID)
	fatal(t, problem)
	if len(holds) != 1 || holds[0].ResultPointer != "weights/model" || holds[0].State != "held" || f.held.Load() == 0 {
		t.Fatalf("the weights output is not held on the machine: %+v", holds)
	}
	// The rental's disk counts what the write actually added, not its declared maximum.
	retained, problem := f.store.RentalRetainedModelBytes(podRental)
	fatal(t, problem)
	if retained != f.written {
		t.Fatalf("the rental counts %d retained bytes, want the %d the write added", retained, f.written)
	}

	// The retained bytes are custody, not a use: maintenance goes past the fence to the
	// rental's own update endpoint, which this stand-in Hub does not publish.
	f.hub.set(podRental, "development", true)
	code, out, errs := runCozyWithin(t, f.root, "rental", "update", "collector", "--json")
	if code == 0 || strings.Contains(out+errs, "rental.maintenance_busy") || !strings.Contains(out+errs, "maintenance endpoint") {
		t.Fatalf("rental update was held by the retained result [exit %d]: %s%s", code, out, errs)
	}
}
