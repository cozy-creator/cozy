package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/installkey"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// conversionInterface is a memoized conversion job: two Model inputs, one weights output.
const conversionInterface = `{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"quantize.models.source"},{"class":"Source","component_use":{},"path":"quantize.models.base"}],"invocable":{"capabilities":["weights"],"context":"ctx","defaults":{"request/base":null,"request/source":null},"enum_members":{},"export":"quantize","memoize":true,"module":"q","parameters":["source","base","steps"],"type_names":{"request":"quantizeRequest"}},"name":"quantize","publishes":false,"request":{"fields":[{"name":"source","type":{"union":["null",{"input":"model"}]},"wire":"optional"},{"name":"base","type":{"union":["null",{"input":"model"}]},"wire":"optional"},{"name":"steps","type":"int"}]},"result":{"input":"model"},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"}]}]}`

// conversionMachine is Runtime on a rented pod. The job returns its fp8 Model, retained on
// the pod. A job root whose weights output grant names `model://org/name` has that Model
// uploaded there before the root finishes, and Runtime journals a `checkpoint` event for
// it. An older Runtime (`publishes` false) ignores the grant and still succeeds.
type conversionMachine struct {
	mu          sync.Mutex
	publishes   bool
	jobRoots    bool // a Runtime that takes jobs by their release (release_root_jobs)
	submissions []*pb.MachineExecutionSubmit
	states      map[string]*pb.MachineExecutionState
	events      map[string][]*pb.MachineExecutionEvent
	outcomes    map[string]*pb.AttemptOutcome
}

func (m *conversionMachine) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId,
		ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *conversionMachine) SubmitMachineExecution(_ context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := submit.Offer.RequestId
	if root := submit.ReleaseRoot; root != nil {
		// The machine mints the job: its identity, and its grant from the root's destination.
		minted := sha256.Sum256([]byte(submit.SubmissionId))
		submit = proto.Clone(submit).(*pb.MachineExecutionSubmit)
		submit.CaptureDigest, submit.Offer.InvocationSpecDigest = minted[:], minted[:]
		if root.Job && root.WeightsDestination != "" {
			submit.Offer.Grant = &pb.DeliveryGrant{Outputs: []*pb.OutputAccess{{OutputId: "fp8", Url: "model://" + root.WeightsDestination}}}
		}
	}
	accepted := &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: submit.CaptureDigest,
		InvocationSpecDigest: submit.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace",
		PublicationAuthorizationId: submit.PublicationAuthorizationId}
	if m.states[id] != nil {
		return accepted, nil
	}
	m.submissions = append(m.submissions, proto.Clone(submit).(*pb.MachineExecutionSubmit))
	events := []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: uint64(time.Now().UnixMilli()), Kind: "running", BodyCanonicalBytes: []byte(`{}`)}}
	for _, access := range submit.Offer.GetGrant().GetOutputs() {
		destination, granted := strings.CutPrefix(access.Url, "model://")
		if !granted || !m.publishes || submit.PublicationAuthorizationId == "" {
			continue
		}
		body, _ := json.Marshal(map[string]string{"destination": destination, "checkpoint": childDigest("c"),
			"output_slot": access.OutputId, "observation": "acknowledged"})
		events = append(events, &pb.MachineExecutionEvent{Sequence: uint64(len(events) + 1), AttemptOrdinal: 1,
			AtMs: uint64(time.Now().UnixMilli()), Kind: "checkpoint", BodyCanonicalBytes: body})
	}
	spec, err := canonical.Spell(submit.Offer.InvocationSpecDigest)
	if err != nil {
		return nil, err
	}
	artifact, err := json.Marshal(records.ModelArtifact{ProducerRequestID: id, OutputSlot: "fp8",
		Manifest: records.ArtifactObjectRef{Digest: childDigest("c"), Length: 123}, TensorFSReceiptDigest: childDigest("b")})
	if err != nil {
		return nil, err
	}
	if artifact, err = canonical.NormalizeJCS(artifact); err != nil {
		return nil, err
	}
	receipt, _ := canonical.Raw(childDigest("b"))
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: spec,
		Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, Result: &pb.ResultEnvelope{InlineResult: artifact,
			RetainedModels: []*pb.RetainedModelResult{{ModelArtifactCanonicalBytes: artifact,
				Retention: &pb.DerivedRetentionRequest{WeightsTransactionId: childDigest("d"), TensorfsReceiptDigest: receipt, RetentionId: childDigest("e")}}}}})
	if err != nil {
		return nil, err
	}
	m.outcomes[id] = &pb.AttemptOutcome{RequestId: id, AttemptOrdinal: 1, InvocationSpecDigest: submit.Offer.InvocationSpecDigest,
		OutcomeId: "outcome-" + id, OutcomeDigest: digest, OutcomeCanonicalBytes: body}
	m.events[id] = append(events, outcomeEvent(uint64(len(events)+1), "succeeded", m.outcomes[id]))
	m.states[id] = &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
		ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded", Sequence: uint64(len(events) + 1)}
	return accepted, nil
}

func (m *conversionMachine) GetMachineExecution(_ context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(m.states[query.RequestId]).(*pb.MachineExecutionState), nil
}

func (m *conversionMachine) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	events := m.events[query.Execution.RequestId]
	page := &pb.MachineExecutionEventPage{NextAfter: uint64(len(events)), HeadSequence: uint64(len(events))}
	for _, event := range events {
		if event.Sequence > query.After {
			page.Events = append(page.Events, event)
		}
	}
	return page, nil
}

func (m *conversionMachine) AcknowledgeMachineExecutionCollection(_ context.Context, ack *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	state := m.states[ack.Execution.RequestId]
	state.Collected = true
	return proto.Clone(state).(*pb.MachineExecutionState), nil
}

// retain is the pod's recipient hold on the returned Model.
func (m *conversionMachine) retain(_ context.Context, call *pb.DerivedRetentionCall) (*pb.DerivedRetentionResult, error) {
	manifest, _ := canonical.Raw(childDigest("c"))
	held := call.Request
	return &pb.DerivedRetentionResult{WeightsTransactionId: held.WeightsTransactionId, TensorfsReceiptDigest: held.TensorfsReceiptDigest,
		RetentionId: held.RetentionId, Manifest: &pb.Ref{Digest: manifest, Length: 123}}, nil
}

func (m *conversionMachine) submitted() []*pb.MachineExecutionSubmit {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*pb.MachineExecutionSubmit(nil), m.submissions...)
}

// retainedIngest records a collected machine ingest on the rental whose converted model
// stays in recipient custody there, as `cozy model upload --rental` leaves it.
func retainedIngest(t *testing.T, store *records.Store) {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "job-retained-ingest", IdemKey: "retained-ingest", Package: "local/upload_model",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true,
		PlannedSourceBytes: 7_000_000_000})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	capture, spec := []byte(`{"capture":"ingest"}`), []byte(`{"invocation":"ingest"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "rented-workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	fatal(t, store.AcceptMachineExecution(request.ID, &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey,
		CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "rented-workspace"}))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
		ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
	specDigest, err := canonical.Spell(submission.Offer.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	fatal(t, store.RecordMachineOutcome(request.ID, &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, OutcomeId: "ingest-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body}))
	artifact, err := json.Marshal(records.ModelArtifact{ProducerRequestID: request.ID, OutputSlot: "model",
		Manifest: records.ArtifactObjectRef{Digest: childDigest("a"), Length: 123}, TensorFSReceiptDigest: childDigest("b")})
	must(t, err)
	receipt, err := canonical.Raw(childDigest("b"))
	must(t, err)
	hold := records.MachineModelRetention{OutcomeID: "ingest-outcome", Artifact: artifact, TransactionID: childDigest("d"),
		ReceiptDigest: receipt, SourceRetentionID: childDigest("e"), RetentionID: childDigest("f")}
	fatal(t, store.FreezeMachineModelRetention(request.ID, hold))
	fatal(t, store.AdvanceMachineModelRetention(request.ID, hold.RetentionID, "held"))
	state.Collected = true
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{}))
}

// conversionRental serves the stand-in Hub's rental routes for one attached pod, and
// records each machine publication grant it is asked for.
func conversionRental(row *map[string]any, grants *[]map[string]any, mu *sync.Mutex) func(*http.ServeMux, *hub.PackageReleaseDetail) {
	return func(mux *http.ServeMux, detail *hub.PackageReleaseDetail) {
		iface := []byte(conversionInterface)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = assessmentDigest(iface)
		detail.Release.PackageInterfaceLength = int64(len(iface))
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			mu.Lock()
			defer mu.Unlock()
			if r.PathValue("id") != podRental {
				w.WriteHeader(http.StatusNotFound)
				return
			}
			_ = json.NewEncoder(w).Encode(*row)
		})
		// The temporary prepared path reads the release's lock (proto-062 R2).
		mux.HandleFunc("GET /v1/packages/{org}/{name}/releases/{release}/locked-requirements", func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("--index-url https://pypi.org/simple\nquantize==1.0.0 --hash=sha256:" + strings.Repeat("11", 32) + "\n"))
		})
		mux.HandleFunc("POST /v1/machine-authorizations", func(w http.ResponseWriter, r *http.Request) {
			var grant map[string]any
			if json.NewDecoder(r.Body).Decode(&grant) != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			mu.Lock()
			*grants = append(*grants, grant)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"token": "machine-grant", "expires_at": time.Now().Add(time.Hour).UTC().Format(time.RFC3339)})
		})
	}
}

// conversionFixture is a Creator home beside a stand-in Hub and one attached pod,
// `tessa`, that already holds a retained ingest.
type conversionFixture struct {
	root    string
	layout  home.Layout
	store   *records.Store
	machine *conversionMachine
	pod     *fakePod
	mu      sync.Mutex
	grants  []map[string]any
	source  string
}

func newConversionFixture(t *testing.T) *conversionFixture {
	f := &conversionFixture{}
	row := map[string]any{}
	root, _, _, source, _ := runModelCatalog(t, conversionRental(&row, &f.grants, &f.mu))
	f.root, f.source = root, source
	raw, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	origin := strings.TrimSpace(strings.TrimPrefix(strings.Split(string(raw), "\n")[0], "tensorhub_url:"))
	var problem *exit.Error
	f.layout, problem = home.Open(root)
	fatal(t, problem)
	f.store, problem = records.Open(f.layout.DB)
	fatal(t, problem)
	retainedIngest(t, f.store)
	owed, problem := f.store.RentalHasMachineObligations(podRental)
	fatal(t, problem)
	live, problem := f.store.RentalHasLiveMachineExecutions(podRental)
	fatal(t, problem)
	if !owed || live {
		t.Fatalf("the retained ingest should be custody only: owed=%v live=%v", owed, live)
	}
	// Its converted model holds pod disk that a later ingest there cannot use.
	retained, problem := f.store.RentalRetainedModelBytes(podRental)
	fatal(t, problem)
	var candidate orchestrator.PlacementCandidate
	if retained != 7_000_000_000 || rental.Standing(&candidate, nil, records.Rental{ID: podRental, State: "ready"}, 0, false, false, true, nil,
		rental.Disk{HaveGB: 30, RetainedBytes: retained, SourceBytes: 2_000_000_000}) || !strings.Contains(candidate.Verdict, "30 GB disk with 7 GB retained") {
		t.Fatalf("placement does not count the retained ingest's disk: retained=%d verdict=%q", retained, candidate.Verdict)
	}
	identity, problem := installkey.Ensure(f.layout.Root)
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	f.machine = &conversionMachine{publishes: true, states: map[string]*pb.MachineExecutionState{},
		events: map[string][]*pb.MachineExecutionEvent{}, outcomes: map[string]*pb.AttemptOutcome{}}
	f.pod = &fakePod{controlKey: public, machine: f.machine, derivedRetain: f.machine.retain,
		releases: map[string]*pb.DescribedRelease{"proof/quantize": {Package: "proof/quantize", Release: "1.0.0", PackageInterface: []byte(conversionInterface)}},
		preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
			placement := podPlacement(download, pkg, release, "quantize")
			placement.PackageInterface = []byte(conversionInterface)
			return placement
		}}
	connection, certPath := startFakePod(t, root, f.pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	f.mu.Lock()
	row = map[string]any{"rental_id": podRental, "name": "tessa", "state": "ready", "requested_accelerator_model": "fake-4090",
		"accelerator_count": 1, "hourly_rate_usd_micros": 1, "worker_address": connection.Addr, "media_address": connection.Media.Addr,
		"worker_id": podWorkerID, "worker_boot_id": podBootID, "cert_pem": string(cert)}
	f.mu.Unlock()
	fatal(t, rental.Attach(f.layout, f.store, records.Rental{ID: podRental, MachineName: "tessa", SKU: "cpu", State: "ready",
		AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: origin,
		Address: connection.Addr, MediaAddress: connection.Media.Addr,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token))
	return f
}

// start hands the home to its daemon.
func (f *conversionFixture) start(t *testing.T) {
	f.store.Close()
	startDaemonProcess(t, f.root)
}

func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	must(t, err)
	return string(raw)
}

// installConversionPackage installs an editable package whose conversion job declares one
// Model input and one weights output, and returns the interface its source describes.
func installConversionPackage(t *testing.T, layout home.Layout, store *records.Store) []byte {
	return installLocalPackage(t, layout, store, "conversion-proof", "convert", `from cozy_runtime.author import App, Context, Model, ModelArtifact, WeightsOutput, invocable

app = App()


class Source(Model[object]):
    def load(self, loader):
        del loader


@invocable(memoize=True)
async def quantize(ctx: Context, *, source: Source, steps: int) -> ModelArtifact:
    raise NotImplementedError


app.job(quantize, weights=(WeightsOutput("fp8", max_new_bytes=1 << 20),))
`)
}

// installLocalPackage installs local/<name>@1.0.0, an editable package of one module, and
// returns the interface its source describes.
func installLocalPackage(t *testing.T, layout home.Layout, store *records.Store, name, module, source string) []byte {
	t.Helper()
	cfg, problem := config.Load()
	fatal(t, problem)
	project := filepath.Join(t.TempDir(), "project")
	must(t, os.MkdirAll(project, 0o700))
	files := map[string]string{
		"pyproject.toml": "[project]\nname = \"" + name + "\"\nversion = \"1.0.0\"\nrequires-python = \">=3.12,<3.13\"\ndependencies = []\n" +
			"[build-system]\nrequires = [\"hatchling\"]\nbuild-backend = \"hatchling.build\"\n[tool.hatch.build.targets.wheel]\nonly-include = [\"" + module + ".py\"]\n",
		"package.toml":    "[application]\nobject='" + module + ":app'\n",
		".python-version": "3.12\n",
		module + ".py":    source,
	}
	for file, body := range files {
		must(t, os.WriteFile(filepath.Join(project, file), []byte(body), 0o600))
	}
	lock := exec.Command("uv", "lock", "--offline", "--python", "3.12")
	lock.Dir, lock.Env = project, cfg.Tool()
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("lock: %s: %s", err, out)
	}
	result, problem := install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/" + name},
		Local: &install.LocalSource{Tree: project, Package: "local/" + name, Release: "1.0.0"}})
	fatal(t, problem)
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(result.Install.Dir))
	fatal(t, problem)
	return surface.Raw
}

// An unpublished package's conversion on the same rental is one release root naming its
// installation: its captured code is installed on the pod, and the root carries the job, its
// input and the destination, with the destination's grant.
func TestRentedConversionOfAnUnpublishedPackagePublishesFromTheMachine(t *testing.T) {
	f := newConversionFixture(t)
	f.machine.jobRoots = true
	prepared := servePrivateCode(t, f, installConversionPackage(t, f.layout, f.store))
	f.start(t)
	code, out := runCozy(t, f.root, "run", "local/conversion-proof/quantize", "proof/source@1.0.0/bf16", "proof/output",
		"steps=7", "--rental=tessa", "--await", "--json", "--idempotency-key", "unpublished-conversion")
	if code != 0 || !strings.Contains(out, childDigest("c")) {
		t.Fatalf("the unpublished conversion did not publish [exit %d]: %s\n%s", code, out, tail(filepath.Join(f.root, "daemon.log")))
	}
	store, problem := records.Open(f.layout.DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("unpublished-conversion")
	fatal(t, problem)
	submissions := f.machine.submitted()
	if len(submissions) != 1 || submissions[0].ReleaseRoot == nil || submissions[0].PublicationAuthorizationId == "" {
		t.Fatalf("the unpublished conversion is not one release root with its grant: %+v", submissions)
	}
	if root := submissions[0].ReleaseRoot; request.LocalInstallationID == "" || root.InstallationId != request.LocalInstallationID ||
		root.Release != "" || !root.Job || root.WeightsDestination != "proof/output" || len(root.Models) == 0 || prepared.Load() == 0 {
		t.Fatalf("the unpublished conversion does not name its installation, input and destination: installation %q, prepared %d, root %+v",
			request.LocalInstallationID, prepared.Load(), root)
	}
}

// servePrivateCode has the pod receive the captured wheels and source archive as it does any
// private code, and answers how many times it was asked to prepare them.
func servePrivateCode(t *testing.T, f *conversionFixture, surface []byte) *atomic.Int32 {
	t.Helper()
	prepared := &atomic.Int32{}
	f.pod.localUpload = func(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		header := frame.GetHeader()
		if header == nil || header.File == nil {
			return status.Error(codes.InvalidArgument, "header required")
		}
		if err := f.pod.verifyClaim(header.Claim, false); err != nil {
			return err
		}
		state := &pb.LocalPackageFileStatus{OperationId: header.OperationId, Digest: header.File.Digest, Filename: header.File.Filename,
			Length: header.File.Length, State: pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING}
		if err := stream.Send(state); err != nil {
			return err
		}
		for state.ReceivedBytes < header.File.Length {
			frame, err := stream.Recv()
			if err != nil {
				return err
			}
			state.ReceivedBytes += uint64(len(frame.GetChunk().GetData()))
			if state.ReceivedBytes == header.File.Length {
				state.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
			}
			if err := stream.Send(state); err != nil {
				return err
			}
		}
		return nil
	}
	f.pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := f.pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		prepared.Add(1)
		selected := call.LocalPackageSet.Package
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, InstalledPackage: &pb.InstalledPackage{
			InstallationId: selected.InstallationId, Package: selected.Package, Release: selected.Release, PackageInterface: surface}})
	}
	f.pod.privatePrepare = func(call *pb.PreparePrivatePlacementCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := f.pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		raw, digest, err := canonical.Identity(&pb.PlacementSet{})
		if err != nil {
			return err
		}
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED,
			PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: digest, PlacementSetCanonicalBytes: raw}})
	}
	return prepared
}

// An unpublished package's provider-source job is one root naming its installation: the pod
// is given the captured code, and the machine makes the source for it; nothing takes the
// classic transfer lane.
func TestAnUnpublishedPackagesProviderSourceIsOneRootNamingItsInstallation(t *testing.T) {
	f := newConversionFixture(t)
	f.machine.jobRoots = true
	prepared := servePrivateCode(t, f, installConversionPackage(t, f.layout, f.store))
	f.start(t)
	source := "hf://example/weights@" + strings.Repeat("c", 40)
	code, out := runCozy(t, f.root, "run", "local/conversion-proof/quantize", "model.source="+source, "steps=7",
		"--rental=tessa", "--json", "--idempotency-key", "local-source")
	if code != 0 {
		t.Fatalf("the unpublished source job was refused [exit %d]: %s\n%s", code, out, tail(filepath.Join(f.root, "daemon.log")))
	}
	var root *pb.ReleaseRoot
	waitFor(t, f.root, "the root naming the installation", func() bool {
		f.machine.mu.Lock()
		defer f.machine.mu.Unlock()
		for _, submitted := range f.machine.submissions {
			root = submitted.ReleaseRoot
		}
		return root != nil
	})
	store, problem := records.Open(f.layout.DB)
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("local-source")
	fatal(t, problem)
	if row.ModelTransfer != nil || row.LocalInstallationID == "" || root.InstallationId != row.LocalInstallationID ||
		root.Release != "" || !root.Job || len(root.Models) != 1 || root.Models[0].Source != source || prepared.Load() == 0 {
		t.Fatalf("the unpublished source job is not one root naming its installation: transfer %+v, installation %q, prepared %d, root %+v",
			row.ModelTransfer, row.LocalInstallationID, prepared.Load(), root)
	}
}

// F18 on a machine that takes jobs by their release: `cozy run <pkg/fn> <input> <org/model>
// --rental=<pod>` is one root naming the job, its input choice and the upload destination, and
// its checkpoint is published, for a release input and an exact checkpoint input alike.
func TestRentedConversionOnAMachineThatTakesJobsByRelease(t *testing.T) {
	f := newConversionFixture(t)
	f.machine.jobRoots = true
	f.start(t)
	for index, input := range []string{"proof/source@1.0.0/bf16", "proof/source#" + f.source} {
		code, out := runCozy(t, f.root, "run", "proof/quantize/quantize", input, "proof/output",
			fmt.Sprintf("steps=%d", 10+index), "model.base=proof/source@1.0.0/bf16", "--rental=tessa", "--await", "--json")
		if code != 0 || !strings.Contains(out, childDigest("c")) {
			t.Fatalf("%s: the release-root conversion did not publish [exit %d]: %s\n%s", input, code, out, tail(filepath.Join(f.root, "daemon.log")))
		}
		submissions := f.machine.submitted()
		root := submissions[len(submissions)-1].ReleaseRoot
		if len(submissions) != index+1 || root == nil || !root.Job || root.WeightsDestination != "proof/output" ||
			len(root.Models) != 2 || submissions[len(submissions)-1].PublicationAuthorizationId == "" {
			t.Fatalf("%s: the conversion is not one release root with its destination and grant: %+v", input, root)
		}
	}
}
