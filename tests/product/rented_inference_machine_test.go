package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// runtimeMachine is the Runtime side of one rented machine: it accepts a submission, holds
// the call waiting for GPUs behind `blocker`, and once released finishes and hands back
// its result.
type runtimeMachine struct {
	mu         sync.Mutex
	blocker    string
	triage     []byte // the bundle Runtime's terminal names; the Host reads it for the owner
	submission *pb.MachineExecutionSubmit
	receipt    *pb.MachineExecutionReceipt
	events     []*pb.MachineExecutionEvent
	state      *pb.MachineExecutionState

	// cpuSlotModelInputs, exactGPUs and devices are what this Runtime reports in its workspace.
	cpuSlotModelInputs bool
	exactGPUs          bool
	devices            int
	sourceCredentials  bool

	submissions []*pb.MachineExecutionSubmit // every submission as sent, resubmissions included
	failure     string                       // a failed terminal's safe message; empty succeeds
}

func (m *runtimeMachine) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	workspace := &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId,
		ExecutionWorkspaceId: "rented-workspace", CpuSlotModelInputs: m.cpuSlotModelInputs, ExactExecutionGpus: m.exactGPUs,
		SourceCredentials: m.sourceCredentials}
	for ordinal := range m.devices {
		workspace.Devices = append(workspace.Devices, &pb.MachineDevice{Ordinal: uint32(ordinal), Name: "fake-4090"})
	}
	return workspace, nil
}

// runtimeExecutionID is Runtime's execution identity grammar (workspace_executions.py `_ID`).
var runtimeExecutionID = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$`)

func (m *runtimeMachine) SubmitMachineExecution(_ context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !runtimeExecutionID.MatchString(submit.SubmissionId) {
		return nil, status.Error(codes.InvalidArgument, "execution identity must be a bounded opaque ID")
	}
	m.submissions = append(m.submissions, proto.Clone(submit).(*pb.MachineExecutionSubmit))
	if m.receipt != nil {
		return m.receipt, nil
	}
	id := submit.Offer.RequestId
	m.submission = proto.Clone(submit).(*pb.MachineExecutionSubmit)
	m.receipt = &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: submit.CaptureDigest,
		InvocationSpecDigest: submit.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}
	m.state = &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
		ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: "running"}
	wait, _ := json.Marshal(map[string]any{"key": id + "#1", "width": 4, "blocked_by": []string{m.blocker}})
	m.record("running", []byte(`{}`))
	m.record("gpu.wait", wait)
	return m.receipt, nil
}

func (m *runtimeMachine) record(kind string, body []byte) {
	m.state.Sequence++
	m.events = append(m.events, &pb.MachineExecutionEvent{Sequence: m.state.Sequence, AttemptOrdinal: 1,
		AtMs: uint64(time.Now().UnixMilli()), Kind: kind, BodyCanonicalBytes: body})
}

// finish grants the GPUs and ends the call.
func (m *runtimeMachine) finish() {
	m.mu.Lock()
	defer m.mu.Unlock()
	grant, _ := json.Marshal(map[string]any{"key": m.state.RequestId + "#1", "ordinals": []int{0, 1, 2, 3}})
	m.record("gpu.grant", grant)
	release, _ := json.Marshal(map[string]any{"key": m.state.RequestId + "#1", "ordinals": []int{0, 1, 2, 3}, "cause": "exited"})
	m.record("gpu.release", release)
	m.state.State = "succeeded"
	if m.failure != "" {
		m.state.State = "failed"
	}
}

func (m *runtimeMachine) status() pb.OutcomeStatus {
	if m.failure == "" {
		return pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED
	}
	return pb.OutcomeStatus_OUTCOME_STATUS_FAILED
}

func (m *runtimeMachine) cause() *pb.OutcomeCause {
	if m.failure == "" {
		return nil
	}
	return &pb.OutcomeCause{Code: pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, Detail: m.failure}
}

func (m *runtimeMachine) submitted() *pb.MachineExecutionSubmit {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.submission
}

func (m *runtimeMachine) GetMachineExecution(context.Context, *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *runtimeMachine) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	page := &pb.MachineExecutionEventPage{NextAfter: m.state.Sequence, HeadSequence: m.state.Sequence}
	for _, event := range m.events {
		if event.Sequence > query.After {
			page.Events = append(page.Events, event)
		}
	}
	return page, nil
}

func (m *runtimeMachine) ReadMachineExecutionTriage(_ context.Context, query *pb.MachineExecutionTriageQuery) (*pb.MachineExecutionTriage, error) {
	return &pb.MachineExecutionTriage{Bundle: &pb.TriageBundleRef{SubjectId: "trb-rented", WriteReceiptDigest: canonical.Digest(m.triage),
		Length: uint64(len(m.triage))}, BundleCanonicalBytes: m.triage}, nil
}

func (m *runtimeMachine) CollectMachineExecution(context.Context, *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	spec, _ := canonical.Spell(m.receipt.InvocationSpecDigest)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: m.state.RequestId, AttemptOrdinal: 1,
		InvocationSpecDigest: spec, Status: m.status(),
		Result: &pb.ResultEnvelope{InlineResult: []byte(`{}`)}, Metrics: &pb.AttemptMetrics{RuntimeMs: 1, WorkingPeakDeviceBytes: 30 << 30},
		TriageBundle: &pb.TriageBundleRef{SubjectId: "trb-rented", WriteReceiptDigest: canonical.Digest(m.triage),
			Length: uint64(len(m.triage))}, SafeMessage: m.failure, Cause: m.cause()})
	if err != nil {
		return nil, err
	}
	return &pb.AttemptOutcome{RequestId: m.state.RequestId, AttemptOrdinal: 1, InvocationSpecDigest: m.receipt.InvocationSpecDigest,
		OutcomeId: "rented-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body}, nil
}

func (m *runtimeMachine) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state.Collected = true
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// Rented inference is a Runtime execution. `cozy run --rental` submits the root against the
// placement the pod prepared, sends that worker no desired state and no device pin — Runtime
// alone picks the GPUs and the width — and a run Runtime holds for GPUs says so, naming the
// run ahead of it.
func TestRentedInferenceIsARuntimeExecution(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &runtimeMachine{triage: bundle}
	pod := &fakePod{machine: machine, deviceCount: 4,
		mediaRequest: func(w http.ResponseWriter, r *http.Request) bool {
			if r.Method != http.MethodGet || r.URL.Path != "/v1/triage/trb-rented" {
				return false
			}
			_, _ = w.Write(bundle)
			return true
		}}
	root, layout := rentedLadderMachine(t, h, pod, func(_ home.Layout, store *records.Store) {
		holder, _, problem := store.Submit(records.Request{ID: "req-gpu-holder", IdemKey: "gpu-holder", BodyDigest: childDigest("7"),
			Package: ladderPackage, Entrypoint: "generate", Payload: []byte(`{}`)})
		fatal(t, problem)
		_, problem = store.FailQueuedRequest(holder.ID, map[string]any{"error_type": "proof", "error": "held elsewhere"})
		fatal(t, problem)
		machine.blocker = holder.ID
	})

	// A resume key names its predecessor with slashes; the machine sees an opaque identity.
	const key = "rented-inference/retry-of/job-1"
	code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json",
		"--idempotency-key", key)
	if code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	list := ""
	for deadline := time.Now().Add(20 * time.Second); !strings.Contains(list, "waiting for GPU (needs 4, behind run 1)"); {
		if time.Now().After(deadline) {
			_, typed := runCozy(t, root, "run", "list", "--json", "--full")
			t.Fatalf("run list does not show the GPU wait:\n%s\n%s", list, typed)
		}
		_, list = runCozy(t, root, "run", "list")
	}
	if _, show := runCozy(t, root, "run", "show", "2"); !strings.Contains(show, "waiting for GPU (needs 4, behind run 1)") {
		t.Fatalf("run show does not say the run waits for GPUs:\n%s", show)
	}
	machine.finish()
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var row *records.Request
	waitFor(t, root, "the collected result", func() bool {
		row, problem = store.RequestByIdempotencyKey(key)
		link, linkProblem := store.MachineExecution(row.ID)
		return problem == nil && linkProblem == nil && row.State == "succeeded" && link != nil && link.Collected
	})

	submission := machine.submitted()
	set := submission.PreparedState.GetPlacementSet()
	if set == nil || len(set.DevicePins) != 0 || set.ExecutionGpus != 0 {
		t.Fatalf("the root was not submitted against its prepared placement alone: %+v", submission.PreparedState)
	}
	placements, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
	must(t, err)
	if id := placements.List("placements")[0].Str("placement_id"); submission.Offer.PlacementId != id {
		t.Fatalf("the offer names placement %q, not the prepared %q", submission.Offer.PlacementId, id)
	}
	var spec pb.InvocationSpec
	must(t, canonical.Unmarshal(submission.Offer.InvocationSpecCanonicalBytes, &spec))
	if spec.GetServing() == nil || spec.GetServing().EntrypointBindingDigest != row.PlanID {
		t.Fatalf("the root is not the prepared inference binding %s: %+v", row.PlanID, spec.Spec)
	}
	if len(row.Models) != 1 || row.Models[0].Manifest == "" {
		t.Fatalf("the rental's lane was not pinned before preparation: %+v", row.Models)
	}
	// Runtime's measured working memory sizes the next selection.
	peaks, problem := store.WorkingPeaks(*row)
	fatal(t, problem)
	if peak := peaks[records.ModelsDigest(row.Models)]; peak.Bytes != 30<<30 || peak.Runs != 1 {
		t.Fatalf("the measured working peak was not kept: %+v", peaks)
	}
	if _, show := runCozy(t, root, "run", "show", "2"); !strings.Contains(show, "ordinals [0 1 2 3]") {
		t.Fatalf("run show does not name the GPUs Runtime granted:\n%s", show)
	}
	// Runtime's triage bundle is kept, so the run still reads as setup, inference and ranks.
	_, show := runCozy(t, root, "run", "show", "2", "--json")
	var report struct {
		Stages []struct{ Name, Kind string } `json:"stages"`
		Steps  []struct {
			Name  string `json:"name"`
			Count int    `json:"count"`
		} `json:"steps"`
		Ranks []struct {
			PID int `json:"pid"`
		} `json:"ranks"`
	}
	must(t, json.Unmarshal([]byte(show), &report))
	kinds := map[string]string{}
	for _, stage := range report.Stages {
		kinds[stage.Name] = stage.Kind
	}
	if kinds["executor boot"] != "setup" || kinds["condition"] != "inference" ||
		len(report.Steps) != 1 || report.Steps[0].Count != 4 || len(report.Ranks) != 1 || report.Ranks[0].PID <= 0 {
		t.Fatalf("run show lost Runtime's evidence: %s", show)
	}
	pod.mu.Lock()
	desired := len(pod.desired)
	pod.mu.Unlock()
	if desired != 0 {
		t.Fatalf("the Runtime-owned worker was sent %d desired state(s); a full replace retires its replicas", desired)
	}
}

// rentedLadderMachine publishes the ladder package, its model slot declaring `degrees`, and
// attaches `pod` as the ready rental "tessa", as many fake-4090 cards wide as its
// deviceCount, then starts the daemon. The pod prepares the placement its own
// preparedPlacement authors (modelBearingPlacement when nil), the ladder package's with
// its release interface. `before` runs against the home while nothing else holds it.
func rentedLadderMachine(t *testing.T, h *ladderHub, pod *fakePod, before func(home.Layout, *records.Store), degrees ...int) (string, home.Layout) {
	t.Helper()
	publishWorkflowRelease(t, h, degrees...)
	var detail hub.PackageReleaseDetail
	response, err := http.Get(h.server.URL + "/v1/packages/proof/h3/releases/1.0.0")
	must(t, err)
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	facts := testPrepareFacts(ladderPackage, "1.0.0")
	inventory, err := json.Marshal(map[string]any{"format": "tensorhub.image_inventory/1",
		"profile": facts.ImageInventory.Profile, "python": facts.ImageInventory.Python,
		"distributions": []map[string]string{{"name": runtimeDistribution, "version": "0.18.41"}}})
	must(t, err)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rentals/"+podRental+"/prepare-facts" {
			served.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.PrepareFactsView{Application: "h3:app", ModelSlotPaths: []string{ladderSlot},
			ImageInventory: inventory, LockedRequirements: string(facts.LockedRequirements)})
	})

	root := ladderRoot(t, h)
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	if before != nil {
		before(layout, store)
	}
	identity, problem := rental.PendingCreatorIdentity(layout, "rented-inference")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod.controlKey = public
	author := pod.preparedPlacement
	if author == nil {
		author = modelBearingPlacement(t)
	}
	pod.preparedPlacement = func(download []byte, pkg, release string) *pb.Placement {
		placement := author(download, pkg, release)
		if pkg == ladderPackage {
			placement.PackageInterface = detail.PackageInterface
		}
		return placement
	}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	cards := int(max(pod.deviceCount, 1))
	h.mu.Lock()
	h.rentals[podRental] = map[string]any{"rental_id": podRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "fake-4090", "accelerator_count": cards, "hourly_rate_usd_micros": 1,
		"worker_address": connection.Addr, "media_address": connection.Media.Addr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "tessa", SKU: "fake-x", State: "ready",
		AcceleratorModel: "fake-4090", AcceleratorCount: cards, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	store.Close()
	startDaemonProcess(t, root)
	return root, layout
}
