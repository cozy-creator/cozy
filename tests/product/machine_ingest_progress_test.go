package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/installkey"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// ingestMachine is a stand-in pod running a rented model ingest. Its execution journal
// carries exactly what Runtime journals for native source and upload work: byte samples
// per stage on the progress lane, then one phase record per finished stage.
type ingestMachine struct {
	mu       sync.Mutex
	state    *pb.MachineExecutionState
	outcome  *pb.AttemptOutcome
	events   []*pb.MachineExecutionEvent
	released int
}

func (m *ingestMachine) release(count int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.released = min(count, len(m.events))
	m.state.Sequence = uint64(m.released)
	if m.released == len(m.events) {
		m.state.State = "succeeded"
	}
}

func (m *ingestMachine) GetMachineExecution(context.Context, *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *ingestMachine) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	page := &pb.MachineExecutionEventPage{NextAfter: query.After, HeadSequence: uint64(m.released)}
	for _, event := range m.events[:m.released] {
		if event.Sequence > query.After && uint32(len(page.Events)) < query.Limit {
			page.Events = append(page.Events, event)
			page.NextAfter = event.Sequence
		}
	}
	return page, nil
}

func (m *ingestMachine) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state.Collected = true
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// ingestJournal is the Runtime lane of one ingest: download, conversion, upload and
// publication, each with its samples and its closing phase record.
func ingestJournal(t *testing.T) []*pb.MachineExecutionEvent {
	const gib = int64(1) << 30
	start := time.Now().UnixMilli()
	var events []*pb.MachineExecutionEvent
	add := func(kind string, document map[string]any) {
		body, err := json.Marshal(document)
		must(t, err)
		events = append(events, &pb.MachineExecutionEvent{Sequence: uint64(len(events) + 1), AttemptOrdinal: 1,
			AtMs: uint64(start + int64(len(events))*1000), Kind: kind, BodyCanonicalBytes: body})
	}
	sample := func(stage string, fields map[string]any) {
		payload := map[string]any{"stage": stage, "step_ms": 0.0}
		for key, value := range fields {
			payload[key] = value
		}
		add("progress", map[string]any{"type": "progress", "payload": payload})
	}
	phase := func(stage string, elapsedMS float64, moved int64) {
		began := start + int64(len(events))*1000
		fields := map[string]any{"phase": stage, "completed": true, "elapsed_ms": elapsedMS,
			"started_unix_ms": began}
		if moved > 0 {
			fields["bytes"], fields["total_bytes"], fields["moved_bytes"] = moved, moved, moved
			fields["rate_bytes_per_second"] = float64(moved) / (elapsedMS / 1000)
		}
		add("log", map[string]any{"type": "log", "payload": map[string]any{"name": stage, "value": "info",
			"at_unix_ms": began + int64(elapsedMS), "fields": fields}})
	}
	for _, stage := range []string{"Downloading source", "Converting to cozytensors", "Uploading checkpoint"} {
		sample(stage, nil)
		// A multi-hour stage journals a sample a second; more than run show's evidence
		// bound precede the download's phase record.
		for range map[bool]int{true: 4200, false: 1}[stage == "Downloading source"] {
			sample(stage, map[string]any{"position": 400 << 20, "total": gib, "unit": "bytes", "rate": float64(100 << 20)})
		}
		sample(stage, map[string]any{"position": gib, "total": gib, "unit": "bytes", "rate": float64(120 << 20)})
		phase(stage, 9500, gib)
	}
	sample("Publishing checkpoint", nil)
	phase("Publishing checkpoint", 1200, 0)
	return events
}

// A rented ingest reports each stage's bytes done/total and measured rate through the
// ordinary run surfaces: list and show while it runs, the watch stream, and per-stage
// timings in `cozy run show` afterwards.
func TestRentedIngestReportsStageBytesAndRate(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := installkey.Ensure(layout.Root)
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	request, _, problem := store.Submit(records.Request{ID: "job-ingest-progress", IdemKey: "ingest-progress",
		Package: "local/upload_model", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "persistent-workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "persistent-workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	machine := &ingestMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "persistent-workspace", Generation: 1, AttemptOrdinal: 1, State: "running"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "ingest-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
		events: ingestJournal(t),
	}
	machine.events = append(machine.events, outcomeEvent(uint64(len(machine.events)+1), "succeeded", machine.outcome))
	machine.release(3)

	pod := &fakePod{controlKey: public, machine: machine}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "ingester")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	row := records.Rental{ID: podRental, MachineName: "ingester", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: ingest-progress-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)

	// Mid-download: the list row carries the stage, its byte coordinates and the rate.
	var running api.Lifecycle
	waitUntil(t, "the download sample in run list", func() bool {
		code, out := runCozy(t, root, "run", "list", "--json", "--full")
		var listed struct {
			Invocations []api.Lifecycle `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || len(listed.Invocations) != 1 {
			return false
		}
		running = listed.Invocations[0]
		return running.Position != nil
	})
	if running.Status != "in_progress" || running.ProgressStage != "Downloading source" || *running.Position != 400<<20 ||
		running.Total == nil || *running.Total != 1<<30 || running.ProgressUnit != "bytes" ||
		running.ProgressRate == nil || *running.ProgressRate != 100<<20 || running.OverallFraction != nil {
		t.Fatalf("run list lost the download's bytes or rate: %+v", running)
	}
	code, listed := runCozy(t, root, "run", "list")
	t.Logf("cozy run list:\n%s", listed)
	if code != 0 || !strings.Contains(listed, "Downloading source · 400.0MiB / 1.0GiB · 100.0MiB/s") {
		t.Fatalf("human run list omitted the byte stage [%d]: %s", code, listed)
	}

	machine.release(len(machine.events))
	code, stdout, stderr := runCozyStreams(t, root, "run", "watch", request.ID, "--json")
	if code != 0 || !strings.Contains(stdout, `"status":"completed"`) {
		t.Fatalf("watch did not follow the ingest [%d]: %s\n%s", code, stdout, stderr)
	}
	for _, stage := range []string{"Downloading source", "Converting to cozytensors", "Uploading checkpoint", "Publishing checkpoint"} {
		if !strings.Contains(stderr, fmt.Sprintf(`"stage":%q`, stage)) {
			t.Fatalf("awaited JSON stream lacks %s progress:\n%s", stage, stderr)
		}
	}
	if !strings.Contains(stderr, `"type":"machine.progress"`) || !strings.Contains(stderr, `"unit":"bytes"`) {
		t.Fatalf("awaited JSON stream lacks byte samples:\n%s", stderr)
	}
	code, human := runCozy(t, root, "run", "watch", request.ID)
	t.Logf("cozy run watch:\n%s", human)
	for _, want := range []string{
		"Downloading source · 400.0MiB / 1.0GiB · 39% stage · 100.0MiB/s · ETA ~6s",
		"Converting to cozytensors · 1.0GiB / 1.0GiB · 100% stage · 120.0MiB/s",
		"Uploading checkpoint", "Publishing checkpoint",
	} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("human watch lacks %q [%d]:\n%s", want, code, human)
		}
	}

	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var report struct {
		Stages []struct {
			Name   string  `json:"name"`
			Kind   string  `json:"kind"`
			MS     float64 `json:"ms"`
			Bytes  int64   `json:"bytes"`
			Detail string  `json:"detail"`
		} `json:"stages"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil {
		t.Fatalf("run show --json [%d]: %s", code, out)
	}
	var names []string
	for _, stage := range report.Stages {
		names = append(names, stage.Name)
		if stage.Kind != "phase" || stage.MS <= 0 {
			t.Fatalf("stage %q lost its timing: %+v", stage.Name, stage)
		}
		if stage.Name != "Publishing checkpoint" && (stage.Bytes != 1<<30 || stage.Detail != "1.0GiB of 1.0GiB at 107.8MiB/s") {
			t.Fatalf("stage %q lost its bytes or rate: %+v", stage.Name, stage)
		}
	}
	if strings.Join(names, ",") != "Downloading source,Converting to cozytensors,Uploading checkpoint,Publishing checkpoint" {
		t.Fatalf("run show stages: %v", names)
	}
	code, human = runCozy(t, root, "run", "show", request.ID)
	t.Logf("cozy run show:\n%s", human)
	if code != 0 || !strings.Contains(human, "Uploading checkpoint") || !strings.Contains(human, "9.5s") {
		t.Fatalf("human run show lacks the phase timings [%d]:\n%s", code, human)
	}
}
