package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Real independent progress/control streams feed the owner. The built CLI reads
// its authenticated API before settlement, after settlement, and from a new daemon.
func TestRunListRetainsTerminalOverallProgress(t *testing.T) {
	for _, test := range []struct {
		name     string
		status   pb.OutcomeStatus
		progress string
		want     *float64
	}{
		{"completed", pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "overall", progressFraction(1)},
		{"failed", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "overall", progressFraction(.42)},
		{"recovered_open", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "overall", progressFraction(.42)},
		{"zero", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "zero", progressFraction(0)},
		{"canceled", pb.OutcomeStatus_OUTCOME_STATUS_CANCELED, "overall", progressFraction(.42)},
		{"stage_only", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "stage", nil},
		{"absent", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "none", nil},
		{"other_attempt", pb.OutcomeStatus_OUTCOME_STATUS_FAILED, "other", nil},
	} {
		t.Run(test.name, func(t *testing.T) {
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			type invocation struct {
				offer *pb.AttemptOffer
				send  func(*pb.WorkerFrame) error
			}
			offered := make(chan invocation, 1)
			frames := make(chan *pb.AttemptProgress, 1)
			acked := make(chan struct{}, 4)
			var current *pb.AttemptOffer
			pod := &fakePod{controlKey: public, serve: true}
			pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
				if offer := frame.GetAttemptOffer(); offer != nil {
					current = offer
					err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptAccepted{AttemptAccepted: &pb.AttemptAccepted{
						RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
						WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
						InvocationSpecDigest: offer.InvocationSpecDigest, PlacementId: offer.PlacementId,
					}}})
					offered <- invocation{offer, send}
					return true, err
				}
				if frame.GetCancelAttempt() != nil {
					return true, send(privateAttemptOutcome(current, pb.OutcomeStatus_OUTCOME_STATUS_CANCELED,
						pb.CauseCode_CAUSE_CODE_DRAIN_CANCEL, pb.CauseOrigin_CAUSE_ORIGIN_RECORD_OWNER))
				}
				if frame.GetOutcomeAck() != nil {
					acked <- struct{}{}
					return true, nil
				}
				return false, nil
			}
			pod.watchProgress = func(open *pb.ProgressOpen, stream pb.WorkerControl_WatchProgressServer) error {
				for {
					select {
					case frame := <-frames:
						frame.RecordOwnerEpoch, frame.ControlStreamEpoch, frame.WorkerBootId = open.RecordOwnerEpoch, open.ControlStreamEpoch, open.WorkerBootId
						if err := stream.Send(frame); err != nil {
							return err
						}
					case <-stream.Context().Done():
						return nil
					}
				}
			}
			connection, _ := startFakePod(t, t.TempDir(), pod)
			o := hostOwner(t, "terminal-progress-"+test.name, rentalWiring(connection, private))
			stopAPI := publicationControlAPI(t, o)
			defer stopAPI()
			const pkg = "proof/progress"
			id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: test.name, Package: pkg,
				Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: []byte(`{}`),
				Worker: podRental, Rental: true, RentalRequired: true})
			fatal(t, problem)
			var call invocation
			select {
			case call = <-offered:
			case <-time.After(10 * time.Second):
				t.Fatal("worker did not receive the request")
			}
			if test.name == "recovered_open" {
				waitUntil(t, "accepted attempt before recovery", func() bool {
					row, problem := o.store.AttemptRow(id, int64(call.offer.AttemptOrdinal))
					fatal(t, problem)
					return row != nil && row.State == "accepted"
				})
				row, problem := o.store.AttemptRow(id, int64(call.offer.AttemptOrdinal))
				fatal(t, problem)
				fatal(t, o.store.Recover(id, row.Attempt, row.SessionID))
			}
			if test.progress != "none" {
				fields := map[string]any{"stage": "denoise", "stage_fraction": .9, "step_ms": 20}
				if test.progress != "stage" {
					fields["overall_fraction"] = .42
					if test.progress == "zero" {
						fields["overall_fraction"] = 0.0
					}
				}
				data, err := json.Marshal(map[string]any{"type": "progress", "payload": fields})
				must(t, err)
				attempt := call.offer.AttemptOrdinal
				if test.progress == "other" {
					attempt++
				}
				frames <- &pb.AttemptProgress{RequestId: id, AttemptOrdinal: attempt, Seq: 1, Data: data}
				waitUntil(t, "real progress observation", func() bool {
					_, ok := o.c.LatestProgress(id, attempt)
					return ok
				})
			}
			readList := func() api.Lifecycle {
				t.Helper()
				code, out := runCozy(t, o.root, "--json", "--full", "run", "list", "--limit", "1")
				var document struct {
					Invocations []api.Lifecycle `json:"invocations"`
				}
				if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Invocations) != 1 {
					t.Fatalf("run list [exit %d]: %s", code, out)
				}
				return document.Invocations[0]
			}
			live := readList()
			if test.name == "recovered_open" {
				row, problem := o.store.AttemptRow(id, int64(call.offer.AttemptOrdinal))
				fatal(t, problem)
				if row.State != "recovered_open" || row.Attempt != int64(call.offer.AttemptOrdinal) {
					t.Fatalf("recovery changed state/ordinal before readback: %s #%d", row.State, row.Attempt)
				}
			}
			wantLive := .42
			if test.progress == "zero" {
				wantLive = 0
			}
			if (test.progress == "overall" || test.progress == "zero") && (live.OverallFraction == nil || *live.OverallFraction != wantLive) {
				t.Fatalf("live overall lost or replaced by stage fraction: %+v", live)
			}
			if test.progress != "overall" && test.progress != "zero" && live.OverallFraction != nil {
				t.Fatalf("invented overall progress from stage/another attempt: %+v", live)
			}
			if test.status == pb.OutcomeStatus_OUTCOME_STATUS_CANCELED {
				code, out := runCozy(t, o.root, "run", "cancel", fmt.Sprint(live.Number))
				if code != 0 {
					t.Fatalf("run cancel [exit %d]: %s", code, out)
				}
			} else {
				cause, origin := pb.CauseCode_CAUSE_CODE_UNSPECIFIED, pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME
				if test.status == pb.OutcomeStatus_OUTCOME_STATUS_FAILED {
					cause, origin = pb.CauseCode_CAUSE_CODE_AUTHOR_EXCEPTION, pb.CauseOrigin_CAUSE_ORIGIN_AUTHOR
				}
				frame := privateAttemptOutcome(call.offer, test.status, cause, origin)
				outcome := frame.GetAttemptOutcome()
				var body pb.AttemptOutcomeBody
				must(t, canonical.Unmarshal(outcome.OutcomeCanonicalBytes, &body))
				body.Metrics = &pb.AttemptMetrics{PeakDeviceMemoryBytes: 7 << 30, RuntimeMs: 42310}
				data, digest, err := canonical.Identity(&body)
				must(t, err)
				outcome.OutcomeCanonicalBytes, outcome.OutcomeDigest = data, digest
				must(t, call.send(frame))
			}
			select {
			case <-acked:
			case <-time.After(10 * time.Second):
				t.Fatal("terminal did not reach durable acknowledgement")
			}
			check := func() {
				t.Helper()
				row := readList()
				if row.Status == "in_progress" || row.StageFraction != nil || row.RemainingMS != nil {
					t.Fatalf("terminal still exposes live telemetry: %+v", row)
				}
				if test.status != pb.OutcomeStatus_OUTCOME_STATUS_CANCELED {
					reader, problem := localapi.Open(o.cfg, daemon.Probe(o.cfg))
					fatal(t, problem)
					detail, problem := reader.Request(id)
					fatal(t, problem)
					if detail.Metrics["peak_vram_bytes"] != float64(7<<30) || detail.Metrics["runtime_ms"] != float64(42310) {
						t.Fatalf("API lost worker-reported terminal metrics: %+v", detail.Metrics)
					}
				}
				if test.want == nil && row.OverallFraction != nil || test.want != nil && (row.OverallFraction == nil || *row.OverallFraction != *test.want) {
					t.Fatalf("terminal overall=%v, want %v", row.OverallFraction, test.want)
				}
				code, text := runCozy(t, o.root, "run", "list", "--limit", "1")
				if code != 0 || !strings.Contains(text, "PROGRESS") {
					t.Fatalf("human progress column missing: %s", text)
				}
				if test.want != nil && !strings.Contains(text, fmt.Sprintf("%.0f%%", *test.want*100)) || test.want == nil && strings.Contains(text, "%") {
					t.Fatalf("human terminal percentage changed: %s", text)
				}
			}
			check()
			events, problem := o.store.EventsAfter(id, 0, 100)
			fatal(t, problem)
			for _, event := range events {
				if event.Type == "request.progress" {
					t.Fatal("lossy progress tick became a durable event")
				}
			}
			stopAPI()
			o.close()
			// A separate daemon owns the reopened SQLite database; no live fanout
			// or progress worker is available to reconstruct the saved percentage.
			_ = startDaemonProcess(t, o.root)
			check()
		})
	}
}

func progressFraction(value float64) *float64 { return &value }

// A CLI update can precede its daemon update. Success is already authoritative
// in the old response; failure without a saved percentage remains unknown.
func TestRunListCompletedProgressWithOldDaemon(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/requests" {
			t.Errorf("unexpected daemon route: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"requests": []api.Lifecycle{
			{Number: 1, Status: "completed", Package: "proof/old-completed", Function: "run"},
			{Number: 2, Status: "failed", Package: "proof/old-failed", Function: "run"},
		}})
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	code, output := runCozy(t, root, "run", "list", "--full")
	if code != 0 {
		t.Fatalf("old daemon list [exit %d]: %s", code, output)
	}
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, "proof/old-completed/run") && !strings.Contains(line, "100%") {
			t.Fatalf("old daemon success did not render 100%%: %s", output)
		}
		if strings.Contains(line, "proof/old-failed/run") && strings.Contains(line, "%") {
			t.Fatalf("old daemon failure invented progress: %s", output)
		}
	}
	if !strings.Contains(output, "proof/old-completed/run") || !strings.Contains(output, "proof/old-failed/run") {
		t.Fatalf("old daemon rows absent: %s", output)
	}
}
