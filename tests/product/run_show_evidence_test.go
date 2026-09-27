package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// One rented run through the real prepare stream, dispatch, output mirror and triage copy:
// `cozy run show` separates the preparation it waited on from the Runtime's own setup and
// inference stages, and names each rank. The bundle is Runtime's own output.
func TestRunShowSeparatesSetupInferenceAndRanks(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, downloadSamples: 3}
	offers := make(chan func(*pb.WorkerFrame) error, 1)
	var offer *pb.AttemptOffer
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		if message := frame.GetAttemptOffer(); message != nil {
			offer = message
			offers <- send
			return true, nil
		}
		return frame.GetOutcomeAck() != nil, nil
	}
	media := []byte("rendered frame bytes")
	pod.mediaRequest = func(w http.ResponseWriter, r *http.Request) bool {
		switch {
		case r.Method == http.MethodGet && strings.HasPrefix(r.URL.Path, "/v1/outputs/"):
			_, _ = w.Write(media)
		case r.Method == http.MethodGet && r.URL.Path == "/v1/triage/trb-evidence":
			_, _ = w.Write(bundle)
		case r.Method == http.MethodDelete:
			_, _ = w.Write([]byte(`{}`))
		default:
			return false
		}
		return true
	}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "run-show-evidence", rentalWiring(connection, private))
	const pkg = "paul/run-show-evidence"
	id, _, problem := o.c.Submit(orchestrator.Submission{IdemKey: "evidence", Package: pkg,
		Release: "1.0.0", Entrypoint: "tile", PlanID: podPlanID(pkg), Payload: []byte(`{}`),
		Outputs: []string{"image"}, Worker: podRental, Rental: true, RentalRequired: true})
	fatal(t, problem)
	waitUntil(t, "the offer", func() bool { return len(offers) == 1 })
	send := <-offers
	identity, err := canonical.Spell(offer.InvocationSpecDigest)
	must(t, err)
	body := &pb.AttemptOutcomeBody{RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: identity, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED,
		ExecutionStarted: true, Cause: &pb.OutcomeCause{Origin: pb.CauseOrigin_CAUSE_ORIGIN_RUNTIME},
		OutputManifest: &pb.OutputManifest{Outputs: []*pb.OutputEntry{{OutputId: "image",
			Digest: canonical.Digest(media), Length: uint64(len(media)), MimeType: "image/png"}}},
		TriageBundle: &pb.TriageBundleRef{SubjectId: "trb-evidence",
			WriteReceiptDigest: canonical.Digest(bundle), Length: uint64(len(bundle))}}
	raw, digest, err := canonical.Identity(body)
	must(t, err)
	must(t, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_AttemptOutcome{AttemptOutcome: &pb.AttemptOutcome{
		RecordOwnerEpoch: offer.RecordOwnerEpoch, ControlStreamEpoch: offer.ControlStreamEpoch,
		WorkerBootId: offer.WorkerBootId, RequestId: offer.RequestId, AttemptOrdinal: offer.AttemptOrdinal,
		InvocationSpecDigest: offer.InvocationSpecDigest, OutcomeId: "out-evidence",
		OutcomeDigest: digest, OutcomeCanonicalBytes: raw}}}))
	waitUntil(t, "the run to settle", func() bool {
		row, e := o.store.RequestRow(id)
		fatal(t, e)
		return row.State == "succeeded"
	})
	defer publicationControlAPI(t, o)()

	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		Status string `json:"status"`
		Stages []struct {
			Name  string  `json:"name"`
			Kind  string  `json:"kind"`
			MS    float64 `json:"ms"`
			Bytes int64   `json:"bytes"`
			Count int     `json:"count"`
		} `json:"stages"`
		Steps []struct {
			Name   string       `json:"name"`
			Count  int          `json:"count"`
			Series [][2]float64 `json:"series"`
		} `json:"steps"`
		Degree int `json:"degree"`
		Ranks  []struct {
			Rank      int   `json:"rank"`
			PID       int   `json:"pid"`
			StartUS   int64 `json:"start_us"`
			EndUS     int64 `json:"end_us"`
			Attention struct {
				Observed string `json:"observed"`
			} `json:"attention"`
		} `json:"ranks"`
		Triage struct {
			Measurements map[string]any `json:"measurements"`
		} `json:"triage"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil {
		t.Fatalf("run show --json [%d]: %s", code, out)
	}
	stages := map[string]string{}
	for _, stage := range report.Stages {
		stages[stage.Name] = stage.Kind
	}
	for name, kind := range map[string]string{"download": "setup", "package environment": "setup",
		"executor boot": "setup", "condition": "inference", "output transfer": "transfer"} {
		if stages[name] != kind {
			t.Fatalf("stage %q is %q, want %q: %+v", name, stages[name], kind, report.Stages)
		}
	}
	for _, stage := range report.Stages {
		if stage.Name == "download" && stage.Bytes <= 0 ||
			stage.Name == "output transfer" && (stage.Bytes != int64(len(media)) || stage.Count != 1) {
			t.Fatalf("stage %q lost its bytes: %+v", stage.Name, stage)
		}
	}
	if len(report.Steps) != 1 || report.Steps[0].Name != "denoise" || report.Steps[0].Count != 4 ||
		len(report.Steps[0].Series) != 4 {
		t.Fatalf("per-step series: %+v", report.Steps)
	}
	if report.Degree != 1 || len(report.Ranks) != 1 || report.Ranks[0].PID <= 0 ||
		report.Ranks[0].StartUS <= 0 || report.Ranks[0].EndUS < report.Ranks[0].StartUS {
		t.Fatalf("ranks: degree %d %+v", report.Degree, report.Ranks)
	}
	if report.Triage.Measurements["execution"] == nil {
		t.Fatalf("JSON dropped the triage bundle: %s", out)
	}

	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	for _, want := range []string{"STAGE", "download", "package environment", "executor boot",
		"condition", "output transfer", "steps denoise: 4 in", "ranks (degree 1)", "unobserved"} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("run show [%d] lacks %q:\n%s", code, want, human)
		}
	}
}

// A bundle written before execution records existed still shows its real stages and
// steps (run 1183, MiniMax H3 on four H100s); nothing is invented for what it lacks.
func TestRunShowReadsAPreRecordBundleTolerantly(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "h3-run-1183-attribution.json"))
	must(t, err)
	o := hostOwner(t, "run-show-legacy")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	code, human := runCozy(t, o.root, "run", "show", id)
	t.Logf("cozy run show:\n%s", human)
	for _, want := range []string{"denoise", "decode_video", "51.1s",
		"steps denoise: 8 in 51.1s; first 11.6s, then mean 5.7s (min 5.6s, max 11.6s)"} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("run show [%d] lacks %q:\n%s", code, want, human)
		}
	}
	if strings.Contains(human, "ranks") {
		t.Fatalf("a bundle with no execution record showed ranks:\n%s", human)
	}
}

// succeededWithTriage records one settled attempt that kept `bundle`, the way the
// orchestrator's terminal transaction does.
func succeededWithTriage(t *testing.T, o *owner, bundle []byte) string {
	t.Helper()
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-evidence",
		Package: "paul/minimax-h3", WorkerID: "local", Devices: []string{"0", "1", "2", "3"}}))
	id := "req-evidence"
	if _, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
		BodyDigest: "sha256:" + sixtyFour("9"), Package: "paul/minimax-h3", Entrypoint: "ref2va_turbo",
		Payload: []byte("{}"), Outputs: "video"}); problem != nil {
		t.Fatal(problem)
	}
	session, digest := "session-evidence", "sha256:"+sixtyFour("a")
	attempt, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: session,
		InstanceID: "ins-evidence", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, attempt, session))
	fatal(t, o.store.Accepted(id, attempt, session))
	if _, problem := o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: attempt,
		SessionID: session, InvocationDigest: digest, TerminalID: "out-evidence",
		TerminalDigest: "sha256:" + sixtyFour("f"), Status: "SUCCEEDED", TriageSubject: "trb-evidence",
		TriageDigest: "sha256:" + sixtyFour("b"), TriageLength: int64(len(bundle)),
		TriageBundle: bundle, EventType: "request.completed", EventPayload: map[string]any{},
		RequestState: "succeeded"}); problem != nil {
		t.Fatal(problem)
	}
	return id
}
