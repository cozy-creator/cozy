package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

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
