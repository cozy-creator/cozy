package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// The pod host's REFUSED event for run 862's failure, as the host now composes it: the
// Runtime's own code and the supervisor's reason, not "refused Runtime disk write".
const refusedWheel = "runtime compatibility check failed (Internal): StorageRefusal: supervisor refused " +
	"locked wheel torch-2.14.0-cp312-cp312-manylinux_2_28_x86_64.whl: fetch sha256:ab12: " +
	"ended at 0 of 851640832 bytes: transfer interrupted: unexpected EOF"

// A rented run whose release its machine cannot prepare fails before any attempt, and
// `cozy run list` and `cozy run watch` both carry the machine's own refusal.
func TestMachinePrepareRefusalReachesRunListAndWatch(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	pod := &fakePod{machine: &runtimeMachine{refusal: "wheel_download_failed: " + refusedWheel}, deviceCount: 4}
	root, _ := rentedLadderMachine(t, h, pod, nil)
	runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json")

	type listed struct {
		Number    int64  `json:"number"`
		ID        string `json:"id"`
		Status    string `json:"status"`
		ErrorType string `json:"error_type"`
		ErrorCode string `json:"error_code"`
		Error     string `json:"error"`
	}
	var run listed
	waitFor(t, root, "the refused run to fail", func() bool {
		var page struct{ Invocations []listed }
		_, out := runCozy(t, root, "run", "list", "--json", "--full")
		if json.Unmarshal([]byte(out), &page) != nil || len(page.Invocations) != 1 {
			return false
		}
		run = page.Invocations[0]
		return run.Status == "failed"
	})
	if run.ErrorType != "machine_execution.refused" || run.Error != "Runtime refused machine execution: wheel_download_failed: "+refusedWheel {
		t.Fatalf("listed failure lost its cause: %+v", run)
	}
	code, out := runCozy(t, root, "run", "list", "--no-watch")
	if code != 0 || !strings.Contains(out, "wheel_download_failed: runtime compatibility check failed (Internal):") || strings.Contains(out, "Runtime refused machine execution:") {
		t.Fatalf("human run list lost the cause or retained its wrapper [%d]:\n%s", code, out)
	}
	code, out = runCozy(t, root, "run", "show", strconv.FormatInt(run.Number, 10), "--json")
	if code != 0 || !strings.Contains(out, run.Error) {
		t.Fatalf("run show changed the full diagnostic [%d]: %s", code, out)
	}
	code, out = runCozy(t, root, "run", "watch", strconv.FormatInt(run.Number, 10), "--json")
	var watched struct {
		Error struct {
			Code    string         `json:"code"`
			Message string         `json:"message"`
			Details map[string]any `json:"details"`
		} `json:"error"`
	}
	if code == 0 || json.Unmarshal([]byte(out), &watched) != nil {
		t.Fatalf("run watch --json of a failed run [%d]: %s", code, out)
	}
	details := watched.Error.Details
	if watched.Error.Code != "failed" ||
		details["error_type"] != "machine_execution.refused" || run.ID == "" || details["request_id"] != run.ID ||
		details["number"] != float64(run.Number) || details["error"] != run.Error ||
		!strings.Contains(watched.Error.Message, refusedWheel) {
		t.Fatalf("run watch --json lost the typed failure: %s", out)
	}
}

// A failed attempt's kept triage bundle is readable from `cozy run watch --json`.
func TestFailedRunWatchCarriesItsTriageBundle(t *testing.T) {
	o := hostOwner(t, "triage-watch")
	fatal(t, o.store.SpawnWorker(records.WorkerProcess{InstanceID: "ins-triage",
		Package: "cozy/sweep", WorkerID: "local", Devices: []string{"cpu"}}))
	id := "req-triage-watch"
	if _, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
		BodyDigest: "sha256:" + sixtyFour("9"), Package: "cozy/sweep", Entrypoint: "generate",
		Payload: []byte("{}"), Outputs: "image"}); problem != nil {
		t.Fatal(problem)
	}
	session, digest := "session-triage", "sha256:"+sixtyFour("a")
	attempt, problem := o.store.Dispatch(records.Attempt{RequestID: id, SessionID: session,
		InstanceID: "ins-triage", InvocationDigest: digest, InvocationCanonical: []byte("{}")})
	fatal(t, problem)
	fatal(t, o.store.OfferDispatch(id, attempt, session))
	fatal(t, o.store.Accepted(id, attempt, session))
	bundle := []byte(`{"terminal":{"traceback":"Traceback\nValueError: bad latent shape"}}`)
	if _, problem := o.store.AcceptTerminal(records.Terminal{RequestID: id, Attempt: attempt,
		SessionID: session, InvocationDigest: digest, TerminalID: "out-triage",
		TerminalDigest: "sha256:" + sixtyFour("f"), Status: "FAILED", Cause: "handler_error",
		TriageSubject: "trb-watch", TriageDigest: "sha256:" + sixtyFour("b"),
		TriageLength: int64(len(bundle)), TriageBundle: bundle, EventType: "run.failed",
		EventPayload: map[string]any{"error_type": "handler_error", "error": "ValueError: bad latent shape"},
		RequestState: "failed"}); problem != nil {
		t.Fatal(problem)
	}
	defer publicationControlAPI(t, o)()
	code, out := runCozy(t, o.root, "run", "watch", id, "--json")
	var watched struct {
		Error struct {
			Details struct {
				Error  string `json:"error"`
				Triage struct {
					SubjectID string `json:"subject_id"`
					Kept      bool   `json:"kept"`
					Bundle    struct {
						Terminal struct {
							Traceback string `json:"traceback"`
						} `json:"terminal"`
					} `json:"bundle"`
				} `json:"triage"`
			} `json:"details"`
		} `json:"error"`
	}
	if code == 0 || json.Unmarshal([]byte(out), &watched) != nil {
		t.Fatalf("run watch --json of a failed run [%d]: %s", code, out)
	}
	details := watched.Error.Details
	if details.Error != "ValueError: bad latent shape" || details.Triage.SubjectID != "trb-watch" ||
		!details.Triage.Kept || !strings.HasSuffix(details.Triage.Bundle.Terminal.Traceback, "bad latent shape") {
		t.Fatalf("run watch --json lost the triage bundle: %s", out)
	}
	// The bundle lives in its attempt row (cl-116): no directory exists for orphans.
	if _, err := os.Stat(filepath.Join(o.root, "triage")); !os.IsNotExist(err) {
		t.Fatalf("a triage directory exists: %v", err)
	}
}

// A pre-attempt failure recorded by an older daemon (no error_code, the old message
// shape; run 862) shows the same reason in `run list` as in `run watch`.
func TestRecordedLegacyFailureReasonIsTheSameInListAndWatch(t *testing.T) {
	o := hostOwner(t, "legacy-failure-reason")
	id := "req-legacy-862"
	if _, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: "idem-" + id,
		BodyDigest: "sha256:" + sixtyFour("9"), Package: "paul/minimax-h3", Entrypoint: "fl2va",
		Payload: []byte("{}"), Outputs: "video"}); problem != nil {
		t.Fatal(problem)
	}
	legacy := "worker rejected desired revision 2 before applying it: runtime_compatibility_failed: " +
		"runtime compatibility check: rpc error: code = Internal desc = StorageRefusal: supervisor refused Runtime disk write"
	applied, problem := o.store.FailQueuedRequest(id, map[string]any{"status": "FAILED",
		"cause": "worker.desired_state_refused", "error_type": "worker.desired_state_refused",
		"error": legacy, "outputs": []any{}, "requeuing": false})
	fatal(t, problem)
	if !applied {
		t.Fatal("the legacy failure was not recorded")
	}
	defer publicationControlAPI(t, o)()

	code, out := runCozy(t, o.root, "run", "list", "--no-watch")
	if code != 0 || !regexp.MustCompile(`(?m)^\d+\s+paul/minimax-h3/fl2va\s.*failed\s.*worker rejected desired revision 2 before applying it: runtime_compatib…$`).MatchString(out) {
		t.Fatalf("the list REASON lost the recorded failure [%d]:\n%s", code, out)
	}
	code, out = runCozy(t, o.root, "run", "list", "--json")
	var listed struct {
		Invocations []struct {
			Number    int64  `json:"number"`
			ErrorType string `json:"error_type"`
			Error     string `json:"error"`
		} `json:"invocations"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &listed) != nil || len(listed.Invocations) != 1 {
		t.Fatalf("run list --json [%d]: %s", code, out)
	}
	run := listed.Invocations[0]
	if run.ErrorType != "worker.desired_state_refused" || run.Error != legacy {
		t.Fatalf("listed failure is not the recorded one: %+v", run)
	}
	code, out = runCozy(t, o.root, "run", "watch", strconv.FormatInt(run.Number, 10), "--json")
	var watched struct {
		Error struct {
			Details map[string]any `json:"details"`
		} `json:"error"`
	}
	if code == 0 || json.Unmarshal([]byte(out), &watched) != nil ||
		watched.Error.Details["error"] != run.Error || watched.Error.Details["error_type"] != run.ErrorType {
		t.Fatalf("run watch disagrees with run list [%d]: %s", code, out)
	}
}
