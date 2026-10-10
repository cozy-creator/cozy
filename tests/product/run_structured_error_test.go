package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Exercise the current machine contract through the real store, daemon API and CLI:
// codes stay typed, messages stay verbatim, and human views render the code once.
func TestRunFailureKeepsCodeSeparateFromMessage(t *testing.T) {
	const errorCode = "model_choice_absent"
	for _, message := range []string{
		"the run names no model for generate_image.models.model",
		"other_code: detail mentions model_choice_absent: literally",
	} {
		t.Run(message, func(t *testing.T) {
			o := hostOwner(t, strings.ReplaceAll(message, " ", "_"))
			id := "job-structured-error"
			_, _, problem := o.store.Submit(records.Request{ID: id, IdemKey: id,
				Package: "local/proof", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`),
				BodyDigest: childDigest("b"), MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, o.store.LinkMachineExecution(id, "pr-unreachable"))
			fatal(t, o.store.AcceptRunV1(id, "pr-unreachable", &v1.RunState{Id: id, Number: 1, State: "running", Attempt: 1}))
			stamp := time.Now().UnixMilli()
			reason := &v1.Reason{Code: errorCode, Message: message, Origin: "runtime"}
			fatal(t, o.store.ObserveRunV1(id, &v1.RunEvent{Sequence: 1, AtMs: stamp,
				Event: &v1.RunEvent_Call{Call: &v1.Call{Run: id + "/0", Function: "generate_image",
					Label: "Creating reference", Status: "failed", CalledAtMs: stamp - 1,
					FinishedAtMs: stamp, Reason: reason}}}, nil))
			fatal(t, o.store.RecordRunOutcomeV1(id, records.RunEndV1{Outcome: &v1.Outcome{Status: "failed", Reason: reason}}))
			defer publicationControlAPI(t, o)()

			code, shown := runCozy(t, o.root, "run", "show", id, "--json")
			var report struct {
				ErrorCode string `json:"error_code"`
				Error     string `json:"error"`
				Calls     []struct {
					ErrorCode string `json:"error_code"`
					Error     string `json:"error"`
				} `json:"calls"`
			}
			if code != 0 || json.Unmarshal([]byte(shown), &report) != nil || report.ErrorCode != errorCode || report.Error != message ||
				len(report.Calls) != 1 || report.Calls[0].ErrorCode != errorCode || report.Calls[0].Error != message {
				t.Fatalf("show lost the typed parent or child failure [%d]: %s", code, shown)
			}
			for _, args := range [][]string{{"run", "show", id}, {"run", "show", id, "--call", "1"}, {"run", "watch", id}} {
				code, text := runCozy(t, o.root, args...)
				wantFailure := args[1] == "watch"
				if (code != 0) != wantFailure || !strings.Contains(text, message) ||
					strings.Count(text, errorCode) != strings.Count(message, errorCode)+1 {
					t.Fatalf("%v duplicated or lost the failure [%d]: %s", args, code, text)
				}
			}
			code, listed := runCozy(t, o.root, "run", "list", "--json")
			var list struct {
				Rows []map[string]any `json:"invocations"`
			}
			if code != 0 || json.Unmarshal([]byte(listed), &list) != nil || len(list.Rows) != 1 ||
				list.Rows[0]["error_code"] != errorCode || list.Rows[0]["error"] != message {
				t.Fatalf("list changed the typed message [%d]: %s", code, listed)
			}
			code, listed = runCozy(t, o.root, "run", "list", "--no-watch")
			if code != 0 || !strings.Contains(listed, message) {
				t.Fatalf("human list changed the plain message [%d]: %s", code, listed)
			}
			code, watched := runCozy(t, o.root, "run", "watch", id, "--json")
			var watch struct {
				Error struct {
					Details map[string]any `json:"details"`
				} `json:"error"`
			}
			if code == 0 || json.Unmarshal([]byte(watched), &watch) != nil ||
				watch.Error.Details["error_code"] != errorCode || watch.Error.Details["error"] != message {
				t.Fatalf("watch changed the typed message [%d]: %s", code, watched)
			}
			link, problem := o.store.MachineExecution(id)
			fatal(t, problem)
			if raw := records.RunV1Outcome(link); raw.GetReason().GetMessage() != message || raw.GetReason().GetCode() != errorCode {
				t.Fatal("viewing changed the machine's original outcome")
			}
		})
	}
}
