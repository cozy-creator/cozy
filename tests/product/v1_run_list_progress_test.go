package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// `cozy run list` shows the progress its machine last reported for a run, read from the
// records: no live sample is needed, as after a client restart. A retried attempt starts with
// none of the last one's, a stage's own fraction never passes for the whole run's, and a
// completed run is 100%.
func TestRunListShowsTheMachinesLastProgress(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	open := func() *records.Store {
		t.Helper()
		store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
		fatal(t, problem)
		return store
	}
	store := open()
	request, _, problem := store.Submit(records.Request{ID: "list-progress", IdemKey: "list-progress", Package: "local/proof",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("b"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "pr-unreachable"))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": "pr-unreachable"}))
	fatal(t, store.AcceptRunV1(request.ID, "pr-unreachable", &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
	sequence := uint64(1)
	report := func(store *records.Store, attempt uint32, progress *v1.Progress) {
		t.Helper()
		sequence++
		if progress != nil {
			fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: sequence, Event: &v1.RunEvent_Progress{Progress: progress}}, nil))
			sequence++
		}
		// A state is written at once, with the progress batched ahead of it.
		fatal(t, store.ObserveRunV1(request.ID, &v1.RunEvent{Sequence: sequence, Event: &v1.RunEvent_State{State: &v1.RunState{
			Id: request.ID, Number: 1, State: "running", Attempt: attempt}}}, nil))
	}
	stage := 0.375
	report(store, 1, &v1.Progress{Stage: "Shot 3 of 4 / denoise", Fraction: 0.57, StageFraction: &stage, Completed: 3, Total: 8, StepMs: 1000})
	store.Close()
	list := func() api.Lifecycle {
		t.Helper()
		code, output := runCozy(t, root, "run", "list", "--json", "--full")
		var response struct {
			Invocations []api.Lifecycle `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(output), &response) != nil || len(response.Invocations) != 1 {
			t.Fatalf("run list [exit %d]: %s", code, output)
		}
		return response.Invocations[0]
	}
	row := list()
	if row.Status != "in_progress" || row.ProgressStage != "Shot 3 of 4 / denoise" || row.OverallFraction == nil || *row.OverallFraction != .57 ||
		row.StageFraction == nil || *row.StageFraction != .375 {
		t.Fatalf("run list dropped the machine's progress: %+v", row)
	}
	if row.RemainingMS != nil {
		t.Fatal("one progress sample invented a time remaining")
	}
	if code, output := runCozy(t, root, "run", "list"); code != 0 || !strings.Contains(output, "57%") || !strings.Contains(output, "Shot 3 of 4") {
		t.Fatalf("the human list omits the run's progress [exit %d]: %s", code, output)
	}

	store = open()
	report(store, 2, nil)
	store.Close()
	if row = list(); row.ProgressStage != "" || row.OverallFraction != nil || row.StageFraction != nil {
		t.Fatalf("a retried attempt took the last one's progress: %+v", row)
	}
	store = open()
	stage = 0.25
	report(store, 2, &v1.Progress{Stage: "Shot 1 of 4 / denoise", Fraction: -1, StageFraction: &stage, StepMs: 1000})
	store.Close()
	if row = list(); row.ProgressStage == "" || row.StageFraction == nil || *row.StageFraction != .25 || row.OverallFraction != nil {
		t.Fatalf("a stage's own fraction passed for the whole run's: %+v", row)
	}
	store = open()
	fatal(t, store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"}}))
	store.Close()
	if row = list(); row.Status != "completed" || row.OverallFraction == nil || *row.OverallFraction != 1 {
		t.Fatalf("a completed run is not 100%%: %+v", row)
	}
}
