package producttest

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Run5026 had a retained Runtime start but no live execution counter. The ordinary
// list must show an advancing elapsed clock without inventing measured execution.
func TestRunListShowsLiveRuntimeElapsedSeparatelyFromMeasuredExecution(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	makeRun := func(id string) records.Request {
		t.Helper()
		row, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Package: "local/elapsed", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("b"), MachineExecutionObserver: true})
		fatal(t, problem)
		fatal(t, store.LinkMachineExecution(id, "pr-live-clock"))
		fatal(t, store.AppendEvent(id, records.RunV1Sent, 0, map[string]any{"machine": "pr-live-clock"}))
		fatal(t, store.AcceptRunV1(id, "pr-live-clock", &v1.RunState{Id: id, Number: 1, State: "queued", Attempt: 1}))
		return row
	}
	type timing struct {
		ID       string `json:"id"`
		Status   string `json:"status"`
		Measured *int64 `json:"execution_ms"`
		Known    bool   `json:"execution_known"`
		Elapsed  *int64 `json:"execution_elapsed_ms"`
	}
	list := func(id string) timing {
		t.Helper()
		code, out := runCozy(t, root, "run", "list", "--json", "--full")
		var doc struct{ Invocations []timing }
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
			t.Fatalf("ordinary list: %d %s", code, out)
		}
		for _, row := range doc.Invocations {
			if row.ID == id {
				return row
			}
		}
		t.Fatalf("missing run %s: %s", id, out)
		return timing{}
	}
	running := func(id string, seq uint64, attempt uint32, at int64, state string) {
		t.Helper()
		fatal(t, store.ObserveRunV1(id, &v1.RunEvent{Sequence: seq, AtMs: at, Event: &v1.RunEvent_State{State: &v1.RunState{Id: id, Number: 1, State: state, Attempt: attempt}}}, nil))
	}
	request := makeRun("live-clock")
	if row := list(request.ID); row.Elapsed != nil {
		t.Fatalf("queued acceptance started an execution clock: %+v", row)
	}
	started := time.Now().Add(-90 * time.Second).UnixMilli()
	running(request.ID, 1, 1, started, "running")
	first := list(request.ID)
	if first.Status != "in_progress" || first.Measured != nil || first.Known || first.Elapsed == nil || *first.Elapsed < 90000 || *first.Elapsed > 105000 {
		t.Fatalf("Runtime start did not produce a separate live clock: %+v", first)
	}
	time.Sleep(120 * time.Millisecond)
	second := list(request.ID)
	if second.Elapsed == nil || *second.Elapsed <= *first.Elapsed {
		t.Fatalf("live elapsed did not advance: first=%+v second=%+v", first, second)
	}
	if code, out := runCozy(t, root, "run", "list"); code != 0 || !strings.Contains(out, "elapsed") {
		t.Fatalf("human list omitted live timer: %d %s", code, out)
	}
	code, out := runCozy(t, root, "run", "show", request.ID, "--json")
	var shown struct {
		timing
		Calls []struct {
			Start int64 `json:"start_unix_ms"`
		} `json:"calls"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || shown.Elapsed == nil || shown.Measured != nil {
		t.Fatalf("show disagreed with live list: %d %s", code, out)
	}
	if len(shown.Calls) == 0 || shown.Calls[0].Start != started {
		t.Fatalf("root call used client receipt time instead of Runtime start: %s", out)
	}
	running(request.ID, 2, 1, time.Now().UnixMilli(), "paused")
	if row := list(request.ID); row.Elapsed != nil {
		t.Fatalf("paused run continued a live execution clock: %+v", row)
	}
	running(request.ID, 3, 2, time.Now().Add(-5*time.Second).UnixMilli(), "running")
	if row := list(request.ID); row.Elapsed == nil || *row.Elapsed < 5000 || *row.Elapsed > 15000 {
		t.Fatalf("resumed interval inherited earlier/paused time: %+v", row)
	}
	fatal(t, store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded", ExecutionMs: 12345}}))
	if row := list(request.ID); row.Status != "completed" || row.Elapsed != nil || row.Measured == nil || *row.Measured != 12345 || !row.Known {
		t.Fatalf("terminal Runtime duration was replaced by a wall guess: %+v", row)
	}
	for _, test := range []struct {
		id string
		at int64
	}{{"no-source-clock", 0}, {"future-source-clock", time.Now().Add(time.Hour).UnixMilli()}} {
		makeRun(test.id)
		running(test.id, 1, 1, test.at, "running")
		if row := list(test.id); row.Elapsed != nil || row.Measured != nil {
			t.Fatalf("receipt/future time passed for Runtime elapsed: %+v", row)
		}
	}
}
