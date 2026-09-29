package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

func TestCallPhaseProgressSeparatesWaitingPreparationAndExecution(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	start := time.Now().Add(-10 * time.Minute).Truncate(time.Second)
	phase := func(request, label, state string, attempt int, at, queue, prep, execute int) {
		p.On(localapi.Event{Type: "machine.call.phase", At: start.Add(time.Duration(at) * time.Second).Format(time.RFC3339Nano), Payload: map[string]any{
			"request": request, "parent": "root", "index": 0, "attempt": attempt, "label": label, "phase": state, "at_unix_ms": start.Add(time.Duration(at) * time.Second).UnixMilli(), "called_unix_ms": start.UnixMilli(),
			"queued_ms": float64(queue * 1000), "preparation_ms": float64(prep * 1000), "execution_ms": float64(execute * 1000), "future_detail": "tolerated",
		}})
	}
	frame := func(at int) string { return liveFrame(p, start.Add(time.Duration(at)*time.Second)) }
	phase("call-one", "Render item", "queued", 1, 0, 0, 0, 0)
	if got := frame(90); !strings.Contains(got, "Render item · queued 1m30s") || strings.Contains(got, "execution 1m30s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "preparing", 1, 90, 90, 0, 0)
	phase("call-one", "Render item", "paused", 1, 100, 90, 10, 0)
	if got := frame(1000); !strings.Contains(got, "Render item · paused · execution 0.0s") || strings.Contains(got, "✓ Render item") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "preparing", 1, 110, 90, 10, 0)
	if got := frame(120); !strings.Contains(got, "Render item · preparing 20s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "running", 1, 120, 90, 20, 0)
	if got := frame(125); !strings.Contains(got, "Render item · execution 5s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "preparing", 1, 126, 90, 20, 6)
	if got := frame(140); !strings.Contains(got, "Render item · preparing 34s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "running", 1, 150, 90, 44, 6)
	if got := frame(152); !strings.Contains(got, "Render item · execution 8s") {
		t.Fatal(got)
	}
	// Duplicated, old and future-unknown phase observations do not restart clocks.
	phase("call-one", "Render item", "running", 1, 150, 90, 44, 6)
	phase("call-one", "Render item", "queued", 1, 90, 90, 0, 0)
	phase("call-one", "Render item", "future_phase", 1, 151, 1000, 0, 0)
	if got := frame(152); !strings.Contains(got, "Render item · execution 8s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "finalizing", 1, 155, 90, 44, 11)
	if got := frame(500); !strings.Contains(got, "Render item · finalizing · execution 11s") || strings.Contains(got, "execution 5m") {
		t.Fatal(got)
	}
	// Dispatch and finalization can share a millisecond with the next transition.
	phase("call-one", "Render item", "terminal", 1, 155, 90, 44, 11)
	got := frame(999)
	if strings.Count(got, "✓ Render item") != 1 || !strings.Contains(got, "execution 11s · queued 1m30s · preparation 44s · wall 2m35s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "terminal", 1, 156, 90, 44, 11)
	phase("call-one", "Render item", "running", 1, 157, 90, 44, 20)
	if got := frame(999); strings.Count(got, "✓ Render item") != 1 || strings.Contains(got, "execution 20s") {
		t.Fatal(got)
	}
	phase("call-one", "Render item", "queued", 2, 160, 0, 0, 0)
	phase("call-one", "Render item", "terminal", 1, 170, 90, 44, 11)
	if got := frame(165); !strings.Contains(got, "Render item · queued 5s") || strings.Contains(got, "✓ Render item") {
		t.Fatal(got)
	}
}

func TestCallPhaseTimingKeepsIdentityAndUnknownExecution(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	start := time.Now().Add(-time.Minute)
	send := func(kind string, seconds int, payload map[string]any) {
		p.On(localapi.Event{Type: kind, At: start.Add(time.Duration(seconds) * time.Second).Format(time.RFC3339Nano), Payload: payload})
	}
	for _, id := range []string{"call-one", "call-two"} {
		send("machine.call.phase", 0, map[string]any{"request": id, "label": "Same label", "attempt": 1, "phase": "running", "at_unix_ms": start.UnixMilli(), "called_unix_ms": start.UnixMilli(), "execution_ms": float64(0)})
	}
	send("machine.call", 4, map[string]any{"request": "call-one", "label": "Same label", "attempt": 1, "called_unix_ms": start.UnixMilli(), "status": "succeeded", "execution_ms": float64(2000)})
	got := liveFrame(p, start.Add(5*time.Second))
	if strings.Count(got, "✓ Same label") != 1 || !strings.Contains(got, "execution 2s") || !strings.Contains(got, "▸ Same label · execution 5s") {
		t.Fatal(got)
	}
	// A final record without the new measurements closes only its own call and
	// never substitutes wall duration for an absent execution measurement.
	send("machine.call", 6, map[string]any{"request": "call-two", "label": "Same label", "attempt": 1, "called_unix_ms": start.UnixMilli(), "status": "succeeded"})
	// An old SDK may omit progress identity; late narration cannot reopen a
	// completed call or choose between two calls carrying the same label.
	send("request.progress", 7, map[string]any{"value": map[string]any{"stage": "Same label / late progress"}})
	got = liveFrame(p, start.Add(100*time.Second))
	if strings.Contains(got, "late progress") {
		t.Fatal(got)
	}
	if strings.Count(got, "✓ Same label") != 2 || !strings.Contains(got, "execution — · wall 6s") || strings.Contains(got, "execution 6s") {
		t.Fatal(got)
	}
}

// The ordinary show command keeps its existing wall interval and raw events,
// exposing execution only when Runtime measured it, across supported versions.
func TestRunShowCallTimingIsAdditiveAndOptional(t *testing.T) {
	o := hostOwner(t, "call-timing")
	id := succeededWithTriage(t, o, []byte(`{}`))
	defer publicationControlAPI(t, o)()
	start := time.Now().Add(-2 * time.Minute).Truncate(time.Millisecond).UnixMilli()
	record := func(kind, request string, attempt int, fields map[string]any) {
		fields["request"], fields["parent"], fields["attempt"] = request, id, attempt
		fields["label"], fields["export"], fields["called_unix_ms"] = request, "render", start
		fatal(t, o.store.AppendEvent(id, kind, 1, fields))
	}
	record("machine.call.phase", "call-measured", 1, map[string]any{"phase": "running", "at_unix_ms": start + 100000, "queued_ms": 80000, "preparation_ms": 20000, "execution_ms": 0})
	record("machine.call", "call-measured", 1, map[string]any{"status": "succeeded", "queued_ms": 80000, "preparation_ms": 34000, "execution_ms": 6000, "execution_started_unix_ms": start + 100000, "future_detail": "retained"})
	record("machine.call", "call-legacy", 1, map[string]any{"status": "succeeded"})
	record("machine.call.phase", "call-missing", 1, map[string]any{"phase": "running", "at_unix_ms": start + 100000, "execution_ms": 5000})
	record("machine.call", "call-missing", 1, map[string]any{"status": "succeeded"})
	// Late history cannot replace a completed or newer attempt's measurements.
	record("machine.call.phase", "call-measured", 1, map[string]any{"phase": "running", "at_unix_ms": start + 200000, "execution_ms": 999000})
	record("machine.call.phase", "call-retry", 2, map[string]any{"phase": "queued", "at_unix_ms": start + 100000, "queued_ms": 0, "preparation_ms": 0, "execution_ms": 0})
	record("machine.call", "call-retry", 1, map[string]any{"status": "failed", "execution_ms": 9000})
	record("machine.call.phase", "call-retry", 2, map[string]any{"phase": "future_phase", "at_unix_ms": start + 110000, "execution_ms": 999000})
	code, out := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		Calls  []map[string]any
		Events []struct {
			Type, At string
			Payload  map[string]any
		}
	}
	if code != 0 || json.Unmarshal([]byte(out), &report) != nil {
		t.Fatalf("run show [%d]: %s", code, out)
	}
	calls := map[string]map[string]any{}
	for _, call := range report.Calls {
		calls[call["request"].(string)] = call
	}
	measured := calls["call-measured"]
	if measured["execution_ms"] != float64(6000) || measured["preparation_ms"] != float64(34000) || measured["queued_ms"] != float64(80000) || measured["execution_started_unix_ms"] != float64(start+100000) || measured["phase"] != "terminal" {
		t.Fatalf("measured call: %+v", measured)
	}
	rawRetained := false
	for _, event := range report.Events {
		if event.Type == "machine.call" && event.Payload["request"] == "call-measured" {
			at, err := time.Parse(time.RFC3339Nano, event.At)
			must(t, err)
			if measured["ms"] != float64(at.UnixMilli()-start) {
				t.Fatalf("wall duration changed: %+v", measured)
			}
			rawRetained = event.Payload["future_detail"] == "retained"
		}
	}
	if !rawRetained {
		t.Fatal("raw additive event field lost")
	}
	for _, name := range []string{"call-legacy", "call-missing"} {
		if call := calls[name]; call["execution_ms"] != nil || call["preparation_ms"] != nil || call["queued_ms"] != nil || call["ms"].(float64) < 100000 {
			t.Fatalf("invented timing for %s: %+v", name, call)
		}
	}
	if call := calls["call-retry"]; call["attempt"] != float64(2) || call["phase"] != "queued" || call["execution_ms"] != float64(0) {
		t.Fatalf("retry accounting: %+v", call)
	}
	code, human := runCozy(t, o.root, "run", "show", id, "--call", "call-measured")
	if code != 0 || !strings.Contains(human, "execution 6s · queued 1m20s · preparation 34s") || !strings.Contains(human, "wall 2m") {
		t.Fatalf("run show --call [%d]: %s", code, human)
	}
	code, human = runCozy(t, o.root, "run", "show", id, "--call", "call-legacy")
	if code != 0 || strings.Contains(human, "execution ") || !strings.Contains(human, "wall 2m") {
		t.Fatalf("legacy run show --call [%d]: %s", code, human)
	}
}

func TestCallPhaseProgressUsesIdentityAndMissingTerminalStaysUnknown(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	start := time.Now().Add(-time.Minute)
	send := func(kind string, at int, fields map[string]any) {
		p.On(localapi.Event{Type: kind, Attempt: 1, At: start.Add(time.Duration(at) * time.Second).Format(time.RFC3339Nano), Payload: fields})
	}
	for _, id := range []string{"one", "two"} {
		send("machine.call.phase", 0, map[string]any{"request": id, "attempt": 1, "label": "Repeated", "phase": "running", "at_unix_ms": start.UnixMilli(), "called_unix_ms": start.UnixMilli(), "execution_ms": 0})
	}
	progress := func(id string, attempt int, leaf string) {
		send("request.progress", 1, map[string]any{"value": map[string]any{"stage": "Repeated / " + leaf, "call_request": id, "call_attempt": attempt, "position": 1, "total": 4}})
	}
	progress("one", 1, "encoding")
	progress("two", 1, "decoding")
	progress("one", 2, "wrong attempt")
	got := liveFrame(p, start.Add(2*time.Second))
	if !strings.Contains(got, "Repeated · encoding · execution 2s") || !strings.Contains(got, "Repeated · decoding · execution 2s") || strings.Contains(got, "wrong attempt") {
		t.Fatal(got)
	}
	send("run.failed", 5, nil)
	got = liveFrame(p, start.Add(time.Hour))
	if strings.Count(got, "execution — · wall 5s") != 2 || strings.Contains(got, "execution 5s") {
		t.Fatal(got)
	}
}
