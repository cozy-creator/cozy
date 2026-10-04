package records

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

func telemetryStore(t *testing.T) (*Store, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "records.sqlite")
	store, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(store.Close)
	_, _, problem = store.Submit(Request{ID: "run", IdemKey: "run", Package: "local/test", Entrypoint: "main",
		Kind: "call", Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem = store.LinkMachineExecution("run", "machine"); problem != nil {
		t.Fatal(problem)
	}
	if problem = store.AcceptRunV1("run", &v1.RunState{Id: "run", Number: 1, State: "running", Attempt: 1}); problem != nil {
		t.Fatal(problem)
	}
	return store, path
}
func telemetryProgress(sequence uint64, fraction float64) *v1.RunEvent {
	return &v1.RunEvent{Sequence: sequence, AtMs: int64(sequence) * 1000,
		Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Fraction: fraction}}}
}
func TestRunTelemetryNeverWaitsOnTheRecordsConnection(t *testing.T) {
	store, _ := telemetryStore(t)
	before, problem := store.LastEventSeq()
	if problem != nil {
		t.Fatal(problem)
	}
	// Hold the one records connection, as a congested FULL fsync does. Thousands of
	// progress/log samples must continue without obtaining that connection or a disk tx.
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan error, 1)
	go func() {
		if problem := store.ObserveRunV1("run", &v1.RunEvent{Event: &v1.RunEvent_State{State: &v1.RunState{State: "running", Attempt: 1}}}, nil); problem != nil {
			done <- fmt.Errorf("%v", problem)
			return
		}
		for i := uint64(1); i <= 2000; i++ {
			if problem := store.ObserveRunV1("run", telemetryProgress(i, float64(i)/2001), nil); problem != nil {
				done <- fmt.Errorf("%v", problem)
				return
			}
		}
		done <- nil
	}()
	select {
	case problem := <-done:
		if problem != nil {
			t.Fatal(problem)
		}
	case <-time.After(2 * time.Second):
		tx.Rollback()
		t.Fatal("progress blocked on the records connection")
	}
	tx.Rollback()
	after, problem := store.LastEventSeq()
	if problem != nil || before != after {
		t.Fatalf("telemetry wrote durable rows: before=%d after=%d problem=%v", before, after, problem)
	}
	link, problem := store.MachineExecution("run")
	if problem != nil || link.RemoteCursor != 0 {
		t.Fatalf("telemetry advanced the durable cursor: %v %v", link, problem)
	}
	live := store.LiveRunEventsV1("run")
	if len(live) != 1 || live[0].Event.Seq != 0 {
		t.Fatalf("unbounded or durable live samples: %v", live)
	}
	progress, problem := store.LatestMachineProgress("run", 1)
	if problem != nil || progress["overall_fraction"] != float64(2000)/2001 {
		t.Fatalf("latest progress missing: %v %v", progress, problem)
	}
}

func TestTerminalFlushesCoalescedTelemetryBeforeItsDurableOutcome(t *testing.T) {
	store, path := telemetryStore(t)
	for i := uint64(1); i <= 1000; i++ {
		if problem := store.ObserveRunV1("run", telemetryProgress(i, float64(i)/1000), nil); problem != nil {
			t.Fatal(problem)
		}
	}
	if problem := store.RecordRunOutcomeV1("run", &v1.Outcome{Status: "succeeded", Result: []byte(`{"answer":42}`)}, nil); problem != nil {
		t.Fatal(problem)
	}
	events, problem := store.EventsAfter("run", 0, 100)
	if problem != nil {
		t.Fatal(problem)
	}
	samples, terminal := 0, false
	for _, event := range events {
		if TerminalEvent(event.Type) {
			terminal = true
		}
		if event.Type == "machine.progress" {
			if terminal {
				t.Fatal("telemetry was appended after the absorbing terminal")
			}
			samples++
		}
	}
	if samples != 2 || !terminal {
		t.Fatalf("coalesced samples=%d terminal=%v", samples, terminal)
	}
	if len(store.LiveRunEventsV1("run")) != 0 {
		t.Fatal("settled run retained live telemetry")
	}
	store.Close()
	reopened, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer reopened.Close()
	row, problem := reopened.RequestRow("run")
	if problem != nil || row.State != "succeeded" {
		t.Fatalf("terminal not durable: %v %v", row, problem)
	}
	link, problem := reopened.MachineExecution("run")
	if problem != nil || RunV1Outcome(link).GetStatus() != "succeeded" {
		t.Fatalf("outcome not durable: %v %v", link, problem)
	}
}

func TestUnsyncedTelemetryDoesNotSkipAProductAfterRestart(t *testing.T) {
	store, path := telemetryStore(t)
	if problem := store.ObserveRunV1("run", &v1.RunEvent{Sequence: 5, Event: &v1.RunEvent_Product{Product: &v1.Product{Output: "image", Rev: 1}}}, nil); problem != nil {
		t.Fatal(problem)
	}
	for i := uint64(6); i <= 100; i++ {
		if problem := store.ObserveRunV1("run", telemetryProgress(i, .5), nil); problem != nil {
			t.Fatal(problem)
		}
	}
	store.Close()
	reopened, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer reopened.Close()
	link, problem := reopened.MachineExecution("run")
	if problem != nil || link.RemoteCursor != 5 {
		t.Fatalf("uncommitted progress skipped the durable product cursor: %v %v", link, problem)
	}
	if len(reopened.LiveRunEventsV1("run")) != 0 {
		t.Fatal("restart invented live progress")
	}
}

func TestOversizedRunTelemetryIsDroppedWithoutAuthorityOrDiskWrites(t *testing.T) {
	store, _ := telemetryStore(t)
	before, problem := store.LastEventSeq()
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.ObserveRunV1("run", telemetryProgress(1, .1), nil); problem != nil {
		t.Fatal(problem)
	}
	oversized := strings.Repeat("x", maxRunTelemetryBytesV1+1)
	if problem := store.ObserveRunV1("run", &v1.RunEvent{Sequence: 2, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: oversized, Fraction: 1}}}, nil); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.ObserveRunV1("run", &v1.RunEvent{Sequence: 3, Event: &v1.RunEvent_Log{Log: &v1.LogLine{Text: oversized}}}, nil); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.AppendEvent("run", "machine.progress", 0, map[string]any{"type": "progress", "payload": map[string]any{"stage": oversized}}); problem != nil {
		t.Fatal(problem)
	}
	samples := store.LiveRunEventsV1("run")
	if len(samples) != 1 || len(samples[0].Event.Raw) > maxRunTelemetryBytesV1 {
		t.Fatalf("oversized sample retained: %v", samples)
	}
	progress, problem := store.LatestMachineProgress("run", 1)
	if problem != nil || progress["overall_fraction"] != .1 {
		t.Fatalf("oversized telemetry displaced the valid sample: %v %v", progress, problem)
	}
	row, problem := store.RequestRow("run")
	if problem != nil || row.State != "dispatching" {
		t.Fatalf("volatile fraction settled a run: %v %v", row, problem)
	}
	after, problem := store.LastEventSeq()
	if problem != nil || before != after {
		t.Fatalf("oversized telemetry wrote rows: %d %d %v", before, after, problem)
	}
}

func TestStageEndpointsSurviveBurstAndFlushWithTerminal(t *testing.T) {
	store, path := telemetryStore(t)
	before, problem := store.LastEventSeq()
	if problem != nil {
		t.Fatal(problem)
	}
	// A held FULL transaction cannot prevent genuine burst samples reaching memory.
	tx, err := store.db.Begin()
	if err != nil {
		t.Fatal(err)
	}
	for _, sample := range []*v1.RunEvent{
		{Sequence: 29, AtMs: 29000, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Completed: 29, Total: 30, Fraction: .9}}},
		{Sequence: 30, AtMs: 30000, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Completed: 30, Total: 30, Fraction: .95}}},
		{Sequence: 31, AtMs: 31000, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "decoding", Fraction: .96}}},
	} {
		if problem := store.ObserveRunV1("run", sample, nil); problem != nil {
			tx.Rollback()
			t.Fatal(problem)
		}
	}
	tx.Rollback()
	after, problem := store.LastEventSeq()
	if problem != nil || before != after {
		t.Fatalf("burst wrote rows: %d %d %v", before, after, problem)
	}
	live := store.LiveRunEventsV1("run")
	if len(live) != 2 {
		t.Fatalf("stage endpoint lost: %v", live)
	}
	endpoint := live[0].Event.Payload["payload"].(map[string]any)
	if endpoint["stage"] != "denoise" || endpoint["position"] != float64(30) || live[0].Event.At != "1970-01-01T00:00:30Z" {
		t.Fatalf("genuine endpoint replaced: %v", live[0])
	}
	for _, sample := range live {
		if sample.Event.Seq != 0 {
			t.Fatal("live stage endpoint acquired durable authority")
		}
	}
	link, problem := store.MachineExecution("run")
	if problem != nil || link.RemoteCursor != 0 {
		t.Fatalf("burst advanced cursor: %v %v", link, problem)
	}
	if problem := store.RecordRunOutcomeV1("run", &v1.Outcome{Status: "succeeded", Result: []byte(`{"answer":42}`)}, nil); problem != nil {
		t.Fatal(problem)
	}
	store.Close()
	reopened, problem := Open(path)
	if problem != nil {
		t.Fatal(problem)
	}
	defer reopened.Close()
	events, problem := reopened.EventsAfter("run", 0, 100)
	if problem != nil {
		t.Fatal(problem)
	}
	seenEndpoint, terminal := false, false
	for _, event := range events {
		if TerminalEvent(event.Type) {
			terminal = true
		}
		if event.Type != "machine.progress" {
			continue
		}
		if terminal {
			t.Fatal("stage sample appended after terminal")
		}
		payload := event.Payload["payload"].(map[string]any)
		if payload["stage"] == "denoise" && payload["position"] == float64(30) {
			seenEndpoint = true
		}
	}
	if !seenEndpoint || !terminal {
		t.Fatalf("endpoint=%v terminal=%v", seenEndpoint, terminal)
	}
}

func TestProgressStageEndpointsStayBoundedAndResetForANewAttempt(t *testing.T) {
	store, _ := telemetryStore(t)
	for i := uint64(1); i <= 2000; i++ {
		if problem := store.ObserveRunV1("run", telemetryProgress(i, .5), nil); problem != nil {
			t.Fatal(problem)
		}
	}
	if got := len(store.LiveRunEventsV1("run")); got != 1 {
		t.Fatalf("same-stage flood retained %d samples", got)
	}
	for i := uint64(2001); i <= 2100; i++ {
		sample := telemetryProgress(i, .5)
		sample.GetProgress().Stage = fmt.Sprintf("stage-%d", i)
		if problem := store.ObserveRunV1("run", sample, nil); problem != nil {
			t.Fatal(problem)
		}
	}
	live := store.LiveRunEventsV1("run")
	if len(live) != maxRunProgressStageEdgesV1+1 {
		t.Fatalf("stage flood retained %d samples", len(live))
	}
	for _, sample := range live {
		if len(sample.Event.Raw) > maxRunTelemetryBytesV1 {
			t.Fatal("oversized stage endpoint")
		}
	}
	sample := telemetryProgress(2101, .5)
	sample.GetProgress().Stage = strings.Repeat("x", maxRunTelemetryBytesV1+1)
	if problem := store.ObserveRunV1("run", sample, nil); problem != nil {
		t.Fatal(problem)
	}
	if got := store.LiveRunEventsV1("run"); len(got) != len(live) || got[len(got)-1].Sequence != live[len(live)-1].Sequence {
		t.Fatal("oversized sample displaced genuine stage endpoint")
	}
	store.seedTelemetryV1("run", 2, "running", 0)
	if len(store.LiveRunEventsV1("run")) != 0 {
		t.Fatal("new attempt inherited previous-stage samples")
	}
}
