package records

import (
	"path/filepath"
	"strings"
	"testing"

	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// This is consumer proof for a queued observer receiving the machine's terminal history.
// The native producer/TLS test independently proves the running event is a durable fact.
func TestQueuedConsumerUsesOnlyTheReplayedRunningFactForTerminalProgress(t *testing.T) {
	for _, withRunning := range []bool{false, true} {
		name := "missing-running-does-not-infer-state"
		if withRunning {
			name = "actual-running-preserves-position"
		}
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "records.sqlite")
			store, problem := Open(path)
			if problem != nil {
				t.Fatal(problem)
			}
			defer store.Close()
			_, _, problem = store.Submit(Request{ID: "run", IdemKey: "run", Package: "local/test", Entrypoint: "main", Kind: "call",
				Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
			if problem != nil {
				t.Fatal(problem)
			}
			if problem = store.LinkMachineExecution("run", "machine"); problem != nil {
				t.Fatal(problem)
			}
			if problem = store.AcceptRunV1("run", &v1.RunState{Id: "run", Number: 1, State: "queued", Attempt: 1}); problem != nil {
				t.Fatal(problem)
			}
			if withRunning {
				// Sequence/time are the original journaled fact, before these later samples.
				if problem = store.ObserveRunV1("run", &v1.RunEvent{Sequence: 10, AtMs: 10000, Event: &v1.RunEvent_State{State: &v1.RunState{State: "running", Sequence: 10, Attempt: 1}}}, nil); problem != nil {
					t.Fatal(problem)
				}
			}
			for _, event := range []*v1.RunEvent{
				{Sequence: 11, AtMs: 11000, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Completed: 30, Total: 30, Fraction: .9}}},
				{Sequence: 12, AtMs: 12000, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "decoding", Fraction: .95}}},
			} {
				if problem = store.ObserveRunV1("run", event, nil); problem != nil {
					t.Fatal(problem)
				}
			}
			row, problem := store.RequestRow("run")
			if problem != nil {
				t.Fatal(problem)
			}
			want := "queued"
			if withRunning {
				want = "dispatching"
			}
			if row.State != want {
				t.Fatalf("progress supplied lifecycle authority: state=%s want=%s", row.State, want)
			}
			link, problem := store.MachineExecution("run")
			if problem != nil {
				t.Fatal(problem)
			}
			cursor := int64(0)
			if withRunning {
				cursor = 10
			}
			if link.RemoteCursor != cursor {
				t.Fatalf("advisory progress advanced durable cursor: %d want%d", link.RemoteCursor, cursor)
			}
			if problem = store.RecordRunOutcomeV1("run", &v1.Outcome{Status: "succeeded", Result: []byte(`{"steps":30}`)}, nil); problem != nil {
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
			found, terminal := false, false
			for _, event := range events {
				if TerminalEvent(event.Type) {
					terminal = true
				}
				if event.Type != "machine.progress" {
					continue
				}
				if terminal {
					t.Fatal("progress was persisted after the absorbing outcome")
				}
				payload := event.Payload["payload"].(map[string]any)
				if payload["stage"] == "denoise" && payload["position"] == float64(30) {
					found = true
				}
			}
			if found != withRunning || !terminal {
				t.Fatalf("actual-running=%v retained-position=%v terminal=%v", withRunning, found, terminal)
			}
		})
	}
}
