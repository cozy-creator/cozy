package records

import (
	"fmt"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Every log line and progress sample is kept (batched, never latest-only), in the log's order
// before the transition that follows it, and a reattach's resent entries are written once.
func TestTelemetryRowsAreBatchedInOrderAndKept(t *testing.T) {
	s, problem := Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	for _, step := range []func() *exit.Error{
		func() *exit.Error {
			_, _, problem := s.Submit(Request{ID: "run", IdemKey: "run", Package: "local/test", Entrypoint: "main", Kind: "call",
				Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
			return problem
		},
		func() *exit.Error { return s.LinkMachineExecution("run", "machine") },
		func() *exit.Error {
			return s.AcceptRunV1("run", &v1.RunState{Id: "run", Number: 1, State: "running", Attempt: 1})
		},
	} {
		if problem := step(); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	log := func(seq uint64, text string) *v1.RunEvent {
		return &v1.RunEvent{Sequence: seq, AtMs: 1, Event: &v1.RunEvent_Log{Log: &v1.LogLine{Level: "info", Text: text}}}
	}
	observe := func(event *v1.RunEvent) {
		if problem := s.ObserveRunV1("run", event, nil); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	for k := uint64(1); k <= 40; k++ {
		observe(log(2*k-1, fmt.Sprint("before ", k)))
		observe(&v1.RunEvent{Sequence: 2 * k, AtMs: 1, Event: &v1.RunEvent_Progress{Progress: &v1.Progress{Stage: "denoise", Fraction: float64(k) / 40}}})
	}
	observe(log(19, "before 10")) // resent by a reattach
	observe(&v1.RunEvent{Sequence: 100, Event: &v1.RunEvent_State{State: &v1.RunState{Id: "run", State: "paused", Attempt: 1}}})
	for k := uint64(1); k <= 5; k++ {
		observe(log(100+k, fmt.Sprint("after ", k)))
	}
	// The writer batches these without any transition; then the outcome follows them.
	for until := time.Now().Add(time.Minute); ; time.Sleep(10 * time.Millisecond) {
		rows, _ := s.EventsAfter("run", 0, 1000)
		if len(rows) > 0 && rows[len(rows)-1].Payload["message"] == "after 5" {
			break
		}
		if time.Now().After(until) {
			t.Fatal("the writer did not batch the pending rows")
		}
	}
	// An unsynced batch leaves the connection synchronous for the transitions after it.
	var level int
	if err := s.db.QueryRow(`PRAGMA synchronous`).Scan(&level); err != nil || level != 2 {
		t.Fatalf("after a batch the connection's synchronous level is %d (%v), not FULL", level, err)
	}
	if problem := s.RecordRunOutcomeV1("run", RunEndV1{Outcome: &v1.Outcome{Status: "succeeded"}}); problem != nil {
		t.Fatal(problem.Message)
	}
	rows, problem := s.EventsAfter("run", 0, 1000)
	if problem != nil {
		t.Fatal(problem.Message)
	}
	var logs []string
	progress, paused, ended := 0, -1, -1
	for i, row := range rows {
		switch row.Type {
		case "request.log":
			line := fmt.Sprint(row.Payload["message"])
			if strings.HasPrefix(line, "before") != (paused < 0) || ended >= 0 {
				t.Fatalf("%s is on the wrong side of a transition", line)
			}
			logs = append(logs, line)
		case "machine.progress":
			progress++
		case "request.paused":
			paused = i
		case "run.completed":
			ended = i
		}
	}
	if len(logs) != 45 || logs[0] != "before 1" || logs[39] != "before 40" || logs[44] != "after 5" || progress != 40 || ended < 0 {
		t.Fatalf("logs %d (%v … %v), progress %d, paused at %d, ended at %d", len(logs), logs[0], logs[len(logs)-1], progress, paused, ended)
	}
	if link, _ := s.MachineExecution("run"); link == nil || link.RemoteCursor != 105 {
		t.Fatalf("cursor %+v", link)
	}
}
