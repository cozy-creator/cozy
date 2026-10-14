package records

import (
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Per-child software survives the protocol observation and durable SQLite event, including
// producer fields a reader does not understand yet.
func TestCallSoftwareSurvivesDurableObservation(t *testing.T) {
	s, problem := Open(filepath.Join(t.TempDir(), "records.sqlite"))
	if problem != nil {
		t.Fatal(problem)
	}
	defer s.Close()
	_, _, problem = s.Submit(Request{ID: "root", IdemKey: "root", Package: "local/test", Entrypoint: "main", Kind: "call",
		Payload: []byte(`{}`), BodyDigest: "sha256:" + strings.Repeat("1", 64), MachineExecutionObserver: true})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := s.LinkMachineExecution("root", "machine"); problem != nil {
		t.Fatal(problem)
	}
	if problem := s.AcceptRunV1("root", "machine", &v1.RunState{Id: "root", Number: 1, State: "running", Attempt: 1}); problem != nil {
		t.Fatal(problem)
	}
	measured := []byte(`{"software":{"attempt_ordinal":2,"runtime":"fallback-sdk","tensord":"old-agent","tensord_commit":"old-build","future_identity":"retained"}}`)
	event := &v1.RunEvent{Sequence: 1, AtMs: 1, Event: &v1.RunEvent_Call{Call: &v1.Call{
		Run: "root/0", Function: "render", Status: "succeeded", Measurements: measured}}}
	if problem := s.ObserveRunV1("root", event, nil); problem != nil {
		t.Fatal(problem)
	}
	rows, problem := s.EventsAfter("root", 0, 100)
	if problem != nil {
		t.Fatal(problem)
	}
	for _, row := range rows {
		if row.Type != "machine.call" {
			continue
		}
		body, err := json.Marshal(row.Payload["software"])
		if err != nil {
			t.Fatal(err)
		}
		var software map[string]any
		if err := json.Unmarshal(body, &software); err != nil {
			t.Fatal(err)
		}
		if software["attempt_ordinal"] != float64(2) || software["runtime"] != "fallback-sdk" ||
			software["tensord_commit"] != "old-build" || software["future_identity"] != "retained" {
			t.Fatalf("lost per-attempt software: %s", body)
		}
		return
	}
	t.Fatal("call observation was not kept")
}
