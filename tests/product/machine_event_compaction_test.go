package producttest

import (
	"context"
	"encoding/json"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// compactingMachine is Runtime's execution journal under a progress burst: only the
// newest `keep` progress events survive, every authored log record remains, and a page
// carries the retained events after its cursor up to its limit (workspace_executions.py).
type compactingMachine struct {
	*runtimeMachine
	keep      int
	compacted uint64
}

// burst journals one log record and one progress sample per tensor, then compacts.
func (m *compactingMachine) burst(tensors int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for i := range tensors {
		log, _ := json.Marshal(map[string]any{"type": "log", "payload": map[string]any{"name": "quantized tensor", "done": i + 1}})
		m.record("log", log)
		progress, _ := json.Marshal(map[string]any{"type": "progress", "payload": map[string]any{"stage": "unet", "done": i + 1}})
		m.record("progress", progress)
	}
	samples := 0
	for i := len(m.events) - 1; i >= 0; i-- {
		if m.events[i].Kind == "progress" {
			if samples++; samples == m.keep+1 {
				m.compacted = m.events[i].Sequence
			}
		}
	}
	kept := m.events[:0]
	for _, event := range m.events {
		if event.Kind != "progress" || event.Sequence > m.compacted {
			kept = append(kept, event)
		}
	}
	m.events = kept
}

func (m *compactingMachine) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	limit := int(query.Limit)
	if limit == 0 {
		limit = 256
	}
	page := &pb.MachineExecutionEventPage{NextAfter: query.After, HeadSequence: m.state.Sequence, CompactedThrough: m.compacted}
	for _, event := range m.events {
		if event.Sequence > query.After && len(page.Events) < limit {
			page.Events = append(page.Events, event)
			page.NextAfter = event.Sequence
		}
	}
	return page, nil
}

// A burst of log records and progress samples compacts the machine's progress past this
// host's cursor while the log records below the compaction stay retained. Every page is
// still read and progress keeps flowing; before, the first such page was refused as
// "skips unaccounted history" and nothing after it was ever relayed (runs 1413, 1414).
func TestProgressCompactionNeverStallsTheRunsEventStream(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := &compactingMachine{runtimeMachine: &runtimeMachine{}, keep: 16}
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, nil)
	const key = "compacted-progress"
	if code, out := cozyWithin(t, root, time.Minute, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", key); code != 0 {
		t.Fatalf("the run was refused [exit %d]: %s", code, out)
	}
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	var id string
	eventually(t, root, "the run executing on the machine", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		if problem == nil && row != nil {
			id = row.ID
		}
		return problem == nil && row != nil && row.State == "dispatching"
	})

	const tensors = 600
	machine.burst(tensors)
	logs, last := 0, 0.0
	for deadline := time.Now().Add(2 * time.Minute); logs < tensors || last < tensors; time.Sleep(100 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("relayed %d of %d log records and progress through %v of %d:\n%s", logs, tensors, last, tensors, tail(filepath.Join(root, "daemon.log")))
		}
		events, problem := store.EventsAfter(id, 0, 10*tensors)
		fatal(t, problem)
		logs, last = 0, 0
		for _, event := range events {
			switch event.Type {
			case "machine.events_missing":
				t.Fatalf("compacted progress was reported as missing history: %+v", event.Payload)
			case "request.log":
				logs++
			case "machine.progress":
				if payload, ok := event.Payload["payload"].(map[string]any); ok {
					last = max(last, payload["done"].(float64))
				}
			}
		}
	}
	machine.finish()
	eventually(t, root, "the run settling", func() bool {
		row, problem := store.RequestByIdempotencyKey(key)
		return problem == nil && row != nil && row.State == "succeeded"
	})
}
