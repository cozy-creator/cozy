package cli

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

// cl-103: the default human wait line says what the queue is DOING — no digests, no
// dispatcher vocabulary. The raw diagnostic lives in --stream/--json/--full, and joins
// the human line only after waitPatience.

const rawDiagnostic = "no claimed worker in paul/anima has a DISPATCHABLE placement for " +
	"sha256:fc1db0000000000000000000000000000000000000000000000000000000000 with a free attempt slot"

// syncBuffer takes the patience timer's own goroutine writes.
type syncBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *syncBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *syncBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func progressSink(mode output.Mode, stream bool) (*runProgress, *syncBuffer) {
	buf := &syncBuffer{}
	ctx := &Context{Inv: &Invocation{Mode: mode}, Out: buf, Err: buf}
	return newProgress(ctx, stream, time.Now()), buf
}

func waitEvent(eventType string, at time.Time, payload map[string]any) localapi.Event {
	return localapi.Event{Type: eventType, RequestID: "req-wait", At: at.UTC().Format(time.RFC3339Nano),
		Payload: payload}
}

func TestWaitLinesAreStageHonest(t *testing.T) {
	cases := []struct {
		name    string
		payload map[string]any
		want    string
	}{
		{"worker_start", map[string]any{"wait": "worker_start", "package": "paul/anima"},
			"starting a worker for paul/anima"},
		{"worker_start_bare", map[string]any{"wait": "worker_start"}, "starting a worker"},
		{"worker_warming", map[string]any{"wait": "worker_warming", "waiting_on": "shidehiko"},
			"warming the model on shidehiko"},
		{"worker_warming_local", map[string]any{"wait": "worker_warming"}, "warming the model"},
		{"slot_busy", map[string]any{"wait": "slot_busy", "waiting_on": "shidehiko"},
			"waiting for a free slot on shidehiko"},
		{"slot_busy_local", map[string]any{"wait": "slot_busy"}, "waiting for a free slot"},
		{"queue_ahead", map[string]any{"wait": "queue_ahead", "position": float64(3)},
			"waiting in line — position 3"},
		{"rental", map[string]any{"wait": "rental"}, "waiting for a rental machine"},
		{"model_transfer", map[string]any{"wait": "model_transfer"}, "downloading the model"},
		{"unclassified", map[string]any{}, "waiting for capacity"},
	}
	for _, c := range cases {
		for _, eventType := range []string{"request.queued", "request.parked"} {
			c.payload["reason"] = rawDiagnostic
			p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
			p.on(waitEvent(eventType, time.Now(), c.payload))
			p.done()
			got := buf.String()
			if !strings.Contains(got, c.want) {
				t.Errorf("%s %s renders %q, want %q", c.name, eventType, got, c.want)
			}
			// The complaint verbatim (cl-103): the default line must not leak the queue's
			// own vocabulary or a digest.
			for _, leak := range []string{"DISPATCHABLE", "sha256:", "placement"} {
				if strings.Contains(got, leak) {
					t.Errorf("%s %s leaks %q into the default human line: %q", c.name, eventType, leak, got)
				}
			}
		}
	}
}

func TestWaitDetailArrivesWithPatience(t *testing.T) {
	payload := map[string]any{"wait": "slot_busy", "waiting_on": "shidehiko", "reason": rawDiagnostic}

	// A fresh wait stays calm.
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	p.on(waitEvent("request.parked", time.Now(), payload))
	if got := buf.String(); strings.Contains(got, "DISPATCHABLE") {
		t.Errorf("the diagnostic surfaced before waitPatience: %q", got)
	}
	p.done()

	// A wait that already outlived waitPatience (the event's own recorded time — a
	// reattached watcher inherits the wait served) carries the raw diagnostic.
	p, buf = progressSink(output.Mode{Human: true, Color: true}, false)
	p.on(waitEvent("request.parked", time.Now().Add(-2*waitPatience), payload))
	got := buf.String()
	if !strings.Contains(got, "waiting for a free slot on shidehiko") || !strings.Contains(got, rawDiagnostic) {
		t.Errorf("a long wait must carry both the calm line and the diagnostic: %q", got)
	}
	p.done()

	// The stuck case: ONE event, then silence. The patience timer re-renders with the
	// detail even though nothing new arrives.
	p, buf = progressSink(output.Mode{Human: true, Color: true}, false)
	p.on(waitEvent("request.parked", time.Now().Add(-waitPatience+50*time.Millisecond), payload))
	if got := buf.String(); strings.Contains(got, "DISPATCHABLE") {
		t.Fatalf("detail arrived before the threshold: %q", got)
	}
	deadline := time.Now().Add(5 * time.Second)
	for !strings.Contains(buf.String(), rawDiagnostic) {
		if time.Now().After(deadline) {
			t.Fatal("the patience timer never surfaced the diagnostic")
		}
		time.Sleep(10 * time.Millisecond)
	}
	p.done()
}

func TestDiagnosticSurfacesKeepTheRawCause(t *testing.T) {
	payload := map[string]any{"wait": "slot_busy", "reason": rawDiagnostic}

	// --full: the lossless human diagnostic stream.
	p, buf := progressSink(output.Mode{Human: true, Full: true}, false)
	p.on(waitEvent("request.queued", time.Now(), payload))
	p.on(waitEvent("request.parked", time.Now(), payload))
	if got := buf.String(); strings.Count(got, rawDiagnostic) != 2 {
		t.Errorf("--full must keep the verbatim diagnostic for queued and parked: %q", got)
	}

	// --stream: the typed envelope, unchanged.
	p, buf = progressSink(output.Mode{Human: true}, true)
	p.on(waitEvent("request.parked", time.Now(), payload))
	var envelope localapi.Event
	if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &envelope); err != nil {
		t.Fatalf("stream did not emit one JSON envelope: %v", err)
	}
	if envelope.Payload["reason"] != rawDiagnostic || envelope.Payload["wait"] != "slot_busy" {
		t.Errorf("the stream payload changed: %v", envelope.Payload)
	}
}
