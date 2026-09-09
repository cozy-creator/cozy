package producttest

import (
	"bytes"
	"encoding/json"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/cli"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

// cl-103's render half: the default human wait line says what the queue is DOING — no
// digests, no dispatcher vocabulary. The raw diagnostic lives in --json/--full
// only. The events fed here are exactly the
// payloads TestQueueWaitCauses proves the orchestrator emits.

const rawDiagnostic = "no claimed worker in paul/anima has a DISPATCHABLE placement for " +
	"sha256:fc1db0000000000000000000000000000000000000000000000000000000000 with a free attempt slot"

// renderBuffer takes the live elapsed timer's goroutine writes.
type renderBuffer struct {
	mu sync.Mutex
	b  bytes.Buffer
}

func (s *renderBuffer) Write(p []byte) (int, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.Write(p)
}

func (s *renderBuffer) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.b.String()
}

func progressSink(mode output.Mode, stream bool) (*cli.RunProgress, *renderBuffer) {
	buf := &renderBuffer{}
	ctx := &cli.Context{Inv: &cli.Invocation{Mode: mode}, Out: buf, Err: buf}
	return cli.NewProgress(ctx, stream, time.Now()), buf
}

func waitEnvelope(eventType string, at time.Time, payload map[string]any) localapi.Event {
	return localapi.Event{Type: eventType, RequestID: "req-wait",
		At: at.UTC().Format(time.RFC3339Nano), Payload: payload}
}

func TestProgressKeepsStageAndOverallFractionsDistinct(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	frame := func(stageFraction, overallFraction float64, position int) localapi.Event {
		return localapi.Event{Type: "request.progress", RequestID: "req-progress", Attempt: 1,
			Payload: map[string]any{"value": map[string]any{
				"stage": "tile_steps", "stage_fraction": stageFraction,
				"overall_fraction": overallFraction, "position": float64(position),
				"total": float64(100), "step_ms": float64(20),
			}}}
	}
	p.On(frame(0.50, 0.10, 50))
	if got := buf.String(); !strings.Contains(got, "overall") ||
		!strings.Contains(got, "tile_steps 50/100") || !strings.Contains(got, "50% stage") ||
		strings.Contains(got, "overall 10% · ETA") {
		t.Fatalf("first progress sample conflated stage and overall facts: %q", got)
	}
	p.On(frame(0.60, 0.20, 60))
	if got := buf.String(); !strings.Contains(got, "20%") || !strings.Contains(got, "ETA ~") {
		t.Fatalf("forward overall progress did not produce a whole-job ETA: %q", got)
	}
	p.Done()

	// Redirecting the same progress retains the measured stage timing, without
	// terminal control bytes or a separate estimate from the human formatter.
	p, buf = progressSink(output.Mode{Human: true}, false)
	p.On(frame(0.50, 0.10, 50))
	if got := buf.String(); !strings.Contains(got, "tile_steps 50/100 · 50% stage · 0.02s/step avg · ETA ~1s · 10% overall") ||
		strings.ContainsAny(got, "\r\033") {
		t.Fatalf("redirected progress dropped measured timing: %q", got)
	}
	p.On(frame(0.60, 0.20, 60))
	if got := buf.String(); !strings.Contains(got, "tile_steps 60/100 · 60% stage · 0.02s/step avg · ETA ~0.8s · 20% overall") {
		t.Fatalf("redirected progress did not advance its stage ETA: %q", got)
	}
	p.Done()

	p, buf = progressSink(output.Mode{Human: true, Color: true}, false)
	p.On(localapi.Event{Type: "request.progress", RequestID: "req-stage", Attempt: 1,
		Payload: map[string]any{"value": map[string]any{
			"stage": "encode", "stage_fraction": float64(0.7), "step_ms": float64(12),
		}}})
	if got := buf.String(); !strings.Contains(got, "70%") || !strings.Contains(got, "stage") ||
		strings.Contains(got, "overall") || strings.Contains(got, "ETA") {
		t.Fatalf("stage-only progress was presented as whole-job progress: %q", got)
	}
	p.Done()
}

func TestProgressOverallETAAccountsForCoalescedSteps(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	for _, position := range []float64{3, 6} {
		p.On(localapi.Event{Type: "request.progress", RequestID: "coalesced-progress", Attempt: 1,
			Payload: map[string]any{"value": map[string]any{
				"stage": "denoise", "position": position, "total": float64(30),
				"stage_fraction": position / 30, "overall_fraction": position / 30,
				"step_ms": float64(42000),
			}}})
	}
	if got := buf.String(); !strings.Contains(got, "overall 20% · ETA ~16m48s") {
		t.Fatalf("whole-job ETA charged one interval to three completed steps: %q", got)
	}
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
		{"blocking_run", map[string]any{"wait": "slot_busy", "waiting_on": "aldra",
			"waiting_for": map[string]any{"number": 429, "request_id": "req-blocking"}}, "waiting for run 429 on aldra"},
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
			p.On(waitEnvelope(eventType, time.Now(), c.payload))
			p.Done()
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

func TestWaitDetailsStayInDiagnosticOutput(t *testing.T) {
	payload := map[string]any{"wait": "slot_busy", "waiting_on": "shidehiko", "reason": rawDiagnostic}

	// A fresh wait stays calm.
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	p.On(waitEnvelope("request.parked", time.Now(), payload))
	if got := buf.String(); strings.Contains(got, "DISPATCHABLE") {
		t.Errorf("the diagnostic surfaced before the wait threshold: %q", got)
	}
	p.Done()

	// A long wait is ordinary while another inference owns the GPU. Reattaching
	// must not turn that into a wall of internal dispatcher diagnostics.
	p, buf = progressSink(output.Mode{Human: true, Color: true}, false)
	p.On(waitEnvelope("request.parked", time.Now().Add(-10*time.Minute), payload))
	got := buf.String()
	if !strings.Contains(got, "waiting for a free slot on shidehiko") || strings.Contains(got, rawDiagnostic) {
		t.Errorf("a long wait exposed the dispatcher diagnostic: %q", got)
	}
	p.Done()

}

func TestDiagnosticSurfacesKeepTheRawCause(t *testing.T) {
	payload := map[string]any{"wait": "slot_busy", "reason": rawDiagnostic}

	// --full: the lossless human diagnostic stream.
	p, buf := progressSink(output.Mode{Human: true, Full: true}, false)
	p.On(waitEnvelope("request.queued", time.Now(), payload))
	p.On(waitEnvelope("request.parked", time.Now(), payload))
	if got := buf.String(); strings.Count(got, rawDiagnostic) != 2 {
		t.Errorf("--full must keep the verbatim diagnostic for queued and parked: %q", got)
	}

	// Awaited --json: the typed envelope, unchanged.
	p, buf = progressSink(output.Mode{JSON: true}, true)
	p.On(waitEnvelope("request.parked", time.Now(), payload))
	var envelope localapi.Event
	if err := json.Unmarshal([]byte(strings.TrimSpace(buf.String())), &envelope); err != nil {
		t.Fatalf("stream did not emit one JSON envelope: %v", err)
	}
	if envelope.Payload["reason"] != rawDiagnostic || envelope.Payload["wait"] != "slot_busy" {
		t.Errorf("the stream payload changed: %v", envelope.Payload)
	}
}
