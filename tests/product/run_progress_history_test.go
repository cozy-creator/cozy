package producttest

import (
	"strings"
	"testing"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

func liveEvent(kind string, fields map[string]any) localapi.Event {
	return localapi.Event{Type: "request." + kind, Attempt: 1,
		At: time.Now().UTC().Format(time.RFC3339Nano), Payload: map[string]any{"value": fields}}
}

func TestLiveProgressKeepsFinishedStagesAndMeasuredRates(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	p.On(liveEvent("phase", map[string]any{
		"phase": "acquiring", "machine": "aldra", "elapsed_ms": 12000,
		"rental": map[string]any{"accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
			"hourly_rate_usd_micros": 3_236_009},
	}))
	acquiring := buf.String()
	for _, want := range []string{"acquiring on aldra", "12s", "NVIDIA H100 NVL", "$3.24/hour"} {
		if !strings.Contains(acquiring, want) {
			t.Fatalf("rental display lacks %q: %q", want, acquiring)
		}
	}
	if strings.Contains(acquiring, "%") || strings.Contains(acquiring, "ETA") {
		t.Fatalf("invented a boot denominator: %q", acquiring)
	}
	p.On(liveEvent("phase", map[string]any{
		"phase": "downloading", "models": []any{
			map[string]any{"model": "paul/minimax-h3", "lane": "fp8-pruned", "moved_bytes": 5 << 30,
				"total_bytes": 50 << 30, "rate_bytes_per_second": 400 << 20},
			map[string]any{"model": "paul/text-encoder", "moved_bytes": 1 << 30,
				"total_bytes": 4 << 30, "rate_bytes_per_second": 80 << 20},
		},
	}))
	download := buf.String()[len(acquiring):]
	for _, want := range []string{"acquiring on aldra", " · done\n", "paul/minimax-h3/fp8-pruned", "paul/text-encoder", "/s", " / "} {
		if !strings.Contains(download, want) {
			t.Fatalf("download history lacks %q: %q", want, download)
		}
	}
	p.On(liveEvent("accepted", nil))
	p.On(liveEvent("progress", map[string]any{"stage": "condition_text"}))
	before := len(buf.String())
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 5, "total": 30, "step_ms": 15000}))
	denoise := buf.String()[before:]
	for _, want := range []string{"conditioning", " · done\n", "denoising 5/30", "15.00s/step avg", "ETA ~6m15s"} {
		if !strings.Contains(denoise, want) {
			t.Fatalf("denoising display lacks %q: %q", want, denoise)
		}
	}
	before = len(buf.String())
	// A heartbeat/old stage end cannot replace the newer work with generic running.
	p.On(liveEvent("accepted", nil))
	p.On(liveEvent("queued", nil))
	p.On(liveEvent("parked", nil))
	p.On(liveEvent("stage", map[string]any{"name": "condition_text", "ms": 1234}))
	p.On(liveEvent("phase", map[string]any{"phase": "warming"}))
	if len(buf.String()) != before {
		t.Fatalf("old lifecycle events replaced the live stage: %q", buf.String()[before:])
	}
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 5, "total": 30, "step_ms": 90000}))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 6, "total": 30, "step_ms": 15000}))
	if !strings.Contains(buf.String()[before:], "ETA ~6m0s") {
		t.Fatalf("duplicate step biased measured ETA: %q", buf.String()[before:])
	}
	before = len(buf.String())
	p.On(liveEvent("failed", nil))
	failure := buf.String()[before:]
	if !strings.Contains(failure, " · failed\n") || strings.Contains(failure, " · done") {
		t.Fatalf("failed stage claimed completion: %q", failure)
	}
}

func TestLiveProgressClockStopsWhenWatcherDetaches(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	p.On(liveEvent("progress", map[string]any{"stage": "condition_media"}))
	waitUntil(t, "quiet stage elapsed clock", func() bool { return strings.Contains(buf.String(), "elapsed 1s") })
	p.Done()
	before := len(buf.String())
	time.Sleep(1100 * time.Millisecond)
	if len(buf.String()) != before {
		t.Fatalf("a detached watcher kept writing to the terminal: %q", buf.String()[before:])
	}
}

func TestReattachedProgressUsesRecordedStageBoundaries(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	start := time.Now().Add(-20 * time.Minute)
	replay := func(kind string, after time.Duration, seq int64, fields map[string]any) localapi.Event {
		e := liveEvent(kind, fields)
		e.At, e.EventID = start.Add(after).UTC().Format(time.RFC3339Nano), seq
		return e
	}
	queued := replay("queued", 0, 1, nil)
	queued.Payload = map[string]any{"wait": "rental", "reason": "historical wait diagnostic"}
	p.On(queued)
	before := len(buf.String())
	p.On(replay("accepted", 12*time.Second, 2, nil))
	accepted := buf.String()[before:]
	for _, want := range []string{"waiting for a rental machine · elapsed 12s · done", "Waiting for a progress update"} {
		if !strings.Contains(accepted, want) {
			t.Fatalf("replayed acceptance lacks %q: %q", want, accepted)
		}
	}
	if strings.Contains(accepted, "starting request") || strings.Contains(accepted, "20m") || strings.Contains(accepted, "historical wait diagnostic") {
		t.Fatalf("historical wait elapsed grew until reattach: %q", accepted)
	}
	p.On(replay("progress", 15*time.Second, 3, map[string]any{"stage": "condition_text"}))
	before = len(buf.String())
	p.On(replay("progress", 25*time.Second, 4, map[string]any{"stage": "denoise", "position": 1, "total": 30, "step_ms": 15000}))
	transition := buf.String()[before:]
	if !strings.Contains(transition, "conditioning · elapsed 10s · done") {
		t.Fatalf("finished stage used reattach walltime instead of recorded boundary: %q", transition)
	}
	before = len(buf.String())
	p.On(replay("completed", 40*time.Second, 5, nil))
	terminal := buf.String()[before:]
	if !strings.Contains(terminal, "elapsed 15s") || strings.Contains(terminal, "19m") {
		t.Fatalf("terminal stage did not freeze at its recorded completion: %q", terminal)
	}
}

func TestCapacityWaitReplacesWarmingButNeverExecution(t *testing.T) {
	p, buf := progressSink(output.Mode{Human: true, Color: true}, false)
	t.Cleanup(p.Done)
	p.On(liveEvent("phase", map[string]any{"phase": "warming", "machine": "aldra"}))
	wait := waitEnvelope("request.parked", time.Now(), map[string]any{
		"wait": "slot_busy", "waiting_on": "aldra", "reason": rawDiagnostic,
		"waiting_for": map[string]any{"number": 429, "request_id": "req-429"},
	})
	before := len(buf.String())
	p.On(wait)
	if got := buf.String()[before:]; !strings.Contains(got, "waiting for run 429 on aldra") || strings.Contains(got, rawDiagnostic) {
		t.Fatalf("fresh queue wait did not replace old preparation: %q", got)
	}
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 1, "total": 30}))
	before = len(buf.String())
	p.On(wait)
	if got := buf.String()[before:]; got != "" {
		t.Fatalf("queue replay replaced running denoising: %q", got)
	}
}
