package producttest

import (
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/cli"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func liveEvent(kind string, fields map[string]any) localapi.Event {
	return localapi.Event{Type: records.StateEvent(kind), Attempt: 1,
		At: time.Now().UTC().Format(time.RFC3339Nano), Payload: map[string]any{"value": fields}}
}

// liveFrame is the live region now, as one string.
func liveFrame(p *cli.RunProgress, at time.Time) string {
	return strings.Join(p.Frame(at, 0), "\n")
}

func TestLiveProgressKeepsFinishedStagesAndMeasuredRates(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	p.On(liveEvent("phase", map[string]any{
		"phase": "acquiring", "machine": "aldra", "elapsed_ms": 12000,
		"rental": map[string]any{"accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
			"hourly_rate_usd_micros": 3_236_009},
	}))
	acquiring := liveFrame(p, time.Now())
	for _, want := range []string{"▸ acquiring on aldra · 12s", "NVIDIA H100 NVL", "$3.24/hour"} {
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
	download := liveFrame(p, time.Now())
	for _, want := range []string{"✓ acquiring on aldra", "paul/minimax-h3/fp8-pruned", "paul/text-encoder", "/s", " / "} {
		if !strings.Contains(download, want) {
			t.Fatalf("download history lacks %q: %q", want, download)
		}
	}
	p.On(liveEvent("accepted", nil))
	p.On(liveEvent("progress", map[string]any{"stage": "condition_text"}))
	// Reported within one frame, conditioning may be a sibling of denoising; denoising
	// reporting again, with conditioning silent, makes them a sequence.
	p.On(liveEvent("progress", map[string]any{"stage": "denoise"}))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 5, "total": 30, "step_ms": 15000}))
	at := time.Now()
	denoise := liveFrame(p, at)
	for _, want := range []string{"✓ conditioning", "▸ denoising", "step 5/30", "15.00s/step avg", "ETA ~6m15s"} {
		if !strings.Contains(denoise, want) {
			t.Fatalf("denoising display lacks %q: %q", want, denoise)
		}
	}
	// A heartbeat/old stage end cannot replace the newer work with generic running.
	p.On(liveEvent("accepted", nil))
	p.On(liveEvent("queued", nil))
	p.On(liveEvent("parked", nil))
	p.On(liveEvent("stage", map[string]any{"name": "condition_text", "ms": 1234}))
	p.On(liveEvent("phase", map[string]any{"phase": "warming"}))
	if got := liveFrame(p, at); got != denoise {
		t.Fatalf("old lifecycle events replaced the live stage: %q", got)
	}
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 5, "total": 30, "step_ms": 90000}))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 6, "total": 30, "step_ms": 15000}))
	if got := liveFrame(p, time.Now()); !strings.Contains(got, "ETA ~6m0s") {
		t.Fatalf("duplicate step biased measured ETA: %q", got)
	}
	p.On(liveEvent("failed", nil))
	if failure := liveFrame(p, time.Now()); !strings.Contains(failure, "✗ denoising") || strings.Contains(failure, "✓ denoising") {
		t.Fatalf("failed stage claimed completion: %q", failure)
	}
}

func TestReattachedProgressUsesRecordedStageBoundaries(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	start := time.Now().Add(-20 * time.Minute)
	replay := func(kind string, after time.Duration, seq int64, fields map[string]any) localapi.Event {
		e := liveEvent(kind, fields)
		e.At, e.SequenceNumber = start.Add(after).UTC().Format(time.RFC3339Nano), seq
		return e
	}
	queued := replay("queued", 0, 1, nil)
	queued.Payload = map[string]any{"wait": "rental", "reason": "historical wait diagnostic"}
	p.On(queued)
	p.On(replay("accepted", 12*time.Second, 2, nil))
	accepted := liveFrame(p, start.Add(12*time.Second))
	for _, want := range []string{"✓ waiting for a rental machine  12s", "▸ starting"} {
		if !strings.Contains(accepted, want) {
			t.Fatalf("replayed acceptance lacks %q: %q", want, accepted)
		}
	}
	if strings.Contains(accepted, "20m") || strings.Contains(accepted, "historical wait diagnostic") {
		t.Fatalf("historical wait elapsed grew until reattach: %q", accepted)
	}
	p.On(replay("progress", 15*time.Second, 3, map[string]any{"stage": "condition_text"}))
	p.On(replay("progress", 25*time.Second, 4, map[string]any{"stage": "denoise", "position": 1, "total": 30, "step_ms": 15000}))
	if transition := liveFrame(p, start.Add(25*time.Second)); !regexp.MustCompile(`✓ conditioning +wall 10s`).MatchString(transition) {
		t.Fatalf("finished stage used reattach walltime instead of recorded boundary: %q", transition)
	}
	p.On(replay("completed", 40*time.Second, 5, nil))
	if terminal := liveFrame(p, time.Now()); !regexp.MustCompile(`✓ denoising +wall 15s`).MatchString(terminal) || strings.Contains(terminal, "19m") {
		t.Fatalf("terminal stage did not freeze at its recorded completion: %q", terminal)
	}
}

func TestCapacityWaitReplacesWarmingButNeverExecution(t *testing.T) {
	p, _ := progressSink(output.Mode{Human: true, Live: true}, false)
	t.Cleanup(p.Done)
	p.On(liveEvent("phase", map[string]any{"phase": "warming", "machine": "aldra"}))
	wait := waitEnvelope("request.parked", time.Now(), map[string]any{
		"wait": "slot_busy", "waiting_on": "aldra", "reason": rawDiagnostic,
		"waiting_for": map[string]any{"number": 429, "request_id": "req-429"},
	})
	p.On(wait)
	if got := liveFrame(p, time.Now()); !strings.Contains(got, "▸ waiting for run 429 on aldra") || strings.Contains(got, rawDiagnostic) {
		t.Fatalf("fresh queue wait did not replace old preparation: %q", got)
	}
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 1, "total": 30}))
	p.On(wait)
	if got := liveFrame(p, time.Now()); strings.Contains(got, "▸ waiting") || !strings.Contains(got, "▸ denoising") {
		t.Fatalf("queue replay replaced running denoising: %q", got)
	}
}
