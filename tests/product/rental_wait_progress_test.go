package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/output"
)

func rentalWaitEvents(at time.Time, machine, spend string) []localapi.Event {
	return []localapi.Event{
		waitEnvelope("request.rentals", at, map[string]any{"line": spend}),
		waitEnvelope("request.placement", at, map[string]any{
			"line":       "placement: wait for " + machine + " (h100-pcie, fp8-adaln-pruned) to attach — balanced",
			"candidates": []any{map[string]any{"machine": machine, "verdict": "attaching"}},
		}),
	}
}

func TestRentalWaitProgressDeduplicatesAlternatingNotices(t *testing.T) {
	const spend = "rentals: 4 remote machines running · $7.08/hour"
	const changedSpend = "rentals: 5 remote machines running · $9.97/hour"
	for _, terminal := range []bool{false, true} {
		t.Run(map[bool]string{false: "redirected", true: "terminal"}[terminal], func(t *testing.T) {
			p, buf := progressSink(output.Mode{Human: true, Live: terminal}, false)
			defer p.Done()
			at := time.Now().Add(-55 * time.Second)
			for i := 0; i < 20; i++ {
				for _, e := range rentalWaitEvents(at, "misery", spend) {
					p.On(e)
				}
			}
			got := buf.String()
			if strings.Count(got, spend) != 1 || strings.Count(got, "placement: wait for misery") != 1 {
				t.Fatalf("stable interleaved notices were appended repeatedly: %q", got)
			}
			if terminal {
				got = liveFrame(p, time.Now())
			}
			if !strings.Contains(got, map[bool]string{false: "waiting for a rental machine · elapsed 55s",
				true: "▸ waiting for a rental machine · 55s"}[terminal]) {
				t.Fatalf("attaching did not become the elapsed rental wait: %q", got)
			}
			if !terminal && (strings.ContainsAny(got, "\r\033") ||
				strings.Count(got, "waiting for a rental machine") != 1) {
				t.Fatalf("redirected waiting was not sparse plain text: %q", got)
			}
			for i := 0; i < 2; i++ {
				for _, e := range rentalWaitEvents(time.Now(), "isao", changedSpend) {
					p.On(e)
				}
			}
			got = buf.String()
			if strings.Count(got, changedSpend) != 1 || strings.Count(got, "placement: wait for isao") != 1 {
				t.Fatalf("meaningful spend/placement changes were lost or repeated: %q", got)
			}
			if terminal {
				// The clock advances even when the provider sends no new event.
				deadline := time.Now().Add(2 * time.Second)
				for !strings.Contains(buf.String(), "machine · 56s") && time.Now().Before(deadline) {
					time.Sleep(10 * time.Millisecond)
				}
				if !strings.Contains(buf.String(), "machine · 56s") {
					t.Fatalf("quiet rental wait did not tick in place: %q", buf.String())
				}
				p.On(waitEnvelope("request.phase", time.Now(), map[string]any{
					"value": map[string]any{"phase": "downloading", "machine": "isao"},
				}))
				p.On(waitEnvelope("request.completed", time.Now(), nil))
				if got := liveFrame(p, time.Now()); strings.Count(got, "✓") != 2 ||
					!strings.Contains(got, "✓ downloading on isao") {
					t.Fatalf("transition discarded finished wait/download stages: %q", got)
				}
			}
		})
	}
}

func TestRentalWaitProgressPreservesJSONAndChosenPlacement(t *testing.T) {
	events := rentalWaitEvents(time.Now(), "misery", "rentals: unchanged")
	p, buf := progressSink(output.Mode{JSON: true}, true)
	for i := 0; i < 2; i++ {
		for _, e := range events {
			p.On(e)
		}
	}
	p.Done()
	lines := strings.Split(strings.TrimSpace(buf.String()), "\n")
	if len(lines) != 4 {
		t.Fatalf("JSON events were deduplicated: %q", buf.String())
	}
	for i, line := range lines {
		want, err := json.Marshal(events[i%len(events)])
		must(t, err)
		if line != string(want) {
			t.Fatalf("JSON envelope changed: got %s, want %s", line, want)
		}
	}
	p, buf = progressSink(output.Mode{Human: true, Live: true}, false)
	defer p.Done()
	chosen := events[1]
	chosen.Payload["line"] = "placement: reuse darkrai (h100-nvl, fp8-adaln-pruned) — balanced"
	chosen.Payload["candidates"] = append(chosen.Payload["candidates"].([]any), map[string]any{
		"machine": "darkrai", "verdict": "chosen",
	})
	p.On(chosen)
	if got := buf.String() + liveFrame(p, time.Now()); strings.Contains(got, "waiting") || !strings.Contains(got, "placement: reuse darkrai") {
		t.Fatalf("an attaching alternative overrode the chosen placement: %q", got)
	}
}
