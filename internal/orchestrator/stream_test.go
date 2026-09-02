package orchestrator

import "testing"

func TestProgressCoordinates(t *testing.T) {
	for _, test := range []struct {
		value any
		want  float64
		ok    bool
	}{
		{value: 0.25, want: 0.25, ok: true},
		{value: map[string]any{"fraction": 0.75}, want: 0.75, ok: true},
		{value: map[string]any{"value": 1.0}, want: 1, ok: true},
		{value: map[string]any{"fraction": 1.1}},
		{value: map[string]any{"name": "denoise"}},
	} {
		_, got, _, _, ok := progressCoordinates(test.value)
		if ok != test.ok || ok && got != test.want {
			t.Fatalf("progressCoordinates(%v) = (%v, %v), want (%v, %v)",
				test.value, got, ok, test.want, test.ok)
		}
	}
}

func TestLatestProgressAveragesMeasuredStepTime(t *testing.T) {
	frames := newFanout()
	orchestrator := &Orchestrator{frames: frames}
	for _, value := range []map[string]any{
		{"name": "denoise", "fraction": 0.1, "position": 1.0, "step_ms": 100.0},
		{"name": "denoise", "fraction": 0.2, "position": 2.0, "step_ms": 300.0},
	} {
		frames.publish(Frame{RequestID: "request", Attempt: 2, Type: "progress", Value: value})
	}
	progress, ok := orchestrator.LatestProgress("request", 2)
	if !ok || progress.Stage != "denoise" || progress.Fraction != 0.2 ||
		!progress.Estimated || progress.RemainingMS != 1600 {
		t.Fatalf("unexpected live progress: %+v, ok=%v", progress, ok)
	}
	if _, ok := orchestrator.LatestProgress("request", 1); ok {
		t.Fatal("a retry inherited the prior attempt's progress")
	}
	frames.forget("request")
	if _, ok := orchestrator.LatestProgress("request", 2); ok {
		t.Fatal("a terminal request retained live progress")
	}
}
