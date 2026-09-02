package orchestrator

import "testing"

func TestProgressFraction(t *testing.T) {
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
		got, ok := progressFraction(test.value)
		if ok != test.ok || ok && got != test.want {
			t.Fatalf("progressFraction(%v) = (%v, %v), want (%v, %v)",
				test.value, got, ok, test.want, test.ok)
		}
	}
}
