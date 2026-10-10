package producttest

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestRunShowRetainsSelectionTimeAttentionFallback(t *testing.T) {
	// The preferred kernel completed during the run. Its final ready state cannot
	// replace the reason recorded when a different kernel started serving.
	fallback := map[string]any{"component": "base/fl2va_dit", "preferred": "sol-attn-sage2-fixed-kmean",
		"selected": "sol-attn", "reason": "kernel build still compiling (42%)"}
	bundle, err := json.Marshal(map[string]any{"measurements": map[string]any{"execution": map[string]any{
		"degree": 1, "gpus": []any{map[string]any{"gpu": 0, "pid": 1234, "arch": "sm_90", "attention": map[string]any{
			"observed": "sol-attn", "fallbacks": []any{fallback}, "kernels": []any{
				map[string]any{"kernel": "sol-attn", "state": "ready", "served": true},
				map[string]any{"kernel": "sol-attn-sage2-fixed-kmean", "state": "ready", "served": false},
			}}}}}}})
	must(t, err)
	o := hostOwner(t, "run-show-attention-fallback")
	id := succeededWithTriage(t, o, bundle)
	defer publicationControlAPI(t, o)()
	want := "GPU 0 attention fallback (base/fl2va_dit): used sol-attn instead of sol-attn-sage2-fixed-kmean: kernel build still compiling (42%)"
	for _, args := range [][]string{{"run", "show", id}, {"run", "show", id, "--full"}} {
		code, human := runCozy(t, o.root, args...)
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("%v [%d] lost selection-time reason:\n%s", args, code, human)
		}
	}
	code, raw := runCozy(t, o.root, "run", "show", id, "--json")
	var report struct {
		GPUs []struct {
			Attention struct {
				Fallbacks []map[string]any `json:"fallbacks"`
			} `json:"attention"`
		} `json:"gpus"`
	}
	must(t, json.Unmarshal([]byte(raw), &report))
	if code != 0 || len(report.GPUs) != 1 || len(report.GPUs[0].Attention.Fallbacks) != 1 {
		t.Fatalf("run show --json [%d] lost fallback receipt:\n%s", code, raw)
	}
	for key, want := range fallback {
		if got := report.GPUs[0].Attention.Fallbacks[0][key]; got != want {
			t.Fatalf("fallback %s = %v, want %v", key, got, want)
		}
	}
}
