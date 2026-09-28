package config

import (
	"reflect"
	"strings"
	"testing"

	"go.yaml.in/yaml/v3"
)

// machine.gpu_budget passes through verbatim, a size or a per-GPU map, into the Runtime's
// runtime.yaml as gpu.budget.
func TestMachineGPUBudgetPassesThroughVerbatim(t *testing.T) {
	for _, row := range []struct{ config, runtime string }{
		{"machine:\n  gpu_budget: 3GiB\n", "gpu:\n    budget: 3GiB\n"},
		{"machine:\n  gpu_budget: 4294967296\n", "gpu:\n    budget: 4294967296\n"},
		{"machine:\n  gpu_budget: {0: 3GiB, 1: 6GiB}\n", "gpu:\n    budget:\n        0: 3GiB\n        1: 6GiB\n"},
	} {
		resolved, err := fileYAML(strings.NewReader(row.config))
		if err != nil {
			t.Fatalf("%q: %v", row.config, err)
		}
		written, err := yaml.Marshal(map[string]any{"gpu": map[string]any{"budget": resolved.gpuBudget}})
		if err != nil || string(written) != row.runtime {
			t.Fatalf("%q wrote %q (%v), want %q", row.config, written, err, row.runtime)
		}
	}
	resolved, err := fileYAML(strings.NewReader("tensorhub_url: local\n"))
	if err != nil || resolved.gpuBudget != nil || !reflect.DeepEqual(resolved.ignored, []string(nil)) {
		t.Fatalf("no budget configured: %v %v %v", resolved.gpuBudget, resolved.ignored, err)
	}
}
