package producttest

import (
	"flag"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

var gpuBudgetChild = flag.Bool("gpu-budget-child", false, "print the runtime.yaml this process's config.yaml gives (a child of TestMachineGPUBudgetPassesThroughVerbatim)")

// machine.gpu_budget passes through config.yaml verbatim, a size or a per-GPU map, into the
// machine Runtime's runtime.yaml as gpu.budget. Each case loads its config in a fresh process,
// as the CLI does once per invocation.
func TestMachineGPUBudgetPassesThroughVerbatim(t *testing.T) {
	if *gpuBudgetChild {
		cfg, problem := config.Load()
		fatal(t, problem)
		if cfg.MachineGPUBudget == nil {
			_, _ = os.Stdout.WriteString("no budget\n")
			return
		}
		written, err := config.RuntimeYAML(cfg.MachineGPUBudget)
		must(t, err)
		_, _ = os.Stdout.Write(written)
		return
	}
	for _, row := range []struct{ config, runtime string }{
		{"machine:\n  gpu_budget: 3GiB\n", "gpu:\n    budget: 3GiB\n"},
		{"machine:\n  gpu_budget: 4294967296\n", "gpu:\n    budget: 4294967296\n"},
		{"machine:\n  gpu_budget: {0: 3GiB, 1: 6GiB}\n", "gpu:\n    budget:\n        0: 3GiB\n        1: 6GiB\n"},
		{"tensorhub_url: local\n", "no budget\n"},
	} {
		root := t.TempDir()
		must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(row.config), 0o600))
		child := exec.Command(os.Args[0], "-test.run=^TestMachineGPUBudgetPassesThroughVerbatim$", "-gpu-budget-child")
		child.Env = append(os.Environ(), "COZY_HOME="+root)
		out, err := child.Output()
		if err != nil || !strings.Contains(string(out), row.runtime) {
			t.Fatalf("%q gave %q (%v), want %q", row.config, out, err, row.runtime)
		}
	}
}
