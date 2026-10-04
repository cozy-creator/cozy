package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// This computer's machine runs a call on its own GPU: the pod's Host binary, launched and
// registered by the daemon, a Runtime that measured the device, and a package environment
// with CUDA Torch that the machine prepared itself.
func TestLocalMachineRunsOnThisComputersGPU(t *testing.T) {
	integration(t)
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	model, count := hostAccelerators()
	if count == 0 {
		t.Skip("no NVIDIA device on this computer")
	}
	h := newMachineHub(t)
	root, err := os.MkdirTemp("", "czg")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("GPU evidence retained at %s\nHost log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	project := filepath.Join(t.TempDir(), "gpu-proof")
	must(t, os.MkdirAll(project, 0o700))
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="gpu-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+runtimeFloor+`", "msgspec>=0.19", "torch==2.13.0"]
[project.entry-points."cozy.application"]
default="gpu_proof:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["gpu_proof.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"gpu_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "gpu_proof.py"), []byte(`import msgspec
from cozy_runtime.author import App


class SumRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class SumResult(msgspec.Struct):
    value: int
    device: str


app = App()


@app.job(accelerator=True)
def gpu_sum(payload: SumRequest) -> SumResult:
    import torch

    x = torch.full((1024,), float(payload.value), device="cuda")
    return SumResult(value=int(x.sum().item()), device=torch.cuda.get_device_name(0))
`), 0o600))
	lock := exec.Command("uv", "lock", "--project", project)
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("locking the GPU package: %v\n%s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "local/gpu-proof/gpu_sum", "value=3", "--await", "--json", "--idempotency-key", "gpu-proof")
	if code != 0 || !strings.Contains(out, `"value":3072`) || !strings.Contains(out, model) {
		t.Fatalf("the GPU call [exit %d]\n%s", code, out)
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("gpu-proof")
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || link.MachineID != machines.Local || !link.Collected {
		t.Fatalf("the GPU call was not a collected execution on this computer's machine: %+v", link)
	}
	events, problem := store.EvidenceEvents(request.ID, 1000)
	fatal(t, problem)
	granted := ""
	for _, event := range events {
		if event.Type == "machine.gpu.grant" {
			raw, _ := json.Marshal(event.Payload["ordinals"])
			granted = string(raw)
		}
	}
	if granted != "[0]" {
		t.Fatalf("the Runtime granted %q, not this computer's one device", granted)
	}
	t.Logf("%s ran on %s (%d device), grant %s: %s", request.ID, model, count, granted, fmt.Sprint(strings.TrimSpace(out)))
}

// hostAccelerators are the NVIDIA devices this run's machines may use: none when the run hides
// them (CUDA_VISIBLE_DEVICES set and empty), as the machines it starts inherit.
func hostAccelerators() (string, int) {
	if visible, named := os.LookupEnv("CUDA_VISIBLE_DEVICES"); named && strings.TrimSpace(visible) == "" {
		return "", 0
	}
	out, err := exec.Command("nvidia-smi", "--query-gpu=name", "--format=csv,noheader").Output()
	names := strings.Split(strings.TrimSpace(string(out)), "\n")
	if err != nil || names[0] == "" {
		return "", 0
	}
	return strings.TrimSpace(names[0]), len(names)
}
