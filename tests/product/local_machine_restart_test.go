package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

// A machine stopped while its Runtime winds down a running job runs a job on its very next
// boot (run 1486): the Host waits for that Runtime to exit, killing it if it stalls, and the
// new Runtime waits for the old one's worker root instead of failing its boot.
func TestAMachineStoppedMidJobRunsAJobOnItsNextBoot(t *testing.T) {
	_, root, _, store := parityMachines(t)
	if code, out := runCozy(t, root, "package", "install", restartProject(t), "--editable"); code != 0 {
		t.Fatalf("installing the restart package [exit %d]\n%s", code, out)
	}
	go runCozy(t, root, "run", "local/restart-proof/slow", "seconds=120", "--json", "--idempotency-key", "slow")
	eventually(t, root, "the slow job running on this computer's machine", func() bool {
		row, problem := store.RequestByIdempotencyKey("slow")
		if problem != nil || row == nil {
			return false
		}
		link, problem := store.MachineExecution(row.ID)
		if problem != nil || link == nil || link.MachineID != machines.Local {
			return false
		}
		_, shown := runCozy(t, root, "run", "show", row.ID, "--json")
		var run struct {
			Status string `json:"status"`
			Events []struct {
				Type string `json:"type"`
			} `json:"events"`
		}
		if json.Unmarshal([]byte(shown), &run) != nil || run.Status != "in_progress" {
			return false
		}
		// Dispatch is the lifecycle fact. Older SDKs report unknown execution
		// duration, and this sleeping handler has no periodic timing samples.
		for _, event := range run.Events {
			if event.Type == "machine.running" {
				return true
			}
		}
		return false
	})
	if code, out := runCozy(t, root, "machine", "stop"); code != 0 {
		t.Fatalf("machine stop [exit %d]\n%s", code, out)
	}
	code, out := runCozy(t, root, "run", "local/restart-proof/add", "value=41", "--await", "--json")
	if code != 0 || !strings.Contains(out, `"value":42`) {
		t.Fatalf("the first job on the next boot failed [exit %d]\n%s", code, out)
	}
}

// restartProject is local/restart-proof: a job that runs until canceled, and a quick one.
func restartProject(t *testing.T) string {
	t.Helper()
	project := filepath.Join(t.TempDir(), "restart-proof")
	must(t, os.MkdirAll(project, 0o700))
	sources := ""
	if *machineRuntimeWheel != "" {
		sources = fmt.Sprintf("[tool.uv.sources]\ncozy-runtime={path=%q}\ntensorfs={path=%q}\n", *machineRuntimeWheel, *machineTensorFSWheel)
	}
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(`[project]
name="restart-proof"
version="0.0.1"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=`+machines.RuntimeFloor+`", "msgspec>=0.19"]
[project.entry-points."cozy.application"]
default="restart_proof:app"
`+sources+`[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["restart_proof.py"]
`), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"restart_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "restart_proof.py"), []byte(`import time

import msgspec
from cozy_runtime.author import App


class SlowRequest(msgspec.Struct, forbid_unknown_fields=True):
    seconds: int


class AddRequest(msgspec.Struct, forbid_unknown_fields=True):
    value: int


class AddResult(msgspec.Struct):
    value: int


app = App()


@app.job
def slow(payload: SlowRequest) -> AddResult:
    time.sleep(payload.seconds)
    return AddResult(value=payload.seconds)


@app.job
def add(payload: AddRequest) -> AddResult:
    return AddResult(value=payload.value + 1)
`), 0o600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("locking the restart package: %v\n%s", err, out)
	}
	return project
}
