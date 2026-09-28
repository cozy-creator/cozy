package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// A real official-index framework is part of local capture, not a private overlay.
func TestOfficialCPUPyTorchInvocableUsesExactCapturedFramework(t *testing.T) {
	integration(t)
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime execution peer")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	control := filepath.Join(t.TempDir(), "control")
	python := filepath.Join(control, "bin", "python")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", python, "--torch-backend", "cpu", wheel, "torch==2.13.0"}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("CPU SDK: %v %s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-pytorch-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if t.Failed() {
			t.Log("CPU PyTorch proof retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "library")
	must(t, os.MkdirAll(library, 0700))
	metadata := fmt.Sprintf(`[project]
name="cpu-torch-operations"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s", "torch>=2.13.0"]
[project.entry-points."cozy.application"]
default="cpu_torch_ops:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["cpu_torch_ops.py"]
[tool.uv.sources]
cozy-runtime={path=%s}
torch={index="pytorch-cpu"}
[[tool.uv.index]]
name="pytorch-cpu"
url="https://download.pytorch.org/whl/cpu"
explicit=true
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject=\"cpu_torch_ops:app\"\n"), 0600))
	module := `import msgspec
from cozy_runtime.author import App, Context, invocable
app=App()
class Result(msgspec.Struct):
    value:int
    device:str
    cuda:bool
@invocable(memoize=False)
async def square_sum(ctx:Context,*,offset:int)->Result:
    import torch
    assert torch.__version__.endswith("+cpu")
    values=torch.arange(4,dtype=torch.int64,device="cpu")
    assert not torch.cuda.is_initialized()
    return Result(int((values*values).sum())+offset,str(values.device),torch.version.cuda is not None)
app.job(square_sum)
`
	must(t, os.WriteFile(filepath.Join(library, "cpu_torch_ops.py"), []byte(module), 0600))
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s","cpu-torch-operations>=1.0.0","torch>=2.13.0"]
# [tool.uv.sources]
# cozy-runtime={path=%s}
# cpu-torch-operations={path="./library",editable=true}
# torch={index="pytorch-cpu"}
# [[tool.uv.index]]
# name="pytorch-cpu"
# url="https://download.pytorch.org/whl/cpu"
# explicit=true
# ///
from cpu_torch_ops import square_sum
async def main()->int:
    result=await square_sum(offset=3)
    assert result.device=="cpu" and not result.cuda
    return result.value
`, version, strconv.Quote(wheel))
	script := filepath.Join(project, "calculate.py")
	must(t, os.WriteFile(script, []byte(body), 0600))
	status, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":17`) {
		t.Fatalf("official CPU invocable [%d]: %s", status, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	calls := machineChildren(t, root, store, "1")
	if len(calls) != 1 || calls[0].Executions != 1 || calls[0].State != "succeeded" {
		t.Fatalf("CPU invocable was not executed by Runtime: %+v", calls)
	}
	var result struct {
		Value  int    `json:"value"`
		Device string `json:"device"`
		CUDA   bool   `json:"cuda"`
	}
	must(t, json.Unmarshal(calls[0].Result, &result))
	if result.Value != 17 || result.Device != "cpu" || result.CUDA {
		t.Fatalf("numerical result was not CPU-only: %+v", result)
	}
	t.Logf("ordinary CLI CPU invocable verified: %+v; %s", result, out)
}
