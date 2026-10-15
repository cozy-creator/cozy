package producttest

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/launch"
)

// descriptor_id is one function's identity in one `entrypoints` list:
// sha256("cozy.runtime.descriptor\0" + its canonical entry). Cozy derives it from the interface
// it installed; the Runtime's own `describe <fn> --json` must print the same bytes for every
// shape: a CPU writer of weights outputs reading a Weights[M] source, a constructed slot, a
// model-less GPU function and an internal invocable.
func TestDescriptorIDAgreesWithRuntimeDescribe(t *testing.T) {
	if *machineRuntimeWheel == "" {
		t.Skip("requires -machine-runtime-wheel=<Runtime with one entrypoints list>")
	}
	tool := filepath.Join(filepath.Dir(machinePython(t)), "cozy-runtime")
	project := t.TempDir()
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"identity_proof:app\"\n"), 0o600))
	must(t, os.WriteFile(filepath.Join(project, "identity_proof.py"), []byte(`from typing import Any
import msgspec
from cozy_runtime.author import App, Context, Model, ModelArtifact, Weights, WeightsOutput, invocable

class Probe(Model):
    pass

class Request(msgspec.Struct):
    prompt: str
    steps: int = 4

class Result(msgspec.Struct):
    text: str

class Count(msgspec.Struct):
    n: int

app = App()

@invocable
async def convert(ctx: Context, *, source: Weights[Any], factor: float = 0.5) -> ModelArtifact:
    raise NotImplementedError

app.entrypoint(convert, weights=(WeightsOutput("fp8", max_new_bytes=1 << 20),), accelerator=False)

@invocable(memoize=True)
async def helper(ctx: Context, *, n: int) -> Count:
    return Count(n)

app.entrypoint(internal=True, accelerator=False)(helper)

@app.entrypoint
def generate(payload: Request, model: Probe) -> Result:
    return Result(payload.prompt)

@app.entrypoint
def plain(payload: Request) -> Result:
    return Result(payload.prompt)
`), 0o600))
	described, err := exec.Command(tool, "--json", "--dir", project, "describe").Output()
	if err != nil {
		t.Fatalf("describe: %v\n%s", err, described)
	}
	var listed struct {
		Document json.RawMessage `json:"document"`
	}
	raw := described
	if json.Unmarshal(described, &listed) == nil && len(listed.Document) > 0 {
		raw = listed.Document
	}
	iface, problem := launch.DecodePackageInterface(raw)
	fatal(t, problem)
	if len(iface.Entrypoints) != 4 || len(iface.Unavailable) != 0 {
		t.Fatalf("the interface lost a function: %d read, unavailable %v\n%s", len(iface.Entrypoints), iface.Unavailable, raw)
	}
	path := filepath.Join(t.TempDir(), launch.PackageInterfaceFile)
	must(t, os.WriteFile(path, raw, 0o600))
	for _, fn := range iface.Entrypoints {
		out, err := exec.Command(tool, "--json", "--package-interface", path, "describe", fn.Name).Output()
		if err != nil {
			t.Fatalf("describe %s: %v\n%s", fn.Name, err, out)
		}
		var one struct {
			DescriptorID string `json:"descriptor_id"`
			Document     struct {
				DescriptorID string `json:"descriptor_id"`
			} `json:"document"`
		}
		must(t, json.Unmarshal(out, &one))
		runtime := one.DescriptorID
		if runtime == "" {
			runtime = one.Document.DescriptorID
		}
		if runtime == "" || runtime != fn.DescriptorID {
			t.Fatalf("%s: Cozy derived %s, the Runtime describes %s\n%s", fn.Name, fn.DescriptorID, runtime, out)
		}
	}
	convert, problem := iface.Function("convert")
	fatal(t, problem)
	if convert.NeedsAccelerator() || len(convert.WeightsOutputs) != 1 || len(convert.Models) != 1 || !convert.Models[0].Weights {
		t.Fatalf("the CPU writer lost its declaration: %+v", convert)
	}
	for _, name := range []string{"generate", "plain"} {
		if fn, _ := iface.Function(name); !fn.NeedsAccelerator() {
			t.Fatalf("%s did not default to the machine's GPU", name)
		}
	}
}
