package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/machines"
)

// A local/ root calling a local/ child whose Model default is org-relative: the child resolves
// it under the account owning the run, as its root would, on this computer's machine and on a
// rental.
func TestALocalChildResolvesItsOwnersModel(t *testing.T) {
	h, root, _, _ := parityMachines(t)
	resolved := seedProbe(t, h, root)
	h.mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"name":"proof"}`))
	})
	doors := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == "proof/probe@1.0.0" && r.URL.Query().Get("lane") == "bf16" {
			_ = json.NewEncoder(w).Encode(resolved)
			return
		}
		doors.ServeHTTP(w, r)
	})

	project := t.TempDir()
	child := filepath.Join(project, "child")
	must(t, os.MkdirAll(child, 0o700))
	write := func(dir, name, body string) {
		t.Helper()
		must(t, os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600))
	}
	source := machines.Source{RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel}
	metadata := func(name, module, extra, sources string) string {
		if source.RuntimeWheel != "" {
			sources += fmt.Sprintf("\ncozy-runtime={path=%q}\ntensorfs={path=%q}", source.RuntimeWheel, source.TensorFSWheel)
		}
		return fmt.Sprintf(`[project]
name=%q
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s", "msgspec>=0.19"%s]
[tool.uv.sources]%s
[project.entry-points."cozy.application"]
default=%q
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=[%q]
`, name, machines.RuntimeFloor, extra, sources, module+":app", module+".py")
	}
	// Serving a Model derives its placement under torch.
	write(child, "pyproject.toml", metadata("owner-child", "owner_child", `, "torch==2.13.0"`, ""))
	write(child, "package.toml", "[application]\nobject=\"owner_child:app\"\n")
	write(child, "owner_child.py", `import msgspec
import torch
from cozy_runtime.author import App, Config, Context, Loader, Model, invocable, uses_components


class Unet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


class Pipeline:
    def __init__(self) -> None:
        self.components = {"unet": Unet()}


def build(config: Config) -> Pipeline:
    return Pipeline()


class Probe(Model[Pipeline]):
    def load(self, loader: Loader) -> None:
        self.pipe = loader.construct(Pipeline, factory=build)

    @uses_components("unet")
    def weight(self) -> int:
        return int(self.pipe.components["unet"].weight.sum().item())


class Result(msgspec.Struct):
    value: int


@invocable(defaults={"source": [{"gpu": "*", "lane": "probe@1.0.0/bf16"}]})
async def touch(ctx: Context, *, value: int, source: Probe) -> Result:
    return Result(value + 1 + source.weight())


app = App()
app.entrypoint(touch)
`)
	write(project, "pyproject.toml", metadata("owner-root", "owner_root", `, "owner-child>=0.1.0"`, "\nowner-child={path=\"./child\"}"))
	write(project, "package.toml", "[application]\nobject=\"owner_root:app\"\n")
	write(project, "owner_root.py", `import msgspec
from cozy_runtime.author import App
from owner_child import Result, touch


class Input(msgspec.Struct):
    value: int


app = App()


@app.job
async def main(payload: Input) -> Result:
    return await touch(value=payload.value)
`)
	for _, dir := range []string{child, project} {
		if out, err := exec.Command("uv", "lock", "--project", dir).CombinedOutput(); err != nil {
			t.Fatalf("locking %s: %v\n%s", dir, err, out)
		}
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("editable install [exit %d]\n%s", code, out)
	}
	for venue, args := range map[string][]string{"local": nil, "tessa": {"--rental=tessa"}} {
		code, out := runCozy(t, root, append([]string{"run", "local/owner-root/main", "value=1", "--await", "--json"}, args...)...)
		if code != 0 || !strings.Contains(out, `"value":2`) {
			t.Fatalf("local/ root calling a local/ child on %s [exit %d]\n%s", venue, code, out)
		}
	}
}
