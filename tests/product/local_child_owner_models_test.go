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

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

// A local/ root calling a local/ child whose Model default is org-relative: the child resolves
// it under the account owning the run, as its root would, on this computer's machine and on a
// rental.
func TestALocalChildResolvesItsOwnersModel(t *testing.T) { localChildOwnerModel(t, false) }
func TestSameInstallationDynamicModelUsesAvailableScopedHub(t *testing.T) {
	localChildOwnerModel(t, true)
}

func localChildOwnerModel(t *testing.T, sameInstallation bool) {
	// Serving this Model executes its components on a measured accelerator.
	// Synthetic inventory covers routing; it does not qualify numerical execution.
	if _, count := hostAccelerators(); count == 0 {
		t.Skip("requires a real NVIDIA device for serving Model defaults; run this hardware gate separately")
	}
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
`, name, runtimeFloor, extra, sources, module+":app", module+".py")
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
	rootMetadata := metadata("owner-root", "owner_root", `, "owner-child>=0.1.0"`, "\nowner-child={path=\"./child\"}")
	if sameInstallation {
		body, err := os.ReadFile(filepath.Join(child, "owner_child.py"))
		must(t, err)
		write(project, "owner_child.py", string(body))
		rootMetadata = metadata("owner-root", "owner_root", `, "torch==2.13.0"`, "")
		rootMetadata = strings.Replace(rootMetadata, `only-include=["owner_root.py"]`, `only-include=["owner_root.py", "owner_child.py"]`, 1)
	}
	write(project, "pyproject.toml", rootMetadata)
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
	if sameInstallation {
		path := filepath.Join(project, "owner_root.py")
		body, err := os.ReadFile(path)
		must(t, err)
		write(project, "owner_root.py", strings.Replace(string(body), "from owner_child import Result, touch", "from owner_child import Result, touch, Probe", 1)+"\napp.entrypoint(touch)\n")
	}
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

// A package may expose unrelated model-bearing functions without making its
// selected scalar root depend on a Hub account or network.
func TestUnselectedModelCallableKeepsLocalRootOffline(t *testing.T) {
	for _, refused := range []bool{false, true} {
		t.Run(fmt.Sprintf("refused_optional=%v", refused), func(t *testing.T) { unselectedModelOffline(t, refused) })
	}
}

func unselectedModelOffline(t *testing.T, refused bool) {
	root, machine, _ := scopedMachine(t, false)
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	settings := "tensorhub_url: http://127.0.0.1:1\n"
	retained := ""
	if refused {
		hub := newScopedAccessHub(t)
		_, problem := machine.Ensure(t.Context(), hub.server.URL, hub.client("first-login"), true)
		fatal(t, problem)
		retained = string(machineScopedFiles(t, machine))
		served := hub.server.Config.Handler
		hub.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Header.Get("Authorization") == "Bearer refused-login" {
				http.Error(w, "login revoked", http.StatusForbidden)
				return
			}
			served.ServeHTTP(w, r)
		})
		settings = "tensorhub_url: " + hub.server.URL + "\ntensorhub_token: refused-login\n"
	}
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(settings), 0600))
	project := t.TempDir()
	metadata := fmt.Sprintf(`[project]
name="offline-model-sibling"
version="0.1.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=0.18.87", "msgspec>=0.19"]
[tool.uv.sources]
cozy-runtime={path=%q}
tensorfs={path=%q}
[project.entry-points."cozy.application"]
default="offline_models:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["offline_models.py"]
`, *machineRuntimeWheel, *machineTensorFSWheel)
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject=\"offline_models:app\"\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "offline_models.py"), []byte(`import msgspec
from cozy_runtime.author import App, Context, Loader, Model, invocable
app = App()
class Unused(Model[object]):
    def load(self, loader: Loader) -> None:
        raise AssertionError("unselected model must never prepare")
class Result(msgspec.Struct):
    value: int
@invocable()
async def unrelated(ctx: Context, *, source: Unused) -> Result:
    return Result(0)
@invocable()
async def scalar(ctx: Context, *, value: int) -> Result:
    return Result(value + 1)
app.entrypoint(unrelated)
app.job(scalar)
`), 0600))
	if out, err := exec.Command("uv", "lock", "--project", project).CombinedOutput(); err != nil {
		t.Fatalf("lock offline sibling: %v %s", err, out)
	}
	if code, out := runCozy(t, root, "package", "install", project, "--editable"); code != 0 {
		t.Fatalf("install offline sibling [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "run", "local/offline-model-sibling/scalar", "value=1", "--idempotency-key=offline-selected", "--await", "--json"); code != 0 || !strings.Contains(out, `"value":2`) {
		t.Fatalf("unselected Model required Hub access [%d]: %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey("offline-selected")
	fatal(t, problem)
	if request == nil {
		t.Fatal("offline run missing")
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil {
		t.Fatal("offline execution missing")
	}
	var frozen pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &frozen))
	if frozen.Hub != "" || frozen.GetReleaseRoot().GetHub() != "" {
		t.Fatal("optional attachment failure borrowed retained authority")
	}
	if refused && string(machineScopedFiles(t, machine)) != retained {
		t.Fatal("refused optional attachment changed retained account authority")
	}
}
