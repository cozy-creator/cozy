package producttest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateImageDependenciesRemainInTheInstalledGraph(t *testing.T) {
	project := t.TempDir()
	metadata := `[project]
name = "image-closure-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["msgspec>=0.21.1"]
[tool.uv]
# This fixture selects exact bytes without publishing an exact requirement.
constraint-dependencies = ["msgspec==0.21.1"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["image_closure_proof.py"]
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "image_closure_proof.py"), []byte("VALUE = 7\n"), 0600))
	command := exec.Command("uv", "lock", "--project", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock image dependency fixture: %v\n%s", err, output)
	}
	venv := filepath.Join(t.TempDir(), "venv")
	environment, problem := install.MaterializeEnvironment(project, venv)
	fatal(t, problem)
	python, problem := install.BasePython(venv)
	fatal(t, problem)
	graph, problem := packagepublish.WheelClosures(context.Background(), project, python,
		environment.Closure, "image-closure-proof", environment.Extra)
	fatal(t, problem)
	if graph["msgspec"]["msgspec"] != "0.21.1" || !strings.Contains(environment.Closure, "msgspec==0.21.1") {
		t.Fatalf("exact installed image dependency was lost: closure=%s graph=%v", environment.Closure, graph)
	}
}

func TestRuntimeCaptureExcludesDefaultDevelopmentGroups(t *testing.T) {
	project := t.TempDir()
	metadata := `[project]
name = "runtime-group-proof"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["packaging>=26.2"]
[dependency-groups]
dev = ["ruff==0.16.4"]
qa = ["mypy==2.3.1"]
[tool.uv]
default-groups = ["dev", "qa"]
constraint-dependencies = ["packaging==26.2"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["runtime_group_proof.py"]
[project.entry-points."cozy.application"]
default = "runtime_group_proof:app"
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject='runtime_group_proof:app'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "runtime_group_proof.py"), []byte(`import msgspec
from cozy_runtime.author import App, Context
app = App()
class Result(msgspec.Struct):
    value: int
class Request(msgspec.Struct):
    value: int = 7
@app.job
def check(ctx: Context, payload: Request) -> Result:
    return Result(payload.value)
`), 0600))
	command := exec.Command("uv", "lock", "--project", project)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock runtime/dev fixture: %v\n%s", err, output)
	}
	lock, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	must(t, err)
	for _, name := range []string{"ruff", "mypy"} {
		if !strings.Contains(string(lock), `name = "`+name+`"`) {
			t.Fatalf("fixture did not lock development package %s", name)
		}
	}
	environment, problem := install.MaterializeEnvironment(project, filepath.Join(t.TempDir(), "venv"))
	fatal(t, problem)
	if !strings.Contains(environment.Closure, "packaging==26.2") {
		t.Fatalf("runtime dependency was lost: %s", environment.Closure)
	}
	for _, name := range []string{"ruff", "mypy", "mypy-extensions"} {
		if strings.Contains(environment.Closure, name+"==") {
			t.Fatalf("development package %s leaked into runtime closure: %s", name, environment.Closure)
		}
	}
}
