package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestPrivateBuildPrunesExplicitCUDAFamilies(t *testing.T) {
	project := t.TempDir()
	metadata := `[project]
name = "base-prefix-fixture"
version = "0.0.1"
requires-python = ">=3.12,<3.13"
dependencies = ["nvidia-cublas==13.1.1.3", "nvidia-cudnn-cu13==9.20.0.48", "cuda-toolkit==13.0.3.0", "packaging==26.3"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["base_prefix_fixture.py"]
[project.entry-points."cozy.application"]
default = "base_prefix_fixture:app"
[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
`
	must(t, os.WriteFile(filepath.Join(project, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(project, "package.toml"), []byte("[application]\nobject = \"base_prefix_fixture:app\"\n"), 0600))
	must(t, os.WriteFile(filepath.Join(project, "base_prefix_fixture.py"), []byte(`import msgspec
from cozy_runtime.author import App, Context
class Request(msgspec.Struct, forbid_unknown_fields=True):
    value: int = 1
class Result(msgspec.Struct):
    value: int
app = App()
@app.job
def run(ctx: Context, payload: Request) -> Result:
    return Result(payload.value)
`), 0600))
	command := exec.Command("uv", "lock", "--project", project)
	if out, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock exact registry fixture: %v\n%s", err, out)
	}
	// These are explicit roots, so pruning torch cannot remove them. Only lock
	// metadata is read; the 789 MB of CUDA wheels are never downloaded by Build.
	export := filepath.Join(t.TempDir(), "pylock.toml")
	command = exec.Command("uv", "export", "--locked", "--no-dev", "--no-emit-project", "--format", "pylock.toml", "--output-file", export, "--project", project, "--prune", "torch")
	if out, err := command.CombinedOutput(); err != nil {
		t.Fatalf("export fixture: %v\n%s", err, out)
	}
	raw, err := os.ReadFile(export)
	must(t, err)
	for _, name := range []string{"nvidia-cublas", "nvidia-cudnn-cu13", "cuda-toolkit"} {
		if !strings.Contains(string(raw), `name = "`+name+`"`) {
			t.Fatalf("fixture did not retain explicit image root %s", name)
		}
	}
	pack, problem := packagepublish.PrepareLocalFrom(project)
	fatal(t, problem)
	defer pack.Close()
	if problem := pack.Build(t.Context()); problem != nil {
		t.Fatalf("build: %s; remedy: %s", problem.Message, problem.Remedy)
	}
	if len(pack.Registry) != 1 || pack.Registry[0].Name != "packaging" || pack.Registry[0].Version != "26.3" || pack.Registry[0].Size != 129956 {
		t.Fatalf("build omitted ordinary wheel or counted image-owned bytes: %+v", pack.Registry)
	}
	for _, name := range []string{"torch", "nvidia-cublas", "cuda-toolkit"} {
		declaration := []byte("lock-version=\"1.0\"\n[[packages]]\nname=\"" + name + "\"\nversion=\"1.0\"\n")
		if _, problem := packagepublish.RegistryRowsFromLock(declaration, nil, ""); problem == nil || problem.Name != "registry_dependency_platform_root_present" {
			t.Fatalf("external image-family declaration %s was not refused: %v", name, problem)
		}
	}
}
