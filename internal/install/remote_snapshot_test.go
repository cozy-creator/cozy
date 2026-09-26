package install

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteSnapshotReusesEnvironmentAndRetainsFreshSource(t *testing.T) {
	if _, problem := config.Load(); problem != nil {
		t.Fatal(problem)
	}
	root := t.TempDir()
	project := filepath.Join(root, "project")
	if err := os.MkdirAll(project, 0700); err != nil {
		t.Fatal(err)
	}
	write := func(name, body string) {
		t.Helper()
		if err := os.WriteFile(filepath.Join(project, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	write("pyproject.toml", `[project]
name = "remote-snapshot-proof"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = []
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["proof.py"]
`)
	write("package.toml", "[application]\nobject='proof:app'\n")
	write(".python-version", "3.12\n")
	source := `from cozy_runtime.author import App
import msgspec
app = App()
class Input(msgspec.Struct):
    value: int = 1
class Result(msgspec.Struct):
    value: int
@app.job
def original(payload: Input) -> Result:
    return Result(payload.value)
`
	write("proof.py", source)
	lock := exec.Command("uv", "lock", "--offline", "--python", "3.12")
	lock.Dir = project
	lock.Env = config.Frozen().Tool()
	if out, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("lock: %s: %s", err, out)
	}
	layout, problem := home.Open(filepath.Join(root, "records"))
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	request := func(remote *records.PackageInstall) Request {
		return Request{Ref: Ref{Package: "local/remote-snapshot-proof"}, Snapshot: remote != nil,
			RemoteEnvironment: remote, Local: &LocalSource{Tree: project, Package: "local/remote-snapshot-proof", Release: "1.0.0"}}
	}
	initial, problem := Run(layout, store, request(nil))
	if problem != nil {
		t.Fatal(problem)
	}
	write("proof.py", strings.ReplaceAll(source, "original", "edited"))
	snapshot, problem := Run(layout, store, request(&initial.Install))
	if problem != nil {
		t.Fatal(problem)
	}
	if !snapshot.RemoteSnapshot {
		t.Fatal("remote source capture rebuilt its environment")
	}
	if _, err := os.Stat(filepath.Join(snapshot.Install.Dir, "venv")); !os.IsNotExist(err) {
		t.Fatalf("remote snapshot retained a local environment: %v", err)
	}
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(snapshot.Install.Dir))
	if problem != nil {
		t.Fatal(problem)
	}
	if _, problem := surface.Function("edited"); problem != nil {
		t.Fatalf("fresh code was not described: %s", problem)
	}
	if _, problem := surface.Function("original"); problem == nil {
		t.Fatal("stale interface reused")
	}
	// Neither capture custody nor rental requirements depend on the old venv.
	if err := os.RemoveAll(initial.Install.Dir); err != nil {
		t.Fatal(err)
	}
	selected, problem := InstalledRequirements(context.Background(), snapshot.Install)
	if problem != nil || selected.RequiresPython == "" {
		t.Fatalf("lost retained requirements: %+v %v", selected, problem)
	}
	if _, problem := localpackage.Stage(context.Background(), layout, snapshot.Install); problem != nil {
		t.Fatal(problem)
	}
	write("proof.py", "raise RuntimeError('later author edit')\n")
	frozen, err := os.ReadFile(filepath.Join(snapshot.Install.ProjectDir, "proof.py"))
	if err != nil || !strings.Contains(string(frozen), "def edited") {
		t.Fatal("accepted source followed a later edit")
	}
}
