package producttest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
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
dependencies = ["snapshot-library"]
[tool.uv.sources]
snapshot-library = {path = "../library"}
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["proof.py"]
`)
	library := filepath.Join(root, "library")
	if err := os.MkdirAll(library, 0700); err != nil {
		t.Fatal(err)
	}
	for name, body := range map[string]string{
		"pyproject.toml": "[project]\nname='snapshot-library'\nversion='1.0.0'\n[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n[tool.hatch.build.targets.wheel]\nonly-include=['library.py']\n",
		"library.py":     "VALUE = 1\n",
	} {
		if err := os.WriteFile(filepath.Join(library, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
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
	request := func(remote *records.PackageInstall) install.Request {
		return install.Request{Ref: install.Ref{Package: "local/remote-snapshot-proof"}, Snapshot: remote != nil,
			RemoteEnvironment: remote, Local: &install.LocalSource{Tree: project, Package: "local/remote-snapshot-proof", Release: "1.0.0"}}
	}
	initial, problem := install.Run(layout, store, request(nil))
	if problem != nil {
		t.Fatal(problem)
	}
	write("proof.py", strings.ReplaceAll(source, "original", "edited"))
	snapshot, problem := install.Run(layout, store, request(&initial.Install))
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
	// A control-file edit cannot borrow dependency declarations from this install.
	metadata := filepath.Join(snapshot.Install.ProjectDir, "pyproject.toml")
	original, err := os.ReadFile(metadata)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(metadata, append(original, []byte("\n[tool.capture-proof]\nchanged = true\n")...), 0600); err != nil {
		t.Fatal(err)
	}
	changed := request(&initial.Install)
	changed.Local.Tree = snapshot.Install.ProjectDir
	fallback, problem := install.Run(layout, store, changed)
	if problem != nil {
		t.Fatal(problem)
	}
	if fallback.RemoteSnapshot {
		t.Fatal("changed project metadata reused an old environment")
	}
	if err := os.WriteFile(metadata, original, 0600); err != nil {
		t.Fatal(err)
	}
	paths, problem := packagepublish.LocalDependencyPaths(snapshot.Install.ProjectDir)
	if problem != nil {
		t.Fatal(problem)
	}
	if err := os.WriteFile(filepath.Join(paths["snapshot-library"], "library.py"), []byte("VALUE = 2\n"), 0600); err != nil {
		t.Fatal(err)
	}
	changed = request(&initial.Install)
	changed.Local.Tree = snapshot.Install.ProjectDir
	fallback, problem = install.Run(layout, store, changed)
	if problem != nil {
		t.Fatal(problem)
	}
	if fallback.RemoteSnapshot {
		t.Fatal("changed dependency source reused an old environment")
	}
	// Neither capture custody nor rental requirements depend on the old venv.
	if err := os.RemoveAll(initial.Install.Dir); err != nil {
		t.Fatal(err)
	}
	selected, problem := install.InstalledRequirements(context.Background(), snapshot.Install)
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
