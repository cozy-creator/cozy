package producttest

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// No package manifest, manual wheel build, install command or publication. Every
// invocation enters through the user's cozy run and captures the edited library.
func TestEditableInvocableLibraryTracksCodeWithoutPackageManifest(t *testing.T) {
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	project := t.TempDir()
	control := filepath.Join(project, "control")
	runtimeInstall, runtimeSource := "cozy-runtime[model-execution]=="+version, ""
	if *privateChildRuntimeWheel != "" {
		wheel, err := filepath.Abs(*privateChildRuntimeWheel)
		must(t, err)
		runtimeInstall = wheel + "[model-execution]"
		runtimeSource = "# cozy-runtime = {path = " + strconv.Quote(wheel) + "}\n"
	}
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", filepath.Join(control, "bin/python"), runtimeInstall}} {
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	root, err := os.MkdirTemp("", "cozy-edit-lib-")
	must(t, err)
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		compositionDown(t, root, path)
		if !t.Failed() {
			must(t, removeAllForce(root))
		} else {
			t.Log("retained editable proof home", root)
		}
	})
	library := filepath.Join(project, "library")
	must(t, os.Mkdir(library, 0700))
	metadata := fmt.Sprintf(`[project]
name="editable-leaf-proof"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime==%s"]
[project.entry-points."cozy.application"]
default="editable_leaf:app"
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["editable_leaf.py"]
`, version)
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	body := `import msgspec
from cozy_runtime.author import App, Context, invocable
class Result(msgspec.Struct):
    value: int
@invocable(memoize=True)
async def value(ctx: Context) -> Result:
    return Result(42)
app = App()
app.job(value)
`
	module := filepath.Join(library, "editable_leaf.py")
	must(t, os.WriteFile(module, []byte(body), 0600))
	script := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==%s", "editable-leaf-proof==1.0.0"]
# [tool.uv.sources]
%s# editable-leaf-proof = {path = %q, editable = true}
# ///
from editable_leaf import value
async def main(ctx):
    assert (await value()).value == 42
`, version, runtimeSource, library)
	firstScript := filepath.Join(project, "first.py")
	must(t, os.WriteFile(firstScript, []byte(script), 0600))
	run := func(file string) {
		t.Helper()
		if code, out := runCozyPath(t, root, path, "run", file, "--await", "--json"); code != 0 {
			t.Fatalf("editable helper failed [%d]: %s", code, out)
		}
	}
	run(firstScript)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	child := func(parentRef string) records.Request {
		t.Helper()
		parent, problem := store.RequestByReference(parentRef)
		fatal(t, problem)
		children, problem := store.Children(parent.ID)
		fatal(t, problem)
		if len(children) != 1 {
			t.Fatalf("expected one child: %+v", children)
		}
		return children[0]
	}
	first := child("1")
	if first.Ordinal != 1 {
		t.Fatalf("first call did not run: %+v", first)
	}
	edited := filepath.Join(project, "edited.py")
	must(t, os.WriteFile(edited, []byte(strings.Replace(script, "    assert", "    ctx.log('caller changed')\n    assert", 1)), 0600))
	run(edited)
	second := child("3")
	if second.Ordinal != 0 || second.ReusedFrom != first.ID || second.ChildTargetDigest != first.ChildTargetDigest {
		t.Fatalf("caller edit invalidated helper: %+v", second)
	}
	must(t, os.WriteFile(module, []byte(strings.Replace(body, "Result(42)", "Result(43)", 1)), 0600))
	must(t, os.WriteFile(edited, []byte(strings.Replace(script, "== 42", "== 43", 1)), 0600))
	run(edited)
	third := child("5")
	if third.Ordinal != 1 || third.ChildTargetDigest == first.ChildTargetDigest {
		t.Fatalf("helper edit reused old code: %+v", third)
	}
	if _, err := os.Stat(filepath.Join(library, "package.toml")); !os.IsNotExist(err) {
		t.Fatal("capture created a package manifest in the user's library")
	}
}
