package producttest

import (
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	pep440 "github.com/aquasecurity/go-pep440-version"
	"github.com/cozy-creator/cozy/internal/records"
	capturedwheel "github.com/cozy-creator/cozy/internal/wheel"
)

// Every invocation uses the actual cozy CLI. A successful import alone does not
// test worker dispatch, native custody, or independently edited caller reuse.
func TestUnpublishedWheelCompositionTracksExecutableNotCaller(t *testing.T) {
	version := runtimeFixtureVersion(t, *privateChildRuntimeWheel)
	parsed, err := pep440.Parse(version)
	must(t, err)
	if parsed.LessThan(pep440.MustParse("0.12.0")) {
		t.Skip("installed App description requires Runtime0.12; select -child-runtime-wheel to qualify a candidate")
	}
	project := t.TempDir()
	control := filepath.Join(project, "control")
	runtimeInstall, runtimeSource := "cozy-runtime[model-execution]=="+version, ""
	if *privateChildRuntimeWheel != "" {
		path, err := filepath.Abs(*privateChildRuntimeWheel)
		must(t, err)
		runtimeInstall = path + "[model-execution]"
		runtimeSource = "# cozy-runtime = {path = " + strconv.Quote(path) + "}\n"
	}
	runUV := func(args ...string) {
		t.Helper()
		if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
			t.Fatalf("uv: %v\n%s", err, out)
		}
	}
	runUV("venv", control, "--python", "3.12")
	runUV("pip", "install", "--python", filepath.Join(control, "bin", "python"), runtimeInstall)
	root, err := os.MkdirTemp("", "cozy-wheel-")
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
			t.Log("captured wheel evidence retained", root)
		}
	})
	library := filepath.Join(project, "library")
	must(t, os.MkdirAll(library, 0o700))
	metadata := fmt.Sprintf(`[project]
name = "unpublished-wheel-proof"
version = "0.1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime[model-execution]==%s", "msgspec"]
[project.entry-points."cozy.application"]
default = "wheel_proof:app"
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["wheel_proof.py"]
`, version)
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0o600))
	body := `import msgspec
from cozy_runtime.author import App, Context, FileAsset, Outputs, invocable
class Result(msgspec.Struct, frozen=True):
    file: FileAsset
    value: int
@invocable(memoize=True)
async def first(ctx: Context, *, value: int, out: Outputs) -> Result:
    return Result(out.save_bytes(str(value).encode(), media_type="text/plain"), value + 1)
@invocable(memoize=True)
async def second(ctx: Context, *, value: int, out: Outputs) -> Result:
    return Result(out.save_bytes(str(value * 2).encode(), media_type="text/plain"), value * 2)
async def compose(value: int) -> Result:
    previous = await first(value=value)
    return await second(value=previous.value)
app = App()
app.job(first)
app.job(second)
`
	module := filepath.Join(library, "wheel_proof.py")
	must(t, os.WriteFile(module, []byte(body), 0o600))
	wheels := filepath.Join(project, "wheels")
	runUV("build", "--wheel", "--out-dir", wheels, library)
	wheel := filepath.Join(wheels, "unpublished_wheel_proof-0.1.0-py3-none-any.whl")
	original, err := os.ReadFile(wheel)
	must(t, err)
	script := filepath.Join(project, "first.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime[model-execution]==%s", "unpublished-wheel-proof==0.1.0"]
# [tool.uv.sources]
%s# unpublished-wheel-proof = {path = %q}
# ///
from wheel_proof import compose
async def main(ctx):
    result = await compose(20)
    assert result.value == 42
`, version, runtimeSource, wheel)
	must(t, os.WriteFile(script, []byte(code), 0o600))
	status, output := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 {
		t.Fatalf("wheel helper failed [%d]: %s", status, output)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	first, problem := store.RequestByReference("1")
	fatal(t, problem)
	children, problem := store.Children(first.ID)
	fatal(t, problem)
	if len(children) != 2 || children[0].Ordinal != 1 || children[1].Ordinal != 1 {
		t.Fatalf("wheel helper did not dispatch two leaves: %+v", children)
	}
	before := append([]records.Request(nil), children...)
	inst, problem := store.Install(children[0].InstallID)
	fatal(t, problem)
	if inst.SourceKind != "wheel" {
		t.Fatalf("captured wheel became a source project: %+v", inst)
	}
	sealed, problem := capturedwheel.Metadata(filepath.Join(inst.Dir, "wheels", filepath.Base(wheel)))
	fatal(t, problem)
	if !strings.Contains(string(sealed), "Requires-Dist: msgspec\n") || strings.Contains(string(sealed), "msgspec==") {
		t.Fatalf("callable wheel replaced its authored image requirement with a client pin: %s", sealed)
	}
	retained := filepath.Join(inst.Dir, "original", filepath.Base(wheel))
	raw, err := os.ReadFile(retained)
	must(t, err)
	if !bytes.Equal(raw, original) {
		t.Fatal("original library wheel was not retained byte-exactly")
	}
	edited := filepath.Join(project, "independent.py")
	must(t, os.WriteFile(edited, []byte(strings.Replace(code, "    result =", "    ctx.log('independent caller')\n    result =", 1)), 0o600))
	status, output = runCozyPath(t, root, path, "run", edited, "--await", "--json")
	if status != 0 {
		t.Fatalf("edited wheel caller failed [%d]: %s", status, output)
	}
	second, problem := store.RequestByReference("4")
	fatal(t, problem)
	children, problem = store.Children(second.ID)
	fatal(t, problem)
	if len(children) != 2 {
		t.Fatalf("edited caller lost children: %+v", children)
	}
	for n, child := range children {
		if child.Ordinal != 0 || child.ReusedFrom != before[n].ID || child.ChildTargetDigest != before[n].ChildTargetDigest || child.LocalPackageDigest != before[n].LocalPackageDigest {
			t.Fatalf("caller edit invalidated wheel computation: %+v", children)
		}
	}
	must(t, os.WriteFile(module, []byte(strings.ReplaceAll(body, "value * 2", "value * 2 + 1")), 0o600))
	runUV("build", "--wheel", "--out-dir", wheels, library)
	must(t, os.WriteFile(edited, []byte(strings.Replace(code, "== 42", "== 43", 1)), 0o600))
	status, output = runCozyPath(t, root, path, "run", edited, "--await", "--json")
	if status != 0 {
		t.Fatalf("same-version wheel edit failed [%d]: %s", status, output)
	}
	third, problem := store.RequestByReference("7")
	fatal(t, problem)
	children, problem = store.Children(third.ID)
	fatal(t, problem)
	if len(children) != 2 {
		t.Fatalf("changed wheel lost children: %+v", children)
	}
	for n, child := range children {
		if child.Ordinal != 1 || child.ChildTargetDigest == before[n].ChildTargetDigest || child.LocalPackageDigest == before[n].LocalPackageDigest {
			t.Fatalf("changed wheel reused old implementation: %+v", children)
		}
	}
	raw, err = os.ReadFile(retained)
	must(t, err)
	if !bytes.Equal(raw, original) {
		t.Fatal("wheel edit changed original source custody")
	}
	// Describing a remote-capable script is not permission to execute a selected
	// wheel's Python startup hooks on the client, in either parent or child venv.
	marker := filepath.Join(project, "startup-hook-ran")
	hook := fmt.Sprintf("import pathlib; pathlib.Path(%q).write_text('unexpected client startup')\n", marker)
	must(t, os.WriteFile(filepath.Join(library, "startup_probe.pth"), []byte(hook), 0o600))
	metadata = strings.Replace(metadata, `only-include = ["wheel_proof.py"]`, `only-include = ["wheel_proof.py", "startup_probe.pth"]`, 1)
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0o600))
	runUV("build", "--wheel", "--out-dir", wheels, library)
	status, output = runCozyPath(t, root, path, "run", edited, "--describe", "--json")
	if status != 0 {
		t.Fatalf("static wheel description failed [%d]: %s", status, output)
	}
	if _, err := os.Stat(marker); !os.IsNotExist(err) {
		t.Fatal("describing the wheel executed its Python startup hook on the client")
	}
}
