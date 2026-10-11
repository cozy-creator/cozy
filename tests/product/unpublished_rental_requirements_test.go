package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// Private dependency versions never constrain a named rental: the machine prepares the
// captured environment itself. The HTTP peer names the rentals only; it never buys one.
func TestUnpublishedNamedRentalUsesPrivateDependencyVersions(t *testing.T) {
	integration(t)
	version := runtimeFixtureVersion(t, *privateScriptRuntimeWheel)
	const current, old, earlierPython = "pr-11111111111111111111", "pr-22222222222222222222", "pr-33333333333333333333"
	root, mu, posts, _, _ := runModelCatalog(t, func(mux *http.ServeMux, _ *hub.PackageReleaseDetail) {
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := r.PathValue("id")
			name := map[string]string{current: "isao", old: "giriko", earlierPython: "priorpython"}[id]
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": id, "name": name, "state": "ready", "accelerator_count": 1,
				"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 100000,
			})
		})
	})
	t.Cleanup(func() { must(t, removeAllForce(root)) })
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, name := range map[string]string{current: "isao", old: "giriko", earlierPython: "priorpython"} {
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: name, AcceleratorModel: "CPU",
			AcceleratorCount: 1, State: "ready", HourlyRateUSDMicros: 100000, Hub: "fixture"}))
	}
	store.Close()
	script := filepath.Join(t.TempDir(), "main.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime>=%s", "msgspec>=0.21,<0.22"]
# [tool.uv]
# constraint-dependencies = ["msgspec==0.21.1"]
# ///
def main(ctx):
    raise AssertionError("an unreachable rental must not execute")
`, version)
	runtimeSource := ""
	if runtimeWheel := *privateScriptRuntimeWheel; runtimeWheel != "" {
		runtimeSource = fmt.Sprintf("# cozy-runtime = {path = %q}\n", runtimeWheel)
	}
	code = strings.Replace(code, "# ///\ndef", "# [tool.uv.sources]\n"+runtimeSource+"# ///\ndef", 1)
	must(t, os.WriteFile(script, []byte(code), 0600))
	request, _, out := submitRun(t, root, "requirements-isao", "run", script, "--rental=isao", "--json", "--full")
	if request == nil || request.RequestedRental != current {
		t.Fatalf("the local exact pin kept the run from its named rental: %s", out)
	}
	if request, _, out := submitRun(t, root, "requirements-giriko", "run", script, "--rental=giriko", "--json"); request == nil {
		t.Fatalf("a private msgspec version kept the run from its named rental: %s", out)
	}
	patchCode := strings.Replace(code, "msgspec>=0.21,<0.22", "msgspec>=0.21,<0.22; python_full_version >= '3.12.5' and (python_full_version >= '3.12.12' or sys_platform == 'win32' or python_full_version < '3.12.4')", 1)
	must(t, os.WriteFile(script, []byte(patchCode), 0600))
	if request, _, out := submitRun(t, root, "requirements-marked", "run", script, "--rental=giriko", "--json"); request == nil {
		t.Fatalf("a private marked requirement kept the run from its named rental: %s", out)
	}
	if request, _, out := submitRun(t, root, "requirements-patch", "run", script, "--rental=priorpython", "--json"); request == nil || request.RequestedRental != earlierPython {
		t.Fatalf("a script within its captured Python minor was refused: %s", out)
	}
	pythonCode := strings.Replace(patchCode, `requires-python = ">=3.12,<3.13"`, `requires-python = ">=3.12.5,<3.13"`, 1)
	must(t, os.WriteFile(script, []byte(pythonCode), 0600))
	// The machine prepares the script's environment from its capture, so the authored patch
	// floor is what it enforces; the pinned rental's image is judged at placement, once the
	// rental is attachable, never at submission.
	floored, _, out := submitRun(t, root, "requirements-python-floor", "run", script, "--rental=priorpython", "--json")
	if floored == nil || floored.RequestedRental != earlierPython {
		t.Fatalf("a patch-floored script was not admitted to its pinned machine: %s", out)
	}
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	capture, problem := store.Install(floored.InstallID)
	store.Close()
	fatal(t, problem)
	if capture == nil {
		t.Fatal("the patch-floored script has no captured install")
	}
	requiresPython, problem := packagepublish.ProjectRequiresPython(capture.ProjectDir)
	fatal(t, problem)
	if clauses := strings.Split(requiresPython, ","); !slices.Contains(clauses, ">=3.12.5") || !slices.Contains(clauses, "<3.13") || len(clauses) != 2 {
		t.Fatalf("authored Python patch floor was replaced in the script's project: %q", requiresPython)
	}
	library := filepath.Join(filepath.Dir(script), "library")
	must(t, os.Mkdir(library, 0700))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(fmt.Sprintf(`[project]
name = "marker-library"
version = "1.0"
requires-python = ">=3.12,<3.13"
dependencies = ["cozy-runtime>=%s"]
[project.entry-points."cozy.application"]
default = "marker_library:app"
[project.optional-dependencies]
gpu = ["msgspec>=0.21,<0.22"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["marker_library.py"]
`, version)), 0600))
	must(t, os.WriteFile(filepath.Join(library, "marker_library.py"), []byte(`import msgspec
from cozy_runtime.author import App, Context, invocable
class Result(msgspec.Struct):
    value: int
@invocable(memoize=True)
async def value(ctx: Context) -> Result:
    return Result(1)
app = App()
app.job(value)
`), 0600))
	wheels := filepath.Join(filepath.Dir(script), "wheels")
	if out, err := exec.Command("uv", "build", "--wheel", "--out-dir", wheels, library).CombinedOutput(); err != nil {
		t.Fatalf("build callable extra fixture: %v\n%s", err, out)
	}
	for i, test := range []struct{ selected, wheel bool }{{true, false}, {false, false}, {true, true}, {false, true}} {
		selected := test.selected
		dependency := "marker-library"
		if selected {
			dependency += "[gpu]"
		}
		extraCode := strings.Replace(code, "msgspec>=0.21,<0.22", dependency, 1)
		source := "./library"
		if test.wheel {
			source = "./wheels/marker_library-1.0-py3-none-any.whl"
		}
		extraCode = strings.Replace(extraCode, "# [tool.uv.sources]\n", "# [tool.uv.sources]\n# marker-library = {path = '"+source+"'}\n", 1)
		must(t, os.WriteFile(script, []byte(extraCode), 0600))
		if request, _, out := submitRun(t, root, fmt.Sprintf("requirements-extra-%d", i), "run", script, "--rental=giriko", "--json"); request == nil {
			t.Fatalf("a private library extra kept the run from its named rental (selected=%v, wheel=%v): %s", selected, test.wheel, out)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("a named rental attempted paid acquisition")
	}
}
