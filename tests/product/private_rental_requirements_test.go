package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivateExtraImageRequirementReachesTheSealedWheel(t *testing.T) {
	for _, extra := range []string{"[gpu]", ""} {
		t.Run(extra, func(t *testing.T) {
			root := t.TempDir()
			library := filepath.Join(root, "library")
			must(t, os.Mkdir(library, 0700))
			metadata := func(dir, name, dependencies string) {
				t.Helper()
				text := fmt.Sprintf(`[project]
name = %q
version = "1.0"
requires-python = ">=3.12,<3.13"
%s
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["code.py"]
`, name, dependencies)
				must(t, os.WriteFile(filepath.Join(dir, "pyproject.toml"), []byte(text), 0600))
				must(t, os.WriteFile(filepath.Join(dir, "code.py"), []byte("VALUE=7\n"), 0600))
			}
			metadata(root, "root-proof", "dependencies = [\"extra-proof"+extra+"\"]\n[tool.uv.sources]\nextra-proof={path='library'}")
			metadata(library, "extra-proof", "dependencies = []\n[project.optional-dependencies]\ngpu = ['msgspec>=0.21,<0.22']")
			run := func(args ...string) {
				t.Helper()
				if out, err := exec.Command("uv", args...).CombinedOutput(); err != nil {
					t.Fatalf("uv: %v\n%s", err, out)
				}
			}
			run("lock", "--project", root)
			environment, problem := install.MaterializeEnvironment(root, filepath.Join(t.TempDir(), "venv"))
			fatal(t, problem)
			wheels := filepath.Join(t.TempDir(), "wheels")
			run("build", "--wheel", "--out-dir", wheels, root)
			run("build", "--wheel", "--out-dir", wheels, library)
			pack := packagepublish.Package{Root: t.TempDir(), Name: "root-proof", Release: "1.0",
				Wheel:            filepath.Join(wheels, "root_proof-1.0-py3-none-any.whl"),
				Files:            map[string]string{"uv.lock": filepath.Join(root, "uv.lock")},
				DependencyWheels: []packagepublish.DependencyWheel{{Path: filepath.Join(wheels, "extra_proof-1.0-py3-none-any.whl")}},
			}
			fatal(t, pack.CapturePrivateClosure(context.Background(), environment.Closure, nil))
			body, problem := wheel.Metadata(pack.Wheel)
			fatal(t, problem)
			text := string(body)
			if extra != "" && (!strings.Contains(text, "Requires-Dist: msgspec<0.22,>=0.21\n") || strings.Contains(text, "msgspec==") || strings.Contains(text, "; extra")) {
				t.Fatalf("selected image range was not committed unconditionally: %s", text)
			}
			if extra == "" && strings.Contains(text, "msgspec") {
				t.Fatalf("unselected extra constrained the worker: %s", text)
			}
		})
	}
}

func TestRentalRequirementsStopAtTheWorkerImage(t *testing.T) {
	venv := t.TempDir()
	must(t, os.WriteFile(filepath.Join(venv, "pyvenv.cfg"), []byte("version_info = 3.12.12\n"), 0600))
	metadata := func(name, version, requirements string) {
		t.Helper()
		dir := filepath.Join(venv, "lib", "python3.12", "site-packages", name+"-"+version+".dist-info")
		must(t, os.MkdirAll(dir, 0700))
		must(t, os.WriteFile(filepath.Join(dir, "METADATA"), []byte("Metadata-Version: 2.3\nName: "+name+"\nVersion: "+version+"\n"+requirements+"\n"), 0600))
	}
	metadata("torch", "2.13.0", "Requires-Dist: cuda-bindings==13.3.1\n")
	metadata("cuda-bindings", "13.3.1", "")
	metadata("numpy", "2.5.3", "")
	metadata("scipy", "1.18.1", "Requires-Dist: numpy>=2.5,<2.6\n")
	metadata("project", "1", "Requires-Dist: torch>=2.13,<3\nRequires-Dist: scipy>=1.18\nRequires-Dist: torch>=3; sys_platform == 'win32'\n")
	metadata("unused-development-library", "1", "Requires-Dist: torch>=99\n")
	inventory := &pb.ImageInventory{Python: "3.12.12", Distributions: []*pb.ImageDistribution{
		{Distribution: "torch", Version: "2.13.0"}, {Distribution: "cuda-bindings", Version: "13.0.3"},
		{Distribution: "numpy", Version: "2.5.2"},
	}}
	selection, problem := install.ExecutionRequirements(context.Background(), venv, "project", nil)
	fatal(t, problem)
	requirements := selection.Requirements
	if why := launch.InventoryMismatch(inventory, requirements, ""); why != "" {
		t.Fatalf("local image implementation pins constrained a compatible worker: %s (%v)", why, requirements)
	}
	if strings.Contains(strings.Join(requirements, "\n"), "torch>=3") {
		t.Fatal("inactive environment marker constrained the worker")
	}
	inventory.Distributions[2].Version = "2.4.0"
	if why := launch.InventoryMismatch(inventory, requirements, ""); !strings.Contains(why, "numpy 2.4.0") {
		t.Fatalf("package-owned SciPy requirement was dropped: %q", why)
	}
	inventory.Distributions[2].Version = "2.5.2"
	metadata("project", "1", "Requires-Dist: torch>=2.13,<3\nRequires-Dist: scipy>=1.18\nRequires-Dist: cuda-bindings>=13.2\n")
	selection, problem = install.ExecutionRequirements(context.Background(), venv, "project", nil)
	fatal(t, problem)
	requirements = selection.Requirements
	if why := launch.InventoryMismatch(inventory, requirements, ""); !strings.Contains(why, "cuda-bindings 13.0.3") {
		t.Fatalf("explicit project CUDA floor was dropped: %q", why)
	}
}

// This exercises actual script capture and named-rental admission through the
// normal CLI. The HTTP peer supplies inventory only; it never buys a machine.
func TestPrivateNamedRentalUsesAuthoredImageRequirements(t *testing.T) {
	version := runtimeFixtureVersion(t, "")
	const current, old = "pr-11111111111111111111", "pr-22222222222222222222"
	root, mu, posts, _, _ := runModelCatalog(t, func(mux *http.ServeMux, _ *hub.PackageReleaseDetail) {
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			id := r.PathValue("id")
			name := map[string]string{current: "isao", old: "giriko"}[id]
			_ = json.NewEncoder(w).Encode(map[string]any{
				"rental_id": id, "name": name, "state": "ready", "accelerator_count": 1,
				"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 100000,
			})
		})
		mux.HandleFunc("GET /v1/rentals/{id}/image-inventory", func(w http.ResponseWriter, r *http.Request) {
			msgspec := "0.21.0"
			if r.PathValue("id") == old {
				msgspec = "0.20.0"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"image_inventory": map[string]any{
				"format": "tensorhub.image_inventory/1", "profile": "python3.12-cpu-linux-x86", "python": "3.12.12",
				"distributions": []map[string]string{{"name": runtimeDistribution, "version": version}, {"name": "msgspec", "version": msgspec}},
			}})
		})
	})
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, name := range map[string]string{current: "isao", old: "giriko"} {
		fatal(t, store.RecordRental(records.Rental{ID: id, MachineName: name, AcceleratorModel: "CPU",
			AcceleratorCount: 1, State: "ready", HourlyRateUSDMicros: 100000, Hub: "fixture"}))
	}
	store.Close()
	script := filepath.Join(t.TempDir(), "main.py")
	code := fmt.Sprintf(`# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = ["cozy-runtime==%s", "msgspec>=0.21,<0.22"]
# [tool.uv]
# constraint-dependencies = ["msgspec==0.21.1"]
# ///
def main(ctx):
    raise AssertionError("dry-run must not execute")
`, version)
	must(t, os.WriteFile(script, []byte(code), 0600))
	status, out := runCozy(t, root, "run", script, "--rental=isao", "--dry-run", "--json", "--full")
	if status != 0 || !strings.Contains(out, current) {
		t.Fatalf("compatible image was held to the local exact pin [%d]: %s", status, out)
	}
	status, out = runCozy(t, root, "run", script, "--rental=giriko", "--dry-run", "--json")
	if status == 0 || !strings.Contains(out, "rental.dependency_mismatch") || !strings.Contains(out, "msgspec 0.20.0") {
		t.Fatalf("authored image requirement was not enforced [%d]: %s", status, out)
	}
	library := filepath.Join(filepath.Dir(script), "library")
	must(t, os.Mkdir(library, 0700))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(`[project]
name = "marker-library"
version = "1.0"
requires-python = ">=3.12,<3.13"
dependencies = []
[project.optional-dependencies]
gpu = ["msgspec>=0.21,<0.22"]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["marker_library.py"]
`), 0600))
	must(t, os.WriteFile(filepath.Join(library, "marker_library.py"), []byte("VALUE=1\n"), 0600))
	for _, selected := range []bool{true, false} {
		dependency := "marker-library"
		if selected {
			dependency += "[gpu]"
		}
		extraCode := strings.Replace(code, "msgspec>=0.21,<0.22", dependency, 1)
		extraCode = strings.Replace(extraCode, "# ///\ndef", "# [tool.uv.sources]\n# marker-library = {path = './library'}\n# ///\ndef", 1)
		must(t, os.WriteFile(script, []byte(extraCode), 0600))
		status, out = runCozy(t, root, "run", script, "--rental=giriko", "--dry-run", "--json")
		if selected && (status == 0 || !strings.Contains(out, "msgspec 0.20.0")) {
			t.Fatalf("selected library extra did not enforce its image floor [%d]: %s", status, out)
		}
		if !selected && status != 0 {
			t.Fatalf("unselected library extra constrained the image [%d]: %s", status, out)
		}
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("named-rental preflight attempted paid acquisition")
	}
}
