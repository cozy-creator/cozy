package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
	metadata("project", "1", "Requires-Dist: torch>=2.13,<3\nRequires-Dist: torch>=3; sys_platform == 'win32'\n")
	inventory := &pb.ImageInventory{Python: "3.12.12", Distributions: []*pb.ImageDistribution{
		{Distribution: "torch", Version: "2.13.0"}, {Distribution: "cuda-bindings", Version: "13.0.3"},
		{Distribution: "numpy", Version: "2.5.2"},
	}}
	requirements, problem := install.ExecutionRequirements(venv)
	fatal(t, problem)
	if why := launch.InventoryMismatch(inventory, requirements, ""); why != "" {
		t.Fatalf("local image implementation pins constrained a compatible worker: %s (%v)", why, requirements)
	}
	if !strings.Contains(strings.Join(requirements, "\n"), "torch>=3; sys_platform == 'win32'") {
		t.Fatal("authored environment marker was discarded")
	}
	inventory.Distributions[2].Version = "2.4.0"
	if why := launch.InventoryMismatch(inventory, requirements, ""); !strings.Contains(why, "numpy 2.4.0") {
		t.Fatalf("package-owned SciPy requirement was dropped: %q", why)
	}
	inventory.Distributions[2].Version = "2.5.2"
	metadata("project", "1", "Requires-Dist: torch>=2.13,<3\nRequires-Dist: cuda-bindings>=13.2\n")
	requirements, problem = install.ExecutionRequirements(venv)
	fatal(t, problem)
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
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("named-rental preflight attempted paid acquisition")
	}
}
