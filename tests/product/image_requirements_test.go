package producttest

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestInstalledClosureAndWorkerRequirementsHaveSeparateMeanings(t *testing.T) {
	root := t.TempDir()
	if out, err := exec.Command("uv", "venv", "--python", "3.12", root).CombinedOutput(); err != nil {
		t.Fatalf("venv: %v\n%s", err, out)
	}
	site := filepath.Join(root, "lib", "python3.12", "site-packages")
	if runtime.GOOS == "windows" {
		site = filepath.Join(root, "Lib", "site-packages")
	}
	metadata := func(name, version, requirements string) {
		t.Helper()
		dir := filepath.Join(site, name+"-"+version+".dist-info")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		body := fmt.Sprintf("Metadata-Version: 2.4\nName: %s\nVersion: %s\n%s", name, version, requirements)
		if err := os.WriteFile(filepath.Join(dir, "METADATA"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	metadata("fixture", "1.0", "Requires-Python: >=3.12,<3.13\nRequires-Dist: cozy-runtime>=0.16.1,<1\nRequires-Dist: helper==1.0\nRequires-Dist: torch==2.13.0\n")
	metadata("helper", "1.0", "Requires-Dist: numpy>=1.26\nRequires-Dist: cuda-bindings==99; extra == 'optional'\n")
	metadata(runtimeDistribution, "0.16.2", "Requires-Dist: packaging>=24\n")
	metadata("torch", "2.13.0", "Requires-Dist: cuda-bindings==13.3.1\n")
	metadata("cuda-bindings", "13.3.1", "")
	metadata("numpy", "2.5.2", "")
	selection, problem := install.ExecutionRequirements(context.Background(), root, "fixture", nil)
	if problem != nil || selection.RequiresPython != ">=3.12,<3.13" || strings.Contains(strings.Join(selection.Requirements, "\n"), "cuda-bindings") {
		t.Fatalf("captured package ranges changed: %+v %v", selection, problem)
	}
	image := &pb.ImageInventory{Python: "3.12.11", Distributions: []*pb.ImageDistribution{
		{Distribution: runtimeDistribution, Version: "0.16.5"}, {Distribution: "torch", Version: "2.13.0+cu130"},
		{Distribution: "cuda-bindings", Version: "13.0.3"}, {Distribution: "numpy", Version: "2.5.1"},
	}}
	if reason := launch.InventoryMismatch(image, selection.Requirements, selection.RequiresPython); reason != "" {
		t.Fatalf("compatible image refused against client-local pins: %s", reason)
	}
	metadata("helper", "1.0", "Requires-Dist: numpy>=2.5.2\n")
	selection, problem = install.ExecutionRequirements(context.Background(), root, "fixture", nil)
	if problem != nil || !strings.Contains(launch.InventoryMismatch(image, selection.Requirements, selection.RequiresPython), "numpy") {
		t.Fatalf("actual transitive package requirement was not enforced: %v %v", selection.Requirements, problem)
	}
}
