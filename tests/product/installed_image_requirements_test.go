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
		t.Fatalf("create real metadata reader environment: %v\n%s", err, out)
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
	metadata("cozy-runtime", "0.16.2", "Requires-Dist: packaging>=24\n") //cozy:allow distribution metadata only; no Runtime process invocation
	metadata("torch", "2.13.0", "Requires-Dist: cuda-bindings==13.3.1\n")
	metadata("cuda-bindings", "13.3.1", "")
	metadata("numpy", "2.5.2", "")
	metadata("packaging", "26.2", "")

	// The captured closure keeps the author's ranges; the rental image is judged on Python.
	selection, problem := install.ExecutionRequirements(context.Background(), root, "fixture", nil)
	joined := strings.Join(selection.Requirements, "\n")
	if problem != nil || selection.RequiresPython != ">=3.12,<3.13" {
		t.Fatalf("captured Python bound changed: %q %v", selection.RequiresPython, problem)
	}
	for _, required := range []string{"cozy-runtime<1,>=0.16.1", "cuda-bindings==13.3.1", "numpy>=1.26", "torch==2.13.0"} {
		if !strings.Contains(joined, required) {
			t.Fatalf("captured package ranges omitted %s: %v", required, selection.Requirements)
		}
	}
	image := &pb.ImageInventory{Python: "3.12.11"}
	if _, reason := launch.InventoryPython(image, selection.RequiresPython, ""); reason != "" {
		t.Fatalf("compatible image refused: %s", reason)
	}
	metadata("helper", "1.0", "Requires-Dist: numpy>=2.5.2\n")
	selection, problem = install.ExecutionRequirements(context.Background(), root, "fixture", nil)
	if problem != nil || !strings.Contains(strings.Join(selection.Requirements, "\n"), "numpy>=2.5.2") {
		t.Fatalf("private transitive requirement was lost: %v %v", selection.Requirements, problem)
	}
	// The installed graph's Python observation must not replace the author's bound.
	metadata("helper", "1.0", "Requires-Dist: numpy>=1.26\n")
	metadata("fixture", "1.0", "Requires-Python: >=3.12.11,<3.13\nRequires-Dist: helper==1.0\nRequires-Dist: torch==2.13.0\n")
	selection, problem = install.ExecutionRequirements(context.Background(), root, "fixture", nil)
	image.Python = "3.12.3"
	if _, reason := launch.InventoryPython(image, selection.RequiresPython, ""); problem != nil ||
		selection.RequiresPython != ">=3.12.11,<3.13" || reason == "" {
		t.Fatalf("authored Python constraint was lost: %q %v", selection.RequiresPython, problem)
	}
}
