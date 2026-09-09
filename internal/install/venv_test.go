package install

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWriteLockedRequirementsOmitsWorkerImageTransitives(t *testing.T) {
	root := t.TempDir()
	exported := filepath.Join(root, "exported.txt")
	target := filepath.Join(root, "locked.txt")
	if err := os.WriteFile(exported, []byte("cuda-bindings==13.3.1 --hash=sha256:"+strings.Repeat("a", 64)+"\nscipy==1.18.1 --hash=sha256:"+strings.Repeat("b", 64)+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if problem := writeLockedRequirements(exported, target, "https://example.invalid/simple", nil); problem != nil {
		t.Fatal(problem)
	}
	data, err := os.ReadFile(target)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "cuda-bindings") || !strings.Contains(string(data), "scipy==1.18.1") {
		t.Fatalf("worker-image transitive survived locked export: %s", data)
	}
}
