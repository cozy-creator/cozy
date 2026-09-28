package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/install"
)

func TestInstalledAppFilterDoesNotPromoteCallerOrAmbientLibraries(t *testing.T) {
	root := t.TempDir()
	if output, err := exec.Command("uv", "venv", "--python", "3.12", root).CombinedOutput(); err != nil {
		t.Fatalf("create real metadata environment: %s: %v", output, err)
	}
	python := filepath.Join(root, "bin", "python")
	site := filepath.Join(root, "lib", "python3.12", "site-packages")
	for _, name := range []string{"caller", "source-lib", "wheel-lib", "ambient"} {
		dir := filepath.Join(site, name+"-1.dist-info")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "entry_points.txt"), []byte("[cozy.application]\ndefault = sample:app\n"), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	for _, test := range []struct {
		installed string
		ignored   map[string]string
		want      bool
	}{
		{"caller==1\ncozy-runtime==0.11.0", nil, false},
		{"caller==1\nsource-lib==1", map[string]string{"source-lib": "already-bound"}, false},
		{"caller==1\nwheel-lib==1", nil, true},
	} {
		found, problem := install.HasInstalledApplications(python, test.installed, "caller", test.ignored)
		if problem != nil || found != test.want {
			t.Fatalf("filter(%q) = %v, %v", test.installed, found, problem)
		}
	}
}
