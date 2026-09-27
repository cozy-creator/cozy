package producttest

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestRequirementMetadataParserDoesNotRequireRuntimeInventory(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX tool-path fixture")
	}
	_, problem := config.Load()
	fatal(t, problem)
	uv, err := exec.LookPath("uv")
	must(t, err)
	tools := t.TempDir()
	must(t, os.Symlink(uv, filepath.Join(tools, "uv")))
	t.Setenv("PATH", tools)
	if _, err := exec.LookPath("cozy-runtime"); err == nil { //cozy:allow asserts the fixture PATH has no Runtime; nothing is invoked
		t.Fatal("fixture exposes a Runtime inventory command")
	}
	selection, problem := packagepublish.ActiveRequirements(context.Background(), "metadata-proof", nil, map[string]string{
		"metadata-proof":  "Metadata-Version: 2.3\nName: metadata-proof\nVersion: 1\nRequires-Python: >=3.12,<3.13\nRequires-Dist: selected-helper>=1; python_full_version < '3.13'\n",
		"selected-helper": "Metadata-Version: 2.3\nName: selected-helper\nVersion: 1\n",
	}, "3.12.12")
	fatal(t, problem)
	if !strings.Contains(selection.RequiresPython, "<3.13") || !strings.Contains(strings.Join(selection.Requirements, "\n"), "selected-helper") {
		t.Fatalf("parser lost the declared target's requirements: %+v", selection)
	}
}
