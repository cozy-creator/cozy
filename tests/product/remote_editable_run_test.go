package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestRemoteEditableRunCapturesFreshSchemaWithoutSnapshotVenv(t *testing.T) {
	cfg, problem := config.Load()
	if problem != nil {
		t.Fatal(problem)
	}
	root := t.TempDir()
	cfg.Home = filepath.Join(root, "records")
	project := filepath.Join(root, "project")
	if err := os.MkdirAll(project, 0700); err != nil {
		t.Fatal(err)
	}
	files := map[string]string{
		"pyproject.toml": `[project]
name = "remote-cli-proof"
version = "1.0.0"
requires-python = ">=3.12,<3.13"
dependencies = []
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
only-include = ["proof.py"]
`,
		"package.toml":    "[application]\nobject='proof:app'\n",
		".python-version": "3.12\n",
		"proof.py": `from cozy_runtime.author import App
import msgspec
app = App()
class Input(msgspec.Struct):
    value: int = 1
class Result(msgspec.Struct):
    value: int
@app.job
def execute(payload: Input) -> Result:
    return Result(payload.value)
`,
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(project, name), []byte(body), 0600); err != nil {
			t.Fatal(err)
		}
	}
	command := exec.Command("uv", "lock", "--offline", "--python", "3.12")
	command.Dir, command.Env = project, cfg.Tool()
	if out, err := command.CombinedOutput(); err != nil {
		t.Fatalf("lock: %s: %s", err, out)
	}
	layout, problem := home.Open(cfg.Home)
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	_, problem = install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/remote-cli-proof"},
		Local: &install.LocalSource{Tree: project, Package: "local/remote-cli-proof", Release: "1.0.0"}})
	if problem != nil {
		t.Fatal(problem)
	}
	// The author edits the request schema after installation. Intake must parse
	// this frozen source, even while it reuses the installed dependency metadata.
	updated := strings.ReplaceAll(files["proof.py"], "value: int = 1", "value: str = 'fresh'")
	if err := os.WriteFile(filepath.Join(project, "proof.py"), []byte(updated), 0600); err != nil {
		t.Fatal(err)
	}
	began := time.Now()
	status, out := runCozy(t, cfg.Home, "run", "local/remote-cli-proof/execute", "value=fresh", "--rental-only", "--dry-run", "--json", "--full")
	if status != 0 {
		t.Fatalf("remote invocation [%d]: %s", status, out)
	}
	if !strings.Contains(out, `"fresh"`) || !strings.Contains(out, `"planned"`) {
		t.Fatalf("fresh schema did not reach remote submission planning: %s", out)
	}
	t.Logf("public cozy run remote capture and planning: %s", time.Since(began))
}
