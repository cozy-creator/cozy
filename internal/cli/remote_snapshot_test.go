package cli

import (
	"bytes"
	"context"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/output"
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
	initial, problem := install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/remote-cli-proof"},
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
	ctx := &Context{Cfg: cfg, Out: io.Discard, Err: io.Discard, Inv: &Invocation{
		Args: []string{"local/remote-cli-proof/execute", "value=fresh"}, Bools: map[string]bool{"--rental-only": true, "--dry-run": true},
		Values: map[string][]string{}, Mode: output.Mode{JSON: true, Full: true},
	}}
	began := time.Now()
	target, surface, problem := snapshotLocalJob(ctx, Target{Package: initial.Install.Package, Release: "1.0.0", InstallID: initial.Install.ID, Function: "execute"})
	if problem != nil {
		t.Fatal(problem)
	}
	defer reclaimSnapshot(ctx, target)
	releaseSnapshotReader(target)
	retained, problem := store.Install(target.InstallID)
	if problem != nil || retained == nil {
		t.Fatalf("snapshot: %v", problem)
	}
	if _, err := os.Stat(filepath.Join(retained.Dir, "venv")); !os.IsNotExist(err) {
		t.Fatalf("intake rebuilt snapshot venv: %v", err)
	}
	if _, problem := install.InstalledRequirements(context.Background(), *retained); problem != nil {
		t.Fatal(problem)
	}
	job, problem := surface.Function("execute")
	if problem != nil {
		t.Fatal(problem)
	}
	var result bytes.Buffer
	ctx.Out = &result
	if problem := handleJobSubmit(ctx, target, job); problem != nil {
		t.Fatal(problem)
	}
	if !strings.Contains(result.String(), `"fresh"`) || !strings.Contains(result.String(), `"planned"`) {
		t.Fatalf("fresh typed request did not reach remote submission boundary: %s", result.String())
	}
	t.Logf("fresh source capture through remote submission planning: %s; no snapshot venv", time.Since(began))
}
