package producttest

import (
	"net/http"
	"net/http/httptest"
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
	// An isolated hub that serves nothing: admission is observed, no machine is bought.
	hub := httptest.NewServer(http.NotFoundHandler())
	defer hub.Close()
	must(t, os.WriteFile(filepath.Join(cfg.Home, config.FileName), []byte("tensorhub_url: "+hub.URL+"\n"), 0600))
	began := time.Now()
	// The daemon's own editable sync may hold the install writer as it starts; the CLI
	// refuses rather than waits, so a retry follows that writer.
	request, _, out := submitRun(t, cfg.Home, "remote-editable-fresh", "run", "local/remote-cli-proof/execute", "value=fresh", "--rental-only", "--json", "--full")
	for attempt := 0; request == nil && strings.Contains(out, "another Cozy writer") && attempt < 20; attempt++ {
		time.Sleep(250 * time.Millisecond)
		request, _, out = submitRun(t, cfg.Home, "remote-editable-fresh", "run", "local/remote-cli-proof/execute", "value=fresh", "--rental-only", "--json", "--full")
	}
	if request == nil || submittedPayload(t, request)["value"] != "fresh" {
		t.Fatalf("fresh schema did not reach remote submission: %+v %s", request, out)
	}
	t.Logf("public cozy run remote capture and admission: %s", time.Since(began))
}
