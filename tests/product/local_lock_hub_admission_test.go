package producttest

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestForeignCalleeLockIsRefusedBeforeRunAdmission(t *testing.T) {
	root := t.TempDir()
	project := filepath.Join(root, "project")
	must(t, os.Mkdir(project, 0o700))
	files := map[string]string{
		"pyproject.toml": "[project]\nname='lock-hub-proof'\nversion='1.0.0'\nrequires-python='>=3.12,<3.13'\ndependencies=[]\n",
		"package.toml":   "[application]\nobject='proof:app'\n",
		"proof.py": "from cozy_runtime.author import App\nimport msgspec\napp=App()\n" +
			"class Input(msgspec.Struct):\n    value: int=1\n" +
			"class Result(msgspec.Struct):\n    value: int\n" +
			"@app.entrypoint(accelerator=False)\ndef execute(payload: Input) -> Result:\n    return Result(payload.value)\n",
		"uv.lock": "version=1\nrequires-python='>=3.12,<3.13'\n[[package]]\nname='lock-hub-proof'\nversion='1.0.0'\nsource={editable='.'}\n" +
			"[[package]]\nname='qwen-image-2'\nversion='0.5.0'\nsource={registry='https://tensorhub.com/v1/index/fidika/simple/'}\n",
	}
	for name, data := range files {
		must(t, os.WriteFile(filepath.Join(project, name), []byte(data), 0o600))
	}
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	_, problem = install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/lock-hub-proof"},
		Local: &install.LocalSource{Tree: project, Package: "local/lock-hub-proof", Release: "1.0.0"}})
	fatal(t, problem)
	var contacted atomic.Int32
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		contacted.Add(1)
		http.NotFound(w, r)
	}))
	defer hub.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\n"), 0o600))
	request, code, output := submitRun(t, root, "foreign-callee-lock", "run", "local/lock-hub-proof/execute", "value=1", "--rental-only", "--json", "--full")
	if request != nil || code == 0 || !strings.Contains(output, "local_package_lock_hub_mismatch") {
		t.Fatalf("foreign lock was admitted: request=%+v exit=%d output=%s", request, code, output)
	}
	if contacted.Load() != 0 {
		t.Fatalf("refusal still contacted the execution Hub %d times", contacted.Load())
	}
}
