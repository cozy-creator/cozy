package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// minimax-h3 1.18.17 published a result field `context: FileAsset | None` that every install
// then refused. Publish reads the staged interface through the install's own decoder, so a
// release no install accepts is refused before anything is declared to the hub.
func TestPublishRefusesWhatInstallRefuses(t *testing.T) {
	project := weightlessProject(t)
	source := filepath.Join(project, "weightless.py")
	raw, err := os.ReadFile(source)
	must(t, err)
	must(t, os.WriteFile(source, append(raw, []byte(`

class OptionalContextOutput(msgspec.Struct):
    context: ImageAsset | None


@app.entrypoint
def optional_context(payload: RefuseInput) -> OptionalContextOutput:
    del payload
    return OptionalContextOutput(None)
`)...), 0o644))

	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	defer server.Close()
	mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.Account{Name: "proof"})
	})
	mux.HandleFunc("GET /v1/packages/proof/cozy-weightless-package/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":"not_found","message":"not published"}}`, http.StatusNotFound)
	})
	mux.HandleFunc("POST /", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("publish reached the hub with an uninstallable release: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusConflict)
	})
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: publication-proof\n"), 0600))
	command := exec.Command(cozyBin, "package", "publish", "--json")
	command.Dir = project
	command.Env = childEnv(t, root)
	output, err := command.CombinedOutput()
	if err == nil {
		t.Fatalf("publish accepted a release every install refuses:\n%s", output)
	}

	// The same staged bytes, read by the install's decoder, give the refusal publish printed.
	pack := buildForPublish(t, project)
	staged, err := os.ReadFile(pack.PackageInterface)
	must(t, err)
	_, installed := callableOf(t, staged, "optional_context")
	if installed == nil || installed.Name != "output_collection_unsupported" {
		t.Fatalf("install decoder answered %v, want output_collection_unsupported", installed)
	}
	for _, want := range []string{"output_collection_unsupported", installed.Message, installed.Remedy} {
		if !strings.Contains(string(output), want) {
			t.Fatalf("publish refusal omits %q:\n%s", want, output)
		}
	}
}
