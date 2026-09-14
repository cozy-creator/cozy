package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Run the actual CLI through cached metadata acquisition, then refuse the body
// walk. The protocol fixture proves presentation and error handling only; it
// does not claim TensorFS custody, model loading, or successful inference.
func TestLocalModelPreparationOutputHidesStorageChatter(t *testing.T) {
	root, tools := t.TempDir(), t.TempDir()
	header := strings.Repeat("7", 64)
	script := `#!/bin/sh
case "$1 $2" in
  'store ensure') exit 0;;
  'repo list') : > "$5"; exit 0;;
  'manifest admit') echo 'admitted=false'; exit 0;;
  'manifest inspect') printf '%s\n' '{"kind":"cozytensors","sha256":"` + header + `","length":64}' > "$5"; exit 0;;
  'manifest walk') echo 'REFUSED TEST_STOP: stopped before model download' >&2; exit 97;;
esac
if [ "$1" = contains ]; then echo 'contains: true'; exit 0; fi
exit 96
`
	must(t, os.WriteFile(filepath.Join(tools, "tfs"), []byte(script), 0700))
	iface := []byte(`{"application":"q:app","entrypoints":[{"models":[{"class":"Source","component_use":{},"path":"generate.models.source"}],"name":"generate","request":{"fields":[]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
	parsed, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/quiet/bindings", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"bindings":[]}`))
	})
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/source", Release: "1.0.0", Lane: "bf16", ManifestID: "sha256:" + strings.Repeat("a", 64), HeaderID: "sha256:" + header, Components: []string{"model"}})
	})
	mux.HandleFunc("GET /v1/models/proof/source/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"fixture":"cached root"}`))
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntfs: "+filepath.Join(tools, "tfs")+"\n"), 0600))
	dir := filepath.Join(root, "installs", "quiet-install")
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	_, problem = store.Activate(records.PackageInstall{ID: "quiet-install", Package: "proof/quiet", Major: 1, Version: "1.0.0", SourceKind: "tensorhub", Dir: dir, PackageInterface: parsed.Digest, SourceDigest: "sha256:" + strings.Repeat("4", 64), Platform: "linux-x86"})
	fatal(t, problem)
	store.Close()
	for _, structured := range []bool{false, true} {
		args := []string{"run", "proof/quiet/generate", "--model.source=proof/source@1.0.0/bf16", "--idempotency-key=quiet-preparation"}
		if structured {
			args = append(args, "--json")
		}
		code, out := runAdmissionCLI(t, root, tools, args...)
		t.Logf("json=%t exit=%d\n%s", structured, code, out)
		if code == 0 || !strings.Contains(out, "TEST_STOP") {
			t.Fatalf("lost actual refusal: %d %s", code, out)
		}
		for _, noise := range []string{"already resident", "nothing to fetch", "Resolving model for", "admitted", "timing:"} {
			if strings.Contains(out, noise) {
				t.Errorf("internal chatter %q: %s", noise, out)
			}
		}
		if !structured {
			target := strings.Index(out, "Execution target: local machine")
			preparing := strings.Index(out, "Preparing model proof/source@1.0.0 for local execution")
			if target < 0 || preparing < target {
				t.Fatalf("preparation hid local target: %s", out)
			}
		} else {
			var body map[string]any
			if json.Unmarshal([]byte(out), &body) != nil {
				t.Fatalf("progress contaminated JSON: %s", out)
			}
		}
	}
	assertAdmissionDidNotSubmit(t, root, "quiet-preparation")
}
