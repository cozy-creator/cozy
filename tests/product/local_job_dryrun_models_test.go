package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

func TestLocalJobDryRunResolvesOnlyModelMetadata(t *testing.T) {
	root, tools := t.TempDir(), t.TempDir()
	activity := filepath.Join(root, "tfs.calls")
	// The metadata-only fixture implements the two native metadata verbs. Any fill,
	// verify, or object read fails and is visible, rather than supplying fake bodies.
	script := "#!/bin/sh\nprintf '%s %s\\n' \"$1\" \"$2\" >> '" + activity + "'\ncase \"$1 $2\" in\n'store ensure') exit 0;;\n'repo list') : > \"$5\"; exit 0;;\nesac\nexit 97\n"
	must(t, os.WriteFile(filepath.Join(tools, "tfs"), []byte(script), 0700))
	manifest := []byte(`{"fixture":"tiny immutable root"}`)
	digest, err := canonical.Spell(canonical.Digest(manifest))
	must(t, err)
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"prepare.models.source"}],"name":"prepare","publishes":false,"request":{"fields":[]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":1,"mime_type":"application/vnd.cozy.model-manifest","output_id":"model"}]}]}`)
	parsed, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	var bodyReads, rootReads atomic.Int32
	var corrupted atomic.Bool
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/dryrun/bindings", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write([]byte(`{"bindings":[]}`)) })
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("ref") != "proof/source@1.0.0" || r.URL.Query().Get("lane") != "bf16" {
			t.Errorf("local selection changed: %s", r.URL.RawQuery)
		}
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/source", Release: "1.0.0", Lane: "bf16", ManifestID: digest, HeaderID: "sha256:" + strings.Repeat("7", 64), Bytes: 195021215555, Objects: 6000, Components: []string{"model"}})
	})
	mux.HandleFunc("GET /v1/models/proof/source/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, r *http.Request) {
		rootReads.Add(1)
		if corrupted.Load() {
			_, _ = w.Write([]byte(`{"changed":true}`))
			return
		}
		_, _ = w.Write(manifest)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		bodyReads.Add(1)
		w.WriteHeader(http.StatusServiceUnavailable)
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntfs: "+filepath.Join(tools, "tfs")+"\n"), 0600))
	dir := filepath.Join(root, "installs", "dryrun")
	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(dir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(dir), iface, 0600))
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	_, problem = st.Activate(records.PackageInstall{ID: "dryrun-install", Package: "proof/dryrun", Major: 1, Version: "1.0.0", SourceKind: "tensorhub", Dir: dir, PackageInterface: parsed.Digest, SourceDigest: "sha256:" + strings.Repeat("4", 64), Platform: "linux-x86"})
	fatal(t, problem)
	st.Close()
	args := []string{"run", "proof/dryrun/prepare", "--model.source=proof/source@1.0.0/bf16", "--dry-run", "--json", "--full", "--idempotency-key=dryrun-only"}
	code, out := runAdmissionCLI(t, root, tools, args...)
	if code != 0 || !strings.Contains(out, `"status":"planned"`) || !strings.Contains(out, digest) {
		t.Fatalf("local dryrun acquired or failed: %d %s", code, out)
	}
	if rootReads.Load() != 1 || bodyReads.Load() != 0 {
		t.Fatalf("dryrun fetched beyond root: roots=%d bodies=%d", rootReads.Load(), bodyReads.Load())
	}
	trace, err := os.ReadFile(activity)
	must(t, err)
	for _, line := range strings.Split(strings.TrimSpace(string(trace)), "\n") {
		if line != "store ensure" && line != "repo list" {
			t.Fatalf("dryrun invoked nonmetadata TensorFS operation: %s", trace)
		}
	}
	assertAdmissionDidNotSubmit(t, root, "dryrun-only")
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("dryrun started daemon")
	}
	// A retained local checkpoint uses native repository metadata too; dry-run
	// neither calls Hub model resolution nor hashes its entire already-local body.
	metadata := `{"checkpoints":[{"manifest":{"sha256":"` + strings.TrimPrefix(digest, "sha256:") + `","length":` + fmt.Sprint(len(manifest)) + `}}]}`
	retainedScript := "#!/bin/sh\nprintf '%s %s\\n' \"$1\" \"$2\" >> '" + activity + "'\ncase \"$1 $2\" in\n'store ensure') exit 0;;\n'repo get') printf '%s' '" + metadata + "' > \"$7\"; exit 0;;\nesac\nexit 97\n"
	must(t, os.WriteFile(filepath.Join(tools, "tfs"), []byte(retainedScript), 0700))
	retainedArgs := append([]string{}, args...)
	retainedArgs[2] = "--model.source=local/retained#" + digest
	priorRoots := rootReads.Load()
	code, out = runAdmissionCLI(t, root, tools, retainedArgs...)
	if code != 0 || !strings.Contains(out, digest) || rootReads.Load() != priorRoots {
		t.Fatalf("local checkpoint dryrun touched bodies/Hub: %d %s", code, out)
	}
	must(t, os.WriteFile(filepath.Join(tools, "tfs"), []byte(script), 0700))
	corrupted.Store(true)
	code, out = runAdmissionCLI(t, root, tools, args...)
	if code == 0 || !strings.Contains(out, "job.model_manifest_changed") {
		t.Fatalf("changed root accepted: %d %s", code, out)
	}
}
