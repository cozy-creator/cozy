package producttest

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// `cozy model delete` never prompts: without --yes it deletes nothing. A model still
// serving a release is refused with the Hub's remedy, and a checkpoint ref deletes
// only that checkpoint.
func TestModelDeleteNeedsYesAndDeletesThroughTheHub(t *testing.T) {
	checkpoint := "sha256:" + strings.Repeat("ab", 32)
	var mu sync.Mutex
	var deletes []string
	live := true
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete || r.Header.Get("Authorization") != "Bearer delete-proof" {
			http.Error(w, `{"error":{"code":"proof.unexpected","message":"unexpected request"}}`, http.StatusBadRequest)
			return
		}
		mu.Lock()
		deletes = append(deletes, r.URL.Path)
		refuse := live
		mu.Unlock()
		switch {
		case r.URL.Path == "/v1/models/proof/bench/checkpoints/"+checkpoint:
			_, _ = w.Write([]byte(`{"checkpoint_id":"` + checkpoint + `","repository_sha256":"sha256:` + strings.Repeat("cd", 32) + `","removed":true}`))
		case refuse:
			w.WriteHeader(http.StatusConflict)
			_, _ = w.Write([]byte(`{"error":{"code":"model.releases_live","message":"proof/bench still serves release(s) 1.0.0",` +
				`"remedy":"yank each first: cozy model yank proof/bench --release <release>"}}`))
		default:
			_, _ = w.Write([]byte(`{"model":"proof/bench","checkpoints":3,"reclaimable_objects":1605,"reclaimable_bytes":49284145561}`))
		}
	}))
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+
		"\ntensorhub_token: delete-proof\n"), 0o600))

	code, out := runCozy(t, root, "model", "delete", "proof/bench")
	if code == 0 || !strings.Contains(out, "--yes") || len(deletes) != 0 {
		t.Fatalf("an unconfirmed delete ran [exit %d, %d deletes]\n%s", code, len(deletes), out)
	}
	code, out = runCozy(t, root, "model", "delete", "proof/bench", "--yes", "--json")
	if code == 0 || !strings.Contains(out, "model.releases_live") || !strings.Contains(out, "cozy model yank proof/bench") {
		t.Fatalf("a model serving a release was not refused with its remedy [exit %d]\n%s", code, out)
	}
	mu.Lock()
	live = false
	mu.Unlock()
	code, out = runCozy(t, root, "model", "delete", "proof/bench", "--yes", "--json")
	if code != 0 || !strings.Contains(out, `"reclaimable_bytes":49284145561`) || !strings.Contains(out, `"status":"deleted"`) {
		t.Fatalf("the model delete did not report the Hub's result [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "model", "delete", "proof/bench#"+checkpoint, "--yes", "--json")
	if code != 0 || !strings.Contains(out, `"checkpoint":"`+checkpoint+`"`) {
		t.Fatalf("the checkpoint delete failed [exit %d]\n%s", code, out)
	}
	code, out = runCozy(t, root, "model", "delete", "proof/bench#latest", "--yes")
	if code == 0 || len(deletes) != 3 {
		t.Fatalf("a malformed checkpoint reached the Hub [exit %d, %d deletes]\n%s", code, len(deletes), out)
	}
	if deletes[1] != "/v1/models/proof/bench" || deletes[2] != "/v1/models/proof/bench/checkpoints/"+checkpoint {
		t.Fatalf("deletes went to %v", deletes)
	}
}
