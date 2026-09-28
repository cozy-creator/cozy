package producttest

import (
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// A run on this computer's machine that may publish (`cozy model upload` without a rental,
// or any job with --allow-upload) asks Tensorhub for the same machine authorization a rental
// gets: the owned machine's id and its Host's own certificate, never a rental id.
func TestLocalMachinePublicationGrantNamesTheOwnedMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	h := newMachineHub(t)
	var mu sync.Mutex
	var grants []map[string]any
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost && r.URL.Path == "/v1/machine-authorizations" {
			var body map[string]any
			if json.NewDecoder(r.Body).Decode(&body) != nil {
				w.WriteHeader(http.StatusBadRequest)
				return
			}
			mu.Lock()
			grants = append(grants, body)
			mu.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"token": "machine-grant", "expires_at": time.Now().Add(time.Hour).UTC().Format(time.RFC3339)})
			return
		}
		served.ServeHTTP(w, r)
	})
	root, err := os.MkdirTemp("", "czg")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "machine", "stop")
		_, _ = runCozy(t, root, "down")
		if t.Failed() {
			log, _ := os.ReadFile(filepath.Join(root, "machine", "host.log"))
			t.Logf("evidence retained at %s\nlocal Host log:\n%s", root, log)
		} else {
			_ = removeAllForce(root)
		}
	})
	script := filepath.Join(root, "publishes.py")
	must(t, os.WriteFile(script, []byte(`# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.18.67,<1"]
# ///

def main() -> dict[str, str]:
    return {"published": "nothing"}
`), 0o600))
	code, out := runCozy(t, root, "run", script, "--allow-upload", "proof/output", "--await", "--json")
	skipWithoutMachine(t, code, out)
	mu.Lock()
	defer mu.Unlock()
	if len(grants) != 1 {
		t.Fatalf("want one machine authorization, got %d [exit %d]\n%s", len(grants), code, out)
	}
	grant, _ := grants[0]["requested_grant"].(map[string]any)
	h.mu.Lock()
	machineID, _ := grant["machine_id"].(string)
	_, registered := h.machines[machineID]
	h.mu.Unlock()
	if !strings.HasPrefix(machineID, "om-") || !registered || grant["rental_id"] != nil {
		t.Fatalf("the grant does not name this computer's registered machine: %v", grants[0])
	}
	leaf, err := base64.RawURLEncoding.DecodeString(grants[0]["delegate_certificate_der_b64url"].(string))
	if err != nil {
		t.Fatalf("the grant carries no Host certificate: %v", err)
	}
	if _, err := x509.ParseCertificate(leaf); err != nil {
		t.Fatalf("the grant's Host certificate is not a certificate: %v", err)
	}
	if repositories, _ := json.Marshal(grant["repositories"]); string(repositories) != `[{"name":"output","org":"proof"}]` {
		t.Fatalf("the grant is not exactly the consented repository: %s", repositories)
	}
	if code != 0 {
		t.Fatalf("the run failed after its grant [exit %d]\n%s", code, out)
	}
}
