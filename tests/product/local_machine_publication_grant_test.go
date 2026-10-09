package producttest

import (
	"encoding/json"
	"net/url"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// A run on this computer's machine that may publish (`cozy model upload` without a rental,
// or any job with --allow-upload) asks Tensorhub for the same publication grant a rental
// gets, bound to its Host's own key, never a rental id.
func TestLocalMachinePublicationGrantBindsCertificateWithoutRegistration(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	h := newMachineHub(t)
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
	var grants []url.Values
	for _, asked := range h.oauth.requests() {
		if publicationRequest(asked) {
			grants = append(grants, asked)
		}
	}
	if len(grants) != 1 {
		t.Fatalf("want one machine publication grant, got %d [exit %d]\n%s", len(grants), code, out)
	}
	var detail []map[string]any
	must(t, json.Unmarshal([]byte(grants[0].Get("authorization_details")), &detail))
	if len(detail) != 1 || detail[0]["machine_id"] != "" || grants[0].Get("scope") != "offline_access" {
		t.Fatalf("local publication grant acquired a registry/rental identity: %v", grants[0])
	}
	if grants[0].Get("dpop_jkt") == "" || grants[0].Get("client_id") != "cozy-machine" {
		t.Fatalf("the grant is not bound to the machine's key: %v", grants[0])
	}
	if repositories, _ := json.Marshal(detail[0]["repositories"]); string(repositories) != `[{"name":"output","org":"proof"}]` {
		t.Fatalf("the grant is not exactly the consented repository: %s", repositories)
	}
	if code != 0 {
		t.Fatalf("the run failed after its grant [exit %d]\n%s", code, out)
	}
}
