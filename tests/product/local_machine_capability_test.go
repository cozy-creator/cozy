package producttest

import (
	"encoding/json"
	"encoding/pem"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// An upload on this computer's machine (`cozy model upload` with no rental) carries the same
// run capability a rental's does (Tensorhub th-241): signed offline by this computer's device
// key for the machine's own leaf, naming exactly its destination. The machine trades it where
// it reads the Hub; nothing registers the machine and no grant is asked of the Hub.
func TestALocalUploadCarriesACapabilityForThisComputersMachine(t *testing.T) {
	if *machineHostBinary == "" {
		t.Skip("requires -machine-host: the pod-supervisor this computer's machine runs")
	}
	h := newMachineHub(t)
	h.hubAccess.refuse = true
	root, err := os.MkdirTemp("", "czg")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	h.hubAccess.signIn(t, root)
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
	provisionMachine(t, root)
	code, out := runCozy(t, root, "model", "upload", strayTensor(t, root), "proof/output", "--await", "--json")
	skipWithoutMachine(t, code, out)
	trades := h.hubAccess.trades()
	if code == 0 || !strings.Contains(out, "capability_refused") || len(trades) != 1 {
		t.Fatalf("want one refused trade of the upload's capability, got %d [exit %d]\n%s", len(trades), code, out)
	}
	header, claims := h.hubAccess.capabilityOf(t, trades[0])
	leaf, err := os.ReadFile(filepath.Join(root, "machine", "leaf.pem"))
	must(t, err)
	block, _ := pem.Decode(leaf)
	cnf, _ := claims["cnf"].(map[string]any)
	if block == nil || cnf["jkt"] != leafJKT(t, block.Bytes) || header["kid"] != h.hubAccess.deviceKeyID ||
		claims["sub"] != h.hubAccess.userID || claims["aud"] != h.server.URL {
		t.Fatalf("the capability is not this computer's machine's, by this device: %v %v", header, claims)
	}
	if ops, _ := json.Marshal(claims["authorization_details"]); string(ops) != `[{"model":"proof/output","type":"tensorhub_model_publish"}]` {
		t.Fatalf("the capability is not exactly the destination: %s", ops)
	}
}
