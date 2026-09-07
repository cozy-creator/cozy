package producttest

import (
	"encoding/json"
	"net/http"
	"testing"
)

// Lifecycle commands operate on an existing request. A miss must reach the
// daemon's typed job lookup, without treating pause/resume as package names or
// creating a replacement transaction.
func TestPrivateTransactionCommandsResolveExistingRequest(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	for _, action := range []string{"pause", "resume"} {
		t.Run(action, func(t *testing.T) {
			response := daemon.call(t, http.MethodPost,
				"/v1/local/jobs/req-private-transaction-absent/"+action,
				map[string]any{"actor": "product proof"})
			if response.Status != http.StatusNotFound || response.code() != "not_found" {
				t.Errorf("%s must resolve an existing transaction: %s", action, response.brief())
			}
			code, stdout, stderr := runCozyStreams(t, root, "run", action,
				"req-private-transaction-absent", "--json")
			var document struct {
				Error struct {
					Code string `json:"code"`
				} `json:"error"`
			}
			if err := json.Unmarshal([]byte(stdout), &document); err != nil ||
				code == 0 || document.Error.Code != "not_found" {
				t.Errorf("run %s must report the owner's missing transaction [exit %d]: stdout=%s stderr=%s",
					action, code, stdout, stderr)
			}
		})
	}
	if requests := listInvocations(t, root); len(requests) != 0 {
		t.Fatalf("lifecycle commands created requests: %+v", requests)
	}
}
