package producttest

import (
	"net/http"
	"strings"
	"testing"
)

// A submission field this running daemon does not implement is a capability it lacks, so
// the request refuses rather than silently dropping intent, and names the recovery.
func TestSubmissionFieldFromNewerClientNamesTheDaemonRestart(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	for _, path := range []string{"/v1/requests", "/v1/local/jobs"} {
		reply := daemon.call(t, http.MethodPost, path, map[string]any{
			"package": "proof/p", "function": "f", "input": map[string]any{}, "future_option": true,
		}, "Idempotency-Key", "stale-daemon"+strings.ReplaceAll(path, "/", "-"))
		if reply.Status != http.StatusBadRequest || reply.code() != "malformed_body" ||
			!strings.Contains(string(reply.Body), "cozy down") {
			t.Fatalf("%s did not name the daemon restart: %s", path, reply.brief())
		}
	}
}
