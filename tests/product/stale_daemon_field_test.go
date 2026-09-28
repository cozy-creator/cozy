package producttest

import (
	"maps"
	"net/http"
	"strings"
	"testing"
)

// A newer cozy may send a field this running daemon predates. The daemon reads the fields it
// knows and ignores the rest: every route answers exactly as it does without the field.
func TestFieldFromNewerClientIsIgnoredByAnOlderDaemon(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	submission := map[string]any{"package": "proof/p", "function": "f", "input": map[string]any{}}
	for path, body := range map[string]map[string]any{
		"/v1/requests":   submission,
		"/v1/local/jobs": submission,
		"/v1/local/rentals/rental-prune-absent/prune": {},
	} {
		key := strings.ReplaceAll(path, "/", "-")
		older := daemon.call(t, http.MethodPost, path, body, "Idempotency-Key", "older"+key)
		newer := maps.Clone(body)
		newer["future_option"] = true
		reply := daemon.call(t, http.MethodPost, path, newer, "Idempotency-Key", "newer"+key)
		if reply.Status != older.Status || reply.code() != older.code() || reply.code() == "malformed_body" || reply.code() == "invalid_request" {
			t.Fatalf("%s: a newer client's field changed the answer: %s; without it: %s", path, reply.brief(), older.brief())
		}
	}
}
