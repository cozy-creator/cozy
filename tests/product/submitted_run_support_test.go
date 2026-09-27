package producttest

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// submitRun sends a run or transfer through the real CLI and daemon admission and returns the durable
// request it recorded (nil when the CLI refused first). The run is canceled afterwards so
// no placement continues once its admitted facts have been read.
func submitRun(t *testing.T, root, key string, args ...string) (*records.Request, int, string) {
	t.Helper()
	// The key goes ahead of any literal `--` tail.
	command := append([]string{}, args...)
	at := slices.Index(command, "--")
	if at < 0 {
		at = len(command)
	}
	command = slices.Insert(command, at, "--idempotency-key="+key)
	code, out := runCozy(t, root, command...)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request != nil && !records.Settled(request.State) {
		_, _ = runCozy(t, root, "run", "cancel", request.ID)
	}
	return request, code, out
}

// submittedPayload is the admitted request payload without its local asset bindings.
func submittedPayload(t *testing.T, request *records.Request) map[string]any {
	t.Helper()
	var payload map[string]any
	must(t, json.Unmarshal(request.Payload, &payload))
	return payload
}
