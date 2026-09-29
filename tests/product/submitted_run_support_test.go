package producttest

import (
	"encoding/json"
	"os/exec"
	"path/filepath"
	"slices"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
)

// submitRun sends a run or transfer through the real CLI and daemon admission and returns the durable
// request it recorded (nil when the CLI refused first). The run is canceled as soon as it is
// recorded, so no placement continues once its admitted facts are read and the CLI's
// optimistic observation ends on that terminal instead of waiting out its window.
func submitRun(t *testing.T, root, key string, args ...string) (*records.Request, int, string) {
	t.Helper()
	return submitRunWith(t, root, key, nil, args...)
}

// submitRunWith is submitRun with extra child environment, such as another HOME.
func submitRunWith(t *testing.T, root, key string, imposed []string, args ...string) (*records.Request, int, string) {
	t.Helper()
	// The key goes ahead of any literal `--` tail.
	command := append([]string{}, args...)
	at := slices.Index(command, "--")
	if at < 0 {
		at = len(command)
	}
	command = slices.Insert(command, at, "--idempotency-key="+key)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	env := childEnv(t, root, imposed...)
	cancel := func(request *records.Request) {
		if request != nil && !records.Settled(request.State) {
			cmd := exec.Command(cozyBin, "run", "cancel", request.ID)
			cmd.Env = env
			_ = cmd.Run()
		}
	}
	submitted, canceled := make(chan struct{}), make(chan struct{})
	go func() {
		defer close(canceled)
		for {
			select {
			case <-submitted:
				return
			case <-time.After(20 * time.Millisecond):
			}
			if request, problem := store.RequestByIdempotencyKey(key); problem == nil && request != nil {
				cancel(request)
				return
			}
		}
	}()
	code, out := runCozyWith(t, root, imposed, command...)
	close(submitted)
	<-canceled
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	cancel(request)
	return request, code, out
}

// submittedPayload is the admitted request payload without its local asset bindings.
func submittedPayload(t *testing.T, request *records.Request) map[string]any {
	t.Helper()
	var payload map[string]any
	must(t, json.Unmarshal(request.Payload, &payload))
	return payload
}
