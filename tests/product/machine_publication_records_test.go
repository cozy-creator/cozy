package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func TestSubmissionIntentCommitsWithItsRequest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request := records.Request{ID: "job-atomic-intent", IdemKey: "atomic-intent", Kind: "job", Package: "local/example", Entrypoint: "main", Payload: []byte(`{}`), BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true}
	if _, _, problem := store.SubmitWithEvent(request, map[string]any{"unserializable": make(chan bool)}); problem == nil {
		t.Fatal("invalid submission intent was accepted")
	}
	row, problem := store.RequestByIdempotencyKey(request.IdemKey)
	fatal(t, problem)
	if row != nil {
		t.Fatal("request survived without its frozen intent")
	}
	_, _, problem = store.SubmitWithEvent(request, map[string]any{"retain_work": true, "allow_publish": []string{"alice/model"}, "timeout_ms": 60000, "deadline_unix_ms": uint64(1900000000123)})
	fatal(t, problem)
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	repositories, problem := store.RequestPublicationRepositories(request.ID)
	fatal(t, problem)
	duration, deadline, problem := store.RequestExecutionTiming(request.ID)
	fatal(t, problem)
	if len(repositories) != 1 || repositories[0] != "alice/model" || duration != 60000 || deadline != 1900000000123 {
		t.Fatal("reopening the committed request lost its consent or deadline")
	}
}
