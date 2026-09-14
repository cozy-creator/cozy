package producttest

import (
	"bytes"
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

func TestMachinePublicationIntentCannotChangeAcrossClientRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	request, _, problem := store.Submit(records.Request{ID: "job-publication-intent", IdemKey: "publication-intent", Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	raw := []byte(`{"authorization_id":"9a4c3c53-564b-4497-8398-ac0f55bcc2cc","rental_id":"private-machine","repositories":[{"org":"alice","name":"model"}],"permissions":["assessment","checkpoint","release"],"expires_at_unix":1900000000,"certificate_der_b64url":"cHVibGljLW1ldGFkYXRh"}`)
	if problem := store.RecordMachinePublicationIntent(request.ID, raw); problem == nil {
		t.Fatal("publication authority was recorded before machine selection")
	}
	fatal(t, store.LinkMachineExecution(request.ID, "private-machine"))
	fatal(t, store.RecordMachinePublicationIntent(request.ID, raw))
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	held, problem := store.MachinePublicationIntent(request.ID)
	fatal(t, problem)
	if !bytes.Equal(held, raw) {
		t.Fatal("recovery changed the frozen publication authority")
	}
	fatal(t, store.RecordMachinePublicationIntent(request.ID, raw))
	changed := bytes.Replace(raw, []byte(`"name":"model"`), []byte(`"name":"another"`), 1)
	if problem := store.RecordMachinePublicationIntent(request.ID, changed); problem == nil {
		t.Fatal("an ambiguous authorization retry expanded its repository scope")
	}
}
