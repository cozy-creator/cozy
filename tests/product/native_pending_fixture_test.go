package producttest

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func pendingNativeRequest(t *testing.T, store *records.Store) records.Request {
	t.Helper()
	request, _, problem := store.Submit(records.Request{ID: "job-observed", IdemKey: "observed",
		Package: "local/observer", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`),
		BodyDigest: childDigest("1"), RetainWork: true, MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	allowed, problem := store.MarkRunV1Sent(request.ID)
	fatal(t, problem)
	if !allowed {
		t.Fatal("live native fixture could not be marked sent")
	}
	return request
}

func pendingNativeFixture(t *testing.T) (*records.Store, records.Request) {
	t.Helper()
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	return store, pendingNativeRequest(t, store)
}
