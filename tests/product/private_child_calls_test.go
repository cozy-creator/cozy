package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

func childDigest(letter string) string { return "sha256:" + strings.Repeat(letter, 64) }

func offerChildParent(t *testing.T, store *records.Store, parent records.Request) records.Request {
	t.Helper()
	_, problem := store.Dispatch(records.Attempt{RequestID: parent.ID, InstanceID: "private-worker", SessionID: "private-boot", InvocationDigest: childDigest("1"), InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(parent.ID, 1, "private-boot"))
	current, problem := store.RequestRow(parent.ID)
	fatal(t, problem)
	return *current
}

func closeChild(t *testing.T, store *records.Store, request records.Request, status, state string) {
	t.Helper()
	if request.Ordinal == 0 {
		request = offerChildParent(t, store, request)
	}
	_, problem := store.AcceptTerminal(records.Terminal{RequestID: request.ID, Attempt: 1, SessionID: "private-boot", InvocationDigest: childDigest("1"), TerminalID: "terminal-" + request.ID, TerminalDigest: childDigest("2"), Status: status, RequestState: state, Body: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, store.Closed(request.ID, 1))
}

func TestPrivateChildrenReuseExactSuccessfulOperationAfterParentEdit(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "parent-original", ""))
	call := records.Request{ID: "req-child-original-a", IdemKey: "child-original-a", BodyDigest: childDigest("3"), Package: "local/operation-a", Entrypoint: "compute", Kind: "job", Payload: []byte(`{"size":100}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}
	child, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if !fresh || child.ParentRequestID != parent.ID || child.ReuseScope != parent.ReuseScope {
		t.Fatalf("child lost owner scope: %+v", child)
	}
	closeChild(t, store, child, "SUCCEEDED", "succeeded")
	closeChild(t, store, parent, "FAILED", "blocked")
	next := parent
	next.ID, next.IdemKey = "req-parent-edited", "parent-edited"
	next.RetryOf = parent.ID
	next.BodyDigest, next.LocalPackageDigest = childDigest("6"), childDigest("7")
	next, _, problem = store.Submit(next)
	fatal(t, problem)
	next = offerChildParent(t, store, next)
	call.ID, call.IdemKey, call.ParentRequestID = "req-child-edited-a", "child-edited-a", next.ID
	reused, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if !fresh || reused.ID == child.ID || reused.State != "succeeded" || reused.Ordinal != 0 || reused.ReusedFrom != child.ID {
		t.Fatalf("edited parent did not acquire previous exact result: %+v", reused)
	}
	history, problem := store.Attempts(child.ID)
	fatal(t, problem)
	if len(history) != 1 || history[0].TerminalStatus != "SUCCEEDED" {
		t.Fatal("child reuse rewrote historical execution")
	}
	newHistory, problem := store.Attempts(reused.ID)
	fatal(t, problem)
	if len(newHistory) != 0 {
		t.Fatal("reused child invented an execution attempt")
	}
	call.ChildTargetDigest = childDigest("8")
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("same parent/index accepted changed implementation")
	}
	call.ParentCallIndex = 1
	call.ID, call.IdemKey = "req-child-edited-b", "child-edited-b"
	changed, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot")
	fatal(t, problem)
	if changed.State != "submitted" || changed.ReusedFrom != "" {
		t.Fatal("changed operation inherited stale result")
	}
	call.ID, call.IdemKey, call.ParentCallIndex = "req-child-forged", "child-forged", 2
	if _, _, problem := store.SubmitChild(call, 1, childDigest("9"), "private-boot"); problem == nil {
		t.Fatal("forged parent invocation admitted child")
	}
	_, problem = store.RequestPause(next.ID, "test")
	fatal(t, problem)
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot"); problem == nil {
		t.Fatal("paused parent admitted new child")
	}
}
