package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func childDigest(letter string) string { return "sha256:" + strings.Repeat(letter, 64) }

func TestJobInstallationUsesRuntimeOwnedHandle(t *testing.T) {
	development := &pb.DevelopmentPackage{Package: "local/script", Release: "0.0.0"}
	placement := &pb.Placement{PackageMode: &pb.Placement_Development{Development: development}}
	set := &pb.PlacementSet{Placements: []*pb.Placement{placement}}
	for _, id := range []string{"installed-original", "installed-edited"} {
		placement.InstallationId = id
		raw, _, err := canonical.Identity(set)
		must(t, err)
		got, problem := orchestrator.JobInstallationID(raw, "local/script")
		fatal(t, problem)
		if got != id {
			t.Fatalf("prepared installation changed: %s", got)
		}
	}
	placement.InstallationId = ""
	raw, _, err := canonical.Identity(set)
	must(t, err)
	if _, problem := orchestrator.JobInstallationID(raw, "local/script"); problem == nil {
		t.Fatal("missing installed resource acquired an invented identity")
	}
}

func offerChildParent(t *testing.T, store *records.Store, parent records.Request) records.Request {
	t.Helper()
	_, problem := store.Dispatch(records.Attempt{RequestID: parent.ID, InstanceID: "private-worker", SessionID: "private-boot", InvocationDigest: childDigest("1"), InvocationCanonical: []byte(`{}`)})
	fatal(t, problem)
	fatal(t, store.OfferDispatch(parent.ID, 1, "private-boot"))
	current, problem := store.RequestRow(parent.ID)
	fatal(t, problem)
	return *current
}

func TestUnpublishedChildReopenPreservesPriorOwnership(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	prior := recordPrivateTransaction(t, store, "reopen", "")
	_, problem = store.RequestPause(prior.ID, "before reopen")
	fatal(t, problem)
	_, problem = store.CompleteRequestPause(prior.ID)
	fatal(t, problem)
	before, problem := store.RequestRow(prior.ID)
	fatal(t, problem)
	store.Close()
	store, problem = records.OpenForDaemon(path, "")
	fatal(t, problem)
	defer store.Close()
	after, problem := store.RequestRow(prior.ID)
	fatal(t, problem)
	if after.State != "paused" || !after.RetainWork || after.ReuseScope != before.ReuseScope || after.ControlRevision != before.ControlRevision || after.ParentCallIndex != -1 {
		t.Fatalf("reopen changed retained ownership: before=%+v after=%+v", before, after)
	}
}

func TestUnpublishedParentRetainsExactOrchestrationContract(t *testing.T) {
	path := filepath.Join(t.TempDir(), "creator.sqlite")
	store, problem := records.Open(path)
	fatal(t, problem)
	parent := recordPrivateTransaction(t, store, "parent-capacity", "")
	directive := &pb.JobDirective{InstallationId: childDigest("a"), JobDescriptorId: parent.PlanID, Orchestration: true, ResourceCaps: &pb.ResourceCaps{MaxRssBytes: 123456}}
	raw, _, err := canonical.Identity(directive)
	must(t, err)
	fatal(t, store.CaptureOrchestrationDirective(parent.ID, raw))
	fatal(t, store.CaptureOrchestrationDirective(parent.ID, raw))
	directive.ResourceCaps.MaxRssBytes++
	changed, _, err := canonical.Identity(directive)
	must(t, err)
	if problem := store.CaptureOrchestrationDirective(parent.ID, changed); problem == nil {
		t.Fatal("parent capacity was replaced after capture")
	}
	store.Close()
	store, problem = records.Open(path)
	fatal(t, problem)
	defer store.Close()
	retained, problem := store.RequestRow(parent.ID)
	fatal(t, problem)
	if string(retained.OrchestrationDirective) != string(raw) {
		t.Fatal("parent restart lost its exact capacity declaration")
	}
	_, problem = store.RequestPause(parent.ID, "test")
	fatal(t, problem)
	_, problem = store.CompleteRequestPause(parent.ID)
	fatal(t, problem)
	retry := *retained
	retry.ID, retry.IdemKey, retry.RetryOf = "req-parent-capacity-new", "parent-capacity-new", parent.ID
	retry, _, problem = store.Submit(retry)
	fatal(t, problem)
	if len(retry.OrchestrationDirective) != 0 {
		t.Fatal("new parent inherited old execution capacity without resolving its new code")
	}
}

func TestUnpublishedChildBindingsAreImmutableAndOwnTheirImplementation(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	parent := cleanupTestInstall(layout, "1111111111111111", "1.0.0")
	child := cleanupTestInstall(layout, "2222222222222222", "1.0.0")
	replacement := cleanupTestInstall(layout, "3333333333333333", "1.0.0")
	for _, inst := range []records.PackageInstall{parent, child, replacement} {
		fatal(t, store.RecordInstall(inst))
	}
	binding := records.ChildBinding{ParentInstallID: parent.ID, Module: "private_ops", Export: "compute", ChildInstallID: child.ID, Entrypoint: "compute"}
	fatal(t, store.RecordChildBindings([]records.ChildBinding{binding}))
	fatal(t, store.RecordChildBindings([]records.ChildBinding{binding}))
	binding.ChildInstallID = replacement.ID
	if problem := store.RecordChildBindings([]records.ChildBinding{binding}); problem == nil {
		t.Fatal("captured dependency accepted a replacement implementation")
	}
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	completed, _, problem := store.Submit(records.Request{ID: "req-bound-result", IdemKey: "bound-result", Kind: "job", Package: child.Package, Entrypoint: "compute", Payload: []byte(`{}`), BodyDigest: childDigest("1"), InstallID: child.ID})
	fatal(t, problem)
	closeChild(t, store, completed, "SUCCEEDED", "succeeded")
	forgotten, problem := store.ForgetIfUnreferenced(child.ID)
	fatal(t, problem)
	if forgotten {
		t.Fatal("GC discarded an implementation still owned by a parent interface")
	}
	completedRow, problem := store.RequestRow(completed.ID)
	fatal(t, problem)
	if completedRow.InstallID != child.ID {
		t.Fatal("rejected GC severed a completed child's exact result schema")
	}
	held, problem := store.LocalInstallationInUse(child.ID)
	fatal(t, problem)
	if !held {
		t.Fatal("GC discarded the frozen implementation wheel revision")
	}
	forgotten, problem = store.ForgetIfUnreferenced(parent.ID)
	fatal(t, problem)
	if !forgotten {
		t.Fatal("unowned parent could not release its dependency references")
	}
	forgotten, problem = store.ForgetIfUnreferenced(child.ID)
	fatal(t, problem)
	if !forgotten {
		t.Fatal("implementation was not reclaimable after its final parent left")
	}
}

func TestUnpublishedChildrenDoNotReuseEffectsByDefault(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "effects-original", ""))
	call := records.Request{ID: "req-effect-original", IdemKey: "effect-original", BodyDigest: childDigest("3"), Package: "local/effect", Entrypoint: "perform", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}
	child, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	closeChild(t, store, child, "SUCCEEDED", "succeeded")
	closeChild(t, store, parent, "FAILED", "blocked")
	parent.ID, parent.IdemKey, parent.RetryOf = "req-effects-edited", "effects-edited", parent.ID
	parent, _, problem = store.Submit(parent)
	fatal(t, problem)
	parent = offerChildParent(t, store, parent)
	call.ID, call.IdemKey, call.ParentRequestID = "req-effect-new", "effect-new", parent.ID
	child, _, problem = store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if child.State != "submitted" || child.ReusedFrom != "" {
		t.Fatal("an undeclared reusable effect was cached across parent revisions")
	}
}

func TestUnpublishedParentPauseWaitsForChildExecutionBarrier(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "pause-parent", ""))
	child, _, problem := store.SubmitChild(records.Request{ID: "req-child-pause", IdemKey: "child-pause", BodyDigest: childDigest("3"), Package: "local/operation", Entrypoint: "run", Kind: "job", Payload: []byte(`{}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5")}, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	child = offerChildParent(t, store, child)
	_, problem = store.RequestPause(parent.ID, "test")
	fatal(t, problem)
	closeChild(t, store, parent, "CANCELED", "pausing")
	paused, problem := store.CompleteRequestPause(parent.ID)
	fatal(t, problem)
	if paused {
		t.Fatal("parent claimed paused while its child still executed")
	}
	closeChild(t, store, child, "CANCELED", "paused")
	paused, problem = store.CompleteRequestPause(parent.ID)
	fatal(t, problem)
	if !paused {
		t.Fatal("parent did not pause after every writer closed")
	}
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

func TestUnpublishedChildHistoryDoesNotActAsOperationCache(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.SpawnWorker(records.WorkerProcess{InstanceID: "private-worker", Package: "local/test", WorkerID: "worker", Devices: []string{"cpu"}}))
	parent := offerChildParent(t, store, recordPrivateTransaction(t, store, "parent-original", ""))
	call := records.Request{ID: "req-child-original-a", IdemKey: "child-original-a", BodyDigest: childDigest("3"), Package: "local/operation-a", Entrypoint: "compute", Kind: "job", Payload: []byte(`{"size":100}`), ParentRequestID: parent.ID, ParentCallIndex: 0, ChildIntentDigest: childDigest("4"), ChildTargetDigest: childDigest("5"), ChildReusable: true}
	child, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if !fresh || child.ParentRequestID != parent.ID || child.ReuseScope != parent.ReuseScope {
		t.Fatalf("child lost owner scope: %+v", child)
	}
	closeChild(t, store, child, "SUCCEEDED", "succeeded")
	closeChild(t, store, parent, "FAILED", "blocked")
	next := parent
	next.ID, next.IdemKey = "req-parent-edited", "parent-edited"
	next.RetryOf = parent.ID
	next.BodyDigest, next.LocalInstallationID = childDigest("6"), childDigest("7")
	next, _, problem = store.Submit(next)
	fatal(t, problem)
	next = offerChildParent(t, store, next)
	call.ID, call.IdemKey, call.ParentRequestID = "req-child-edited-a", "child-edited-a", next.ID
	reused, fresh, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if !fresh || reused.ID == child.ID || reused.State != "submitted" || reused.Ordinal != 0 || reused.ReusedFrom != "" {
		t.Fatalf("run history invented a cache hit without its workspace owner: %+v", reused)
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
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("same parent/index accepted changed implementation")
	}
	call.ParentCallIndex = 1
	call.ID, call.IdemKey = "req-child-edited-b", "child-edited-b"
	changed, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if changed.State != "submitted" || changed.ReusedFrom != "" {
		t.Fatal("changed operation inherited stale result")
	}
	call.ID, call.IdemKey, call.ParentCallIndex, call.ChildTargetDigest = "req-child-reordered", "child-reordered", 2, childDigest("5")
	reordered, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil)
	fatal(t, problem)
	if reordered.ReusedFrom != "" {
		t.Fatal("matching call inputs bypassed the workspace cache authority")
	}
	call.ID, call.IdemKey, call.ParentCallIndex = "req-child-forged", "child-forged", 3
	if _, _, problem := store.SubmitChild(call, 1, childDigest("9"), "private-boot", nil); problem == nil {
		t.Fatal("forged parent invocation admitted child")
	}
	_, problem = store.RequestPause(next.ID, "test")
	fatal(t, problem)
	if _, _, problem := store.SubmitChild(call, 1, childDigest("1"), "private-boot", nil); problem == nil {
		t.Fatal("paused parent admitted new child")
	}
}
