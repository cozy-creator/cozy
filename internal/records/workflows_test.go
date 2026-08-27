package records

import (
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
)

const workflowDigest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

func testStore(t *testing.T) *Store {
	t.Helper()
	store, problem := Open(filepath.Join(t.TempDir(), "records.db"))
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(store.Close)
	return store
}

func TestWorkflowRecordsTransitionsAndOwnership(t *testing.T) {
	store := testStore(t)
	_, problem := store.Activate(EndpointInstall{
		ID: "install-workflow", Endpoint: "org/ep", Major: 1, Version: "1.0.0",
		SourceKind: "dir", SourceRef: ".", SourceDigest: workflowDigest,
		Dir: ".", Python: "python", UV: "uv", LockDigest: workflowDigest,
		Platform: "test", LinkMode: "copy", Closure: "none", Descriptor: workflowDigest,
	})
	if problem != nil {
		t.Fatal(problem)
	}
	asset := AssetBinding{FieldPath: "image", LocalPath: "/private/staged", Digest: workflowDigest,
		Length: 4, MediaType: "image/png"}
	created, fresh, problem := store.CreateWorkflow(WorkflowExecution{
		ID: "wfl-one", IdemKey: "workflow-key", BodyDigest: workflowDigest,
		ExecutionDigest: workflowDigest, CreativePlanDigest: workflowDigest, Plan: []byte(`{"x":1}`),
	}, map[int][]AssetBinding{1: {asset}}, map[int]string{1: "install-workflow"}, nil, 1)
	if problem != nil || !fresh || created.StepCount != 1 {
		t.Fatalf("create=%#v fresh=%v problem=%v", created, fresh, problem)
	}
	replayed, fresh, problem := store.CreateWorkflow(WorkflowExecution{
		ID: "wfl-other", IdemKey: "workflow-key", BodyDigest: workflowDigest,
	}, nil, nil, nil, 1)
	if problem != nil || fresh || replayed.ID != created.ID {
		t.Fatalf("replay=%#v fresh=%v problem=%v", replayed, fresh, problem)
	}
	if _, _, problem := store.CreateWorkflow(WorkflowExecution{
		ID: "wfl-changed", IdemKey: "workflow-key", BodyDigest: "sha256:" + string(make([]byte, 64)),
	}, nil, nil, nil, 1); problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("changed body: %#v", problem)
	}
	used, problem := store.AssetInUse(workflowDigest)
	if problem != nil || !used {
		t.Fatalf("asset ownership=%v problem=%v", used, problem)
	}
	resolved := []ResolvedBinding{{FieldPath: "image", PriorStep: 1, Digest: workflowDigest}}
	if problem := store.PrepareWorkflowStep(created.ID, 1, "install-workflow",
		[]byte(`{"materialized":true}`), workflowDigest, resolved, workflowDigest); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.PrepareWorkflowStep(created.ID, 1, "install-workflow",
		[]byte(`{"materialized":true}`), workflowDigest, resolved, workflowDigest); problem != nil {
		t.Fatalf("prepare replay: %v", problem)
	}
	child, fresh, problem := store.SubmitWorkflowChild(created.ID, 1, workflowDigest,
		Request{ID: "req-child", IdemKey: workflowDigest,
			BodyDigest: workflowDigest, Endpoint: "org/ep", Entrypoint: "run",
			PlanID: workflowDigest, Payload: []byte(`{}`)}, map[string]any{"endpoint": "org/ep"})
	if problem != nil || !fresh {
		t.Fatalf("child=%#v fresh=%v problem=%v", child, fresh, problem)
	}
	replayedChild, fresh, problem := store.SubmitWorkflowChild(created.ID, 1, workflowDigest,
		Request{ID: "req-other", IdemKey: workflowDigest,
			BodyDigest: workflowDigest, Endpoint: "org/ep", Entrypoint: "run",
			PlanID: workflowDigest, Payload: []byte(`{}`)}, map[string]any{})
	if problem != nil || fresh || replayedChild.ID != child.ID {
		t.Fatalf("child replay=%#v fresh=%v problem=%v", replayedChild, fresh, problem)
	}
	events, problem := store.EventsAfter(child.ID, 0, 10)
	if problem != nil || len(events) != 1 || events[0].Type != "request.submitted" {
		t.Fatalf("child events=%#v problem=%v", events, problem)
	}
	if problem := store.SettleWorkflow(created.ID, "succeeded", "", ""); problem != nil {
		t.Fatal(problem)
	}
	used, problem = store.AssetInUse(workflowDigest)
	if problem != nil || used {
		t.Fatalf("settled asset ownership=%v problem=%v", used, problem)
	}
}

func TestQueuedCancellationCommitsTerminalEvent(t *testing.T) {
	store := testStore(t)
	request, _, problem := store.Submit(Request{ID: "req-queued", IdemKey: "queued",
		BodyDigest: workflowDigest, Endpoint: "org/ep", Entrypoint: "run",
		PlanID: workflowDigest, Payload: []byte(`{}`)})
	if problem != nil {
		t.Fatal(problem)
	}
	applied, problem := store.CancelQueuedRequest(request.ID, map[string]any{"status": "CANCELED"})
	if problem != nil || !applied {
		t.Fatalf("cancel applied=%v problem=%v", applied, problem)
	}
	row, problem := store.RequestRow(request.ID)
	if problem != nil || row.State != "canceled" {
		t.Fatalf("request=%#v problem=%v", row, problem)
	}
	events, problem := store.EventsAfter(request.ID, 0, 10)
	if problem != nil || len(events) != 1 || events[0].Type != "request.canceled" {
		t.Fatalf("events=%#v problem=%v", events, problem)
	}
	applied, problem = store.CancelQueuedRequest(request.ID, map[string]any{})
	if problem != nil || applied {
		t.Fatalf("cancel replay applied=%v problem=%v", applied, problem)
	}
}

func TestWorkflowCancellationFencesDispatchAndRequeueTransactions(t *testing.T) {
	store := testStore(t)
	workflow, _, problem := store.CreateWorkflow(WorkflowExecution{ID: "wfl-fenced",
		IdemKey: "wfl-fenced", BodyDigest: workflowDigest, ExecutionDigest: workflowDigest,
		CreativePlanDigest: workflowDigest, Plan: []byte(`{}`)}, nil, nil, nil, 1)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.PrepareWorkflowStep(workflow.ID, 1, "",
		[]byte(`{}`), workflowDigest, nil, workflowDigest); problem != nil {
		t.Fatal(problem)
	}
	request, _, problem := store.SubmitWorkflowChild(workflow.ID, 1, workflowDigest,
		Request{ID: "req-fenced", IdemKey: workflowDigest,
			BodyDigest: workflowDigest, Endpoint: "org/ep", Entrypoint: "run",
			PlanID: workflowDigest, Payload: []byte(`{}`)}, map[string]any{})
	if problem != nil {
		t.Fatal(problem)
	}
	if _, _, problem := store.RequestWorkflowCancel(workflow.ID); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.AttachWorker(WorkerProcess{InstanceID: "worker-fenced",
		Endpoint: "org/ep", ReleaseID: "org/ep@1", WorkerID: "test"}); problem != nil {
		t.Fatal(problem)
	}
	if _, problem := store.Dispatch(Attempt{RequestID: request.ID, InstanceID: "worker-fenced",
		SessionID: "session", InvocationDigest: workflowDigest, InvocationCanonical: []byte(`{}`)}); problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("dispatch after workflow cancel: %#v", problem)
	}
	attempts, _ := store.Attempts(request.ID)
	if len(attempts) != 0 {
		t.Fatalf("attempt minted after workflow cancel: %#v", attempts)
	}

	secondWorkflow, _, problem := store.CreateWorkflow(WorkflowExecution{ID: "wfl-requeue",
		IdemKey: "wfl-requeue", BodyDigest: workflowDigest, ExecutionDigest: workflowDigest,
		CreativePlanDigest: workflowDigest, Plan: []byte(`{}`)}, nil, nil, nil, 1)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.PrepareWorkflowStep(secondWorkflow.ID, 1, "",
		[]byte(`{}`), workflowDigest, nil, "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"); problem != nil {
		t.Fatal(problem)
	}
	secondKey := "sha256:eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee"
	second, _, problem := store.SubmitWorkflowChild(secondWorkflow.ID, 1, secondKey,
		Request{ID: "req-requeue", IdemKey: secondKey,
			BodyDigest: workflowDigest, Endpoint: "org/ep", Entrypoint: "run",
			PlanID: workflowDigest, Payload: []byte(`{}`)}, map[string]any{})
	if problem != nil {
		t.Fatal(problem)
	}
	if _, err := store.db.Exec(`INSERT INTO attempts(request_id,attempt,attempt_key,instance_id,
		session_id,invocation_digest,invocation,state,dispatched_at,closed_at)
		VALUES(?,1,'closed-history','worker-fenced','old-session',?,x'7b7d','closed','now','now')`,
		second.ID, workflowDigest); err != nil {
		t.Fatal(err)
	}
	if _, err := store.db.Exec(`UPDATE requests SET state='requeue_pending',ordinal=1 WHERE id=?`, second.ID); err != nil {
		t.Fatal(err)
	}
	if _, _, problem := store.RequestWorkflowCancel(secondWorkflow.ID); problem != nil {
		t.Fatal(problem)
	}
	_, started, canceled, problem := store.BeginRequeue(second.ID, 3)
	if problem != nil || started || !canceled {
		t.Fatalf("requeue cancel: started=%v canceled=%v problem=%v", started, canceled, problem)
	}
	row, _ := store.RequestRow(second.ID)
	if row.State != "canceled" {
		t.Fatalf("requeue child state=%s", row.State)
	}
}
