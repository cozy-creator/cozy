package workflow

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

type fixedResolver struct{ placement orchestrator.DesiredPlacement }

func (r fixedResolver) ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error) {
	if endpoint != r.placement.Endpoint {
		return orchestrator.DesiredPlacement{}, exit.New(exit.NotFound, "unknown endpoint")
	}
	return r.placement, nil
}

func succeedChild(t *testing.T, store *records.Store, requestID string, output records.Output) {
	t.Helper()
	if problem := store.AttachWorker(records.WorkerProcess{InstanceID: "worker-test",
		Endpoint: "org/ep", ReleaseID: "org/ep@1.0.0", WorkerID: "test"}); problem != nil {
		t.Fatal(problem)
	}
	if problem := store.BindSession("worker-test", "session-test", 1); problem != nil {
		t.Fatal(problem)
	}
	ordinal, problem := store.Dispatch(records.Attempt{RequestID: requestID,
		InstanceID: "worker-test", SessionID: "session-test",
		InvocationDigest: testDigest, InvocationCanonical: []byte(`{}`)})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.OfferDispatch(requestID, ordinal, "session-test"); problem != nil {
		t.Fatal(problem)
	}
	_, problem = store.AcceptTerminal(records.Terminal{RequestID: requestID, Attempt: ordinal,
		SessionID: "session-test", InvocationDigest: testDigest,
		TerminalID: "terminal-test", TerminalDigest: testDigest,
		Status: "SUCCEEDED", RequestState: "succeeded", EventType: "request.completed",
		Outputs: []records.Output{output}})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.Closed(requestID, ordinal); problem != nil {
		t.Fatal(problem)
	}
}

func (r fixedResolver) ResolveInstall(installID string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	if installID != r.placement.InstallID {
		return orchestrator.WorkerLaunchSpec{}, exit.New(exit.NotFound, "unknown install")
	}
	return orchestrator.WorkerLaunchSpec{Placement: r.placement}, nil
}

func (r fixedResolver) Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error) {
	if installID != r.placement.InstallID || name != "run" {
		return nil, exit.New(exit.NotFound, "unknown entrypoint")
	}
	asset := launch.Field{Name: "first_frame", Type: []byte(`{"asset":"video"}`), Wire: "optional"}
	asset.AssetBound.MaxBytes = 128 << 20
	return &launch.Entrypoint{Name: "run", Request: launch.Struct{Fields: []launch.Field{
		{Name: "prompt", Type: []byte(`"str"`), Wire: "optional"}, asset,
	}}}, nil
}

func testEngine(t *testing.T) (*Engine, *records.Store) {
	t.Helper()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	t.Cleanup(store.Close)
	_, problem = store.Activate(records.EndpointInstall{
		ID: "install-1", Endpoint: "org/ep", Major: 1, Version: "1.0.0",
		SourceKind: "dir", SourceRef: ".", SourceDigest: testDigest,
		Dir: ".", Python: "python", UV: "uv", LockDigest: testDigest,
		Platform: "test", LinkMode: "copy", Closure: "none", Descriptor: testDigest,
	})
	if problem != nil {
		t.Fatal(problem)
	}
	binding := &orchestrator.Binding{Entrypoint: "run", Outputs: []string{"video"},
		RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: testDigest, Kind: "plan",
			Digest: testDigest, Length: 1}}
	resolver := fixedResolver{placement: orchestrator.DesiredPlacement{
		Endpoint: "org/ep", ReleaseID: "org/ep@1.0.0", InstallID: "install-1",
		DescriptorDigest: testDigest, Bindings: []*orchestrator.Binding{binding},
	}}
	owner, problem := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	if problem != nil {
		t.Fatal(problem)
	}
	engine, problem := Open(Options{Store: store, Owner: owner, Resolver: resolver, Layout: layout})
	if problem != nil {
		t.Fatal(problem)
	}
	return engine, store
}

func TestEngineMintsOneOrdinaryChildPerOrdinal(t *testing.T) {
	engine, store := testEngine(t)
	plan := testPlan(2)
	row, fresh, problem := engine.Submit(Submission{
		IdempotencyKey: "workflow-one", Plan: encodePlan(t, plan),
	})
	if problem != nil || !fresh {
		t.Fatalf("submit: fresh=%v problem=%v", fresh, problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, problem := store.WorkflowSteps(row.ID)
	if problem != nil || len(steps) != 2 || steps[0].ChildRequestID == "" ||
		steps[1].ChildRequestID != "" {
		t.Fatalf("first child only: %#v %v", steps, problem)
	}
	firstID := steps[0].ChildRequestID
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ = store.WorkflowSteps(row.ID)
	if steps[0].ChildRequestID != firstID {
		t.Fatal("reconcile changed the first child")
	}
	if problem := store.SettleRequest(firstID, "succeeded"); problem != nil {
		t.Fatal(problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ = store.WorkflowSteps(row.ID)
	if steps[1].ChildRequestID == "" || steps[1].ChildRequestID == firstID {
		t.Fatalf("second child: %#v", steps)
	}
	requests, problem := store.Requests("", 100)
	if problem != nil || len(requests) != 2 {
		t.Fatalf("ordinary request count=%d problem=%v", len(requests), problem)
	}
}

func prepareFirstStep(t *testing.T, store *records.Store,
	row records.WorkflowExecution, plan Plan) records.WorkflowStep {
	t.Helper()
	steps, problem := store.WorkflowSteps(row.ID)
	if problem != nil {
		t.Fatal(problem)
	}
	materialized, data, digest, resolved, problem := MaterializeStep(&plan, 1, nil, nil)
	if problem != nil {
		t.Fatal(problem)
	}
	key, problem := ChildKey(row.ExecutionDigest, 1, materialized, digest)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := store.PrepareWorkflowStep(row.ID, 1, steps[0].InstallID,
		data, digest, resolved, key); problem != nil {
		t.Fatal(problem)
	}
	steps, problem = store.WorkflowSteps(row.ID)
	if problem != nil {
		t.Fatal(problem)
	}
	return steps[0]
}

func TestReconcileAdoptsPreparedStepAfterRestart(t *testing.T) {
	engine, store := testEngine(t)
	plan := testPlan(1)
	row, _, problem := engine.Submit(Submission{
		IdempotencyKey: "prepared-crash", Plan: encodePlan(t, plan),
	})
	if problem != nil {
		t.Fatal(problem)
	}
	prepared := prepareFirstStep(t, store, row, plan)
	if prepared.MaterializedDigest == "" || prepared.ChildRequestID != "" {
		t.Fatalf("prepared step = %#v", prepared)
	}
	restarted, problem := Open(engine.opt)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := restarted.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ := store.WorkflowSteps(row.ID)
	requests, _ := store.Requests("", 100)
	if steps[0].ChildRequestID == "" || steps[0].MaterializedDigest != prepared.MaterializedDigest ||
		len(requests) != 1 {
		t.Fatalf("recovered step=%#v requests=%#v", steps[0], requests)
	}
}

func TestReconcileAdoptsRecordedUnlinkedChildAfterRestart(t *testing.T) {
	engine, store := testEngine(t)
	plan := testPlan(1)
	row, _, problem := engine.Submit(Submission{
		IdempotencyKey: "recorded-crash", Plan: encodePlan(t, plan),
	})
	if problem != nil {
		t.Fatal(problem)
	}
	prepared := prepareFirstStep(t, store, row, plan)
	materialized, problem := DecodeMaterialized(prepared.MaterializedSubmission)
	if problem != nil {
		t.Fatal(problem)
	}
	submission, problem := materialized.Submission(prepared.ChildKey,
		prepared.MaterializedDigest, prepared.InstallID, "", map[string]string{}, map[string]int64{})
	if problem != nil {
		t.Fatal(problem)
	}
	child, fresh, problem := engine.opt.Owner.RecordSubmission(submission)
	if problem != nil || !fresh {
		t.Fatalf("record child: fresh=%v problem=%v", fresh, problem)
	}
	steps, _ := store.WorkflowSteps(row.ID)
	if steps[0].ChildRequestID != "" {
		t.Fatal("child was linked before the simulated crash window")
	}
	restarted, problem := Open(engine.opt)
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := restarted.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ = store.WorkflowSteps(row.ID)
	requests, _ := store.Requests("", 100)
	if steps[0].ChildRequestID != child.ID || len(requests) != 1 {
		t.Fatalf("adopted step=%#v child=%s requests=%d", steps[0], child.ID, len(requests))
	}
}

func TestEngineQueuedCancellationPreventsLaterChild(t *testing.T) {
	engine, store := testEngine(t)
	row, _, problem := engine.Submit(Submission{
		IdempotencyKey: "workflow-cancel", Plan: encodePlan(t, testPlan(2)),
	})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	if _, changed, problem := engine.Cancel(row.ID); problem != nil || !changed {
		t.Fatalf("cancel changed=%v problem=%v", changed, problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	snapshot, problem := engine.State(row.ID)
	if problem != nil || snapshot.Execution.State != "canceled" {
		t.Fatalf("state=%#v problem=%v", snapshot, problem)
	}
	steps, _ := store.WorkflowSteps(row.ID)
	if steps[1].ChildRequestID != "" {
		t.Fatalf("later child minted after cancellation: %#v", steps[1])
	}
	requests, _ := store.Requests("", 100)
	if len(requests) != 1 || requests[0].State != "canceled" {
		t.Fatalf("child requests: %#v", requests)
	}
}

func TestEngineWorkflowIdempotency(t *testing.T) {
	engine, _ := testEngine(t)
	plan := encodePlan(t, testPlan(1))
	first, fresh, problem := engine.Submit(Submission{IdempotencyKey: "same", Plan: plan})
	if problem != nil || !fresh {
		t.Fatal(problem)
	}
	second, fresh, problem := engine.Submit(Submission{IdempotencyKey: "same", Plan: plan})
	if problem != nil || fresh || second.ID != first.ID {
		t.Fatalf("replay=%#v fresh=%v problem=%v", second, fresh, problem)
	}
	changed := testPlan(1)
	changed.Steps[0].PayloadBase64 = "eyJwcm9tcHQiOiJjaGFuZ2VkIn0="
	if _, _, problem := engine.Submit(Submission{
		IdempotencyKey: "same", Plan: encodePlan(t, changed),
	}); problem == nil || problem.Code != exit.Conflict {
		t.Fatalf("changed body: %#v", problem)
	}
}

func TestEqualPlanUnderDifferentWorkflowKeysMintsDistinctChildren(t *testing.T) {
	engine, store := testEngine(t)
	plan := encodePlan(t, testPlan(1))
	first, _, problem := engine.Submit(Submission{IdempotencyKey: "run-one", Plan: plan})
	if problem != nil {
		t.Fatal(problem)
	}
	second, _, problem := engine.Submit(Submission{IdempotencyKey: "run-two", Plan: plan})
	if problem != nil {
		t.Fatal(problem)
	}
	if first.ExecutionDigest == second.ExecutionDigest {
		t.Fatal("distinct workflow executions share an execution digest")
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	firstSteps, _ := store.WorkflowSteps(first.ID)
	secondSteps, _ := store.WorkflowSteps(second.ID)
	if firstSteps[0].ChildRequestID == "" || secondSteps[0].ChildRequestID == "" ||
		firstSteps[0].ChildRequestID == secondSteps[0].ChildRequestID ||
		firstSteps[0].ChildKey == secondSteps[0].ChildKey {
		t.Fatalf("children coalesced: %#v %#v", firstSteps[0], secondSteps[0])
	}
}

func TestRemoteStepPersistsAndSubmitsExactRentalTarget(t *testing.T) {
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	binding := &orchestrator.Binding{Entrypoint: "run", Outputs: []string{"video"},
		RuntimePlan: &orchestrator.BindingPlanSubject{SubjectID: testDigest, Kind: "plan",
			Digest: testDigest, Length: 1}}
	remote := orchestrator.DesiredPlacement{Endpoint: "org/ep", ReleaseID: "org/ep@1.0.0",
		InstallID: "acquisition-attempt", Bindings: []*orchestrator.Binding{binding}}
	resolver := fixedResolver{placement: remote}
	owner, _ := orchestrator.Open(orchestrator.Options{Layout: layout, Store: store})
	engine, problem := Open(Options{Store: store, Owner: owner, Resolver: resolver, Layout: layout,
		Rentals: func(id string) (*orchestrator.DesiredPlacement, *exit.Error) {
			if id != "rnt-exact" {
				t.Fatalf("rental=%s", id)
			}
			copy := remote
			return &copy, nil
		},
		RemoteEntrypoint: func(worker, name string) (*launch.Entrypoint, *exit.Error) {
			return resolver.Entrypoint(remote.InstallID, name)
		},
	})
	if problem != nil {
		t.Fatal(problem)
	}
	row, _, problem := engine.Submit(Submission{IdempotencyKey: "remote-workflow",
		Plan: encodePlan(t, testPlan(1)), Workers: map[int]string{1: "rnt-exact"}})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ := store.WorkflowSteps(row.ID)
	child, problem := store.RequestRow(steps[0].ChildRequestID)
	if problem != nil || child == nil || child.Worker != "rnt-exact" || child.InstallID != "" ||
		steps[0].Worker != "rnt-exact" || steps[0].InstallID != "" {
		t.Fatalf("step=%#v child=%#v problem=%v", steps[0], child, problem)
	}
}

func TestLargeWinningOutputMaterializesIntoNextChild(t *testing.T) {
	engine, store := testEngine(t)
	plan := testPlan(2)
	plan.Steps[1].PayloadBase64 = "eyJmaXJzdF9mcmFtZSI6IiJ9"
	plan.Steps[1].Bindings = []OutputBinding{{FieldPath: "first_frame", PriorStep: 1,
		OutputName: "video", ExpectedMediaKind: "video"}}
	row, _, problem := engine.Submit(Submission{IdempotencyKey: "large-output",
		Plan: encodePlan(t, plan)})
	if problem != nil {
		t.Fatal(problem)
	}
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ := store.WorkflowSteps(row.ID)
	path := filepath.Join(t.TempDir(), "large.mp4")
	const length = int64(65 << 20)
	if err := os.WriteFile(path, nil, 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Truncate(path, length); err != nil {
		t.Fatal(err)
	}
	_, digest, _, fingerprintProblem := inputasset.Fingerprint(path, 128<<20)
	if fingerprintProblem != nil {
		t.Fatal(fingerprintProblem)
	}
	succeedChild(t, store, steps[0].ChildRequestID, records.Output{OutputID: "video",
		Path: path, Digest: digest, Length: length, MimeType: "video/mp4"})
	if problem := engine.Reconcile(); problem != nil {
		t.Fatal(problem)
	}
	steps, _ = store.WorkflowSteps(row.ID)
	child, problem := store.RequestRow(steps[1].ChildRequestID)
	if problem != nil || child == nil || len(child.Assets) != 1 ||
		child.Assets[0].Length != length || child.Assets[0].Digest != digest ||
		child.Assets[0].MaxBytes != 128<<20 {
		t.Fatalf("large child=%#v problem=%v", child, problem)
	}
	if problem := inputasset.Verify(child.Assets[0], child.Assets[0].MaxBytes); problem != nil {
		t.Fatal(problem)
	}
}

func TestWorkflowAssetKindMustMatchPinnedRequestField(t *testing.T) {
	engine, store := testEngine(t)
	plan := testPlan(1)
	plan.Steps[0].PayloadBase64 = "eyJmaXJzdF9mcmFtZSI6IiJ9"
	plan.Steps[0].Assets = []AssetClaim{{FieldPath: "first_frame", Digest: testDigest,
		Length: 1, MediaType: "image/png", Kind: "image"}}
	if _, _, problem := engine.Submit(Submission{IdempotencyKey: "wrong-kind",
		Plan: encodePlan(t, plan), Assets: map[int][]records.AssetBinding{1: {{
			FieldPath: "first_frame", Digest: testDigest, Length: 1, MediaType: "image/png",
		}}}}); problem == nil || problem.ErrName() != "workflow_asset_field" {
		t.Fatalf("kind mismatch: %#v", problem)
	}
	if row, problem := store.WorkflowByIdempotencyKey("wrong-kind"); problem != nil || row != nil {
		t.Fatalf("invalid workflow recorded: %#v %v", row, problem)
	}
}
