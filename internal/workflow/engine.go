package workflow

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/home"
	"github.com/cozy-creator/cozy-creator-v2/internal/inputasset"
	"github.com/cozy-creator/cozy-creator-v2/internal/launch"
	"github.com/cozy-creator/cozy-creator-v2/internal/orchestrator"
	"github.com/cozy-creator/cozy-creator-v2/internal/records"
)

type Resolver interface {
	ResolvePlacement(endpoint string) (orchestrator.DesiredPlacement, *exit.Error)
	ResolveInstall(installID string) (orchestrator.WorkerLaunchSpec, *exit.Error)
	Entrypoint(installID, name string) (*launch.Entrypoint, *exit.Error)
}

type Options struct {
	Store            *records.Store
	Owner            *orchestrator.Orchestrator
	Resolver         Resolver
	Rentals          func(id string) (*orchestrator.DesiredPlacement, *exit.Error)
	RemoteEntrypoint func(worker, name string) (*launch.Entrypoint, *exit.Error)
	Layout           home.Layout
	Log              io.Writer
}

// Engine serializes every workflow transition. Ordinary request attempts remain owned by
// Orchestrator; this loop only decides whether one next ordinary child may be minted.
type Engine struct {
	opt       Options
	mu        sync.Mutex
	submitMu  sync.Mutex
	done      chan struct{}
	wake      chan struct{}
	closeOnce sync.Once
	closing   atomic.Bool
	wg        sync.WaitGroup
}

type Submission struct {
	IdempotencyKey string
	Plan           []byte
	Assets         map[int][]records.AssetBinding
	Workers        map[int]string
}

type Snapshot struct {
	Execution records.WorkflowExecution
	Steps     []StepSnapshot
}

type StepSnapshot struct {
	Ordinal            int
	State              string
	Child              *records.Request
	Outputs            []records.Output
	Materialized       bool
	MaterializedDigest string
	MaterializedAssets []MaterializedAsset
	ResolvedBindings   []records.ResolvedBinding
	ChildKey           string
	RentalID           string
}

type Receipt struct {
	Snapshot     Snapshot
	Plan         []byte
	CreativePlan []byte
	Steps        []ReceiptStep
	Rentals      []records.WorkflowRentalControl
}

type ReceiptStep struct {
	Ordinal                int
	MaterializedSubmission []byte
}

func Open(opt Options) (*Engine, *exit.Error) {
	if opt.Store == nil || opt.Owner == nil || opt.Resolver == nil {
		return nil, exit.Internalf("workflow engine needs the records, request owner, and resolver")
	}
	if opt.Log == nil {
		opt.Log = io.Discard
	}
	return &Engine{opt: opt, done: make(chan struct{}), wake: make(chan struct{}, 1)}, nil
}

func (e *Engine) Start() {
	e.wg.Add(1)
	go func() {
		defer e.wg.Done()
		for {
			select {
			case <-e.done:
				return
			case <-e.wake:
			}
			if err := e.Reconcile(); err != nil {
				fmt.Fprintf(e.opt.Log, "[workflow] reconcile: %s\n", err.Message)
			}
		}
	}()
}

func (e *Engine) Close() {
	e.closeOnce.Do(func() {
		e.closing.Store(true)
		e.submitMu.Lock()
		e.submitMu.Unlock()
		e.mu.Lock()
		e.mu.Unlock()
		close(e.done)
	})
	e.wg.Wait()
}

func (e *Engine) Wake() {
	select {
	case e.wake <- struct{}{}:
	default:
	}
}

func (e *Engine) Submit(sub Submission) (records.WorkflowExecution, bool, *exit.Error) {
	e.submitMu.Lock()
	defer e.submitMu.Unlock()
	if e.closing.Load() {
		return records.WorkflowExecution{}, false,
			exit.Unavailablef("the workflow engine is shutting down and admits no new workflow")
	}
	if strings.TrimSpace(sub.IdempotencyKey) == "" {
		return records.WorkflowExecution{}, false,
			exit.New(exit.Validation, "a workflow submission needs an idempotency key")
	}
	plan, canonicalPlan, digest, problem := DecodePlan(sub.Plan)
	if problem != nil {
		return records.WorkflowExecution{}, false, problem
	}
	existing, problem := e.opt.Store.WorkflowByIdempotencyKey(sub.IdempotencyKey)
	if problem != nil {
		return records.WorkflowExecution{}, false, problem
	}
	if existing != nil {
		if existing.BodyDigest != digest {
			return records.WorkflowExecution{}, false, exit.New(exit.Conflict,
				"idempotency key %s already names a workflow with a different body",
				sub.IdempotencyKey)
		}
		got, readProblem := e.opt.Store.WorkflowTargets(existing.ID)
		if readProblem != nil {
			return records.WorkflowExecution{}, false, readProblem
		}
		want := normalizedTargets(sub.Workers, len(plan.Steps))
		if want == nil || !sameTargets(got, want) {
			return records.WorkflowExecution{}, false, exit.New(exit.Conflict,
				"idempotency key %s already names different rental targets", sub.IdempotencyKey)
		}
		return *existing, false, nil
	}
	installs, workers := map[int]string{}, map[int]string{}
	controls := map[string]records.WorkflowRentalControl{}
	for index, step := range plan.Steps {
		ordinal := index + 1
		worker := strings.TrimSpace(sub.Workers[ordinal])
		var placement orchestrator.DesiredPlacement
		if worker != "" {
			if e.opt.Rentals == nil {
				return records.WorkflowExecution{}, false,
					exit.Unavailablef("this LocalService resolves no rented workflow targets")
			}
			remote, problem := e.opt.Rentals(worker)
			if problem != nil {
				return records.WorkflowExecution{}, false, problem
			}
			if remote == nil {
				return records.WorkflowExecution{}, false,
					exit.New(exit.NotFound, "no attached rental %s", worker)
			}
			if controls[worker].RentalID == "" {
				row, readProblem := e.opt.Store.RentalRow(worker)
				if readProblem != nil {
					return records.WorkflowExecution{}, false, readProblem
				}
				if row == nil || row.State != "ready" || row.ObservedAccelerator == "" ||
					row.ObservedAccelerator != row.AcceleratorModel ||
					row.ObservedAcceleratorCount != 1 || row.ObservedBackend == "" ||
					row.ObservedWorkerInstance == "" || row.ObservedWorkerBootID == "" {
					return records.WorkflowExecution{}, false, exit.Named(exit.Conflict,
						"workflow.rental_unprobed",
						"rental %s has no complete actual-hardware ClaimAck readback", worker).
						WithRemedy("run `cozy rent probe %s` before submitting the workflow", worker)
				}
				controls[worker] = records.WorkflowRentalControl{
					RentalID: worker, EndpointRef: row.EndpointRef,
					AcceleratorModel:         row.AcceleratorModel,
					ObservedAccelerator:      row.ObservedAccelerator,
					ObservedAcceleratorCount: row.ObservedAcceleratorCount,
					ObservedBackend:          row.ObservedBackend,
					ObservedWorkerInstance:   row.ObservedWorkerInstance,
					ObservedWorkerBootID:     row.ObservedWorkerBootID, ObservedAt: row.ObservedAt,
					ControlSnapshotDigest: row.ControlSnapshotDigest,
					ControlSnapshotLength: row.ControlSnapshotLength,
					ControlSnapshotBytes:  append([]byte(nil), row.ControlSnapshotBytes...),
				}
			}
			placement, workers[ordinal] = *remote, worker
		} else {
			var problem *exit.Error
			placement, problem = e.opt.Resolver.ResolvePlacement(step.Endpoint)
			if problem != nil {
				return records.WorkflowExecution{}, false, problem
			}
		}
		if problem := validatePlacement(step, placement); problem != nil {
			return records.WorkflowExecution{}, false, problem
		}
		if worker == "" && placement.InstallID == "" {
			return records.WorkflowExecution{}, false, exit.Named(exit.Structural,
				"workflow_install_unpinned", "workflow step %d resolves no immutable local install", index+1)
		}
		if worker == "" {
			installs[ordinal] = placement.InstallID
		}
		var entrypoint *launch.Entrypoint
		if worker != "" {
			if e.opt.RemoteEntrypoint == nil {
				return records.WorkflowExecution{}, false,
					exit.Unavailablef("this LocalService resolves no rented endpoint descriptors")
			}
			entrypoint, problem = e.opt.RemoteEntrypoint(worker, step.Entrypoint)
		} else {
			entrypoint, problem = e.opt.Resolver.Entrypoint(placement.InstallID, step.Entrypoint)
		}
		if problem != nil {
			return records.WorkflowExecution{}, false, problem
		}
		payload, problem := validationPayload(step)
		if problem != nil {
			return records.WorkflowExecution{}, false, problem
		}
		if problem := launch.ValidatePayload(entrypoint, payload); problem != nil {
			return records.WorkflowExecution{}, false, problem
		}
		populatedAssets, problem := launch.PopulatedAssetPaths(entrypoint, payload)
		if problem != nil {
			return records.WorkflowExecution{}, false, problem
		}
		if expected := stepAssetFields(step); !sameStrings(populatedAssets, expected) {
			return records.WorkflowExecution{}, false, exit.Named(exit.Validation,
				"workflow_asset_resolution",
				"workflow step %d payload asset references do not exactly match its declared grants",
				ordinal)
		}
		for _, asset := range step.Assets {
			assetSpec, ok := launch.AssetSpec(entrypoint, asset.FieldPath)
			if !ok || assetSpec.Kind != asset.Kind {
				return records.WorkflowExecution{}, false, exit.Named(exit.Validation,
					"workflow_asset_field", "workflow step %d target %s is not a %s asset field in %s",
					ordinal, asset.FieldPath, asset.Kind, step.Entrypoint)
			}
			if assetSpec.MaxBytes <= 0 || asset.Length > assetSpec.MaxBytes {
				return records.WorkflowExecution{}, false, exit.Named(exit.Validation,
					"workflow_asset_over_bound",
					"workflow step %d asset %s is %d B; its pinned field admits %d B",
					ordinal, asset.FieldPath, asset.Length, assetSpec.MaxBytes)
			}
			if !assetSpec.AcceptsMediaType(asset.MediaType) {
				return records.WorkflowExecution{}, false, exit.Named(exit.Validation,
					"workflow_asset_media_type",
					"workflow step %d asset %s is %s, which its pinned field does not admit",
					ordinal, asset.FieldPath, asset.MediaType)
			}
		}
		for _, binding := range step.Bindings {
			assetSpec, ok := launch.AssetSpec(entrypoint, binding.FieldPath)
			if !ok || assetSpec.Kind != binding.ExpectedMediaKind {
				return records.WorkflowExecution{}, false, exit.Named(exit.Validation,
					"workflow_asset_field", "workflow step %d target %s is not a %s asset field in %s",
					ordinal, binding.FieldPath, binding.ExpectedMediaKind, step.Entrypoint)
			}
		}
	}
	for ordinal := range sub.Workers {
		if ordinal < 1 || ordinal > len(plan.Steps) {
			return records.WorkflowExecution{}, false,
				exit.New(exit.Validation, "workflow target names nonexistent step %d", ordinal)
		}
	}

	unlock := inputasset.Guard()
	defer unlock()
	if e.closing.Load() {
		return records.WorkflowExecution{}, false,
			exit.Unavailablef("the workflow engine is shutting down and admits no new workflow")
	}
	if problem := verifyAuthoredAssets(plan, sub.Assets); problem != nil {
		return records.WorkflowExecution{}, false, problem
	}
	id := records.NewID("wfl")
	executionDigest, problem := workflowExecutionDigest(id, digest)
	if problem != nil {
		return records.WorkflowExecution{}, false, problem
	}
	row, fresh, problem := e.opt.Store.CreateWorkflow(records.WorkflowExecution{
		ID: id, IdemKey: sub.IdempotencyKey,
		BodyDigest: digest, ExecutionDigest: executionDigest,
		CreativePlanDigest: plan.CreativePlanDigest, Plan: canonicalPlan,
	}, sub.Assets, installs, workers, controls, len(plan.Steps))
	if problem != nil {
		return records.WorkflowExecution{}, false, problem
	}
	if fresh {
		e.Wake()
	}
	return row, fresh, nil
}

func normalizedTargets(workers map[int]string, steps int) map[int]string {
	out := map[int]string{}
	for ordinal, worker := range workers {
		worker = strings.TrimSpace(worker)
		if ordinal < 1 || ordinal > steps || worker == "" {
			return nil
		}
		out[ordinal] = worker
	}
	return out
}

func sameTargets(left, right map[int]string) bool {
	if len(left) != len(right) {
		return false
	}
	for ordinal, worker := range left {
		if right[ordinal] != worker {
			return false
		}
	}
	return true
}

func validationPayload(step Step) ([]byte, *exit.Error) {
	payload, err := base64.StdEncoding.Strict().DecodeString(step.PayloadBase64)
	if err != nil {
		return nil, exit.Internalf("validated workflow payload no longer decodes: %s", err)
	}
	var document map[string]any
	decoder := json.NewDecoder(bytes.NewReader(payload))
	decoder.UseNumber()
	if err := decoder.Decode(&document); err != nil {
		return nil, exit.Internalf("validated workflow payload no longer parses: %s", err)
	}
	for _, asset := range step.Assets {
		if problem := setPayloadRef(document, asset.FieldPath, asset.Digest); problem != nil {
			return nil, problem
		}
	}
	for _, binding := range step.Bindings {
		if problem := setPayloadRef(document, binding.FieldPath,
			"sha256:0000000000000000000000000000000000000000000000000000000000000000"); problem != nil {
			return nil, problem
		}
	}
	rendered, err := json.Marshal(document)
	if err != nil {
		return nil, exit.Internalf("cannot render workflow validation payload: %s", err)
	}
	return rendered, nil
}

func stepAssetFields(step Step) []string {
	fields := make([]string, 0, len(step.Assets)+len(step.Bindings))
	for _, asset := range step.Assets {
		fields = append(fields, asset.FieldPath)
	}
	for _, binding := range step.Bindings {
		fields = append(fields, binding.FieldPath)
	}
	return fields
}

func workflowExecutionDigest(id, planDigest string) (string, *exit.Error) {
	data, err := canonical.Write(map[string]canonical.Value{
		"format": "cozy.workflow.ExecutionIdentity/1", "workflow_id": id,
		"plan_digest": planDigest,
	})
	if err != nil {
		return "", exit.Internalf("cannot canonicalize workflow execution identity: %s", err)
	}
	digest, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot spell workflow execution identity: %s", err)
	}
	return digest, nil
}

func verifyAuthoredAssets(plan *Plan, supplied map[int][]records.AssetBinding) *exit.Error {
	for index, step := range plan.Steps {
		ordinal := index + 1
		byField := map[string]records.AssetBinding{}
		for _, asset := range supplied[ordinal] {
			if byField[asset.FieldPath].FieldPath != "" {
				return exit.New(exit.Validation,
					"workflow step %d asset %s was resolved twice", ordinal, asset.FieldPath)
			}
			byField[asset.FieldPath] = asset
		}
		for _, claim := range step.Assets {
			asset, ok := byField[claim.FieldPath]
			if !ok || asset.Digest != claim.Digest || asset.Length != claim.Length ||
				asset.MediaType != claim.MediaType || asset.Order != claim.Order {
				return exit.Named(exit.Validation, "workflow_asset_resolution",
					"workflow step %d asset %s has no exact staged resolution", ordinal, claim.FieldPath)
			}
			if problem := inputasset.Verify(asset, asset.Length); problem != nil {
				return problem
			}
		}
		if len(byField) != len(step.Assets) {
			return exit.Named(exit.Validation, "workflow_asset_resolution",
				"workflow step %d supplies assets its plan does not declare", ordinal)
		}
	}
	for ordinal := range supplied {
		if ordinal < 1 || ordinal > len(plan.Steps) {
			return exit.New(exit.Validation, "workflow assets name nonexistent step %d", ordinal)
		}
	}
	return nil
}

func validatePlacement(step Step, placement orchestrator.DesiredPlacement) *exit.Error {
	if placement.Endpoint != step.Endpoint || placement.ReleaseID != step.EndpointReleaseID {
		return exit.Named(exit.Conflict, "workflow_release_mismatch",
			"workflow step %s/%s pins release %s; this host resolves %s",
			step.Endpoint, step.Entrypoint, step.EndpointReleaseID, placement.ReleaseID)
	}
	for _, binding := range placement.Bindings {
		if binding.Entrypoint != step.Entrypoint {
			continue
		}
		planID, problem := binding.PlanID()
		if problem != nil {
			return problem
		}
		if planID != step.EntrypointBindingPlanID {
			return exit.Named(exit.Conflict, "workflow_plan_mismatch",
				"workflow step %s/%s pins plan %s; this release resolves %s",
				step.Endpoint, step.Entrypoint, step.EntrypointBindingPlanID, planID)
		}
		if !sameStrings(binding.Outputs, step.Outputs) {
			return exit.Named(exit.Conflict, "workflow_outputs_mismatch",
				"workflow step %s/%s declares outputs that differ from the pinned release",
				step.Endpoint, step.Entrypoint)
		}
		return nil
	}
	return exit.New(exit.NotFound, "%s has no function %q", step.Endpoint, step.Entrypoint)
}

func sameStrings(left, right []string) bool {
	if len(left) != len(right) {
		return false
	}
	a, b := append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(a)
	sort.Strings(b)
	for index := range a {
		if a[index] != b[index] {
			return false
		}
	}
	return true
}

func (e *Engine) Reconcile() *exit.Error {
	e.mu.Lock()
	defer e.mu.Unlock()
	rows, problem := e.opt.Store.ActiveWorkflows()
	if problem != nil {
		return problem
	}
	for _, row := range rows {
		if problem := e.advance(row.ID); problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] %s: %s\n", row.ID, problem.Message)
		}
	}
	return nil
}

// ReconcileCancellations is the pre-orchestrator boot fence. It mints nothing: it only
// makes already-durable workflow cancellation visible to queued/requeueing children before
// ordinary request recovery can dispatch them.
func (e *Engine) ReconcileCancellations() *exit.Error {
	e.mu.Lock()
	defer e.mu.Unlock()
	rows, problem := e.opt.Store.ActiveWorkflows()
	if problem != nil {
		return problem
	}
	for _, row := range rows {
		if row.State != "canceling" && row.CancelRequestedAt == "" {
			continue
		}
		steps, problem := e.opt.Store.WorkflowSteps(row.ID)
		if problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] %s cancellation recovery: %s\n", row.ID, problem.Message)
			continue
		}
		if problem := e.cancelActive(row, steps); problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] %s cancellation recovery: %s\n", row.ID, problem.Message)
		}
	}
	return nil
}

func (e *Engine) advance(id string) *exit.Error {
	row, problem := e.opt.Store.Workflow(id)
	if problem != nil || row == nil || workflowDone(row.State) {
		return problem
	}
	steps, problem := e.opt.Store.WorkflowSteps(id)
	if problem != nil {
		return problem
	}
	if row.State == "canceling" || row.CancelRequestedAt != "" {
		return e.cancelActive(*row, steps)
	}
	plan, _, digest, problem := DecodePlan(row.Plan)
	if problem != nil || digest != row.BodyDigest {
		if problem == nil {
			problem = exit.Internalf("workflow %s plan digest changed in the records authority", id)
		}
		return e.fail(*row, steps, problem)
	}
	for index := range steps {
		stepRow := &steps[index]
		planStep := plan.Steps[index]
		if stepRow.ChildRequestID == "" {
			if stepRow.MaterializedDigest == "" {
				prior, problem := e.resolvePrior(plan, steps, index+1)
				if problem != nil {
					return e.fail(*row, steps, problem)
				}
				materialized, data, materializedDigest, resolved, problem :=
					MaterializeStep(plan, index+1, stepRow.AuthoredAssets, prior)
				if problem != nil {
					return e.fail(*row, steps, problem)
				}
				key, problem := ChildKey(row.ExecutionDigest, index+1, materialized, materializedDigest)
				if problem != nil {
					return e.fail(*row, steps, problem)
				}
				if problem := e.opt.Store.PrepareWorkflowStep(id, index+1, stepRow.InstallID,
					data, materializedDigest, resolved, key); problem != nil {
					return problem
				}
				stepRow.MaterializedSubmission, stepRow.MaterializedDigest = data, materializedDigest
				stepRow.ResolvedBindings, stepRow.ChildKey = resolved, key
			}
			return e.submitChild(*row, *stepRow, planStep)
		}
		child, problem := e.opt.Store.RequestRow(stepRow.ChildRequestID)
		if problem != nil {
			return problem
		}
		if child == nil {
			return e.fail(*row, steps, exit.Internalf(
				"workflow %s step %d links missing child %s", id, index+1, stepRow.ChildRequestID))
		}
		if !requestDone(child.State) {
			return nil
		}
		if child.State == "canceled" {
			// An honest projection (cl-028): a canceled child ends the workflow as
			// CANCELED, not as a failure the child never had.
			return e.settle(*row, steps, "canceled", "CHILD_CANCELED",
				fmt.Sprintf("workflow step %d child %s was canceled", index+1, child.ID))
		}
		if child.State != "succeeded" {
			return e.fail(*row, steps, exit.Named(exit.Failed, "workflow_child_failed",
				"workflow step %d child %s settled %s", index+1, child.ID, child.State))
		}
	}
	return e.settle(*row, steps, "succeeded", "", "")
}

func (e *Engine) resolvePrior(plan *Plan, steps []records.WorkflowStep,
	ordinal int) ([]ResolvedOutput, *exit.Error) {
	var resolved []ResolvedOutput
	for _, binding := range plan.Steps[ordinal-1].Bindings {
		prior := steps[binding.PriorStep-1]
		if prior.ChildRequestID == "" {
			return nil, exit.Internalf("workflow step %d prior step %d has no child",
				ordinal, binding.PriorStep)
		}
		request, problem := e.opt.Store.RequestRow(prior.ChildRequestID)
		if problem != nil {
			return nil, problem
		}
		if request == nil || request.State != "succeeded" {
			return nil, exit.Named(exit.Failed, "workflow_prior_unsuccessful",
				"workflow step %d prior step %d is not succeeded", ordinal, binding.PriorStep)
		}
		outputs, problem := e.opt.Store.VisibleOutputsOf(request.ID, request.Ordinal)
		if problem != nil {
			return nil, problem
		}
		var found *records.Output
		for index := range outputs {
			if outputs[index].OutputID == binding.OutputName {
				if found != nil {
					return nil, exit.Internalf("prior output %s appears more than once", binding.OutputName)
				}
				copy := outputs[index]
				found = &copy
			}
		}
		if found == nil {
			return nil, exit.Named(exit.Validation, "workflow_output_missing",
				"workflow step %d prior step %d published no output %s",
				ordinal, binding.PriorStep, binding.OutputName)
		}
		resolved = append(resolved, ResolvedOutput{Binding: binding,
			RequestID: request.ID, Attempt: request.Ordinal, Output: *found})
	}
	return resolved, nil
}

func (e *Engine) submitChild(workflow records.WorkflowExecution, step records.WorkflowStep,
	planStep Step) *exit.Error {
	materialized, problem := DecodeMaterialized(step.MaterializedSubmission)
	if problem != nil {
		return e.fail(workflow, []records.WorkflowStep{step}, problem)
	}
	var placement orchestrator.DesiredPlacement
	var entrypoint *launch.Entrypoint
	if step.Worker != "" {
		if e.opt.Rentals == nil || e.opt.RemoteEntrypoint == nil {
			return e.fail(workflow, []records.WorkflowStep{step},
				exit.Unavailablef("this LocalService resolves no complete rented workflow target"))
		}
		remote, resolveProblem := e.opt.Rentals(step.Worker)
		if resolveProblem != nil || remote == nil {
			if resolveProblem == nil {
				resolveProblem = exit.New(exit.NotFound, "no attached rental %s", step.Worker)
			}
			return e.fail(workflow, []records.WorkflowStep{step}, resolveProblem)
		}
		placement = *remote
		entrypoint, resolveProblem = e.opt.RemoteEntrypoint(step.Worker, planStep.Entrypoint)
		if resolveProblem != nil {
			return e.fail(workflow, []records.WorkflowStep{step}, resolveProblem)
		}
	} else {
		spec, resolveProblem := e.opt.Resolver.ResolveInstall(step.InstallID)
		if resolveProblem != nil {
			return e.fail(workflow, []records.WorkflowStep{step}, resolveProblem)
		}
		placement = spec.Placement
		entrypoint, resolveProblem = e.opt.Resolver.Entrypoint(step.InstallID, planStep.Entrypoint)
		if resolveProblem != nil {
			return e.fail(workflow, []records.WorkflowStep{step}, resolveProblem)
		}
	}
	if problem := validatePlacement(planStep, placement); problem != nil {
		return e.fail(workflow, []records.WorkflowStep{step}, problem)
	}
	paths, problem := e.assetPaths(step, materialized)
	if problem != nil {
		return e.fail(workflow, []records.WorkflowStep{step}, problem)
	}
	limits := map[string]int64{}
	for _, asset := range materialized.Assets {
		assetSpec, ok := launch.AssetSpec(entrypoint, asset.FieldPath)
		if !ok || assetSpec.MaxBytes <= 0 {
			return e.fail(workflow, []records.WorkflowStep{step}, exit.Named(exit.Structural,
				"workflow_asset_bound_missing",
				"workflow step %d field %s has no effective compressed-byte bound",
				step.Ordinal, asset.FieldPath))
		}
		if !assetSpec.AcceptsMediaType(asset.MediaType) {
			return e.fail(workflow, []records.WorkflowStep{step}, exit.Named(exit.Validation,
				"workflow_asset_media_type",
				"workflow step %d asset %s is %s, which its pinned field does not admit",
				step.Ordinal, asset.FieldPath, asset.MediaType))
		}
		limits[asset.FieldPath] = assetSpec.MaxBytes
	}
	submission, problem := materialized.Submission(step.ChildKey,
		step.MaterializedDigest, step.InstallID, step.Worker, paths, limits)
	if problem != nil {
		return e.fail(workflow, []records.WorkflowStep{step}, problem)
	}
	child, _, problem := e.opt.Owner.RecordWorkflowChild(
		workflow.ID, step.Ordinal, step.ChildKey, submission)
	if problem != nil {
		return problem
	}
	go func() {
		if problem := e.opt.Owner.ActivateRecorded(child.ID); problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] child %s activation: %s\n", child.ID, problem.Message)
		}
	}()
	return nil
}

func (e *Engine) assetPaths(step records.WorkflowStep,
	materialized MaterializedSubmission) (map[string]string, *exit.Error) {
	paths := map[string]string{}
	authored := map[string]records.AssetBinding{}
	for _, asset := range step.AuthoredAssets {
		paths[asset.FieldPath] = asset.LocalPath
		authored[asset.FieldPath] = asset
	}
	for _, binding := range step.ResolvedBindings {
		outputs, problem := e.opt.Store.VisibleOutputsOf(binding.PriorRequestID, binding.PriorAttempt)
		if problem != nil {
			return nil, problem
		}
		for _, output := range outputs {
			if output.OutputID == binding.OutputName && output.Digest == binding.Digest &&
				output.Length == binding.Length && output.MimeType == binding.MediaType {
				paths[binding.FieldPath] = output.Path
			}
		}
	}
	for _, asset := range materialized.Assets {
		path := paths[asset.FieldPath]
		if path == "" {
			return nil, exit.Named(exit.Conflict, "workflow_asset_resolution",
				"workflow step %d asset %s lost its private resolution", step.Ordinal, asset.FieldPath)
		}
		if binding, ok := authored[asset.FieldPath]; ok {
			if problem := inputasset.Verify(binding, binding.Length); problem != nil {
				return nil, problem
			}
		}
	}
	return paths, nil
}

func (e *Engine) Cancel(id string) (*Snapshot, bool, *exit.Error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	row, changed, problem := e.opt.Store.RequestWorkflowCancel(id)
	if problem != nil || row == nil {
		return nil, changed, problem
	}
	if !workflowDone(row.State) {
		if problem := e.advance(id); problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] %s cancellation delivery deferred: %s\n",
				id, problem.Message)
		}
	}
	snapshot, problem := e.snapshot(id)
	return snapshot, changed, problem
}

func (e *Engine) cancelActive(workflow records.WorkflowExecution,
	steps []records.WorkflowStep) *exit.Error {
	for _, step := range steps {
		if step.ChildRequestID == "" {
			continue
		}
		request, problem := e.opt.Store.RequestRow(step.ChildRequestID)
		if problem != nil {
			return problem
		}
		if request == nil || requestDone(request.State) {
			continue
		}
		attempts, problem := e.opt.Store.Attempts(request.ID)
		if problem != nil {
			return problem
		}
		if len(attempts) == 0 {
			return e.opt.Owner.CancelQueued(request.ID)
		}
		last := attempts[len(attempts)-1]
		switch last.State {
		case "closed", "dispatch_aborted":
			return e.opt.Owner.CancelQueued(request.ID)
		case "terminal":
			return nil // wait for the ack/requeue projection before deciding what remains
		default:
			return e.opt.Owner.CancelClient(request.ID, uint64(last.Attempt),
				orchestrator.ClientCancelGraceMS)
		}
	}
	return e.settle(workflow, steps, "canceled", "CLIENT_CANCELED",
		"workflow cancellation prevented every later child")
}

func (e *Engine) fail(workflow records.WorkflowExecution, steps []records.WorkflowStep,
	problem *exit.Error) *exit.Error {
	if problem == nil {
		problem = exit.Internalf("workflow %s failed without a cause", workflow.ID)
	}
	if len(steps) != workflow.StepCount {
		all, readProblem := e.opt.Store.WorkflowSteps(workflow.ID)
		if readProblem != nil {
			return readProblem
		}
		steps = all
	}
	if settle := e.settle(workflow, steps, "failed", problem.ErrName(), problem.Message); settle != nil {
		return settle
	}
	return nil
}

func (e *Engine) settle(workflow records.WorkflowExecution, steps []records.WorkflowStep,
	state, code, message string) *exit.Error {
	unlock := inputasset.Guard()
	defer unlock()
	if problem := e.opt.Store.SettleWorkflow(workflow.ID, state, code, message); problem != nil {
		return problem
	}
	for _, step := range steps {
		if problem := inputasset.DropUnowned(e.opt.Layout, e.opt.Store, step.AuthoredAssets); problem != nil {
			fmt.Fprintf(e.opt.Log, "[workflow] %s asset cleanup deferred: %s\n",
				workflow.ID, problem.Message)
		}
	}
	return nil
}

func (e *Engine) State(id string) (*Snapshot, *exit.Error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.snapshot(id)
}

// Receipt returns only durable workflow-owned evidence. In particular it never
// resolves the current rental table: paid dial authority may already have been
// released while the exact execution snapshot remains part of this workflow.
func (e *Engine) Receipt(id string) (*Receipt, *exit.Error) {
	e.mu.Lock()
	defer e.mu.Unlock()
	snapshot, problem := e.snapshot(id)
	if problem != nil || snapshot == nil {
		return nil, problem
	}
	steps, problem := e.opt.Store.WorkflowSteps(id)
	if problem != nil {
		return nil, problem
	}
	controls, problem := e.opt.Store.WorkflowRentalControls(id)
	if problem != nil {
		return nil, problem
	}
	out := &Receipt{Snapshot: *snapshot, Plan: append([]byte(nil), snapshot.Execution.Plan...),
		Rentals: controls, Steps: make([]ReceiptStep, 0, len(steps))}
	for _, step := range steps {
		out.Steps = append(out.Steps, ReceiptStep{Ordinal: step.Ordinal,
			MaterializedSubmission: append([]byte(nil), step.MaterializedSubmission...)})
	}
	if snapshot.Execution.CreativePlanDigest != "" {
		composition, readProblem := e.opt.Store.VideoCompositionByCreativePlan(
			snapshot.Execution.CreativePlanDigest)
		if readProblem != nil {
			return nil, readProblem
		}
		if composition != nil {
			out.CreativePlan = append([]byte(nil), composition.CreativePlan...)
		}
	}
	return out, nil
}

func (e *Engine) snapshot(id string) (*Snapshot, *exit.Error) {
	row, problem := e.opt.Store.Workflow(id)
	if problem != nil || row == nil {
		return nil, problem
	}
	steps, problem := e.opt.Store.WorkflowSteps(id)
	if problem != nil {
		return nil, problem
	}
	out := &Snapshot{Execution: *row, Steps: make([]StepSnapshot, 0, len(steps))}
	for _, step := range steps {
		one := StepSnapshot{Ordinal: step.Ordinal, Materialized: step.MaterializedDigest != "",
			MaterializedDigest: step.MaterializedDigest,
			ResolvedBindings:   append([]records.ResolvedBinding(nil), step.ResolvedBindings...),
			ChildKey:           step.ChildKey, RentalID: step.Worker}
		if len(step.MaterializedSubmission) > 0 {
			materialized, decodeProblem := DecodeMaterialized(step.MaterializedSubmission)
			if decodeProblem != nil {
				return nil, decodeProblem
			}
			one.MaterializedAssets = append([]MaterializedAsset(nil), materialized.Assets...)
		}
		if !one.Materialized {
			one.State = "pending"
		} else if step.ChildRequestID == "" {
			one.State = "prepared"
		} else {
			request, problem := e.opt.Store.RequestRow(step.ChildRequestID)
			if problem != nil {
				return nil, problem
			}
			one.Child = request
			if request == nil {
				one.State = "missing"
			} else {
				one.State = request.State
				if request.State == "succeeded" {
					one.Outputs, problem = e.opt.Store.VisibleOutputsOf(request.ID, request.Ordinal)
					if problem != nil {
						return nil, problem
					}
				}
			}
		}
		out.Steps = append(out.Steps, one)
	}
	return out, nil
}

func requestDone(state string) bool {
	switch state {
	case "succeeded", "failed", "canceled", "refused", "abandoned":
		return true
	}
	return false
}

func workflowDone(state string) bool {
	return state == "succeeded" || state == "failed" || state == "canceled"
}
