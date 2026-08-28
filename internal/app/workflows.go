package app

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/canonical"
	localapi "github.com/cozy-creator/cozy-creator-v2/internal/client"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

const workflowPollCadence = 2 * time.Second

func handleWorkflowSubmit(ctx *Context) *exit.Error {
	path := ctx.Inv.Value("--in")
	if path == "" {
		return exit.Usagef("workflow submit requires --in <canonical-plan.json>")
	}
	plan, err := os.ReadFile(path)
	if err != nil {
		return exit.New(exit.NotFound, "cannot read workflow plan %s: %s", path, err)
	}
	if !json.Valid(plan) {
		return exit.New(exit.Validation, "workflow plan %s is not JSON", path)
	}
	targets, problem := workflowTargets(ctx.Inv.Values["--worker"])
	if problem != nil {
		return problem
	}
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	key := ctx.Inv.Value("--idempotency-key")
	if key == "" {
		key = mintKey()
	}
	handle, problem := client.SubmitWorkflow(api.WorkflowSubmission{Plan: plan, Targets: targets}, key)
	if problem != nil {
		return problem
	}
	notes := []string{}
	if handle.Replay {
		notes = append(notes,
			"this key was already recorded — the SAME workflow answered, nothing new started")
	}
	return emit(ctx, render.Record{Kind: "workflow", Fields: []render.Field{
		{K: "workflow", V: handle.WorkflowID}, {K: "status", V: handle.Status},
	}, Notes: notes, Next: []string{"cozy workflow status " + handle.WorkflowID}})
}

func workflowTargets(values []string) ([]api.WorkflowTargetResolution, *exit.Error) {
	var out []api.WorkflowTargetResolution
	seen := map[int]bool{}
	for _, value := range values {
		left, worker, ok := strings.Cut(value, "=")
		step, err := strconv.Atoi(left)
		if !ok || err != nil || step < 1 || strings.TrimSpace(worker) == "" || seen[step] {
			return nil, exit.Usagef("--worker %q is not one unique <step>=<rental-id>", value)
		}
		seen[step] = true
		out = append(out, api.WorkflowTargetResolution{Step: step, Worker: strings.TrimSpace(worker)})
	}
	return out, nil
}

func handleWorkflowStatus(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.Workflow(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	return emit(ctx, workflowRecord(state, ctx.Inv.Mode.JSON))
}

func handleWorkflowFollow(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	id, last := ctx.Inv.Args[0], ""
	for {
		state, problem := client.Workflow(id)
		if problem != nil {
			return problem
		}
		snapshot, err := json.Marshal(state)
		if err != nil {
			return exit.Internalf("cannot render workflow progress: %s", err)
		}
		if string(snapshot) != last {
			last = string(snapshot)
			record := workflowRecord(state, ctx.Inv.Mode.JSON)
			if ctx.Inv.Mode.JSON {
				if err := record.Emit(ctx.Out, ctx.Mode()); err != nil {
					return exit.As(err)
				}
			} else if state.Status == "running" || state.Status == "canceling" {
				fmt.Fprintf(ctx.Err, "workflow %s %s: %s\n", id, state.Status,
					workflowStepSummary(state.Steps))
			}
		}
		switch state.Status {
		case "succeeded":
			if ctx.Inv.Mode.JSON {
				return nil
			}
			return emit(ctx, workflowRecord(state, false))
		case "failed":
			if !ctx.Inv.Mode.JSON {
				_ = workflowRecord(state, false).Emit(ctx.Out, ctx.Mode())
			}
			return exit.Named(exit.Failed, "workflow_failed", "workflow %s failed: %s",
				id, state.TerminalMessage)
		case "canceled":
			if !ctx.Inv.Mode.JSON {
				_ = workflowRecord(state, false).Emit(ctx.Out, ctx.Mode())
			}
			return exit.Named(exit.Canceled, "workflow_canceled", "workflow %s was canceled", id)
		}
		// Sampling only. The worker's observed progress/liveness facts decide failure;
		// this loop has no elapsed-time verdict or implicit cancellation.
		time.Sleep(workflowPollCadence)
	}
}

func workflowStepSummary(steps []api.WorkflowStepState) string {
	parts := make([]string, 0, len(steps))
	for _, step := range steps {
		parts = append(parts, fmt.Sprintf("%d=%s", step.Ordinal, step.Status))
	}
	return strings.Join(parts, " ")
}

func handleWorkflowDownload(ctx *Context) *exit.Error {
	dir := strings.TrimSpace(ctx.Inv.Value("--out"))
	if dir == "" {
		return exit.Usagef("workflow download requires --out <new-dir>")
	}
	absolute, err := filepath.Abs(dir)
	if err != nil {
		return exit.New(exit.NotFound, "cannot resolve workflow output directory %s: %s", dir, err)
	}
	if _, err := os.Lstat(absolute); err == nil || !os.IsNotExist(err) {
		return exit.New(exit.Conflict, "workflow output directory %s must not exist", absolute)
	}
	if err := os.MkdirAll(filepath.Dir(absolute), 0o755); err != nil {
		return exit.Internalf("cannot create workflow output parent: %s", err)
	}
	parent, base := filepath.Dir(absolute), filepath.Base(absolute)
	staging, err := os.MkdirTemp(parent, "."+base+".staging-*")
	if err != nil {
		return exit.Internalf("cannot stage workflow output directory: %s", err)
	}
	published := false
	defer func() {
		if !published {
			_ = os.RemoveAll(staging)
		}
	}()

	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	receiptSet, problem := client.WorkflowReceipt(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	state := receiptSet.Workflow
	if state.Status == "running" || state.Status == "canceling" {
		return exit.New(exit.Conflict,
			"workflow %s is %s; download only a terminal receipt set", state.WorkflowID, state.Status).
			WithNext("cozy workflow follow " + state.WorkflowID)
	}
	if problem := writeWorkflowJSON(filepath.Join(staging, "workflow.json"), state); problem != nil {
		return problem
	}
	planDigest, problem := writeExactEvidence(filepath.Join(staging, "workflow-plan.json"),
		receiptSet.CanonicalPlan, "")
	if problem != nil {
		return problem
	}
	creativePath, creativeDigest := "", ""
	if len(receiptSet.CanonicalCreative) > 0 {
		creativePath = "creative-plan.json"
		creativeDigest, problem = writeExactEvidence(filepath.Join(staging, creativePath),
			receiptSet.CanonicalCreative, state.CreativePlanDigest)
		if problem != nil {
			return problem
		}
	}

	type requestReceipt struct {
		Ordinal   int                 `json:"ordinal"`
		Lifecycle api.Lifecycle       `json:"lifecycle"`
		Events    []localapi.Event    `json:"events"`
		Triage    any                 `json:"triage,omitempty"`
		Saved     []map[string]string `json:"saved_outputs"`
	}
	type stepIndex struct {
		Ordinal            int    `json:"ordinal"`
		ReceiptPath        string `json:"receipt_path"`
		ReceiptDigest      string `json:"receipt_digest"`
		MaterializedPath   string `json:"materialized_path"`
		MaterializedDigest string `json:"materialized_digest"`
	}
	materialized := map[int][]byte{}
	for _, step := range receiptSet.Steps {
		materialized[step.Ordinal] = step.MaterializedSubmission
	}
	steps := make([]stepIndex, 0, len(state.Steps))
	requestCount := 0
	for _, step := range state.Steps {
		index := stepIndex{Ordinal: step.Ordinal}
		if len(materialized[step.Ordinal]) > 0 {
			name := fmt.Sprintf("step-%02d-materialized.json", step.Ordinal)
			digest, writeProblem := writeExactEvidence(filepath.Join(staging, name),
				materialized[step.Ordinal], step.MaterializedDigest)
			if writeProblem != nil {
				return writeProblem
			}
			index.MaterializedPath, index.MaterializedDigest = name, digest
		}
		if step.ChildRequest == "" {
			steps = append(steps, index)
			continue
		}
		life, problem := client.Request(step.ChildRequest)
		if problem != nil {
			return problem
		}
		events := []localapi.Event{}
		if _, problem := client.Watch(step.ChildRequest, 0, func(event localapi.Event) bool {
			events = append(events, event)
			return true
		}); problem != nil {
			return problem
		}
		var triage any
		if life.Triage != nil && life.Triage.Kept {
			triage, problem = client.Triage(life.Triage.AttemptKey)
			if problem != nil {
				return problem
			}
		}
		stepDir := filepath.Join(staging, fmt.Sprintf("step-%02d", step.Ordinal))
		saved, problem := saveOutputsAt(client, life, stepDir)
		if problem != nil {
			return problem
		}
		for _, output := range saved {
			relative, err := filepath.Rel(staging, output["path"])
			if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(filepath.Separator)) {
				return exit.Internalf("downloaded output escaped the workflow staging root")
			}
			output["path"] = filepath.ToSlash(relative)
		}
		receipt := requestReceipt{Ordinal: step.Ordinal, Lifecycle: life,
			Events: events, Triage: triage, Saved: saved}
		name := fmt.Sprintf("step-%02d-receipt.json", step.Ordinal)
		path := filepath.Join(staging, name)
		if problem := writeWorkflowJSON(path, receipt); problem != nil {
			return problem
		}
		receiptDigest, digestProblem := digestEvidenceFile(path)
		if digestProblem != nil {
			return digestProblem
		}
		index.ReceiptPath, index.ReceiptDigest = name, receiptDigest
		steps, requestCount = append(steps, index), requestCount+1
	}

	controlDir := filepath.Join(staging, "rental-controls")
	if len(receiptSet.Rentals) > 0 {
		if err := os.Mkdir(controlDir, 0o755); err != nil {
			return exit.Internalf("cannot create rental-control evidence directory: %s", err)
		}
	}
	controls := map[string]map[string]any{}
	for _, control := range receiptSet.Rentals {
		name := strings.TrimPrefix(control.ControlSnapshotDigest, "sha256:") + ".json"
		path := filepath.Join(controlDir, name)
		if _, problem := writeExactEvidence(path, control.ExactControlSnapshotBytes,
			control.ControlSnapshotDigest); problem != nil {
			return problem
		}
		controls[control.RentalID] = map[string]any{
			"endpoint":                             control.Endpoint,
			"accelerator":                          control.Accelerator,
			"observed_accelerator":                 control.ObservedAccelerator,
			"observed_accelerator_count":           control.ObservedAcceleratorCount,
			"observed_backend":                     control.ObservedBackend,
			"observed_worker_instance":             control.ObservedWorkerInstance,
			"observed_worker_boot_id":              control.ObservedWorkerBootID,
			"observed_at":                          control.ObservedAt,
			"snapshot_path":                        filepath.ToSlash(filepath.Join("rental-controls", name)),
			"control_snapshot_digest":              control.ControlSnapshotDigest,
			"control_snapshot_length":              control.ControlSnapshotLength,
			"endpoint_execution_digest":            control.EndpointExecutionDigest,
			"artifact_object_set_digest":           control.ArtifactObjectSetDigest,
			"model_root_digests":                   control.ModelRootDigests,
			"endpoint_release_id":                  control.EndpointReleaseID,
			"descriptor_digest":                    control.DescriptorDigest,
			"environment_spec_digest":              control.EnvironmentSpecDigest,
			"installed_environment_receipt_digest": control.InstalledReceiptDigest,
			"placement_set_digest":                 control.PlacementSetDigest,
			"binding_plan_digests":                 control.BindingPlanDigests,
		}
	}
	controlSummary := filepath.Join(staging, "rental-control.json")
	if problem := writeWorkflowJSON(controlSummary, controls); problem != nil {
		return problem
	}
	workflowDigest, problem := digestEvidenceFile(filepath.Join(staging, "workflow.json"))
	if problem != nil {
		return problem
	}
	controlDigest, problem := digestEvidenceFile(controlSummary)
	if problem != nil {
		return problem
	}
	manifest := map[string]any{
		"version":        1,
		"workflow":       map[string]any{"path": "workflow.json", "digest": workflowDigest},
		"workflow_plan":  map[string]any{"path": "workflow-plan.json", "digest": planDigest},
		"creative_plan":  map[string]any{"path": creativePath, "digest": creativeDigest},
		"steps":          steps,
		"rental_control": map[string]any{"path": "rental-control.json", "digest": controlDigest},
	}
	if problem := writeWorkflowJSON(filepath.Join(staging, "download-manifest.json"), manifest); problem != nil {
		return problem
	}
	if problem := syncTree(staging); problem != nil {
		return problem
	}
	if _, err := os.Lstat(absolute); err == nil || !os.IsNotExist(err) {
		return exit.New(exit.Conflict, "workflow output directory %s appeared during publication", absolute)
	}
	if err := os.Rename(staging, absolute); err != nil {
		return exit.Internalf("cannot publish complete workflow output %s: %s", absolute, err)
	}
	published = true
	if problem := syncDirectory(parent); problem != nil {
		return problem
	}
	return emit(ctx, render.Record{Kind: "workflow_download", Fields: []render.Field{
		{K: "workflow", V: state.WorkflowID}, {K: "status", V: state.Status},
		{K: "directory", V: absolute}, {K: "request_receipts", V: requestCount},
		{K: "rental_controls", V: len(controls)},
	}, Notes: []string{"the complete bundle appeared in one rename after every exact byte and media object was verified"}})
}

func writeExactEvidence(path string, data []byte, expected string) (string, *exit.Error) {
	if len(data) == 0 {
		return "", exit.Internalf("cannot publish empty exact evidence %s", filepath.Base(path))
	}
	digest, err := canonical.Spell(canonical.Digest(data))
	if err != nil {
		return "", exit.Internalf("cannot digest exact evidence %s: %s", filepath.Base(path), err)
	}
	if expected != "" && digest != expected {
		return "", exit.Named(exit.Conflict, "workflow.receipt_digest_mismatch",
			"exact evidence %s hashes to %s, expected %s", filepath.Base(path), digest, expected)
	}
	file, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".staging-*")
	if err != nil {
		return "", exit.Internalf("cannot stage %s: %s", filepath.Base(path), err)
	}
	staged := file.Name()
	keep := false
	defer func() {
		_ = file.Close()
		if !keep {
			_ = os.Remove(staged)
		}
	}()
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(staged, path)
	}
	if err != nil {
		return "", exit.Internalf("cannot publish %s: %s", filepath.Base(path), err)
	}
	keep = true
	return digest, nil
}

func digestEvidenceFile(path string) (string, *exit.Error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", exit.Internalf("cannot digest workflow evidence %s: %s", filepath.Base(path), err)
	}
	sum := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(sum[:]), nil
}

func writeWorkflowJSON(path string, value any) *exit.Error {
	data, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return exit.Internalf("cannot render %s: %s", filepath.Base(path), err)
	}
	data = append(data, '\n')
	file, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+".staging-*")
	if err != nil {
		return exit.Internalf("cannot stage %s: %s", filepath.Base(path), err)
	}
	staging := file.Name()
	keep := false
	defer func() {
		_ = file.Close()
		if !keep {
			_ = os.Remove(staging)
		}
	}()
	if _, err = file.Write(data); err == nil {
		err = file.Sync()
	}
	if closeErr := file.Close(); err == nil {
		err = closeErr
	}
	if err == nil {
		err = os.Rename(staging, path)
	}
	if err != nil {
		return exit.Internalf("cannot publish %s: %s", filepath.Base(path), err)
	}
	keep = true
	return nil
}

func handleWorkflowCancel(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.CancelWorkflow(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	record := workflowRecord(state, ctx.Inv.Mode.JSON)
	record.Notes = append(record.Notes,
		"cancellation is durable; no later child can be minted")
	if state.Status == "canceling" {
		record.Next = []string{"cozy workflow status " + state.WorkflowID}
	}
	return emit(ctx, record)
}

func workflowRecord(state api.WorkflowState, structured bool) render.Record {
	steps := make([]string, 0, len(state.Steps))
	for _, step := range state.Steps {
		child := step.ChildRequest
		if child == "" {
			child = "-"
		}
		steps = append(steps, fmt.Sprintf("%d %s child=%s outputs=%d",
			step.Ordinal, step.Status, child, len(step.Outputs)))
	}
	var renderedSteps any = steps
	if structured {
		renderedSteps = state.Steps
	}
	fields := []render.Field{
		{K: "workflow", V: state.WorkflowID}, {K: "status", V: state.Status},
		{K: "execution_digest", V: state.ExecutionDigest},
		{K: "creative_plan_digest", V: state.CreativePlanDigest},
		{K: "steps", V: renderedSteps},
	}
	if state.TerminalCode != "" {
		fields = append(fields, render.Field{K: "terminal", V: state.TerminalCode},
			render.Field{K: "message", V: state.TerminalMessage})
	}
	next := []string{}
	if state.Status == "running" {
		next = []string{"cozy workflow cancel " + state.WorkflowID}
	}
	return render.Record{Kind: "workflow", Fields: fields, Next: next}
}
