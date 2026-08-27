package app

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	localapi "github.com/cozy-creator/cozy-creator-v2/internal/client"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
	"github.com/cozy-creator/cozy-creator-v2/internal/rental"
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
	if err := os.Mkdir(absolute, 0o755); err != nil {
		return exit.Internalf("cannot create workflow output directory: %s", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = os.RemoveAll(absolute)
		}
	}()

	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.Workflow(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	if state.Status == "running" || state.Status == "canceling" {
		return exit.New(exit.Conflict,
			"workflow %s is %s; download only a terminal receipt set", state.WorkflowID, state.Status).
			WithNext("cozy workflow follow " + state.WorkflowID)
	}
	if problem := writeWorkflowJSON(filepath.Join(absolute, "workflow.json"), state); problem != nil {
		return problem
	}

	type requestReceipt struct {
		Ordinal   int                 `json:"ordinal"`
		Lifecycle api.Lifecycle       `json:"lifecycle"`
		Events    []localapi.Event    `json:"events"`
		Triage    any                 `json:"triage,omitempty"`
		Saved     []map[string]string `json:"saved_outputs"`
	}
	receipts := make([]requestReceipt, 0, len(state.Steps))
	rentalIDs := map[string]bool{}
	for _, step := range state.Steps {
		if step.RentalID != "" {
			rentalIDs[step.RentalID] = true
		}
		if step.ChildRequest == "" {
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
		stepDir := filepath.Join(absolute, fmt.Sprintf("step-%02d", step.Ordinal))
		saved, problem := saveOutputsAt(client, life, stepDir)
		if problem != nil {
			return problem
		}
		receipt := requestReceipt{Ordinal: step.Ordinal, Lifecycle: life,
			Events: events, Triage: triage, Saved: saved}
		receipts = append(receipts, receipt)
		if problem := writeWorkflowJSON(filepath.Join(absolute,
			fmt.Sprintf("step-%02d-receipt.json", step.Ordinal)), receipt); problem != nil {
			return problem
		}
	}

	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	controls := map[string]any{}
	for id := range rentalIDs {
		row, control, inspectProblem := rental.Inspect(store, id)
		if inspectProblem != nil {
			return inspectProblem
		}
		controls[id] = map[string]any{
			"endpoint": row.EndpointRef, "accelerator": row.AcceleratorModel,
			"control": control,
			"exact_control_snapshot": map[string]any{
				"canonical_bytes": row.ControlSnapshotBytes,
				"digest":          row.ControlSnapshotDigest, "length": row.ControlSnapshotLength,
			},
		}
	}
	if problem := writeWorkflowJSON(filepath.Join(absolute, "rental-control.json"), controls); problem != nil {
		return problem
	}
	manifest := map[string]any{
		"workflow": state, "requests": receipts, "rental_control": controls,
	}
	if problem := writeWorkflowJSON(filepath.Join(absolute, "download-manifest.json"), manifest); problem != nil {
		return problem
	}
	keep = true
	return emit(ctx, render.Record{Kind: "workflow_download", Fields: []render.Field{
		{K: "workflow", V: state.WorkflowID}, {K: "status", V: state.Status},
		{K: "directory", V: absolute}, {K: "request_receipts", V: len(receipts)},
		{K: "rental_controls", V: len(controls)},
	}, Notes: []string{"every media object was length- and digest-verified before local publication"}})
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
