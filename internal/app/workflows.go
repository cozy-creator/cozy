package app

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy-creator-v2/internal/api"
	"github.com/cozy-creator/cozy-creator-v2/internal/exit"
	"github.com/cozy-creator/cozy-creator-v2/internal/render"
)

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
	return emit(ctx, workflowRecord(state))
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
	record := workflowRecord(state)
	record.Notes = append(record.Notes,
		"cancellation is durable; no later child can be minted")
	if state.Status == "canceling" {
		record.Next = []string{"cozy workflow status " + state.WorkflowID}
	}
	return emit(ctx, record)
}

func workflowRecord(state api.WorkflowState) render.Record {
	steps := make([]string, 0, len(state.Steps))
	for _, step := range state.Steps {
		child := step.ChildRequest
		if child == "" {
			child = "-"
		}
		steps = append(steps, fmt.Sprintf("%d %s child=%s outputs=%d",
			step.Ordinal, step.Status, child, len(step.Outputs)))
	}
	fields := []render.Field{
		{K: "workflow", V: state.WorkflowID}, {K: "status", V: state.Status},
		{K: "execution_digest", V: state.ExecutionDigest},
		{K: "creative_plan_digest", V: state.CreativePlanDigest},
		{K: "steps", V: steps},
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
