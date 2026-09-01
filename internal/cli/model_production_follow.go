package cli

import (
	"fmt"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
)

func followModelProduction(ctx *Context, client *localclient.Client,
	state api.ModelProductionState,
) *exit.Error {
	progress := NewModelProductionProgress(ctx.Err, !ctx.Mode().JSON, state.Steps)
	if state.Changed {
		progress.Accepted(state.ID, len(state.Outputs))
	} else {
		progress.Resume(state.ID, modelProductionStage(state))
	}
	interrupt := make(chan os.Signal, 1)
	signal.Notify(interrupt, syscall.SIGINT, syscall.SIGTERM)
	defer signal.Stop(interrupt)
	ticker := time.NewTicker(750 * time.Millisecond)
	defer ticker.Stop()
	last := state.Status
	for !modelProductionSettled(state.Status) {
		select {
		case <-interrupt:
			if !ctx.Mode().JSON {
				fmt.Fprintf(ctx.Err, "\ndetached from model production %s; durable work continues\n", state.ID)
			}
			state.Changed = false
			return emitModelProductionState(ctx, state, false)
		case <-ticker.C:
			current, problem := client.ModelProduction(state.ID)
			if problem != nil {
				return problem.WithRemedy("the model production remains durable; reconnect with the same publish instruction")
			}
			state = current
			if state.Status != last {
				progress.Resume(state.ID, modelProductionStage(state))
				last = state.Status
			}
		}
	}
	if state.Status == "completed" || state.Status == "partial" {
		return emitModelProductionState(ctx, state, state.Changed)
	}
	if state.Status == "canceled" {
		return exit.Named(exit.Canceled, firstNonempty(state.ErrorCode, "model_production.canceled"),
			"model production %s was canceled: %s", state.ID, state.Error)
	}
	return exit.Named(exit.Failed, firstNonempty(state.ErrorCode, "model_production.failed"),
		"model production %s failed: %s", state.ID, state.Error)
}

func emitModelProductionState(ctx *Context, state api.ModelProductionState,
	changed bool,
) *exit.Error {
	fields := []output.Field{
		{K: "id", V: state.ID}, {K: "kind", V: state.Kind},
		{K: "model", V: state.Model}, {K: "status", V: state.Status},
		{K: "outputs", V: state.Outputs}, {K: "checkpoints", V: state.Checkpoints},
		{K: "rental", V: state.Rental}, {K: "cleanup", V: state.Cleanup},
		{K: "changed", V: changed},
	}
	record := compactRecord(fields, "id", "kind", "model", "status", "outputs",
		"checkpoints", "cleanup", "changed")
	if !modelProductionSettled(state.Status) {
		record.Next = []string{"cozy run cancel " + state.ID}
	}
	return emit(ctx, record)
}

func handleModelProductionCancel(ctx *Context) *exit.Error {
	client, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	state, problem := client.ModelProduction(ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	if modelProductionSettled(state.Status) {
		return emitModelProductionState(ctx, state, false)
	}
	state, problem = client.CancelModelProduction(state.ID)
	if problem != nil {
		return problem
	}
	for !modelProductionSettled(state.Status) {
		time.Sleep(500 * time.Millisecond)
		state, problem = client.ModelProduction(state.ID)
		if problem != nil {
			return problem.WithRemedy("cancellation remains durable; `cozy run list` shows cleanup progress")
		}
	}
	return emitModelProductionState(ctx, state, true)
}

func modelProductionSettled(state string) bool {
	switch state {
	case "completed", "partial", "failed", "canceled":
		return true
	}
	return false
}

func modelProductionStage(state api.ModelProductionState) string {
	switch state.Status {
	case "resolving":
		return "resolving immutable source and package releases"
	case "accepted":
		return "exact plan accepted; rental selection"
	case "source_preparing":
		return "source preparation on rental " + state.Rental
	case "source_prepared":
		return "source prepared; step execution follows"
	case "step_running":
		return fmt.Sprintf("step %d/%d", min(int(state.StepIndex)+1, state.Steps), state.Steps)
	case "outputs_preparing":
		return "required output checkpoints are being retained"
	case "cleanup_pending":
		return "checkpoints retained; confirming provider absence"
	default:
		return state.Status
	}
}
