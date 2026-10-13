package cli

import (
	"encoding/json"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

// handleRentalPackageInstall asks the Hub nothing: the machine reads the release at the hub the
// operation names, and one it already holds is the answer with no read anywhere. No model
// bindings are resolved or forwarded by package installation.
func handleRentalPackageInstall(ctx *Context) *exit.Error {
	ref, release, problem := registryPackageRef(ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	op := records.Operation{Package: ref.String(), Release: release, Hub: ctx.Cfg.HubURL}
	return runOperation(ctx, ctx.Inv.Value("--rental"), op, !ctx.Inv.Bool("--no-wait"))
}

// runOperation records op on a machine (`local`, or a rental by name) as a numbered run and,
// awaited, follows it as `cozy run watch` does: Ctrl-C detaches and the run goes on.
func runOperation(ctx *Context, machine string, op records.Operation, await bool) *exit.Error {
	client, handle, problem := submitOperation(ctx, machine, op)
	if problem != nil {
		return problem
	}
	if !await {
		return emitOperation(ctx, machine, op, handle.Number, handle.RequestID, handle.Status, nil, 0)
	}
	began := time.Now()
	life, detached, problem := followOperation(ctx, client, handle)
	if problem != nil {
		return problem
	}
	if detached || !invocationSettled(life.Status) {
		return emitOperation(ctx, machine, op, life.Number, life.RequestID, life.Status, nil, 0)
	}
	result, problem := operationResult(machine, life)
	if problem != nil {
		return problem
	}
	return emitOperation(ctx, machine, op, life.Number, life.RequestID, life.Status, result, time.Since(began))
}

// settleOperation is runOperation awaited and its result answered: what `package update` and
// `package remove` do once per package.
func settleOperation(ctx *Context, machine string, op records.Operation) (json.RawMessage, *exit.Error) {
	client, handle, problem := submitOperation(ctx, machine, op)
	if problem != nil {
		return nil, problem
	}
	life, detached, problem := followOperation(ctx, client, handle)
	if problem == nil && detached {
		problem = exit.New(exit.Canceled, "detached from run %d (%s); it continues on %s; `cozy run cancel %d` stops it",
			life.Number, op.Target(), machine, life.Number)
	}
	if problem != nil {
		return nil, problem
	}
	return operationResult(machine, life)
}

func submitOperation(ctx *Context, machine string, op records.Operation) (*localapi.Client, api.Handle, *exit.Error) {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return nil, api.Handle{}, problem
	}
	id := machines.Local
	if !machines.IsLocal(machine) {
		id, problem = selectedRental(ctx, store, machine)
	}
	store.Close()
	if problem != nil {
		return nil, api.Handle{}, problem
	}
	client, problem := dial(ctx)
	if problem != nil {
		return nil, api.Handle{}, problem
	}
	op.Hub = either(op.Hub, ctx.Cfg.HubURL)
	handle, problem := client.SubmitOperation(id, op)
	return client, handle, problem
}

// followOperation watches the run until it settles or Ctrl-C detaches from it.
func followOperation(ctx *Context, client *localapi.Client, handle api.Handle) (api.Lifecycle, bool, *exit.Error) {
	notice := &reattachNotice{ctx: ctx}
	client = client.Following(notice.say)
	life, problem := client.Request(handle.RequestID)
	if problem != nil {
		return life, false, problem
	}
	_, detached, problem := watchRunStream(ctx, client, life, notice)
	if problem != nil {
		return life, detached, problem
	}
	life, problem = client.Request(handle.RequestID)
	return life, detached, problem
}

// operationResult is a settled operation's result, or why it has none.
func operationResult(machine string, life api.Lifecycle) (json.RawMessage, *exit.Error) {
	switch life.Status {
	case "completed":
		raw, _ := json.Marshal(life.Result)
		return raw, nil
	case "canceled":
		return nil, exit.Named(exit.Canceled, "run.canceled", "%s: run %d was canceled", machine, life.Number)
	}
	return nil, exit.Named(exit.Failed, either(life.ErrorCode, "operation.failed"), "%s: %s", machine, either(life.Error, life.Status))
}

// emitOperation is an operation's run: its number and state, and once it succeeded what it
// produced (the release installed, an upload's checkpoint, a warm member's level).
func emitOperation(ctx *Context, machine string, op records.Operation, number int64, id, status string, raw json.RawMessage, elapsed time.Duration) *exit.Error {
	var result struct {
		Package, Release string
		Models           []struct {
			Published *struct{ Destination, Checkpoint string } `json:"published"`
		} `json:"models"`
		Set []struct {
			Package, Entrypoint, Level string
			HeldBack                   string `json:"held_back"`
		} `json:"set"`
	}
	_ = json.Unmarshal(raw, &result)
	target := op.Target()
	if result.Package != "" && result.Release != "" {
		target = result.Package + "@" + result.Release
	}
	fields := []output.Field{{K: "number", V: number}, {K: "id", V: id}, {K: "kind", V: op.Kind()}, {K: "machine", V: machine},
		{K: "status", V: status}, {K: "target", V: target}}
	shown := []string{"number", "id", "kind", "machine", "status", "target"}
	if elapsed > 0 {
		fields, shown = append(fields, output.Field{K: "elapsed", V: elapsed.Round(time.Second).String()}), append(shown, "elapsed")
	}
	for _, member := range result.Set {
		if member.Package == op.Package && member.Entrypoint == op.Entrypoint {
			fields, shown = append(fields, output.Field{K: "level", V: member.Level}), append(shown, "level")
			if member.HeldBack != "" {
				fields, shown = append(fields, output.Field{K: "held_back", V: member.HeldBack}), append(shown, "held_back")
			}
		}
	}
	if len(result.Models) == 1 && result.Models[0].Published != nil {
		published := result.Models[0].Published
		fields = append(fields, output.Field{K: "destination", V: published.Destination}, output.Field{K: "checkpoint", V: published.Checkpoint})
		shown = append(shown, "destination", "checkpoint")
	}
	record := compactRecord(fields, shown...)
	if raw == nil {
		n := strconv.FormatInt(number, 10)
		record.Next = []string{"cozy run watch " + n, "cozy run cancel " + n}
	} else if op.Package != "" && !machines.IsLocal(machine) && op.Warm == "" {
		record.Next = []string{"cozy package list --rental=" + machine}
	}
	return emit(ctx, record)
}
