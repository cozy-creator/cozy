package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPackageInstall(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	ref, plan, problem := resolveRegistryPackage(ctx, ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	// No model bindings are resolved or forwarded by package installation.
	return enqueueRentalInstall(ctx, ctx.Inv.Value("--rental"), records.InstallSelection{Package: ref.String(), Release: plan.Release})
}

// enqueueRentalInstall journals a download or installation on a machine and answers its
// number; with --await it follows the operation to its end.
func enqueueRentalInstall(ctx *Context, machineName string, selection records.InstallSelection) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	machine, problem := machines.InstallTarget(store, machineName)
	store.Close()
	if problem != nil {
		return problem
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	life, problem := client.Install(machine, api.InstallRequest{InstallSelection: selection,
		IdempotencyKey: strings.TrimSpace(ctx.Inv.Value("--idempotency-key"))})
	if problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") {
		return watchOperation(ctx, client, life)
	}
	record := operationRecord(life)
	record.Notes = append(record.Notes, "Creator delivers it when the machine is ready")
	if len(selection.Models) == 0 {
		record.Notes = append(record.Notes, "no model weights were requested")
	}
	return emit(ctx, record)
}
