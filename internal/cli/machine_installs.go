package cli

import (
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
)

func handleMachineInstallShow(ctx *Context) *exit.Error {
	client, machine, problem := rentalInstallClient(ctx, ctx.Inv.Value("--rental"))
	if problem != nil {
		return problem
	}
	status, problem := client.RentalInstall(machine, ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	if ctx.Inv.Bool("--await") && status.Active() {
		began := time.Now()
		settled, detached, problem := watchRentalInstall(ctx, client, machine, status.RentalInstall)
		if problem != nil {
			return problem
		}
		if !detached {
			return emitInstalled(ctx, settled, began)
		}
		status.RentalInstall = settled
	}
	return emitInstallStatus(ctx, status)
}

func handleMachineInstallCancel(ctx *Context) *exit.Error {
	client, machine, problem := rentalInstallClient(ctx, ctx.Inv.Value("--rental"))
	if problem != nil {
		return problem
	}
	status, problem := client.CancelRentalInstall(machine, ctx.Inv.Args[0])
	if problem != nil {
		return problem
	}
	return emitInstallStatus(ctx, status)
}

func emitInstallStatus(ctx *Context, status machines.InstallStatus) *exit.Error {
	state := status.State
	if state == "failed" && status.ErrorCode == "install.canceled" {
		state = "canceled"
	}
	fields := []output.Field{
		{K: "id", V: status.ID}, {K: "rental", V: status.RentalID},
		{K: "status", V: state}, {K: "target", V: status.Selection.Target()},
		{K: "selection", V: status.Selection}, {K: "progress", V: status.Progress},
		{K: "result", V: status.Result}, {K: "error_code", V: status.ErrorCode},
		{K: "error", V: status.Error}, {K: "created_at", V: status.CreatedAt},
		{K: "updated_at", V: status.UpdatedAt},
	}
	return emit(ctx, compactRecord(fields, "id", "rental", "status", "target", "progress", "error"))
}
