package cli

import (
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPackageInstall(ctx *Context) *exit.Error {
	adoptRentalHub(ctx, ctx.Inv.Value("--rental"))
	ref, plan, problem := resolveRegistryPackage(ctx, ctx.Inv.Args[0], ctx.Inv.Value("--version"))
	if problem != nil {
		return problem
	}
	// No model bindings are resolved or forwarded by package installation.
	return enqueueRentalInstall(ctx, ctx.Inv.Value("--rental"), records.RentalInstallSelection{Package: ref.String(), Release: plan.Release})
}

func enqueueRentalInstall(ctx *Context, rentalName string, selection records.RentalInstallSelection) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	rentalID, problem := machines.InstallTarget(store, rentalName)
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
	result, problem := client.PrepareRentalPackage(rentalID, selection)
	if problem != nil {
		return problem
	}
	fields := []output.Field{{K: "id", V: result.ID}, {K: "rental", V: result.RentalID}, {K: "status", V: result.State}, {K: "target", V: rentalInstallTarget(result.Selection)}, {K: "package", V: result.Selection.Package}, {K: "release", V: result.Selection.Release}, {K: "models", V: result.Selection.Models}}
	record := compactRecord(fields, "id", "rental", "status", "target")
	record.Notes = []string{"installation accepted; Creator delivers it when the machine is ready"}
	if len(selection.Models) == 0 {
		record.Notes = append(record.Notes, "no model weights were requested")
	}
	return emit(ctx, record)
}

func rentalInstallTarget(selection records.RentalInstallSelection) string {
	if selection.Package != "" {
		return selection.Package + "@" + selection.Release
	}
	models := make([]string, 0, len(selection.Models))
	for _, model := range selection.Models {
		name := model.Model
		if model.Release != "" {
			name += "@" + model.Release
		}
		if model.Lane != "" {
			name += "/" + model.Lane
		}
		if model.Release == "" {
			name += "#" + model.Manifest
		}
		models = append(models, name)
	}
	return strings.Join(models, ", ")
}
