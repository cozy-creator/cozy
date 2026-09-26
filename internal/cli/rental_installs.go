package cli

import (
	"fmt"
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPackageInstall(ctx *Context) *exit.Error {
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
	row, problem := store.RentalByMachine(strings.TrimSpace(rentalName))
	store.Close()
	if problem != nil {
		return problem
	}
	if row == nil {
		return exit.New(exit.NotFound, "no rental %q on this host", rentalName)
	}
	rentalID := row.ID
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
	record.Notes = []string{"installation accepted; Creator will deliver it when this rental is ready"}
	if len(selection.Models) == 0 {
		record.Notes = append(record.Notes, "no model weights were requested")
	}
	record.Next = []string{"cozy rental installs " + rentalID}
	return emit(ctx, record)
}

func handleRentalInstalls(ctx *Context) *exit.Error {
	name := strings.TrimSpace(ctx.Inv.Args[0])
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	row, problem := store.RentalByMachine(name)
	store.Close()
	if problem != nil {
		return problem
	}
	id := name
	if row != nil {
		id = row.ID
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	rows, problem := client.RentalInstalls(id)
	if problem != nil {
		return problem
	}
	list := output.List{Name: "installs", Fields: []string{"id", "status", "target", "error"}, AllFields: []string{"id", "rental", "status", "target", "package", "release", "models", "error_code", "error"}, Rows: []map[string]string{}, TypedRows: []map[string]any{}, Total: len(rows)}
	for _, row := range rows {
		list.Rows = append(list.Rows, map[string]string{"id": row.ID, "rental": row.RentalID, "status": row.State, "target": rentalInstallTarget(row.Selection), "package": row.Selection.Package, "release": row.Selection.Release, "models": fmt.Sprint(len(row.Selection.Models)), "error_code": row.ErrorCode, "error": row.Error})
		list.TypedRows = append(list.TypedRows, map[string]any{"id": row.ID, "rental": row.RentalID, "status": row.State, "target": rentalInstallTarget(row.Selection), "package": row.Selection.Package, "release": row.Selection.Release, "models": row.Selection.Models, "error_code": row.ErrorCode, "error": row.Error})
	}
	return emit(ctx, list)
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
