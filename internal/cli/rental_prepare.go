package cli

import (
	"fmt"
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

func handleRentalPrepare(ctx *Context) *exit.Error {
	packageName := strings.TrimSpace(ctx.Inv.Args[1])
	ref, problem := hub.ParseRef(packageName)
	if problem != nil {
		return problem
	}
	version := strings.TrimSpace(ctx.Inv.Value("--version"))
	if version == "" {
		return exit.Usagef("rental package preparation requires --version for the exact package release").
			WithRemedy("use `cozy rental prepare %s %s --version 1.2.3`", ctx.Inv.Args[0], packageName)
	}
	models := make([]orchestrator.ModelRef, 0, len(ctx.Inv.Values["--model"]))
	for _, selection := range ctx.Inv.Values["--model"] {
		slotPath, raw, ok := strings.Cut(selection, "=")
		if !ok || strings.TrimSpace(slotPath) == "" || strings.TrimSpace(raw) == "" {
			return exit.Usagef("--model must be SLOT=org/model@release[/lane][#sha256:digest]").
				WithRemedy("repeat --model generate.models.model=org/model@1.0.0/fp8")
		}
		model, release, lane, manifest, parseProblem := hub.ParseModelRef(strings.TrimSpace(raw))
		if parseProblem != nil {
			return parseProblem
		}
		if release == "" || lane == "" {
			return exit.Usagef("--model %q must pin a model release and lane", selection)
		}
		selected, resolveProblem := resolveRemoteModel(ctx, packageName,
			launch.Slot{Path: strings.TrimSpace(slotPath)}, strings.TrimSpace(raw), lane, nil)
		if resolveProblem != nil {
			return resolveProblem
		}
		if selected.Model != model || selected.Release != release || selected.Lane != lane || (manifest != "" && selected.Manifest != manifest) {
			return exit.New(exit.Conflict, "model selection changed while resolving %s", raw)
		}
		models = append(models, selected)
	}
	return enqueueRentalInstall(ctx, ctx.Inv.Args[0], records.RentalInstallSelection{Package: ref.String(), Release: version, Models: models})
}

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
	fields := []output.Field{{K: "id", V: result.ID}, {K: "rental", V: result.RentalID}, {K: "status", V: result.State}, {K: "package", V: result.Selection.Package}, {K: "release", V: result.Selection.Release}, {K: "models", V: result.Selection.Models}}
	record := compactRecord(fields, "id", "rental", "status", "package", "release")
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
	list := output.List{Name: "installs", Fields: []string{"id", "status", "package", "release", "error"}, AllFields: []string{"id", "rental", "status", "package", "release", "models", "error_code", "error"}, Rows: []map[string]string{}, TypedRows: []map[string]any{}, Total: len(rows)}
	for _, row := range rows {
		list.Rows = append(list.Rows, map[string]string{"id": row.ID, "rental": row.RentalID, "status": row.State, "package": row.Selection.Package, "release": row.Selection.Release, "models": fmt.Sprint(len(row.Selection.Models)), "error_code": row.ErrorCode, "error": row.Error})
		list.TypedRows = append(list.TypedRows, map[string]any{"id": row.ID, "rental": row.RentalID, "status": row.State, "package": row.Selection.Package, "release": row.Selection.Release, "models": row.Selection.Models, "error_code": row.ErrorCode, "error": row.Error})
	}
	return emit(ctx, list)
}
