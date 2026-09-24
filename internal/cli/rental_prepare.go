package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
)

func handleRentalPrepare(ctx *Context) *exit.Error {
	rentalID := strings.TrimSpace(ctx.Inv.Args[0])
	packageName := strings.TrimSpace(ctx.Inv.Args[1])
	ref, problem := hub.ParseRef(packageName)
	if problem != nil {
		return problem
	}
	version := strings.TrimSpace(ctx.Inv.Value("--version"))
	if version == "" {
		return exit.Usagef("rental package preparation requires --version for the exact package release").
			WithRemedy("use `cozy rental prepare %s %s --version 1.2.3`", rentalID, packageName)
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
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	result, problem := client.PrepareRentalPackage(rentalID, api.RentalPackagePrepareRequest{
		Package: ref.String(), Release: version, Models: models,
	})
	if problem != nil {
		return problem
	}
	fields := []output.Field{{K: "rental", V: result.Rental}, {K: "package", V: result.Package},
		{K: "release", V: result.Release}, {K: "status", V: result.Status}, {K: "models", V: len(models)}}
	record := compactRecord(fields, "rental", "package", "release", "status")
	record.Notes = []string{"the worker retained this package and its model inputs; repeating this command reuses preparation"}
	record.Next = []string{"cozy run " + packageName + "/<function> --rental=" + rentalID}
	return emit(ctx, record)
}
