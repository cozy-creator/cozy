package cli

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// nativeModelUpload is the ordinary private-script path for complete model recipes.
// Local files, existing checkpoints and unregistered families retain their transfer path.
func nativeModelUpload(ctx *Context) (bool, *exit.Error) {
	cwd, err := os.Getwd()
	if err != nil {
		return true, exit.Internalf("cannot resolve working directory: %s", err)
	}
	parsed, problem := modelsource.Parse(ctx.Inv.Args[0], cwd)
	if problem != nil || parsed.Kind != modelsource.HuggingFace || parsed.Member != "" {
		return false, nil
	}
	if problem := validateRunPlacement(ctx); problem != nil {
		return true, problem
	}
	if ctx.Inv.Bool("--dry-run") && ctx.Inv.Bool("--await") {
		return true, exit.Usagef("--dry-run and --await conflict")
	}
	if ctx.Inv.Value("--lane") != "" {
		return true, exit.Usagef("--lane selects only a Tensorhub model release")
	}
	if name := ctx.Inv.Value("--rental"); name != "" {
		_, store, problem := rentalStores(ctx)
		if problem != nil {
			return true, problem
		}
		selected, problem := rental.Resolve(store, name)
		store.Close()
		if problem != nil {
			return true, problem
		}
		if selected.Row == nil || records.RentalTerminalState(selected.Row.State) {
			return true, exit.Named(exit.NotFound, "rental.selection_unavailable", "--rental=%s does not name an existing usable rental", name)
		}
	}
	destination, problem := hub.ParseRef(ctx.Inv.Args[1])
	if problem != nil {
		return true, problem
	}
	if destination.Org == "local" {
		return true, exit.Usagef("local/ is reserved for private aliases")
	}
	// The provider resolves moving URL spellings to an immutable commit before the
	// model module selects a recipe. No weights are downloaded by this inventory.
	source, problem := resolvePublishSource(ctx, parsed.Canonical, nil)
	if problem != nil {
		return true, problem
	}
	pinned := source.Resolution.Source
	tool, problem := launch.BuiltinOperationsTool(cwd, ctx.Cfg.Home, ctx.Cfg.Tool())
	if problem != nil {
		return true, problem
	}
	recipe, problem := tool.ModelIngestionPlan(context.Background(), pinned.Org+"/"+pinned.Repo, pinned.Revision)
	if problem != nil {
		return true, problem
	}
	if recipe == nil {
		return false, nil
	}
	if _, problem := ownedPublication(ctx, destination); problem != nil {
		return true, problem
	}
	source, problem = narrowPublishSource(ctx, source, []string{recipe.Profile})
	if problem != nil {
		return true, problem
	}
	runCtx, cancel := hub.LongContext()
	defer cancel()
	conversion, problem := preflightConversionPlan(runCtx, ctx, source, map[string]string{"model": recipe.Profile})
	if problem != nil {
		return true, problem
	}
	if !conversion.decided() {
		return true, exit.Named(exit.Unavailable, "model_source.preflight_unavailable", "source headers must be inspected before ingestion: %s", conversion.Undecided)
	}
	if problem := source.Resolver.VerifyMetadata(runCtx, source.Resolution.Source, recipe.Metadata); problem != nil {
		return true, problem
	}
	if ctx.Inv.Bool("--dry-run") {
		return true, emit(ctx, compactRecord([]output.Field{
			{K: "kind", V: "model-upload"}, {K: "model", V: destination.String()},
			{K: "source", V: source.Canonical}, {K: "recipe", V: recipe.Name},
			{K: "source_profile", V: recipe.Profile}, {K: "source_files", V: source.Files},
			{K: "source_bytes", V: output.Bytes(source.Bytes)}, {K: "metadata", V: recipe.Metadata},
			{K: "conversion", V: "planned"}, {K: "conversion_sessions", V: conversion.Sessions()},
			{K: "rental", V: ctx.Inv.Value("--rental")},
			{K: "steps", V: []string{"download", "convert", "prepare-metadata", "upload-checkpoint"}},
			{K: "status", V: "planned"}, {K: "changed", V: false},
		}, "kind", "model", "source", "recipe", "conversion", "status", "changed"))
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return true, problem
	}
	work, problem := scratch.Temp(layout.Tmp, "model-ingestion-")
	if problem != nil {
		return true, problem
	}
	defer work.Release()
	script := filepath.Join(work.Path, "upload_model.py")
	if err := os.WriteFile(script, modelUploadScript(recipe.Repository, recipe.Revision, destination.String()), 0600); err != nil {
		return true, exit.Internalf("cannot prepare model ingestion: %s", err)
	}
	// Run capture owns the exact script and SDK. The same native receipts survive
	// edited callers, cancellation and retry; the destination is the sole grant.
	ctx.Inv.Args = []string{script}
	ctx.Inv.Values["--allow-publish"] = []string{destination.String()}
	return true, handleRunExecute(ctx)
}

func modelUploadScript(repository, revision, destination string) []byte {
	quote := func(value string) string { raw, _ := json.Marshal(value); return string(raw) }
	return []byte(fmt.Sprintf(`# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=0.18.23,<1", "tensorfs>=0.3.51,<0.4"]
# ///
from cozy_runtime.author.sources import ingest_huggingface
from cozy_runtime.author.publication import upload_checkpoint

async def main() -> dict[str, str]:
    model = await ingest_huggingface(%s, revision=%s)
    checkpoint = await upload_checkpoint(model, destination=%s)
    return {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
`, quote(repository), quote(revision), quote(destination)))
}
