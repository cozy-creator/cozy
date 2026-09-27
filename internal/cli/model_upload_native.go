package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"

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

// Runtime floors for the generated ingest script: 0.18.24 supplies native download,
// conversion and upload; 0.18.40 composes several reviewed profiles into one model.
const (
	ingestRuntimeFloor         = "0.18.24"
	composedIngestRuntimeFloor = "0.18.40"
)

// nativeModelUpload runs a rented provider ingest as an ordinary local script on the
// rental: the pod downloads the source, converts it to CozyTensors with reviewed TensorFS
// profiles and uploads the checkpoint. A Runtime recipe, where one exists, adds its
// model-owned metadata; without one the profiles the source headers match are used.
func nativeModelUpload(ctx *Context) (bool, *exit.Error) {
	if !rentalRequested(ctx) {
		return false, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return true, exit.Internalf("cannot resolve working directory: %s", err)
	}
	parsed, problem := modelsource.Parse(ctx.Inv.Args[0], cwd)
	if problem != nil || (parsed.Kind != modelsource.HuggingFace && parsed.Kind != modelsource.Civitai) {
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
	rentalID := ""
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
		rentalID = selected.Row.ID
	}
	destination, problem := hub.ParseRef(ctx.Inv.Args[1])
	if problem != nil {
		return true, problem
	}
	if destination.Org == "local" {
		return true, exit.Usagef("local/ is reserved for private aliases")
	}
	// The provider resolves moving URL spellings to an immutable identity first. No
	// weights are downloaded by this inventory.
	source, problem := resolvePublishSource(ctx, parsed.Canonical, nil)
	if problem != nil {
		return true, problem
	}
	pinned := source.Resolution.Source
	profiles := ctx.Inv.Values["--source-profile"]
	var recipe *launch.ModelIngestionRecipe
	if parsed.Kind == modelsource.HuggingFace && parsed.Member == "" && len(profiles) == 0 {
		tool, problem := launch.BuiltinOperationsTool(cwd, ctx.Cfg.Home, ctx.Cfg.Tool())
		if problem != nil {
			return true, problem
		}
		if recipe, problem = tool.ModelIngestionPlan(context.Background(), pinned.Org+"/"+pinned.Repo, pinned.Revision); problem != nil {
			return true, problem
		}
	}
	if _, problem := ownedPublication(ctx, destination); problem != nil {
		return true, problem
	}
	if recipe != nil {
		profiles = []string{recipe.Profile}
	}
	if len(profiles) > 0 {
		if source, problem = narrowSourceToProfiles(ctx, source, profiles); problem != nil {
			return true, problem
		}
	}
	runCtx, cancel := hub.LongContext()
	defer cancel()
	conversion, problem := preflightConversionPlan(runCtx, ctx, source, profileSlots(profiles))
	if problem != nil {
		if strings.Contains(problem.Message, "AMBIGUOUS_CLASSIFICATION") {
			problem = problem.WithRemedy("choose the reviewed profile(s) with --source-profile; profiles over different files compose one model")
		}
		return true, problem
	}
	if !conversion.decided() {
		return true, exit.Named(exit.Unavailable, "model_source.preflight_unavailable", "source headers must be inspected before ingestion: %s", conversion.Undecided)
	}
	if len(profiles) == 0 {
		profiles = []string{conversion.Plans["model"].Profile}
		if source, problem = narrowSourceToProfiles(ctx, source, profiles); problem != nil {
			return true, problem
		}
	}
	var script []byte
	if recipe != nil {
		script = recipeUploadScript(recipe.Repository, recipe.Revision, destination.String())
	} else {
		script = genericUploadScript(parsed.Kind, pinned, profiles, destination.String())
	}
	if ctx.Inv.Bool("--dry-run") {
		fields := []output.Field{
			{K: "kind", V: "model-upload"}, {K: "model", V: destination.String()},
			{K: "source", V: source.Canonical}, {K: "source_profiles", V: profiles},
			{K: "source_files", V: source.Files}, {K: "source_bytes", V: output.Bytes(source.Bytes)},
			{K: "conversion", V: "planned"}, {K: "conversion_sessions", V: conversion.Sessions()},
			{K: "rental", V: ctx.Inv.Value("--rental")},
			{K: "steps", V: []string{"download", "convert", "upload-checkpoint"}},
			{K: "status", V: "planned"}, {K: "changed", V: false},
		}
		if recipe != nil {
			fields = append(fields, output.Field{K: "recipe", V: recipe.Name},
				output.Field{K: "metadata", V: recipe.Metadata},
				output.Field{K: "steps", V: []string{"download", "convert", "prepare-metadata", "upload-checkpoint"}})
		}
		return true, emit(ctx, compactRecord(fields, "kind", "model", "source", "source_profiles", "conversion", "status", "changed"))
	}
	layout, problem := home.Open(ctx.Cfg.Home)
	if problem != nil {
		return true, problem
	}
	if ctx.Inv.Value("--idempotency-key") == "" {
		key, retryOf, problem := ingestRunKey(layout, script, rentalID)
		if problem != nil {
			return true, problem
		}
		ctx.Inv.Values["--idempotency-key"] = []string{key}
		if retryOf != "" {
			ctx.Inv.Values["--retry"] = []string{retryOf}
		}
	}
	work, problem := scratch.Temp(layout.Tmp, "model-ingestion-")
	if problem != nil {
		return true, problem
	}
	defer work.Release()
	path := filepath.Join(work.Path, "upload_model.py")
	if err := os.WriteFile(path, script, 0600); err != nil {
		return true, exit.Internalf("cannot prepare model ingestion: %s", err)
	}
	// Run capture owns the exact script and SDK. The same native receipts survive
	// edited callers, cancellation and retry; the destination is the sole grant.
	ctx.Inv.Args = []string{path}
	ctx.Inv.Values["--allow-publish"] = []string{destination.String()}
	return true, handleRunExecute(ctx)
}

func profileSlots(profiles []string) map[string]string {
	if len(profiles) == 0 {
		return map[string]string{"model": ""}
	}
	slots := make(map[string]string, len(profiles))
	for i, profile := range profiles {
		slots[fmt.Sprintf("part%d", i)] = profile
	}
	return slots
}

// narrowSourceToProfiles keeps only the members the reviewed profiles name; a single
// carrier is already exact.
func narrowSourceToProfiles(ctx *Context, source publishSource, profiles []string) (publishSource, *exit.Error) {
	if len(source.Resolution.Files) <= 1 {
		return source, nil
	}
	return narrowPublishSource(ctx, source, profiles)
}

// ingestRunKey names this ingest on this rental; records.ResumableRun walks its history.
func ingestRunKey(layout home.Layout, script []byte, rentalID string) (string, string, *exit.Error) {
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return "", "", problem
	}
	defer store.Close()
	digest := sha256.Sum256(append(append([]byte(nil), script...), "\x00"+rentalID...))
	return store.ResumableRun("model-upload-" + hex.EncodeToString(digest[:]))
}

func quote(value string) string { raw, _ := json.Marshal(value); return string(raw) }

func scriptHeader(floor string) string {
	return fmt.Sprintf(`# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=%s,<1", "tensorfs>=0.3.51,<0.4"]
# ///
`, floor)
}

func recipeUploadScript(repository, revision, destination string) []byte {
	return []byte(scriptHeader(ingestRuntimeFloor) + fmt.Sprintf(`from cozy_runtime.author.sources import ingest_huggingface
from cozy_runtime.author.publication import upload_checkpoint

async def main() -> dict[str, str]:
    model = await ingest_huggingface(%s, revision=%s)
    checkpoint = await upload_checkpoint(model, destination=%s)
    return {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
`, quote(repository), quote(revision), quote(destination)))
}

// genericUploadScript downloads exactly the reviewed carriers, converts them with the
// profiles the owner-side header preflight decided, and uploads one checkpoint.
func genericUploadScript(kind modelsource.Kind, pinned modelsource.Source, profiles []string,
	destination string,
) []byte {
	download := ""
	switch {
	case kind == modelsource.Civitai:
		download = fmt.Sprintf("download_civitai(%d)", pinned.VersionID)
	case pinned.Member != "":
		download = fmt.Sprintf("download_huggingface(%s, revision=%s, carriers=(%s,))",
			quote(pinned.Org+"/"+pinned.Repo), quote(pinned.Revision), quote(pinned.Member))
	default:
		download = fmt.Sprintf("download_huggingface(%s, revision=%s, profiles=%s)",
			quote(pinned.Org+"/"+pinned.Repo), quote(pinned.Revision), pythonTuple(profiles))
	}
	floor, convert := ingestRuntimeFloor, "profile="+quote(profiles[0])
	if len(profiles) > 1 {
		floor, convert = composedIngestRuntimeFloor, "profiles="+pythonTuple(profiles)
	}
	return []byte(scriptHeader(floor) + fmt.Sprintf(`from cozy_runtime.author.sources import convert_cozytensors, download_civitai, download_huggingface
from cozy_runtime.author.publication import upload_checkpoint

async def main() -> dict[str, str]:
    source = await %s
    model = await convert_cozytensors(source, %s)
    checkpoint = await upload_checkpoint(model, destination=%s)
    return {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
`, download, convert, quote(destination)))
}

func pythonTuple(values []string) string {
	quoted := make([]string, len(values))
	for i, value := range values {
		quoted[i] = quote(value)
	}
	return "(" + strings.Join(quoted, ", ") + ",)"
}
