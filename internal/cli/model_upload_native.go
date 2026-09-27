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
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/scratch"
)

// Floors for the generated ingest script: Runtime 0.18.49 and TensorFS 0.3.60 stream the
// source through conversion into the publication within the rental's free disk.
const (
	ingestRuntimeFloor  = "0.18.49"
	ingestTensorFSFloor = "0.3.60"
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
	if _, problem := ownedPublication(ctx, destination); problem != nil {
		return true, problem
	}
	plan, problem := planNativeIngest(ctx, cwd, parsed, ctx.Inv.Values["--source-profile"])
	if problem != nil {
		return true, problem
	}
	ctx.ingestBytes = plan.source.Bytes
	var script []byte
	if plan.recipe != nil {
		script = recipeUploadScript(plan.recipe.Repository, plan.recipe.Revision, destination.String())
	} else {
		script = genericUploadScript(parsed.Kind, plan.source.Resolution.Source, plan.profiles, destination.String())
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
	// The script already names its profiles; the run must not reread them as job
	// model-slot bindings (slot=profile), which refused every profiled ingest. The
	// ingest consumed --lane too; the run has nothing left to read from either.
	delete(ctx.Inv.Values, "--source-profile")
	delete(ctx.Inv.Values, "--lane")
	return true, handleRunExecute(ctx)
}

// nativeIngestPlan is one provider source as a rented ingest converts it: pinned, narrowed
// to its reviewed profiles, with the Runtime recipe that owns its metadata when one exists.
// Upload runs it; `cozy rental new --model` sizes the pod's disk from its bytes.
type nativeIngestPlan struct {
	source   publishSource
	profiles []string
	recipe   *launch.ModelIngestionRecipe
}

func planNativeIngest(ctx *Context, cwd string, parsed modelsource.Source, profiles []string) (nativeIngestPlan, *exit.Error) {
	// The provider resolves moving URL spellings to an immutable identity first. No
	// weights are downloaded by this inventory.
	source, problem := resolvePublishSource(ctx, parsed.Canonical, nil)
	if problem != nil {
		return nativeIngestPlan{}, problem
	}
	pinned := source.Resolution.Source
	var recipe *launch.ModelIngestionRecipe
	if parsed.Kind == modelsource.HuggingFace && parsed.Member == "" && len(profiles) == 0 {
		tool, problem := launch.BuiltinOperationsTool(cwd, ctx.Cfg.Home, ctx.Cfg.Tool())
		if problem != nil {
			return nativeIngestPlan{}, problem
		}
		if recipe, problem = tool.ModelIngestionPlan(context.Background(), pinned.Org+"/"+pinned.Repo, pinned.Revision); problem != nil {
			return nativeIngestPlan{}, problem
		}
	}
	if recipe != nil {
		profiles = []string{recipe.Profile}
	}
	if len(profiles) > 0 {
		if source, problem = narrowSourceToProfiles(ctx, source, profiles); problem != nil {
			return nativeIngestPlan{}, problem
		}
	}
	runCtx, cancel := hub.LongContext()
	defer cancel()
	conversion, problem := preflightConversionPlan(runCtx, ctx, source, profileSlots(profiles))
	if problem != nil {
		if strings.Contains(problem.Message, "AMBIGUOUS_CLASSIFICATION") {
			problem = problem.WithRemedy("choose the reviewed profile(s) with --source-profile; profiles over different files compose one model")
		}
		return nativeIngestPlan{}, problem
	}
	if !conversion.decided() {
		return nativeIngestPlan{}, exit.Named(exit.Unavailable, "model_source.preflight_unavailable", "source headers must be inspected before ingestion: %s", conversion.Undecided)
	}
	if len(profiles) == 0 {
		profiles = []string{conversion.Plans["model"].Profile}
		if source, problem = narrowSourceToProfiles(ctx, source, profiles); problem != nil {
			return nativeIngestPlan{}, problem
		}
	}
	return nativeIngestPlan{source: source, profiles: profiles, recipe: recipe}, nil
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

// ingestRunKey names this ingest wherever it runs: a completed ingest is a memo hit on any
// rental, and records.ResumableRun resumes a stopped one on the same rental.
func ingestRunKey(layout home.Layout, script []byte, rentalID string) (string, string, *exit.Error) {
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return "", "", problem
	}
	defer store.Close()
	digest := sha256.Sum256(script)
	return store.ResumableRun("model-upload-"+hex.EncodeToString(digest[:]), rentalID)
}

func quote(value string) string { raw, _ := json.Marshal(value); return string(raw) }

func scriptHeader() string {
	return fmt.Sprintf(`# /// script
# requires-python = ">=3.12"
# dependencies = ["cozy-runtime>=%s,<1", "tensorfs>=%s,<0.4"]
# ///
`, ingestRuntimeFloor, ingestTensorFSFloor)
}

func uploadScript(call string) []byte {
	return []byte(scriptHeader() + fmt.Sprintf(`from cozy_runtime.author.sources import upload_civitai, upload_huggingface

async def main() -> dict[str, str]:
    checkpoint = await %s
    return {"destination": checkpoint.destination, "checkpoint": checkpoint.checkpoint}
`, call))
}

// recipeUploadScript lets the model's reviewed recipe choose carriers, profile and metadata.
func recipeUploadScript(repository, revision, destination string) []byte {
	return uploadScript(fmt.Sprintf("upload_huggingface(%s, revision=%s, destination=%s)",
		quote(repository), quote(revision), quote(destination)))
}

// genericUploadScript uploads exactly the reviewed carriers, converted with the profiles
// the owner-side header preflight decided, as one checkpoint. The rental streams source
// windows through conversion into the publication, so the model need not fit its disk.
func genericUploadScript(kind modelsource.Kind, pinned modelsource.Source, profiles []string,
	destination string,
) []byte {
	var call string
	switch {
	case kind == modelsource.Civitai:
		call = fmt.Sprintf("upload_civitai(%d, profiles=%s, destination=%s)",
			pinned.VersionID, pythonTuple(profiles), quote(destination))
	case pinned.Member != "":
		call = fmt.Sprintf("upload_huggingface(%s, revision=%s, carriers=(%s,), profiles=%s, destination=%s)",
			quote(pinned.Org+"/"+pinned.Repo), quote(pinned.Revision), quote(pinned.Member),
			pythonTuple(profiles), quote(destination))
	default:
		call = fmt.Sprintf("upload_huggingface(%s, revision=%s, profiles=%s, destination=%s)",
			quote(pinned.Org+"/"+pinned.Repo), quote(pinned.Revision), pythonTuple(profiles),
			quote(destination))
	}
	return uploadScript(call)
}

func pythonTuple(values []string) string {
	quoted := make([]string, len(values))
	for i, value := range values {
		quoted[i] = quote(value)
	}
	return "(" + strings.Join(quoted, ", ") + ",)"
}
