package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// Floors for the generated ingest script: Runtime 0.18.67 and TensorFS 0.3.73 stream the
// source through conversion into the publication within the rental's free disk.
const (
	ingestRuntimeFloor  = "0.18.67"
	ingestTensorFSFloor = "0.3.73"
)

// nativeModelUpload runs a provider ingest as an ordinary script on a machine, rented or
// this computer's: the machine downloads the source, converts it to CozyTensors with
// reviewed TensorFS profiles and uploads the checkpoint under the run's capability. A Runtime recipe, where one exists, adds its
// model-owned metadata; without one the profiles the source headers match are used.
func nativeModelUpload(ctx *Context) (bool, *exit.Error) {
	cwd, err := os.Getwd()
	if err != nil {
		return true, exit.Internalf("cannot resolve working directory: %s", err)
	}
	parsed, problem := modelsource.Parse(ctx.Inv.Args[0], cwd)
	if problem != nil && !catalogModelSpelling(strings.TrimSpace(ctx.Inv.Args[0])) {
		return true, problem
	}
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
	if taken, problem := machineUpload(ctx, rentalID, parsed, destination.String()); taken || problem != nil {
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
		key, retryOf, problem := ingestRunKey(layout, ctx.Cfg.HubURL, script, rentalID)
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
	// edited callers, cancellation and retry.
	ctx.Inv.Args = []string{path}
	// The script already names its profiles; the run must not reread them as job
	// model-slot bindings (slot=profile), which refused every profiled ingest. The
	// ingest consumed --lane too; the run has nothing left to read from either.
	delete(ctx.Inv.Values, "--source-profile")
	delete(ctx.Inv.Values, "--lane")
	return true, handleRunExecute(ctx)
}

// machineUpload hands the upload to a cozy.machine.v1 machine that takes uploads, as a warm
// run of the pinned source with the destination: the machine makes it with TensorFS profiles
// and puts the checkpoint there itself. false: the machine, or a rental the fleet has yet to
// choose, takes the generated ingest script instead.
func machineUpload(ctx *Context, rentalID string, parsed modelsource.Source, destination string) (bool, *exit.Error) {
	machine, name := rentalID, ctx.Inv.Value("--rental")
	if machine == "" {
		if rentalRequested(ctx) {
			return false, nil
		}
		machine, name = machines.Local, machines.Local
	}
	if machine != machines.Local {
		// Under a daemon that predates cozy.machine.v1 the upload is this command's own warm run.
		ep, problem := foregroundRental(ctx, name)
		if problem != nil || ep != nil {
			if problem != nil {
				return true, problem
			}
			selection, problem := uploadSelection(ctx, parsed, destination)
			if problem != nil {
				return true, problem
			}
			return foregroundInstall(ctx, ep, name, selection)
		}
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return true, problem
	}
	ctx.Daemon = state
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return true, problem
	}
	if status, problem := client.MachineStatus(machine); problem != nil || !slices.Contains(status.Capabilities, "upload/1") {
		return false, nil
	}
	selection, problem := uploadSelection(ctx, parsed, destination)
	if problem != nil {
		return true, problem
	}
	return true, enqueueRentalInstall(ctx, name, selection, ctx.Inv.Bool("--await"))
}

// uploadSelection is the installation an upload is: the pinned provider source, made into the
// destination.
func uploadSelection(ctx *Context, parsed modelsource.Source, destination string) (records.RentalInstallSelection, *exit.Error) {
	source, problem := pinnedProviderSource(ctx, parsed.Canonical)
	if problem != nil {
		return records.RentalInstallSelection{}, problem
	}
	model := records.ModelRef{Slot: "model", Source: source, Profiles: ctx.Inv.Values["--source-profile"]}
	return records.RentalInstallSelection{Models: []records.ModelRef{model}, Destination: destination}, nil
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
	named := nativeIngestPlan{source: source, profiles: profiles, recipe: recipe}
	asIs := slices.Contains(profiles, tfs.AsIsProfile)
	if len(profiles) > 0 && !asIs {
		narrowed, problem := narrowSourceToProfiles(ctx, source, profiles)
		if hostUnaware(problem) {
			return deferToRental(ctx, named, problem), nil
		}
		if problem != nil {
			return nativeIngestPlan{}, problem
		}
		source, named.source = narrowed, narrowed
	}
	runCtx, cancel := hub.LongContext()
	defer cancel()
	conversion, problem := preflightConversionPlan(runCtx, ctx, source, profileSlots(profiles))
	if hostUnaware(problem) {
		return deferToRental(ctx, named, problem), nil
	}
	if problem != nil {
		if tfs.Refused(problem, "AMBIGUOUS_CLASSIFICATION") {
			problem = problem.WithRemedy("choose the reviewed profile(s) with --source-profile; profiles over different files compose one model")
		}
		return nativeIngestPlan{}, problem
	}
	if !conversion.decided() {
		return nativeIngestPlan{}, exit.Named(exit.Unavailable, "model_source.preflight_unavailable", "source headers must be inspected before ingestion: %s", conversion.Undecided)
	}
	// Sized by exactly the members this host's plan uses. The rental's TensorFS, which may
	// know more or less than this host's, selects and plans on its own registry: it is told
	// only the profiles the owner named.
	if source, problem = selectPublishMembers(source, conversion.Members); problem != nil {
		return nativeIngestPlan{}, problem
	}
	return nativeIngestPlan{source: source, profiles: profiles, recipe: recipe}, nil
}

// hostUnaware is this host's TensorFS not recognizing a profile or fingerprint. The host
// and the rental install TensorFS independently, so the rental's may know what this one
// does not: that is the rental's decision, never a veto here.
func hostUnaware(problem *exit.Error) bool {
	return tfs.Refused(problem, "UNREGISTERED_FINGERPRINT")
}

// deferToRental keeps the named profiles, or none, and whatever narrowing already happened;
// the rental's TensorFS selects, plans and refuses on its own registry.
func deferToRental(ctx *Context, plan nativeIngestPlan, problem *exit.Error) nativeIngestPlan {
	version := "tfs (unavailable)"
	if tool, _, opened := localTensorFS(ctx); opened == nil {
		version = tool.Version()
	}
	said := problem.Message
	if at := strings.Index(said, "REFUSED "); at >= 0 {
		said = said[at:]
	}
	what := strings.Join(plan.profiles, ", ")
	if what == "" {
		what = "this source"
	}
	fmt.Fprintf(ctx.Err, "note: this host's %s does not recognize %s (%s); the rental's TensorFS decides\n",
		version, what, said)
	return plan
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

// narrowSourceToProfiles keeps only the members the reviewed profiles name. A single carrier
// is already exact, and a Civitai primary with its companions is narrowed by its header plan.
func narrowSourceToProfiles(ctx *Context, source publishSource, profiles []string) (publishSource, *exit.Error) {
	if len(source.Resolution.Files) <= 1 || slices.ContainsFunc(source.Resolution.Files, func(file modelsource.File) bool { return file.Companion }) {
		return source, nil
	}
	return narrowPublishSource(ctx, source, profiles)
}

// ingestRunKey names this ingest wherever it runs: a live ingest is reattached and a stopped
// one resumed on its own rental. A completed one runs again: the pod's Runtime, not this
// script, decides the result (its version and the source's configs are in its memo key),
// and its memo answers when nothing changed.
func ingestRunKey(layout home.Layout, origin string, script []byte, rentalID string) (string, string, *exit.Error) {
	store, problem := records.Open(layout.DB)
	if problem != nil {
		return "", "", problem
	}
	defer store.Close()
	digest := sha256.Sum256(script)
	return store.ResumableRun("model-upload-"+hex.EncodeToString(digest[:]), origin, rentalID, false)
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
	if len(quoted) == 0 {
		return "()"
	}
	return "(" + strings.Join(quoted, ", ") + ",)"
}
