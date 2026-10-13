package cli

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"

	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// nativeModelUpload puts a provider source (Hugging Face, Civitai) in a Tensorhub destination:
// the machine, this computer's or --rental's, downloads it, converts it with TensorFS profiles
// and uploads the checkpoint itself.
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
	if ctx.Inv.Value("--lane") != "" {
		return true, exit.Usagef("--lane selects only a Tensorhub model release")
	}
	machine, name := machines.Local, ctx.Inv.Value("--rental")
	if name != "" {
		_, store, problem := rentalStores(ctx)
		if problem != nil {
			return true, problem
		}
		machine, problem = selectedRental(ctx, store, name)
		store.Close()
		if problem != nil {
			return true, problem
		}
	} else {
		name = machines.Local
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
	return true, machineUpload(ctx, machine, name, parsed, destination.String())
}

// machineUpload hands the upload to the machine as a warm run of the pinned source with the
// destination: the machine makes it with TensorFS profiles and puts the checkpoint there.
func machineUpload(ctx *Context, machine, name string, parsed modelsource.Source, destination string) *exit.Error {
	state, _, problem := ensureCalcifer(ctx)
	if problem != nil {
		return problem
	}
	ctx.Calcifer = state
	client, problem := localapi.Open(ctx.Cfg, state)
	if problem != nil {
		return problem
	}
	status, problem := client.MachineStatus(machine)
	if problem != nil {
		return problem
	}
	if !slices.Contains(status.Capabilities, "upload/1") {
		return exit.Named(exit.Unavailable, "model_upload.machine_unsupported", "%s does not take provider uploads; its machine software predates them", name).
			WithRemedy("update the machine software: `cozy machine install` on this computer, or `cozy rental update <rental>` for a rental")
	}
	op, problem := uploadOperation(ctx, parsed, destination)
	if problem != nil {
		return problem
	}
	return runOperation(ctx, name, op, ctx.Inv.Bool("--await"))
}

// uploadOperation is the operation an upload is: the pinned provider source, made into the
// destination.
func uploadOperation(ctx *Context, parsed modelsource.Source, destination string) (records.Operation, *exit.Error) {
	source, problem := pinnedProviderSource(ctx, parsed.Canonical)
	if problem != nil {
		return records.Operation{}, problem
	}
	model := records.ModelRef{Slot: "model", Source: source, Profiles: ctx.Inv.Values["--source-profile"]}
	return records.Operation{Models: []records.ModelRef{model}, Destination: destination}, nil
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
	source, problem := resolvePublishSource(ctx, parsed.Canonical)
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
	runCtx, cancel := hub.Context()
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
