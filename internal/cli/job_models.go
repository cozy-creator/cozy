package cli

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

func providerModelSource(raw string) (string, bool, *exit.Error) {
	if !strings.Contains(raw, "://") {
		return "", false, nil
	}
	cwd, err := os.Getwd()
	if err != nil {
		return "", false, exit.Internalf("cannot resolve the current directory: %s", err)
	}
	source, problem := modelsource.Parse(raw, cwd)
	if problem != nil {
		return "", false, problem
	}
	if source.Kind != modelsource.HuggingFace && source.Kind != modelsource.Civitai {
		return "", false, exit.Usagef("model input %q is not a Tensorhub, Hugging Face, or Civitai reference", raw)
	}
	return source.Canonical, true, nil
}

// A provider selection is acquired once, with an explicit reviewed profile for
// each model slot. The existing durable source owner holds one source inventory;
// it must never silently widen a selection or replace independently bound inputs.
func resolveJobModelInputs(ctx *Context, target Target, job *launch.Entrypoint,
	overrides map[string]string,
) (string, map[string]string, []orchestrator.ModelRef, *exit.Error) {
	profiles, problem := parseSourceProfileFlags(ctx)
	if problem != nil {
		return "", nil, nil, problem
	}
	if len(overrides) == 0 && len(profiles) == 0 {
		models, problem := resolveInvocationModels(ctx, target, job, overrides)
		if problem == nil {
			models, problem = jobManifestInputs(ctx, job, models)
		}
		return "", nil, models, problem
	}
	selected, problem := invocationModelSpecs(ctx, target, job, overrides)
	if problem != nil {
		return "", nil, nil, problem
	}
	source, foreign := "", 0
	for _, spec := range selected {
		canonical, provider, problem := providerModelSource(spec.Ref)
		if problem != nil {
			return "", nil, nil, problem
		}
		if !provider {
			continue
		}
		if source != "" && source != canonical {
			return "", nil, nil, exit.Named(exit.Unavailable, "model_source.multiple_sources_unsupported",
				"one invocation currently prepares one provider source; all foreign model inputs must name the same source")
		}
		source, foreign = canonical, foreign+1
	}
	if foreign == 0 {
		if len(profiles) > 0 {
			return "", nil, nil, exit.Usagef("--source-profile applies only to foreign model inputs")
		}
		models, problem := resolveSelectedInvocationModels(ctx, target, job, selected)
		if problem == nil {
			models, problem = jobManifestInputs(ctx, job, models)
		}
		return "", nil, models, problem
	}
	if foreign != len(job.Models) {
		return "", nil, nil, exit.Named(exit.Unavailable, "model_source.mixed_inputs_unsupported",
			"foreign source preparation currently requires every model slot to name that source")
	}
	if problem := modeltransfer.ValidateProducer(target.Package+"/"+target.Function, job, profiles); problem != nil {
		return "", nil, nil, problem
	}
	return source, profiles, nil, nil
}

// jobModelChoices are a published job's explicit Model choices, unresolved, when it runs on
// a named rental and no slot names a provider source: that machine resolves them (release
// roots), or the daemon does at dispatch for one without. It answers false when the client
// must resolve them itself: choosing a machine to rent reads the ladders.
func jobModelChoices(ctx *Context, target Target, job *launch.Entrypoint, overrides map[string]string,
	trees bool, rental string) ([]orchestrator.ModelRef, bool, *exit.Error) {
	if trees || rental == "" || len(ctx.Inv.Values["--source-profile"]) > 0 {
		return nil, false, nil
	}
	for _, raw := range overrides {
		if strings.Contains(raw, "://") {
			return nil, false, nil
		}
	}
	return modelChoices(ctx, target, job, overrides, nil)
}

func jobOutputDestination(ctx *Context, job *launch.Entrypoint, sub *api.JobSubmission) *exit.Error {
	destination := strings.TrimSpace(ctx.Inv.Value("--upload-to"))
	if destination == "" {
		return nil
	}
	if len(job.WeightsOutputs) == 0 {
		return exit.Usagef("--upload-to requires a job with declared weight outputs")
	}
	ref, problem := hub.ParseRef(destination)
	if problem != nil {
		return problem
	}
	if sub.Org != "" && sub.Org != ref.Org {
		return exit.Usagef("--org must match the --upload-to organization")
	}
	if _, problem := ownedPublication(ctx, ref); problem != nil {
		return problem
	}
	intent := &records.ModelTransferIntent{Kind: "model-upload", Destination: ref.String()}
	for _, output := range job.WeightsOutputs {
		intent.Outputs = append(intent.Outputs, records.ModelTransferOutput{Name: output.OutputID})
	}
	if problem := records.NormalizeModelTransferIntent(intent); problem != nil {
		return problem
	}
	sub.Org, sub.ModelTransfer = ref.Org, intent
	return nil
}

// conversionRunKey names a rented conversion by what it computes and where it publishes,
// as a rented ingest is named: a completed one is a memo hit, a running one reattaches and
// a stopped one resumes on its rental (records.ResumableRun).
func conversionRunKey(ctx *Context, sub api.JobSubmission) (string, string, *exit.Error) {
	models := make([]string, 0, len(sub.Models))
	for _, model := range sub.Models {
		models = append(models, model.Slot+"="+model.Manifest)
	}
	sort.Strings(models)
	identity, err := json.Marshal([]any{sub.Package, sub.Function, sub.Release, sub.InstallID, sub.Org,
		sub.Input, models, sub.ModelTransfer.Destination})
	if err != nil {
		return "", "", exit.Internalf("cannot name the conversion: %s", err)
	}
	digest := sha256.Sum256(identity)
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return "", "", problem
	}
	defer store.Close()
	return store.ResumableRun("conversion-"+hex.EncodeToString(digest[:]), sub.RequestedRental, true)
}

// A job grants exact manifest bytes, whereas a serving binding names a model
// identity. Read the small root through the existing release route, verify its
// selected digest, and retain its actual length; closure bytes never stand in
// for manifest bytes. Runtime continues to own native format interpretation.
func jobManifestInputs(ctx *Context, job *launch.Entrypoint, models []orchestrator.ModelRef) ([]orchestrator.ModelRef, *exit.Error) {
	params := make(map[string]string, len(job.Models))
	for _, slot := range job.Models {
		params[slot.Path] = slot.Param
	}
	for i := range models {
		model := &models[i]
		if model.ManifestLength == 0 {
			ref, problem := hub.ParseRef(model.Model)
			if problem != nil {
				return nil, problem
			}
			hctx, cancel := hub.Context()
			raw, problem := client(ctx).ReleaseManifest(hctx, ref, model.Release, model.Lane)
			if problem == nil && !manifestBytes(raw, model.Manifest) {
				// The lane moved since selection; read the selected checkpoint itself.
				if checkpoint, fallback := client(ctx).CheckpointManifest(hctx, ref, model.Manifest); fallback == nil {
					raw = checkpoint
				}
			}
			cancel()
			if problem != nil {
				return nil, problem
			}
			if !manifestBytes(raw, model.Manifest) {
				return nil, exit.Named(exit.Conflict, "job.model_manifest_changed",
					"Tensorhub serves no bytes for the selected checkpoint %s", model.Manifest)
			}
			model.ManifestLength = int64(len(raw))
		}
		param, ok := params[model.Slot]
		if !ok {
			return nil, exit.Internalf("resolved model slot %s is absent from the job", model.Slot)
		}
		// Preparation names the interface's full binding path; invocation inputs
		// name its parameter. Preserve both before replacing the serving slot key.
		model.BindingPath, model.Slot = model.Slot, param
	}
	return models, nil
}

func manifestBytes(raw []byte, manifest string) bool {
	digest, err := canonical.Spell(canonical.Digest(raw))
	return err == nil && len(raw) > 0 && digest == manifest
}
