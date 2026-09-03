package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

type localPreparedSource struct {
	manifestID     string
	manifestLength int64
	plan           *tfs.SourcePlan
}

// prepareLocalTransferSources stages the source under root — the request's own
// `tmp/<request-id>/`, claimed and released by the caller — and leaves its CozyTensors in
// the CAS. Nothing here outlives the caller's Release.
func prepareLocalTransferSources(runCtx context.Context, ctx *Context, root string,
	intent records.ModelTransferIntent, slots map[string]string,
) (map[string]localPreparedSource, *exit.Error) {
	tool, _, problem := localTensorFS(ctx)
	if problem != nil {
		return nil, problem
	}
	if strings.HasPrefix(intent.Source, "local/") {
		name := strings.TrimPrefix(intent.Source, "local/")
		alias, problem := tool.ResolveLocal(name)
		if problem != nil {
			return nil, problem
		}
		return samePreparedSource(slots, localPreparedSource{manifestID: alias.ManifestDigest,
			manifestLength: alias.ManifestLength}), nil
	}
	if catalogModelSpelling(intent.Source) {
		fetch := &transfer.Fetch{Tool: tool, Hub: client(ctx), Spec: intent.Source,
			Lane: intent.InputLane, Progress: progress(ctx)}
		hctx, cancel := hub.LongContext()
		defer cancel()
		row, problem := fetch.Resolve(hctx)
		if problem != nil {
			return nil, problem
		}
		if row.ManifestID != intent.SourceSelection {
			return nil, exit.Named(exit.Conflict, "model_transfer.source_changed",
				"Tensorhub source resolved to a different Manifest than the accepted request")
		}
		fetch.Scratch = filepath.Join(root, "fetch")
		fetched, problem := fetch.Acquire(hctx, row)
		if problem != nil {
			return nil, problem
		}
		return samePreparedSource(slots, localPreparedSource{manifestID: fetched.ManifestID,
			manifestLength: fetched.ManifestLength}), nil
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, exit.Internalf("cannot resolve current directory: %s", err)
	}
	parsed, problem := modelsource.Parse(intent.Source, cwd)
	if problem != nil {
		return nil, problem
	}
	selected, headerFiles, resolver, problem := stageLocalTransferHeaders(runCtx, ctx, parsed, root)
	if problem != nil {
		return nil, problem
	}
	headerPlans, members, problem := planLocalSourceProfiles(tool, ctx, slots, headerFiles,
		parsed.Kind != modelsource.LocalFile, filepath.Join(root, "header-plans"))
	if problem != nil {
		return nil, problem
	}
	if parsed.Kind != modelsource.LocalFile {
		selected, problem = selected.Select(members)
		if problem != nil {
			return nil, problem
		}
		headerFiles, problem = resolver.Stage(runCtx, selected, filepath.Join(root, "files"), false,
			progress(ctx))
		if problem != nil {
			return nil, problem
		}
	}
	carriers := sourceCarriers(headerFiles, parsed.Kind != modelsource.LocalFile)
	out := map[string]localPreparedSource{}
	for _, slot := range sortedMapKeys(slots) {
		path := filepath.Join(root, "full-plan-"+slot+".json")
		var full tfs.SourcePlan
		if profile := slots[slot]; profile != "" {
			full, problem = tool.PlanSourceProfile(ctx.Cfg.TensorFSRegistry, profile, carriers, path)
		} else {
			full, problem = tool.PlanSource(ctx.Cfg.TensorFSRegistry, carriers, path)
		}
		if problem != nil {
			return nil, problem
		}
		if !headerPlans[slot].SameSelection(full) {
			return nil, exit.Named(exit.Conflict, "model_source_plan_changed",
				"downloaded source no longer matches reviewed headers for input %s", slot)
		}
		manifestID, problem := tool.RunSource(runCtx, path)
		if problem != nil {
			return nil, problem
		}
		manifestPath := filepath.Join(root, "manifest-"+slot)
		if problem := tool.Manifest(manifestID, manifestPath); problem != nil {
			return nil, problem
		}
		info, err := os.Stat(manifestPath)
		if err != nil || info.Size() <= 0 {
			return nil, exit.Internalf("prepared source Manifest %s is unreadable", manifestID)
		}
		copyPlan := full
		out[slot] = localPreparedSource{manifestID: manifestID,
			manifestLength: info.Size(), plan: &copyPlan}
	}
	return out, nil
}

func samePreparedSource(slots map[string]string, source localPreparedSource) map[string]localPreparedSource {
	out := make(map[string]localPreparedSource, len(slots))
	for slot := range slots {
		out[slot] = source
	}
	return out
}

func stageLocalTransferHeaders(runCtx context.Context, ctx *Context, source modelsource.Source,
	root string,
) (modelsource.Plan, []modelsource.StagedFile, *modelsource.Resolver, *exit.Error) {
	if source.Kind == modelsource.LocalFile {
		plan, staged, problem := modelsource.StageLocal(runCtx, source, filepath.Join(root, "files"))
		return plan, []modelsource.StagedFile{staged}, nil, problem
	}
	token := ctx.Cfg.HuggingFaceToken
	if source.Kind == modelsource.Civitai {
		token = ctx.Cfg.CivitaiToken
	}
	resolver, problem := modelsource.NewResolver(source.Kind, token)
	if problem != nil {
		return modelsource.Plan{}, nil, nil, problem
	}
	plan, problem := resolver.Resolve(runCtx, source)
	if problem != nil {
		return modelsource.Plan{}, nil, nil, problem
	}
	staged, problem := resolver.Stage(runCtx, plan, filepath.Join(root, "headers"), true, progress(ctx))
	return plan, staged, resolver, problem
}

func planLocalSourceProfiles(tool *tfs.Tool, ctx *Context, slots map[string]string,
	files []modelsource.StagedFile, labelled bool, root string,
) (map[string]tfs.SourcePlan, []string, *exit.Error) {
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, nil, exit.Internalf("cannot create source-plan staging: %s", err)
	}
	carriers := sourceCarriers(files, labelled)
	plans := map[string]tfs.SourcePlan{}
	members := map[string]bool{}
	for _, slot := range sortedMapKeys(slots) {
		path := filepath.Join(root, slot+".json")
		var plan tfs.SourcePlan
		var problem *exit.Error
		if profile := slots[slot]; profile != "" {
			plan, problem = tool.PlanSourceProfile(ctx.Cfg.TensorFSRegistry, profile, carriers, path)
		} else {
			plan, problem = tool.PlanSource(ctx.Cfg.TensorFSRegistry, carriers, path)
		}
		if problem != nil {
			return nil, nil, problem
		}
		if problem := tool.PreviewSource(path); problem != nil {
			return nil, nil, problem
		}
		plans[slot] = plan
		for _, member := range sourcePlanMembers(plan, files) {
			members[member] = true
		}
	}
	selected := make([]string, 0, len(members))
	for member := range members {
		selected = append(selected, member)
	}
	sort.Strings(selected)
	return plans, selected, nil
}

func sortedMapKeys(values map[string]string) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

func sourceCarriers(files []modelsource.StagedFile, labelled bool) []tfs.SourceCarrier {
	carriers := make([]tfs.SourceCarrier, 0, len(files))
	for _, file := range files {
		if !file.Carrier {
			continue
		}
		carrier := tfs.SourceCarrier{Path: file.Path}
		if labelled {
			carrier.Member = file.Member
		}
		carriers = append(carriers, carrier)
	}
	return carriers
}

func sourcePlanMembers(plan tfs.SourcePlan, staged []modelsource.StagedFile) []string {
	byPath := make(map[string]string, len(staged))
	for _, file := range staged {
		byPath[file.Path] = file.Member
	}
	members := make([]string, 0, len(plan.Sources))
	for _, source := range plan.Sources {
		member := source.SourceMember
		if member == "" {
			member = byPath[source.Path]
		}
		members = append(members, member)
	}
	return members
}

func localFinalizationSelection(requestID, slot string) string {
	sum := sha256.Sum256([]byte("cozy-local-model-finalization/1\x00" + requestID + "\x00" + slot))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortTransferID(id string) string {
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:8])
}
