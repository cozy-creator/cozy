package cli

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

type localPreparedSource struct {
	productionManifest
	plan *tfs.SourcePlan
}

func runLocalModelTransfer(ctx *Context, runCtx context.Context, plan modelproduction.Plan,
	_ publishSource,
) *exit.Error {
	_, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	planBytes, err := plan.Bytes()
	if err != nil {
		return exit.Internalf("cannot encode model transfer plan: %s", err)
	}
	planDigest, err := plan.Digest()
	if err != nil {
		return exit.Internalf("cannot digest model transfer plan: %s", err)
	}
	operation, replay, problem := store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: planDigest, Plan: planBytes})
	if problem != nil {
		return problem
	}
	if modelProductionSettled(operation.State) {
		return emitCompletedProduction(ctx, store, plan, &operation, true)
	}
	if operation.State == "cleanup_pending" {
		settled, settleProblem := settleProductionUpload(store, plan, &operation)
		if settleProblem != nil {
			return settleProblem
		}
		return emitCompletedProduction(ctx, store, plan, settled, replay)
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	ctx.Daemon = state
	local, problem := dial(ctx)
	if problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	if problem = productionCancellation(runCtx, store, operation.ID); problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	prepared, problem := prepareLocalTransferSources(runCtx, ctx, store, plan)
	if problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	if plan.Job == nil {
		return runLocalPassThrough(ctx, runCtx, local, store, plan, prepared, replay)
	}
	return runLocalProducer(ctx, runCtx, local, store, plan, prepared, replay)
}

func prepareLocalTransferSources(runCtx context.Context, ctx *Context, store *records.Store,
	plan modelproduction.Plan,
) (map[string]localPreparedSource, *exit.Error) {
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return nil, problem
	}
	if current.State == "accepted" {
		if problem := store.AdvanceModelProduction(plan.ID(), "accepted", "source_preparing", 0, ""); problem != nil {
			return nil, problem
		}
	}
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return nil, problem
	}
	if err := os.MkdirAll(layout.Transfer, 0o700); err != nil {
		return nil, exit.Internalf("cannot create model transfer root: %s", err)
	}
	slots := plan.SourceProfiles
	if plan.Job == nil {
		slots = map[string]string{"model": ""}
	}
	if strings.HasPrefix(plan.Source, "local/") {
		aliasName := strings.TrimPrefix(plan.Source, "local/")
		alias, problem := tool.ResolveLocal(aliasName)
		if problem != nil {
			return nil, problem
		}
		evidence, problem := tool.CheckpointEvidence("local", aliasName, alias.ManifestDigest,
			filepath.Join(layout.Transfer, "model-transfer-local-evidence.jsonl"))
		if problem != nil {
			return nil, problem
		}
		out := map[string]localPreparedSource{}
		for slot := range slots {
			out[slot] = localPreparedSource{productionManifest: productionManifest{
				ID: alias.ManifestDigest, Length: alias.ManifestLength,
				Evidence: base64.StdEncoding.EncodeToString(evidence)}}
		}
		return finishLocalSourceState(store, plan, out)
	}
	if catalogModelSpelling(plan.Source) {
		fetch := &transfer.Fetch{Tool: tool, Hub: client(ctx), Spec: plan.Source,
			Lane: plan.InputLane, Progress: progress(ctx)}
		hctx, cancel := hub.LongContext()
		defer cancel()
		row, problem := fetch.Resolve(hctx)
		if problem != nil {
			return nil, problem
		}
		if row.ManifestID != plan.SourceSelection {
			return nil, exit.Named(exit.Conflict, "model_transfer.source_changed",
				"Tensorhub source resolved to a different Manifest than the accepted plan")
		}
		fetch.Scratch = scratch(layout, row.ManifestID)
		fetched, problem := fetch.Acquire(hctx, row)
		if problem != nil {
			return nil, problem
		}
		out := map[string]localPreparedSource{}
		for slot := range slots {
			out[slot] = localPreparedSource{productionManifest: productionManifest{
				ID: fetched.ManifestID, Length: fetched.ManifestLength,
				Evidence: base64.StdEncoding.EncodeToString(row.CheckpointEvidence)}}
		}
		return finishLocalSourceState(store, plan, out)
	}

	cwd, err := os.Getwd()
	if err != nil {
		return nil, exit.Internalf("cannot resolve current directory: %s", err)
	}
	parsed, problem := modelsource.Parse(plan.Source, cwd)
	if problem != nil {
		return nil, problem
	}
	root := filepath.Join(layout.Transfer, "model-transfer-"+strings.TrimPrefix(plan.ID(), "modelupload-")[:16])
	if err := os.MkdirAll(root, 0o700); err != nil {
		return nil, exit.Internalf("cannot create model transfer staging: %s", err)
	}
	defer os.RemoveAll(root)
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
	carriers := importCarriers(headerFiles, parsed.Kind != modelsource.LocalFile)
	out := map[string]localPreparedSource{}
	ordered := sortedMapKeys(slots)
	for _, slot := range ordered {
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
		out[slot] = localPreparedSource{productionManifest: productionManifest{
			ID: manifestID, Length: info.Size()}, plan: &copyPlan}
	}
	return finishLocalSourceState(store, plan, out)
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
	carriers := importCarriers(files, labelled)
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
		for _, member := range planMembers(plan, files) {
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

func finishLocalSourceState(store *records.Store, plan modelproduction.Plan,
	prepared map[string]localPreparedSource,
) (map[string]localPreparedSource, *exit.Error) {
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return nil, problem
	}
	if current.State == "source_preparing" {
		if problem := store.AdvanceModelProduction(plan.ID(), "source_preparing", "source_prepared", 0, ""); problem != nil {
			return nil, problem
		}
	}
	return prepared, nil
}

func runLocalProducer(ctx *Context, runCtx context.Context, local *localclient.Client,
	store *records.Store, plan modelproduction.Plan, prepared map[string]localPreparedSource,
	replay bool,
) *exit.Error {
	pin := *plan.Job
	stepName := pin.Function
	row, problem := store.BeginModelProductionStep(records.ModelProductionStep{
		OperationID: plan.ID(), StepIndex: 0, StepName: stepName, State: "pending"})
	if problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	if row.State != "completed" {
		models := make([]orchestrator.ModelRef, 0, len(prepared))
		for slot, source := range prepared {
			models = append(models, orchestrator.ModelRef{Package: pin.Package, Slot: slot,
				Model: plan.ID() + "/" + slot, Manifest: source.ID,
				ManifestLength: source.Length})
		}
		sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
		org := "local"
		if plan.Instruction.Kind == "model-upload" {
			org = strings.Split(plan.Destination, "/")[0]
		}
		handle, problem := local.SubmitJob(api.JobSubmission{Package: pin.Package,
			Function: pin.Function, Input: []byte("{}"), InstallID: pin.InstallID,
			Release: pin.Release, ReleaseDigest: pin.ReleaseDigest, Models: models, Org: org,
			ProductionOperationID: plan.ID(), ProductionStep: stepName},
			productionRequestKey(plan.ID(), stepName))
		if problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
		progress := NewModelProductionProgress(ctx.Err, !ctx.Mode().JSON, 1)
		if problem := waitProductionJob(runCtx, local, store, plan.ID(), handle.JobID,
			progress, 0, stepName); problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
	}
	artifacts, problem := waitStepArtifacts(runCtx, store, plan.ID(), stepName, plan.OutputNames())
	if problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	for _, artifact := range artifacts {
		if problem := finalizeLocalArtifact(runCtx, ctx, store, plan, artifact); problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
	}
	if problem := store.SetModelProductionStepRequest(plan.ID(), 0, stepName,
		artifacts[0].RequestID, "completed"); problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	if problem := advanceProductionStep(store, plan.ID(), "", 1, true); problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	return finishLocalTransfer(ctx, runCtx, store, plan, replay)
}

func runLocalPassThrough(ctx *Context, runCtx context.Context, _ *localclient.Client,
	store *records.Store, plan modelproduction.Plan, prepared map[string]localPreparedSource,
	replay bool,
) *exit.Error {
	source := prepared["model"]
	tool, layout, problem := localTensorFS(ctx)
	if problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	if source.plan != nil {
		temporary := "transfer-" + shortTransferID(plan.ID())
		observed, problem := tool.ObserveLocal(temporary, filepath.Join(layout.Transfer, "temp-observed.jsonl"))
		if problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
		if problem := tool.InstallLocal(source.plan.Session, temporary, plan.SourceSelection,
			observed, plan.Source, plan.SourceLicense); problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
		alias, problem := tool.ResolveLocal(temporary)
		if problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
		evidence, problem := tool.CheckpointEvidence("local", temporary, alias.ManifestDigest,
			filepath.Join(layout.Transfer, "temp-evidence.jsonl"))
		if problem != nil {
			return failProduction(store, plan.ID(), problem)
		}
		source.productionManifest = productionManifest{ID: alias.ManifestDigest,
			Length: alias.ManifestLength, Evidence: base64.StdEncoding.EncodeToString(evidence)}
		defer tool.RemoveLocal(temporary, alias.RepositoryDigest)
	}
	if problem := recordPassThroughArtifact(store, plan, source.productionManifest); problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	artifact := records.ModelProductionArtifact{OperationID: plan.ID(), StepName: "pass-through",
		OutputSlot: "model", ManifestID: source.ID, ManifestLength: source.Length,
		CheckpointEvidence: mustDecodeEvidence(source.Evidence)}
	if problem := finalizeLocalArtifact(runCtx, ctx, store, plan, artifact); problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	if problem := advanceProductionStep(store, plan.ID(), "", 1, true); problem != nil {
		return failProduction(store, plan.ID(), problem)
	}
	return finishLocalTransfer(ctx, runCtx, store, plan, replay)
}

func recordPassThroughArtifact(store *records.Store, plan modelproduction.Plan,
	manifest productionManifest,
) *exit.Error {
	if _, problem := store.BeginModelProductionStep(records.ModelProductionStep{
		OperationID: plan.ID(), StepIndex: 0, StepName: "pass-through", State: "pending"}); problem != nil {
		return problem
	}
	receipt := []byte(`{"kind":"builtin-pass-through"}`)
	digest := sha256.Sum256(receipt)
	artifact := records.ModelProductionArtifact{OperationID: plan.ID(), StepName: "pass-through",
		OutputSlot: "model", RequestID: "builtin-" + shortTransferID(plan.ID()), Attempt: 1,
		InvocationDigest: "builtin-pass-through/1", TransactionID: plan.ID() + "/model",
		WriterGeneration: 1, ReceiptDigest: "sha256:" + hex.EncodeToString(digest[:]),
		Receipt: receipt, ManifestID: manifest.ID, ManifestLength: manifest.Length,
		CheckpointEvidence: mustDecodeEvidence(manifest.Evidence)}
	return store.RecordModelProductionArtifact(artifact, nil)
}

func finalizeLocalArtifact(runCtx context.Context, ctx *Context, store *records.Store,
	plan modelproduction.Plan, artifact records.ModelProductionArtifact,
) *exit.Error {
	if artifact.PublicationID != "" {
		return nil
	}
	if plan.Instruction.Kind == "model-download" {
		tool, layout, problem := localTensorFS(ctx)
		if problem != nil {
			return problem
		}
		name := strings.TrimPrefix(plan.Destination, "local/")
		observed, problem := tool.ObserveLocal(name, filepath.Join(layout.Transfer, "local-observed.jsonl"))
		if problem != nil {
			return problem
		}
		evidencePath := filepath.Join(layout.Transfer, "local-evidence-"+artifact.OutputSlot+".json")
		if err := os.WriteFile(evidencePath, artifact.CheckpointEvidence, 0o600); err != nil {
			return exit.Internalf("cannot stage local checkpoint evidence: %s", err)
		}
		defer os.Remove(evidencePath)
		alias, problem := tool.ReplaceLocal(name, localFinalizationSelection(plan.ID(), artifact.OutputSlot),
			observed, artifact.ManifestID, artifact.ManifestLength, evidencePath)
		if problem != nil {
			return problem
		}
		if alias.ManifestDigest != artifact.ManifestID {
			return exit.Named(exit.Conflict, "model_download.local_replace_changed",
				"TensorFS retained a different Manifest under %s", plan.Destination)
		}
		return store.MarkModelProductionArtifactPublished(plan.ID(), artifact.StepName,
			artifact.OutputSlot, "local:"+alias.RepositoryDigest)
	}
	tool, _, layout, problem := tooling(ctx)
	if problem != nil {
		return problem
	}
	ref, problem := hub.ParseRef(plan.Destination)
	if problem != nil {
		return problem
	}
	publicationClient, problem := ownedPublication(ctx, ref)
	if problem != nil {
		return problem
	}
	upload := &transfer.Upload{Tool: tool, Hub: publicationClient, Ref: ref,
		ManifestID: artifact.ManifestID, CheckpointEvidence: artifact.CheckpointEvidence,
		Session: productionPublicationOperation(plan.ID(), artifact.StepName, artifact.OutputSlot),
		Reason:  "cozy model upload " + plan.Source + " " + plan.Destination,
		Scratch: scratch(layout, artifact.ManifestID), Progress: progress(ctx)}
	hctx, cancel := productionHubContext(runCtx)
	defer cancel()
	result, problem := upload.Run(hctx)
	if problem != nil {
		return problem
	}
	contract, ok := productionContract(plan, artifact.StepName+"."+artifact.OutputSlot)
	if plan.Job != nil && (!ok || result.TopologyDigest != contract.TopologyDigest ||
		!sameStrings(result.EncodingSet, contract.Encodings)) {
		return exit.Named(exit.Conflict, "model_producer.contract_mismatch",
			"Tensorhub-derived contract for output %s does not match the producer descriptor",
			artifact.OutputSlot)
	}
	return store.MarkModelProductionArtifactPublished(plan.ID(), artifact.StepName,
		artifact.OutputSlot, result.PublishID)
}

func finishLocalTransfer(ctx *Context, runCtx context.Context, store *records.Store,
	plan modelproduction.Plan, replay bool,
) *exit.Error {
	progress := NewModelProductionProgress(ctx.Err, !ctx.Mode().JSON, 1)
	if problem := finishProductionOutputs(runCtx, store, plan, progress); problem != nil {
		return problem
	}
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return problem
	}
	settled, problem := settleProductionUpload(store, plan, current)
	if problem != nil {
		return problem
	}
	return emitCompletedProduction(ctx, store, plan, settled, replay)
}

func localFinalizationSelection(operationID, slot string) string {
	sum := sha256.Sum256([]byte("cozy-local-model-finalization/1\x00" + operationID + "\x00" + slot))
	return "sha256:" + hex.EncodeToString(sum[:])
}

func shortTransferID(id string) string {
	sum := sha256.Sum256([]byte(id))
	return hex.EncodeToString(sum[:8])
}

func mustDecodeEvidence(value string) []byte {
	decoded, _ := base64.StdEncoding.DecodeString(value)
	return decoded
}

func importCarriers(files []modelsource.StagedFile, labelled bool) []tfs.SourceCarrier {
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

func planMembers(plan tfs.SourcePlan, staged []modelsource.StagedFile) []string {
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
