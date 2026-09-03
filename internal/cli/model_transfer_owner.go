package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
	"github.com/cozy-creator/cozy/internal/transfer"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type modelTransferOwner struct {
	cfg   config.Config
	store *records.Store
	log   io.Writer
	auth  *accountauth.Manager
}

func newModelTransferOwner(cfg config.Config, store *records.Store, log io.Writer,
	auth *accountauth.Manager,
) *modelTransferOwner {
	if log == nil {
		log = io.Discard
	}
	return &modelTransferOwner{cfg: cfg, store: store, log: log, auth: auth}
}

func (o *modelTransferOwner) cliContext(intent records.ModelTransferIntent, rental bool) *Context {
	return &Context{Inv: &Invocation{Args: []string{intent.Source, intent.Destination},
		Bools: bools("--rental", rental), Values: values("--lane", intent.InputLane)},
		Out: io.Discard, Err: o.log, Cfg: o.cfg, AccountAuth: o.auth}
}

func (o *modelTransferOwner) requestContext(parent context.Context, requestID string) (
	context.Context, context.CancelFunc,
) {
	ctx, cancel := context.WithCancel(parent)
	go func() {
		ticker := time.NewTicker(500 * time.Millisecond)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				request, problem := o.store.RequestRow(requestID)
				if problem != nil || request == nil || request.State == "canceled" {
					cancel()
					return
				}
			}
		}
	}()
	return ctx, cancel
}

func (o *modelTransferOwner) MaterializeLocal(parent context.Context, requestID string,
	intent records.ModelTransferIntent,
) ([]orchestrator.ModelRef, *exit.Error) {
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	work, problem := o.requestScratch(requestID)
	if problem != nil {
		return nil, problem
	}
	defer work.Release()
	prepared, problem := prepareLocalTransferSources(ctx, o.cliContext(intent, false), work.Path,
		intent, intent.SourceProfiles)
	if problem != nil {
		return nil, problem
	}
	models := make([]orchestrator.ModelRef, 0, len(prepared))
	for slot, source := range prepared {
		models = append(models, orchestrator.ModelRef{Slot: slot,
			Model: requestID + "/" + slot, Manifest: source.manifestID,
			ManifestLength: source.manifestLength})
	}
	sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
	return models, nil
}

func (o *modelTransferOwner) RefreshRemoteSource(parent context.Context,
	intent records.ModelTransferIntent,
) ([]orchestrator.ModelSourceCapability, *exit.Error) {
	ctx := o.cliContext(intent, true)
	resolved, problem := resolvePublishSource(ctx, intent.Source,
		sourceProfileNames(intent.SourceProfiles))
	if problem != nil {
		return nil, problem
	}
	if resolved.Canonical != intent.Source || resolved.Selection != intent.SourceSelection ||
		resolved.License != intent.SourceLicense || resolved.Lane != intent.InputLane ||
		!sameSourceFiles(resolved.Exact, intent.SourceFiles) {
		return nil, exit.Named(exit.Conflict, "model_transfer.source_changed",
			"refreshed provider source no longer matches the accepted request")
	}
	capabilities := make([]orchestrator.ModelSourceCapability, 0, len(resolved.Access))
	for _, access := range resolved.Access {
		provider := map[string]pb.ModelSourceProvider{
			"huggingface": pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_HUGGING_FACE,
			"civitai":     pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_CIVITAI,
		}[access.Provider]
		if provider == pb.ModelSourceProvider_MODEL_SOURCE_PROVIDER_UNSPECIFIED {
			return nil, exit.Named(exit.Validation, "model_transfer.provider_invalid",
				"source capability names unsupported provider %s", access.Provider)
		}
		capabilities = append(capabilities, orchestrator.ModelSourceCapability{
			Member: access.Member, ObjectID: access.ObjectID, Length: access.Length,
			Provider: provider, URL: access.URL, ExpiresAtUnix: access.ExpiresAtUnix})
	}
	_ = parent
	return capabilities, nil
}

func sameSourceFiles(left []modeltransfer.SourceFile,
	right []records.ModelTransferSourceFile,
) bool {
	if len(left) != len(right) {
		return false
	}
	left = append([]modeltransfer.SourceFile(nil), left...)
	right = append([]records.ModelTransferSourceFile(nil), right...)
	sort.Slice(left, func(i, j int) bool { return left[i].Member < left[j].Member })
	sort.Slice(right, func(i, j int) bool { return right[i].Member < right[j].Member })
	for i := range left {
		if left[i].Member != right[i].Member || left[i].SHA256 != right[i].SHA256 ||
			left[i].Length != right[i].Length {
			return false
		}
	}
	return true
}

func (o *modelTransferOwner) PassThrough(parent context.Context, requestID string,
	intent records.ModelTransferIntent,
) *exit.Error {
	if problem := o.store.BeginModelTransferMaterialization(requestID); problem != nil {
		return problem
	}
	ctx, cancel := o.requestContext(parent, requestID)
	defer cancel()
	// The request's scratch holds the downloaded source only until its CozyTensors are in
	// the CAS; it is released before finalization, which claims it again for its own
	// small exchange files.
	work, problem := o.requestScratch(requestID)
	if problem != nil {
		return problem
	}
	defer work.Release()
	prepared, problem := prepareLocalTransferSources(ctx, o.cliContext(intent, false), work.Path,
		intent, map[string]string{"model": ""})
	if problem != nil {
		return problem
	}
	source := prepared["model"]
	tool, _, problem := localTensorFS(o.cliContext(intent, false))
	if problem != nil {
		return problem
	}
	if source.plan != nil {
		temporary := "transfer-" + shortTransferID(requestID)
		observed, problem := tool.ObserveLocal(temporary,
			filepath.Join(work.Path, "temp-observed.jsonl"))
		if problem != nil {
			return problem
		}
		if problem := tool.InstallLocal(source.plan.Session, temporary, intent.SourceSelection,
			observed); problem != nil {
			return problem
		}
		alias, problem := tool.ResolveLocal(temporary)
		if problem != nil {
			return problem
		}
		source.manifestID, source.manifestLength = alias.ManifestDigest, alias.ManifestLength
		defer tool.RemoveLocal(temporary, alias.RepositoryDigest)
	}
	weights := records.ModelTransferWeights{RequestID: requestID, OutputSlot: "model",
		ManifestID: source.manifestID, ManifestLength: source.manifestLength}
	if problem := o.store.RecordModelTransferWeights(weights); problem != nil {
		return problem
	}
	if problem := o.store.CompleteModelTransferMaterialization(requestID, nil); problem != nil {
		return problem
	}
	if problem := o.store.BeginModelTransferFinalization(requestID); problem != nil {
		return problem
	}
	work.Release()
	return o.Finalize(ctx, requestID, nil)
}

// requestScratch is `tmp/<request-id>/`: the one directory a model transfer may write
// bytes in flight to, released by the phase that claimed it.
func (o *modelTransferOwner) requestScratch(requestID string) (*scratch.Dir, *exit.Error) {
	layout, problem := home.Open(o.cfg.Home)
	if problem != nil {
		return nil, problem
	}
	return scratch.Named(layout.Tmp, requestID)
}

func (o *modelTransferOwner) Finalize(ctx context.Context, requestID string,
	mover orchestrator.ModelTransferMover,
) *exit.Error {
	request, problem := o.store.RequestRow(requestID)
	if problem != nil || request == nil {
		return problem
	}
	intent := request.ModelTransfer
	if intent == nil {
		return exit.Internalf("request %s lost its model transfer intent", requestID)
	}
	rows, problem := o.store.AllModelTransferWeights(requestID, request.Ordinal)
	if problem != nil {
		return problem
	}
	if len(rows) != len(intent.Outputs) {
		return exit.Named(exit.Unavailable, "model_transfer.outputs_pending",
			"model transfer %s has %d of %d required outputs", requestID, len(rows),
			len(intent.Outputs))
	}
	declared := make(map[string]bool, len(intent.Outputs))
	for _, output := range intent.Outputs {
		declared[output.Name] = true
	}
	checkpoints := make(map[string]string, len(rows))
	for _, weights := range rows {
		if !declared[weights.OutputSlot] {
			return exit.Named(exit.Conflict, "model_transfer.output_undeclared",
				"weights output %s is absent from the accepted producer package interface",
				weights.OutputSlot)
		}
		if weights.FinalID == "" {
			finalID, problem := o.finalizeOutput(ctx, *intent, request.Worker, weights, mover)
			if problem != nil {
				return problem
			}
			if problem := o.store.CompleteModelTransferOutput(requestID, weights.Attempt, weights.OutputSlot,
				finalID); problem != nil {
				return problem
			}
		}
		checkpoints[weights.OutputSlot] = weights.ManifestID
	}
	return o.store.CompleteModelTransfer(requestID, checkpoints)
}

func (o *modelTransferOwner) finalizeOutput(ctx context.Context,
	intent records.ModelTransferIntent, worker string, weights records.ModelTransferWeights,
	mover orchestrator.ModelTransferMover,
) (string, *exit.Error) {
	cli := o.cliContext(intent, worker != "")
	if intent.Kind == "model-download" {
		tool, _, problem := localTensorFS(cli)
		if problem != nil {
			return "", problem
		}
		work, problem := o.requestScratch(weights.RequestID)
		if problem != nil {
			return "", problem
		}
		defer work.Release()
		name := strings.TrimPrefix(intent.Destination, "local/")
		observed, problem := tool.ObserveLocal(name,
			filepath.Join(work.Path, "local-observed.jsonl"))
		if problem != nil {
			return "", problem
		}
		alias, problem := tool.ReplaceLocal(name,
			localFinalizationSelection(weights.RequestID, weights.OutputSlot), observed,
			weights.ManifestID, weights.ManifestLength)
		if problem != nil {
			return "", problem
		}
		return "local:" + alias.RepositoryDigest, nil
	}
	ref, problem := hub.ParseRef(intent.Destination)
	if problem != nil {
		return "", problem
	}
	publicationClient, problem := ownedPublication(cli, ref)
	if problem != nil {
		return "", problem
	}
	if worker == "" {
		tool, _, _, problem := tooling(cli)
		if problem != nil {
			return "", problem
		}
		work, problem := o.requestScratch(weights.RequestID)
		if problem != nil {
			return "", problem
		}
		defer work.Release()
		upload := &transfer.Upload{Tool: tool, Hub: publicationClient, Ref: ref,
			ManifestID: weights.ManifestID,
			Session:    transferOutputOperation(weights.RequestID, weights.OutputSlot),
			Reason:     "cozy model upload " + intent.Source + " " + intent.Destination,
			Scratch:    work.Path, Progress: progress(cli)}
		result, problem := upload.Run(ctx)
		if problem != nil {
			return "", problem
		}
		return result.PublishID, nil
	}
	if mover == nil {
		return "", exit.Internalf("remote model transfer has no host mover")
	}
	objects := make([]hub.Object, 0, len(weights.Objects))
	for _, object := range weights.Objects {
		objects = append(objects, hub.Object{ID: object.ObjectID, Length: object.Length})
	}
	operation := transferOutputOperation(weights.RequestID, weights.OutputSlot)
	opened, problem := publicationClient.OpenPublication(ctx, ref, operation, objects,
		"cozy model upload "+intent.Source+" "+intent.Destination)
	if problem != nil {
		return "", problem
	}
	totals, problem := transfer.ValidateOpenedPublication(opened, operation, objects)
	if problem != nil {
		return "", problem
	}
	// THE MOVER MINTS, THIS ONLY SIGNS. Grants used to be asked for here, 128 at a time,
	// before the mover moved a single byte: one expiry stamped on the whole batch, the last
	// object's URL signed beside the first and dead before its turn came. The mover now
	// carries the cursor, so it asks for the objects whose bytes are about to move and asks
	// again once the hub's own declared life is half spent. Nothing about the publication
	// protocol changed -- the hub always re-minted for a still-claimed object.
	if opened.Publication.State == "open" {
		if problem := mover(ctx, weights, operation,
			func(ctx context.Context, objectIDs []string) (orchestrator.WeightsGrantMint, *exit.Error) {
				return mintWeightsGrants(ctx, publicationClient, ref, operation, intent, objectIDs)
			}); problem != nil {
			return "", problem
		}
	}
	checkpoint, problem := publicationClient.FinalizePublication(ctx, ref, operation,
		hub.FinalizePublicationRequest{ManifestID: weights.ManifestID,
			ManifestLength: weights.ManifestLength},
		"cozy model upload "+intent.Source+" "+intent.Destination)
	if problem != nil {
		return "", problem
	}
	if problem := transfer.ValidateFinalizedCheckpoint(checkpoint, operation,
		weights.ManifestID, weights.ManifestLength, totals); problem != nil {
		return "", problem
	}
	return checkpoint.PublishID, nil
}

// mintWeightsGrants authorizes one window of objects and reports the hub's clock beside them.
// The window is whatever the mover asked for; this neither batches nor caches, because the
// only place that knows which object is about to move is the walk.
func mintWeightsGrants(ctx context.Context, client *hub.Client, ref hub.Ref, operation string,
	intent records.ModelTransferIntent, objectIDs []string,
) (orchestrator.WeightsGrantMint, *exit.Error) {
	var window orchestrator.WeightsGrantMint
	granted, problem := client.GrantKnownTransfers(ctx, ref, operation, objectIDs,
		"cozy model upload "+intent.Source+" "+intent.Destination)
	if problem != nil {
		return window, problem
	}
	if granted.ServerTimeUnix <= 0 {
		return window, exit.Named(exit.Conflict, "model_transfer.grant_clock_absent",
			"Tensorhub authorized these objects without stating its own clock, "+
				"so no client can tell how much of the grant is left")
	}
	window.ServerTimeUnix = granted.ServerTimeUnix
	window.Decisions = make([]orchestrator.WeightsTransferDecision, 0, len(objectIDs))
	for _, grant := range granted.Grants {
		if grant.ExpiresAtUnix <= granted.ServerTimeUnix {
			return window, exit.Named(exit.Conflict, "model_transfer.grant_expiry_invalid",
				"Tensorhub signed %s to expire at or before the moment it signed it",
				grant.ObjectID)
		}
		window.Decisions = append(window.Decisions, orchestrator.WeightsTransferDecision{
			ObjectID: grant.ObjectID, Length: grant.Length, URL: grant.URL,
			Headers: grant.Headers, ExpiresAtUnix: uint64(grant.ExpiresAtUnix)})
	}
	for _, held := range granted.Held {
		window.Decisions = append(window.Decisions, orchestrator.WeightsTransferDecision{
			ObjectID: held.ObjectID, Length: held.Length, Held: true})
	}
	return window, nil
}

func transferOutputOperation(requestID, slot string) string {
	sum := sha256.Sum256([]byte(requestID + "\x00" + slot))
	return "model-artifact-" + hex.EncodeToString(sum[:])
}
