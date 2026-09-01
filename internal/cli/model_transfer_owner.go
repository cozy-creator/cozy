package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/accountauth"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modeltransfer"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/tfs"
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
	prepared, problem := prepareLocalTransferSources(ctx, o.cliContext(intent, false), requestID,
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
	prepared, problem := prepareLocalTransferSources(ctx, o.cliContext(intent, false), requestID,
		intent, map[string]string{"model": ""})
	if problem != nil {
		return problem
	}
	source := prepared["model"]
	tool, layout, problem := localTensorFS(o.cliContext(intent, false))
	if problem != nil {
		return problem
	}
	if source.plan != nil {
		temporary := "transfer-" + shortTransferID(requestID)
		observed, problem := tool.ObserveLocal(temporary,
			filepath.Join(layout.Transfer, "temp-observed.jsonl"))
		if problem != nil {
			return problem
		}
		if problem := tool.InstallLocal(source.plan.Session, temporary, intent.SourceSelection,
			observed, intent.Source, intent.SourceLicense); problem != nil {
			return problem
		}
		alias, problem := tool.ResolveLocal(temporary)
		if problem != nil {
			return problem
		}
		evidence, problem := tool.CheckpointEvidence("local", temporary, alias.ManifestDigest,
			filepath.Join(layout.Transfer, "temp-evidence.jsonl"))
		if problem != nil {
			return problem
		}
		source.manifestID, source.manifestLength, source.evidence = alias.ManifestDigest,
			alias.ManifestLength, evidence
		defer tool.RemoveLocal(temporary, alias.RepositoryDigest)
	}
	weights := records.ModelTransferWeights{RequestID: requestID, OutputSlot: "model",
		ManifestID: source.manifestID, ManifestLength: source.manifestLength,
		Evidence: source.evidence}
	if problem := o.store.RecordModelTransferWeights(weights); problem != nil {
		return problem
	}
	if problem := o.store.CompleteModelTransferMaterialization(requestID, nil); problem != nil {
		return problem
	}
	if problem := o.store.BeginModelTransferFinalization(requestID); problem != nil {
		return problem
	}
	return o.Finalize(ctx, requestID, nil)
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
	contracts := make(map[string]*records.ModelTransferContract, len(intent.Outputs))
	for _, output := range intent.Outputs {
		contracts[output.Name] = output.RequiredContract
	}
	checkpoints := make(map[string]string, len(rows))
	for _, weights := range rows {
		contract, declared := contracts[weights.OutputSlot]
		if !declared {
			return exit.Named(exit.Conflict, "model_transfer.output_undeclared",
				"weights output %s is absent from the accepted producer descriptor",
				weights.OutputSlot)
		}
		if weights.FinalID == "" {
			finalID, problem := o.finalizeOutput(ctx, *intent, request.Worker, weights,
				contract, mover)
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
	contract *records.ModelTransferContract, mover orchestrator.ModelTransferMover,
) (string, *exit.Error) {
	cli := o.cliContext(intent, worker != "")
	if intent.Kind == "model-download" {
		tool, layout, problem := localTensorFS(cli)
		if problem != nil {
			return "", problem
		}
		name := strings.TrimPrefix(intent.Destination, "local/")
		observed, problem := tool.ObserveLocal(name,
			filepath.Join(layout.Transfer, "local-observed.jsonl"))
		if problem != nil {
			return "", problem
		}
		evidencePath := filepath.Join(layout.Transfer, "local-evidence-"+weights.OutputSlot+".json")
		if err := os.WriteFile(evidencePath, weights.Evidence, 0o600); err != nil {
			return "", exit.Internalf("cannot stage local checkpoint evidence: %s", err)
		}
		defer os.Remove(evidencePath)
		alias, problem := tool.ReplaceLocal(name,
			localFinalizationSelection(weights.RequestID, weights.OutputSlot), observed,
			weights.ManifestID, weights.ManifestLength, evidencePath, expectedTFSContract(contract))
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
		tool, _, layout, problem := tooling(cli)
		if problem != nil {
			return "", problem
		}
		upload := &transfer.Upload{Tool: tool, Hub: publicationClient, Ref: ref,
			ManifestID: weights.ManifestID, CheckpointEvidence: weights.Evidence,
			Session: transferOutputOperation(weights.RequestID, weights.OutputSlot),
			Reason:  "cozy model upload " + intent.Source + " " + intent.Destination,
			Scratch: scratch(layout, weights.ManifestID), Progress: progress(cli),
			ExpectedContract: expectedHubContract(contract)}
		result, problem := upload.Run(ctx)
		if problem != nil {
			return "", problem
		}
		if contract != nil && (result.TopologyDigest != contract.TopologyDigest ||
			!sameStrings(result.EncodingSet, contract.Encodings)) {
			return "", exit.Named(exit.Conflict, "model_transfer.contract_mismatch",
				"Tensorhub-derived contract for output %s differs from the producer descriptor",
				weights.OutputSlot)
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
	const batch = 128
	for start := 0; opened.Publication.State == "open" && start < len(objects); start += batch {
		end := min(start+batch, len(objects))
		ids := make([]string, 0, end-start)
		for _, object := range objects[start:end] {
			ids = append(ids, object.ID)
		}
		granted, problem := publicationClient.GrantKnownTransfers(ctx, ref, operation, ids,
			"cozy model upload "+intent.Source+" "+intent.Destination)
		if problem != nil {
			return "", problem
		}
		decisions := make([]orchestrator.WeightsTransferDecision, 0, len(ids))
		for _, grant := range granted.Grants {
			expires, err := time.Parse(time.RFC3339, grant.Expires)
			if err != nil {
				return "", exit.Named(exit.Conflict, "model_transfer.grant_expiry_invalid",
					"Tensorhub returned an invalid grant expiry")
			}
			decisions = append(decisions, orchestrator.WeightsTransferDecision{
				ObjectID: grant.ObjectID, Length: grant.Length, URL: grant.URL,
				Headers: grant.Headers, ExpiresAtUnix: uint64(expires.Unix())})
		}
		for _, held := range granted.Held {
			decisions = append(decisions, orchestrator.WeightsTransferDecision{
				ObjectID: held.ObjectID, Length: held.Length, Held: true})
		}
		if problem := mover(ctx, weights, operation, decisions); problem != nil {
			return "", problem
		}
	}
	checkpoint, problem := publicationClient.FinalizePublication(ctx, ref, operation,
		hub.FinalizePublicationRequest{ManifestID: weights.ManifestID,
			ManifestLength:           weights.ManifestLength,
			CheckpointEvidenceBase64: hub.B64(weights.Evidence),
			ExpectedContract:         expectedHubContract(contract)},
		"cozy model upload "+intent.Source+" "+intent.Destination)
	if problem != nil {
		return "", problem
	}
	if problem := transfer.ValidateFinalizedCheckpoint(checkpoint, operation,
		weights.ManifestID, weights.ManifestLength, weights.Evidence, totals); problem != nil {
		return "", problem
	}
	if contract != nil && (checkpoint.TopologyDigest != contract.TopologyDigest ||
		!sameStrings(checkpoint.Contract.Encoding.Set, contract.Encodings)) {
		return "", exit.Named(exit.Conflict, "model_transfer.contract_mismatch",
			"Tensorhub-derived contract for output %s differs from the producer descriptor",
			weights.OutputSlot)
	}
	return checkpoint.PublishID, nil
}

func expectedHubContract(contract *records.ModelTransferContract) *hub.ExpectedModelContract {
	if contract == nil {
		return nil
	}
	return &hub.ExpectedModelContract{TopologyDigest: contract.TopologyDigest,
		Encodings: append([]string(nil), contract.Encodings...)}
}

func expectedTFSContract(contract *records.ModelTransferContract) *tfs.ExpectedModelContract {
	if contract == nil {
		return nil
	}
	return &tfs.ExpectedModelContract{TopologyDigest: contract.TopologyDigest,
		Encodings: append([]string(nil), contract.Encodings...)}
}

func transferOutputOperation(requestID, slot string) string {
	sum := sha256.Sum256([]byte(requestID + "\x00" + slot))
	return "model-artifact-" + hex.EncodeToString(sum[:])
}

func sameStrings(left, right []string) bool {
	left, right = append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(left)
	sort.Strings(right)
	return strings.Join(left, "\x00") == strings.Join(right, "\x00")
}
