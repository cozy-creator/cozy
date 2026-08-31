package cli

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"os/signal"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
)

const productionTransferBatch = 128

type productionManifest struct {
	ID, Evidence string
	Length       int64
}

func runRentedModelProduction(ctx *Context, plan modelproduction.Plan,
	source publishSource,
) *exit.Error {
	if plan.Production == nil || len(plan.Jobs) == 0 || len(source.Access) == 0 {
		return exit.Named(exit.Structural, "model_production.plan_incomplete",
			"rented model production requires a reviewed graph, exact jobs, and foreign source files")
	}
	planBytes, err := plan.Bytes()
	if err != nil {
		return exit.Internalf("cannot encode the model production restart plan: %s", err)
	}
	planDigest, err := plan.Digest()
	if err != nil {
		return exit.Internalf("cannot digest the model production restart plan: %s", err)
	}
	layout, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	operation, replay, problem := store.BeginModelProduction(records.ModelProductionOperation{
		ID: plan.ID(), PlanDigest: planDigest, Plan: planBytes,
	})
	if problem != nil {
		return problem
	}
	if operation.State == "completed" {
		return emitCompletedProduction(ctx, plan, &operation, true)
	}
	if operation.State == "failed" || operation.State == "canceled" {
		return exit.Named(exit.Conflict, "model_production.settled",
			"model production %s is already %s: %s", operation.ID, operation.State,
			operation.SafeDetail)
	}
	if operation.State == "release_cut" || operation.State == "cleanup_pending" {
		return resumeProductionCleanup(ctx, store, plan, operation, replay)
	}
	state, _, problem := ensureDaemon(ctx)
	if problem != nil {
		return problem
	}
	ctx.Daemon = state
	local, problem := dial(ctx)
	if problem != nil {
		return problem
	}
	runCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	rentalID, problem := ensureProductionRental(ctx, layout, store, plan, operation)
	if problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	releaseOwed := true
	defer func() {
		if releaseOwed {
			_ = endRentalSilently(ctx, rentalID)
		}
	}()
	if _, problem = local.EnsureRental(rentalID); problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}

	if problem = prepareProductionSource(local, store, plan, source, rentalID); problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	manifests, problem := productionSourceManifests(store, plan)
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	ordered, problem := plan.Production.OrderedNodes()
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	pins := make(map[string]modelproduction.JobPin, len(plan.Jobs))
	for _, pin := range plan.Jobs {
		pins[pin.Node] = pin
	}

	for index, node := range ordered {
		if runCtx.Err() != nil {
			_ = store.RequestModelProductionCancel(operation.ID)
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				exit.New(exit.Canceled, "model production %s was interrupted", operation.ID))
		}
		pin, ok := pins[node.Name]
		if !ok {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				exit.Internalf("model production node %s has no exact job pin", node.Name))
		}
		packageName, function, _ := splitProductionCallable(node.Callable)
		models := make([]orchestrator.ModelRef, 0, len(node.Models))
		for parameter, reference := range node.Models {
			manifest, ok := manifests[reference]
			if !ok {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
					exit.Named(exit.Conflict, "model_production.input_absent",
						"node %s input %s has no prepared Manifest", node.Name, reference))
			}
			models = append(models, orchestrator.ModelRef{Package: packageName, Slot: parameter,
				Model: operation.ID + "/" + reference, Manifest: manifest.ID,
				ManifestLength: manifest.Length})
		}
		sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
		row, problem := store.BeginModelProductionNode(records.ModelProductionNode{
			OperationID: operation.ID, NodeIndex: int64(index), NodeName: node.Name,
			State: "pending",
		})
		if problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
		}
		if row.State != "completed" {
			handle, submitProblem := local.SubmitJob(api.JobSubmission{
				Package: packageName, Function: function, Input: []byte("{}"),
				Release: pin.Release, ReleaseDigest: pin.ReleaseDigest,
				Rental: true, Worker: rentalID, Models: models,
				Org:                   strings.Split(plan.Destination, "/")[0],
				ProductionOperationID: operation.ID, ProductionNode: node.Name,
				ProductionNodeIndex: int64(index),
			}, productionRequestKey(operation.ID, node.Name))
			if submitProblem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, submitProblem)
			}
			if problem = waitProductionJob(runCtx, local, store, operation.ID, handle.JobID); problem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
			}
		}
		artifacts, problem := waitNodeArtifacts(runCtx, store, operation.ID, node.Name,
			node.Outputs)
		if problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
		}
		for _, artifact := range artifacts {
			manifest, publicationProblem := publishProductionArtifact(ctx, local,
				store, plan, rentalID, artifact)
			if publicationProblem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, publicationProblem)
			}
			manifests[node.Name+"."+artifact.OutputSlot] = manifest
		}
		if problem = store.SetModelProductionNodeRequest(operation.ID, int64(index), node.Name,
			artifacts[0].RequestID, "completed"); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
		}
		if problem = advanceProductionNode(store, operation.ID, rentalID, int64(index+1),
			index == len(ordered)-1); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
		}
	}

	if problem = cutProductionRelease(ctx, store, plan); problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	current, problem := store.ModelProduction(operation.ID)
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	if current != nil && current.State == "release_cut" {
		problem = store.AdvanceModelProduction(operation.ID, "release_cut", "cleanup_pending",
			current.NodeIndex, rentalID)
	}
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem)
	}
	if problem = endRentalSilently(ctx, rentalID); problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	releaseOwed = false
	current, problem = store.ModelProduction(operation.ID)
	if problem != nil {
		return problem
	}
	if current != nil && current.State == "cleanup_pending" {
		problem = store.AdvanceModelProduction(operation.ID, "cleanup_pending", "completed",
			current.NodeIndex, rentalID)
	}
	if problem != nil {
		return problem
	}
	current, _ = store.ModelProduction(operation.ID)
	return emitCompletedProduction(ctx, plan, current, replay)
}

func resumeProductionCleanup(ctx *Context, store *records.Store, plan modelproduction.Plan,
	operation records.ModelProductionOperation, replay bool,
) *exit.Error {
	if operation.State == "release_cut" {
		if problem := store.AdvanceModelProduction(operation.ID, "release_cut", "cleanup_pending",
			operation.NodeIndex, operation.RentalID); problem != nil {
			return problem
		}
		operation.State = "cleanup_pending"
	}
	if operation.RentalID != "" {
		if problem := endRentalSilently(ctx, operation.RentalID); problem != nil {
			return problem.WithRemedy("the model release is already cut; confirm rental %s absence to finish cleanup",
				operation.RentalID).WithNext("cozy rental end " + operation.RentalID)
		}
	}
	if problem := store.AdvanceModelProduction(operation.ID, "cleanup_pending", "completed",
		operation.NodeIndex, operation.RentalID); problem != nil {
		return problem
	}
	completed, problem := store.ModelProduction(operation.ID)
	if problem != nil {
		return problem
	}
	return emitCompletedProduction(ctx, plan, completed, replay)
}

func ensureProductionRental(ctx *Context, layout home.Layout, store *records.Store,
	plan modelproduction.Plan, operation records.ModelProductionOperation,
) (string, *exit.Error) {
	if operation.RentalID != "" {
		return operation.RentalID, nil
	}
	hctx, cancel := hub.Context()
	skus, problem := client(ctx).RentalSKUs(hctx)
	cancel()
	if problem != nil {
		return "", problem
	}
	eligible := make([]hub.RentalSKU, 0, len(skus))
	for _, sku := range skus {
		sm, parseProblem := computeSM(sku.ComputeCapability)
		if parseProblem == nil && sm >= plan.Resources.MinSM && sku.VRAMGB >= plan.Resources.VRAMGB &&
			sku.MinimumRAMPerGPUGB >= plan.Resources.RAMGB {
			eligible = append(eligible, sku)
		}
	}
	if len(eligible) == 0 {
		return "", exit.Named(exit.Capacity, "model_production.no_compatible_sku",
			"Tensorhub offers no GPU meeting sm%d+, %d GiB VRAM, and %d GiB RAM",
			plan.Resources.MinSM, plan.Resources.VRAMGB, plan.Resources.RAMGB)
	}
	sort.Slice(eligible, func(i, j int) bool {
		if eligible[i].PriceUSDMicrosPerHour != eligible[j].PriceUSDMicrosPerHour {
			return eligible[i].PriceUSDMicrosPerHour < eligible[j].PriceUSDMicrosPerHour
		}
		return eligible[i].Name < eligible[j].Name
	})
	sku := eligible[0]
	if operation.SelectedSKU != "" {
		found := false
		for _, candidate := range eligible {
			if candidate.Name == operation.SelectedSKU {
				sku, found = candidate, true
				break
			}
		}
		if !found {
			return "", exit.Named(exit.Capacity, "model_production.selected_sku_unavailable",
				"selected rental SKU %s no longer meets the production resource floor",
				operation.SelectedSKU)
		}
	}
	operationKey := "model-production-rental-" + operation.ID
	rentalOperation, problem := store.RentalOperation(operationKey)
	if problem != nil {
		return "", problem
	}
	rate := sku.PriceUSDMicrosPerHour
	if rentalOperation == nil {
		fleet := &managedRentals{ctx: ctx, layout: layout, store: store}
		line, admittedRate, admitProblem := fleet.admit(sku.Name)
		if admitProblem != nil {
			return "", admitProblem
		}
		fmt.Fprintln(ctx.Err, line)
		rate = admittedRate
	} else {
		rate = rentalOperation.HourlyRateUSDMicros
	}
	if problem = store.SelectModelProductionSKU(operation.ID, sku.Name); problem != nil {
		return "", problem
	}
	row, _, _, problem := acquireRental(ctx, layout, store, sku.Name, "",
		operationKey, "cozy model publish "+plan.Destination,
		rate, ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, "")
	if problem != nil {
		return "", problem
	}
	if problem = store.AdvanceModelProduction(operation.ID, "accepted", "source_preparing", 0,
		row.ID); problem != nil {
		return "", problem
	}
	return row.ID, nil
}

func computeSM(capability string) (int64, error) {
	parts := strings.Split(capability, ".")
	if len(parts) != 2 {
		return 0, fmt.Errorf("invalid compute capability")
	}
	major, err := strconv.ParseInt(parts[0], 10, 64)
	if err != nil {
		return 0, err
	}
	minor, err := strconv.ParseInt(parts[1], 10, 64)
	return major*10 + minor, err
}

func prepareProductionSource(local *localclient.Client, store *records.Store,
	plan modelproduction.Plan, source publishSource, rentalID string,
) *exit.Error {
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil {
		return problem
	}
	if current != nil && current.State == "accepted" {
		if problem := store.AdvanceModelProduction(plan.ID(), "accepted", "source_preparing", 0,
			rentalID); problem != nil {
			return problem
		}
	}
	current, problem = store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return problem
	}
	if current.State != "source_preparing" {
		return nil
	}
	files := make([]orchestrator.ProductionSourceFile, 0, len(plan.SourceFiles))
	for _, file := range plan.SourceFiles {
		files = append(files, orchestrator.ProductionSourceFile{Member: file.Member,
			ObjectID: "sha256:" + file.SHA256, Length: file.Length})
	}
	result, problem := local.ModelProductionAction(plan.ID(), api.ModelProductionAction{
		Action: "prepare_source", RentalID: rentalID,
		Source: &api.ModelProductionSourceAction{SelectionDigest: plan.SourceSelection,
			SourceURI: plan.Source, DeclaredLicense: plan.SourceLicense, Files: files,
			Profiles: plan.Production.Sources, Capabilities: source.Access},
	})
	if problem != nil {
		return problem
	}
	if len(result.Prepared) != len(plan.Production.Sources) {
		return exit.Named(exit.Conflict, "model_production.source_result_incomplete",
			"worker prepared %d of %d source profiles", len(result.Prepared),
			len(plan.Production.Sources))
	}
	return store.AdvanceModelProduction(plan.ID(), "source_preparing", "source_prepared", 0,
		rentalID)
}

func productionSourceManifests(store *records.Store, plan modelproduction.Plan) (
	map[string]productionManifest, *exit.Error,
) {
	rows, problem := store.PreparedModelSources(plan.ID())
	if problem != nil {
		return nil, problem
	}
	out := make(map[string]productionManifest, len(rows))
	for _, row := range rows {
		out[row.Slot] = productionManifest{ID: row.ManifestID, Length: row.ManifestLength,
			Evidence: base64.StdEncoding.EncodeToString(row.ReleaseEvidence)}
	}
	return out, nil
}

func splitProductionCallable(value string) (string, string, bool) {
	parts := strings.Split(value, "/")
	if len(parts) != 3 {
		return "", "", false
	}
	return parts[0] + "/" + parts[1], parts[2], true
}

func productionRequestKey(operationID, node string) string {
	return "model-production:" + operationID + ":" + node
}

func waitProductionJob(runCtx context.Context, local *localclient.Client, store *records.Store,
	operationID, requestID string,
) *exit.Error {
	for {
		operation, problem := store.ModelProduction(operationID)
		if problem != nil {
			return problem
		}
		if operation != nil && operation.CancelRequested {
			_ = local.CancelJob(requestID)
			return exit.New(exit.Canceled, "model production %s was canceled", operationID)
		}
		job, problem := local.Job(requestID)
		if problem != nil {
			return problem
		}
		switch job.Status {
		case "completed":
			return nil
		case "failed", "canceled":
			return exit.Named(exit.Failed, firstNonempty(job.ErrorType, "model_production.job_failed"),
				"production job %s %s: %s", requestID, job.Status, job.Error)
		}
		select {
		case <-runCtx.Done():
			_ = local.CancelJob(requestID)
			return exit.New(exit.Canceled, "model production interrupted while job %s was active",
				requestID)
		case <-time.After(2 * time.Second):
		}
	}
}

func waitNodeArtifacts(runCtx context.Context, store *records.Store, operationID,
	nodeName string, outputs []string,
) ([]records.ModelProductionArtifact, *exit.Error) {
	want := make(map[string]bool, len(outputs))
	for _, outputSlot := range outputs {
		want[outputSlot] = true
	}
	for {
		rows, problem := store.ModelProductionArtifacts(operationID)
		if problem != nil {
			return nil, problem
		}
		found := make([]records.ModelProductionArtifact, 0, len(want))
		for _, row := range rows {
			if row.NodeName == nodeName && want[row.OutputSlot] {
				found = append(found, row)
			}
		}
		if len(found) == len(want) {
			sort.Slice(found, func(i, j int) bool { return found[i].OutputSlot < found[j].OutputSlot })
			return found, nil
		}
		select {
		case <-runCtx.Done():
			return nil, exit.New(exit.Canceled, "model production interrupted waiting for node %s artifacts",
				nodeName)
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func publishProductionArtifact(ctx *Context, local *localclient.Client,
	store *records.Store, plan modelproduction.Plan, rentalID string,
	artifact records.ModelProductionArtifact,
) (productionManifest, *exit.Error) {
	source := artifact.NodeName + "." + artifact.OutputSlot
	contract, final := productionContract(plan, source)
	if !final {
		// Intermediate Manifests stay under the worker's adopted artifact root and
		// feed the next node directly. Final lane closures contain their inherited
		// bytes, so opening extra Hub publications here would add no durability to
		// the release and would leave uncut prepared sessions behind.
		return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
			Evidence: base64.StdEncoding.EncodeToString(artifact.ReleaseEvidence)}, nil
	}
	if artifact.PublicationID != "" && artifact.State == "prepared" {
		return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
			Evidence: base64.StdEncoding.EncodeToString(artifact.ReleaseEvidence)}, nil
	}
	objects, problem := store.ModelProductionObjects(plan.ID(), artifact.NodeName,
		artifact.OutputSlot)
	if problem != nil {
		return productionManifest{}, problem
	}
	hubObjects := make([]hub.Object, 0, len(objects))
	for _, object := range objects {
		hubObjects = append(hubObjects, hub.Object{ID: object.ObjectID, Length: object.Length})
	}
	lane := contract.LaneKey
	publicationOperation := productionPublicationOperation(plan.ID(), artifact.NodeName,
		artifact.OutputSlot)
	reason := "cozy model publish " + plan.Destination + "@" + plan.Release
	ref, parseProblem := hub.ParseRef(plan.Destination)
	if parseProblem != nil {
		return productionManifest{}, parseProblem
	}
	hctx, cancel := hub.LongContext()
	opened, problem := client(ctx).OpenPublication(hctx, ref, publicationOperation,
		plan.Release, lane, hubObjects, reason)
	cancel()
	if problem != nil {
		return productionManifest{}, problem
	}
	_ = opened
	for start := 0; start < len(objects); start += productionTransferBatch {
		end := min(start+productionTransferBatch, len(objects))
		ids := make([]string, 0, end-start)
		for _, object := range objects[start:end] {
			ids = append(ids, object.ObjectID)
		}
		hctx, cancel = hub.LongContext()
		granted, grantProblem := client(ctx).GrantKnownTransfers(hctx, ref,
			publicationOperation, ids, reason)
		cancel()
		if grantProblem != nil {
			return productionManifest{}, grantProblem
		}
		decisions := make([]orchestrator.ArtifactTransferDecision, 0, len(ids))
		for _, grant := range granted.Grants {
			expires, err := time.Parse(time.RFC3339, grant.Expires)
			if err != nil {
				return productionManifest{}, exit.Named(exit.Conflict,
					"model_production.grant_expiry_invalid", "Tensorhub returned an invalid grant expiry")
			}
			decisions = append(decisions, orchestrator.ArtifactTransferDecision{
				ObjectID: grant.ObjectID, Length: grant.Length, URL: grant.URL,
				Headers: grant.Headers, ExpiresAtUnix: uint64(expires.Unix()),
			})
		}
		for _, held := range granted.Held {
			decisions = append(decisions, orchestrator.ArtifactTransferDecision{
				ObjectID: held.ObjectID, Length: held.Length, Held: true,
			})
		}
		_, problem = local.ModelProductionAction(plan.ID(), api.ModelProductionAction{
			Action: "transfer_artifact", RentalID: rentalID,
			Artifact: &api.ModelProductionArtifactAction{NodeName: artifact.NodeName,
				OutputSlot: artifact.OutputSlot, TransferOperationID: publicationOperation,
				Decisions: decisions},
		})
		if problem != nil {
			return productionManifest{}, problem
		}
	}
	hctx, cancel = hub.LongContext()
	prepared, problem := client(ctx).FinalizePublication(hctx, ref, publicationOperation,
		hub.FinalizePublicationRequest{ManifestID: artifact.ManifestID,
			ManifestLength:        artifact.ManifestLength,
			ReleaseEvidenceBase64: base64.StdEncoding.EncodeToString(artifact.ReleaseEvidence)}, reason)
	cancel()
	if problem != nil {
		return productionManifest{}, problem
	}
	if prepared.Manifest.SHA256 != strings.TrimPrefix(artifact.ManifestID, "sha256:") ||
		prepared.Manifest.Length != artifact.ManifestLength {
		return productionManifest{}, exit.Named(exit.Conflict, "model_production.finalize_changed",
			"Tensorhub finalized a different Manifest for %s.%s", artifact.NodeName,
			artifact.OutputSlot)
	}
	if prepared.TopologyDigest != contract.TopologyDigest ||
		!sameStrings(prepared.Contract.Encoding.Set, contract.Encodings) {
		return productionManifest{}, exit.Named(exit.Conflict,
			"model_production.contract_mismatch",
			"Tensorhub-derived contract for lane %s does not match the reviewed production", lane)
	}
	if problem = store.MarkModelProductionArtifactPublished(plan.ID(), artifact.NodeName,
		artifact.OutputSlot, prepared.PublishID); problem != nil {
		return productionManifest{}, problem
	}
	return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
		Evidence: base64.StdEncoding.EncodeToString(artifact.ReleaseEvidence)}, nil
}

func productionPublicationOperation(operationID, node, slot string) string {
	sum := sha256.Sum256([]byte(operationID + "\x00" + node + "\x00" + slot))
	return "model-artifact-" + hex.EncodeToString(sum[:])
}

func productionContract(plan modelproduction.Plan, source string) (
	modelproductionContract, bool,
) {
	for _, output := range plan.Production.Outputs {
		if output.Source == source {
			return modelproductionContract{LaneKey: output.LaneKey,
				TopologyDigest: output.RequiredContract.TopologyDigest,
				Encodings:      output.RequiredContract.Encodings}, true
		}
	}
	return modelproductionContract{}, false
}

type modelproductionContract struct {
	LaneKey        string
	TopologyDigest string
	Encodings      []string
}

func sameStrings(left, right []string) bool {
	left, right = append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(left)
	sort.Strings(right)
	return strings.Join(left, "\x00") == strings.Join(right, "\x00")
}

func advanceProductionNode(store *records.Store, operationID, rentalID string,
	next int64, last bool,
) *exit.Error {
	current, problem := store.ModelProduction(operationID)
	if problem != nil || current == nil {
		return problem
	}
	if current.State == "node_running" && current.NodeIndex >= next ||
		current.State == "outputs_preparing" || current.State == "release_cut" ||
		current.State == "cleanup_pending" || current.State == "completed" {
		return nil
	}
	if current.State == "source_prepared" {
		if problem = store.AdvanceModelProduction(operationID, "source_prepared", "node_running",
			0, rentalID); problem != nil {
			return problem
		}
		current, problem = store.ModelProduction(operationID)
	}
	if problem != nil || current == nil {
		return problem
	}
	to := "node_running"
	if last {
		to = "outputs_preparing"
	}
	return store.AdvanceModelProduction(operationID, "node_running", to, next, rentalID)
}

func cutProductionRelease(ctx *Context, store *records.Store, plan modelproduction.Plan) *exit.Error {
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return problem
	}
	if current.State == "release_cut" || current.State == "cleanup_pending" ||
		current.State == "completed" {
		return nil
	}
	if current.State != "outputs_preparing" {
		return exit.Named(exit.Conflict, "model_production.cut_state_invalid",
			"model production %s is %s before release cut", plan.ID(), current.State)
	}
	artifacts, problem := store.ModelProductionArtifacts(plan.ID())
	if problem != nil {
		return problem
	}
	bySource := make(map[string]records.ModelProductionArtifact, len(artifacts))
	for _, artifact := range artifacts {
		bySource[artifact.NodeName+"."+artifact.OutputSlot] = artifact
	}
	publicationIDs := make([]string, 0, len(plan.Production.Outputs))
	for _, output := range plan.Production.Outputs {
		artifact, ok := bySource[output.Source]
		if !ok || artifact.PublicationID == "" {
			return exit.Named(exit.Conflict, "model_production.output_not_prepared",
				"required lane %s has no prepared publication", output.LaneKey)
		}
		publicationIDs = append(publicationIDs, artifact.PublicationID)
	}
	sort.Strings(publicationIDs)
	ref, parseProblem := hub.ParseRef(plan.Destination)
	if parseProblem != nil {
		return parseProblem
	}
	hctx, cancel := hub.LongContext()
	cut, problem := client(ctx).CutRelease(hctx, ref, plan.Release, plan.ID(), publicationIDs,
		"cozy model publish "+plan.Destination+"@"+plan.Release)
	cancel()
	if problem != nil {
		return problem
	}
	if cut.Release != plan.Release || len(cut.Lanes) != len(plan.Production.Outputs) {
		return exit.Named(exit.Conflict, "model_production.cut_changed",
			"Tensorhub cut an incomplete or different model release")
	}
	return store.AdvanceModelProduction(plan.ID(), "outputs_preparing", "release_cut",
		current.NodeIndex, current.RentalID)
}

func failAndReleaseProduction(ctx *Context, store *records.Store, operationID, rentalID string,
	cause *exit.Error,
) *exit.Error {
	failed := failProduction(store, operationID, cause)
	if release := endRentalSilently(ctx, rentalID); release != nil {
		return release.WithRemedy("model production failed, and rental %s may still be billing; release it explicitly",
			rentalID).WithNext("cozy rental end " + rentalID)
	}
	return failed
}

func failProduction(store *records.Store, operationID string, cause *exit.Error) *exit.Error {
	current, problem := store.ModelProduction(operationID)
	if problem == nil && current != nil && current.State != "failed" && current.State != "canceled" &&
		current.State != "completed" {
		if cause.Code == exit.Canceled {
			_ = store.CancelModelProduction(operationID, current.State, cause.Message)
		} else {
			_ = store.FailModelProduction(operationID, current.State, cause.Name, cause.Message)
		}
	}
	return cause
}

func emitCompletedProduction(ctx *Context, plan modelproduction.Plan,
	operation *records.ModelProductionOperation, replay bool,
) *exit.Error {
	rentalID := ""
	if operation != nil {
		rentalID = operation.RentalID
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "id", V: plan.ID()}, {K: "kind", V: "model-publication"},
		{K: "model", V: plan.Destination}, {K: "release", V: plan.Release},
		{K: "lanes", V: plan.Lanes()}, {K: "rental", V: rentalID},
		{K: "status", V: "completed"}, {K: "changed", V: !replay},
	}, "model", "release", "lanes", "status", "changed"))
}

func firstNonempty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return "unknown"
}
