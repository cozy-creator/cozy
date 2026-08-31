package cli

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	localclient "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/modelproduction"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/privatepackage"
	"github.com/cozy-creator/cozy/internal/records"
)

const productionTransferBatch = 128

type productionManifest struct {
	ID, Evidence string
	Length       int64
}

func runRentedModelProduction(ctx *Context, runCtx context.Context, plan modelproduction.Plan,
	source publishSource,
) *exit.Error {
	if plan.Production == nil || len(plan.Jobs) == 0 {
		return exit.Named(exit.Structural, "model_production.plan_incomplete",
			"rented model production requires a reviewed graph and exact jobs")
	}
	planBytes, err := plan.Bytes()
	if err != nil {
		return exit.Internalf("cannot encode the model production restart plan: %s", err)
	}
	planDigest, err := plan.Digest()
	if err != nil {
		return exit.Internalf("cannot digest the model production restart plan: %s", err)
	}
	ordered, problem := plan.Production.OrderedNodes()
	if problem != nil {
		return problem
	}
	progress := NewModelProductionProgress(ctx.Err, !ctx.Mode().JSON, len(ordered))
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
	existingArtifacts, problem := productionArtifactSnapshot(store, operation.ID)
	if problem != nil {
		return problem
	}
	if replay {
		files, readProblem := store.ModelProductionSourceFiles(operation.ID)
		if readProblem != nil {
			return readProblem
		}
		transferred, total := productionSourceProgress(files)
		progress.SeedSourceProgress(transferred, total)
		progress.Resume(operation.ID, productionResumeStage(operation, ordered, transferred, total,
			len(plan.Production.Outputs)))
	} else {
		progress.Accepted(operation.ID, len(plan.Production.Outputs))
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
		return resumeProductionCleanup(ctx, store, plan, operation, replay, progress)
	}
	if operation.State == "outputs_preparing" {
		if problem := cutProductionRelease(runCtx, ctx, store, plan, progress); problem != nil {
			return problem
		}
		current, problem := store.ModelProduction(operation.ID)
		if problem != nil || current == nil {
			return problem
		}
		return resumeProductionCleanup(ctx, store, plan, *current, replay, progress)
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
	if problem = productionCancellation(runCtx, store, operation.ID); problem != nil {
		return failProduction(store, operation.ID, problem)
	}
	if len(source.Access) == 0 {
		return exit.Named(exit.Structural, "model_production.source_capabilities_absent",
			"rented model production requires refreshable foreign source file capabilities")
	}
	rentalID, problem := ensureProductionRental(runCtx, ctx, layout, store, plan, operation, progress)
	if problem != nil {
		if rentalID != "" {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		return failProduction(store, operation.ID, problem)
	}
	releaseOwed := true
	defer func() {
		if releaseOwed {
			_ = endRentalSilently(ctx, rentalID)
		}
	}()
	progress.WorkerWaiting(rentalID)
	_, workerProblem := local.EnsureRental(rentalID)
	if workerProblem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, workerProblem, progress)
	}
	progress.WorkerReady(rentalID)

	announceSource := !replay || operation.State != "source_preparing"
	if problem = prepareProductionSource(runCtx, local, store, plan, source, rentalID,
		progress, announceSource); problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	manifests, problem := productionSourceManifests(store, plan)
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	pins := make(map[string]modelproduction.JobPin, len(plan.Jobs))
	for _, pin := range plan.Jobs {
		pins[pin.Node] = pin
	}

	for index, node := range ordered {
		if cancelProblem := productionCancellation(runCtx, store, operation.ID); cancelProblem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				cancelProblem, progress)
		}
		pin, ok := pins[node.Name]
		if !ok {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				exit.Internalf("model production node %s has no exact job pin", node.Name), progress)
		}
		packageName, function, _ := splitProductionCallable(node.Callable)
		models := make([]orchestrator.ModelRef, 0, len(node.Models))
		for parameter, reference := range node.Models {
			manifest, ok := manifests[reference]
			if !ok {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
					exit.Named(exit.Conflict, "model_production.input_absent",
						"node %s input %s has no prepared Manifest", node.Name, reference), progress)
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
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		nodeCompleted := row.State == "completed"
		if !nodeCompleted {
			progress.NodeStarting(index, node.Name, node.Callable, row.RequestID != "")
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
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, submitProblem, progress)
			}
			if problem = waitProductionJob(runCtx, local, store, operation.ID, handle.JobID,
				progress, index, node.Name); problem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
			}
		}
		artifacts, problem := waitNodeArtifacts(runCtx, store, operation.ID, node.Name,
			node.Outputs)
		if problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		for artifactIndex, artifact := range artifacts {
			source := artifact.NodeName + "." + artifact.OutputSlot
			if !existingArtifacts[source] {
				progress.ArtifactAdopted(index, artifactIndex, len(artifacts), node.Name)
			}
			manifest, publicationProblem := publishProductionArtifact(runCtx, ctx, local,
				store, plan, rentalID, artifact, progress)
			if publicationProblem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, publicationProblem, progress)
			}
			manifests[node.Name+"."+artifact.OutputSlot] = manifest
		}
		if problem = store.SetModelProductionNodeRequest(operation.ID, int64(index), node.Name,
			artifacts[0].RequestID, "completed"); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		if problem = advanceProductionNode(store, operation.ID, rentalID, int64(index+1),
			index == len(ordered)-1); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		if !nodeCompleted {
			progress.NodeCompleted(index, node.Name, node.Callable)
		}
	}

	if problem = cutProductionRelease(runCtx, ctx, store, plan, progress); problem != nil {
		if problem.Name == "model_production.cut_verdict_unknown" {
			// The cut may already be committed. Keep the exact rental/source/artifact
			// joins intact until replay learns the verdict; releasing here would turn
			// an ambiguous response into an unrecoverable second topology.
			releaseOwed = false
			return problem
		}
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	current, problem := store.ModelProduction(operation.ID)
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	if current != nil && current.State == "release_cut" {
		problem = store.AdvanceModelProduction(operation.ID, "release_cut", "cleanup_pending",
			current.NodeIndex, rentalID)
	}
	if problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	progress.RentalReleaseStarting(rentalID)
	if problem = endRentalSilently(ctx, rentalID); problem != nil {
		progress.RentalReleaseUnconfirmed(rentalID)
		return failProduction(store, operation.ID, problem)
	}
	progress.RentalReleased(rentalID)
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
	operation records.ModelProductionOperation, replay bool, progress *productionProgress,
) *exit.Error {
	if operation.State == "release_cut" {
		if problem := store.AdvanceModelProduction(operation.ID, "release_cut", "cleanup_pending",
			operation.NodeIndex, operation.RentalID); problem != nil {
			return problem
		}
		operation.State = "cleanup_pending"
	}
	if operation.RentalID != "" {
		progress.RentalReleaseStarting(operation.RentalID)
		if problem := endRentalSilently(ctx, operation.RentalID); problem != nil {
			progress.RentalReleaseUnconfirmed(operation.RentalID)
			return problem.WithRemedy("the model release is already cut; confirm rental %s absence to finish cleanup",
				operation.RentalID).WithNext("cozy rental end " + operation.RentalID)
		}
		progress.RentalReleased(operation.RentalID)
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

func ensureProductionRental(runCtx context.Context, ctx *Context, layout home.Layout, store *records.Store,
	plan modelproduction.Plan, operation records.ModelProductionOperation,
	progress *productionProgress,
) (string, *exit.Error) {
	if problem := productionCancellation(runCtx, store, operation.ID); problem != nil {
		return "", problem
	}
	if operation.RentalID != "" {
		return operation.RentalID, nil
	}
	hctx, cancel := productionHubContext(runCtx)
	skus, problem := client(ctx).RentalSKUs(hctx)
	cancel()
	if problem != nil {
		if cancelProblem := productionCancellation(runCtx, store, operation.ID); cancelProblem != nil {
			return "", cancelProblem
		}
		return "", problem
	}
	if problem := productionCancellation(runCtx, store, operation.ID); problem != nil {
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
	var acceptableBases []string
	if rentalOperation == nil {
		acceptableBases, problem = productionAcceptableBases(runCtx, ctx, store, plan)
		if problem != nil {
			return "", problem
		}
	}
	progress.RentalSelecting(sku.Name)
	row, _, _, problem := acquireRentalContext(runCtx, ctx, layout, store, sku.Name, "",
		operationKey, "cozy model publish "+plan.Destination,
		rate, ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, "", acceptableBases)
	if problem != nil {
		operation, readProblem := store.RentalOperation(operationKey)
		if readProblem == nil && operation != nil {
			return operation.RentalID, problem
		}
		if readProblem != nil {
			return "", readProblem
		}
		return "", problem
	}
	if problem := productionCancellation(runCtx, store, operation.ID); problem != nil {
		return row.ID, problem
	}
	progress.RentalReady(row.ID, row.State, sku.Name)
	if problem = store.AdvanceModelProduction(operation.ID, "accepted", "source_preparing", 0,
		row.ID); problem != nil {
		return "", problem
	}
	return row.ID, nil
}

func productionAcceptableBases(runCtx context.Context, ctx *Context, store *records.Store,
	plan modelproduction.Plan,
) ([]string, *exit.Error) {
	resolver := NewResolver(store, ctx.Cfg)
	byRelease := map[string][]string{}
	var intersection []string
	for _, pin := range plan.Jobs {
		accepted, ok := byRelease[pin.ReleaseDigest]
		if !ok {
			packageName, _, valid := splitProductionCallable(pin.Callable)
			if !valid {
				return nil, exit.Named(exit.Structural, "model_production_callable_invalid",
					"production node %s has invalid callable %s", pin.Node, pin.Callable)
			}
			if pin.InstallID != "" {
				unlock := privatepackage.Guard()
				revision, compatible, problem := resolver.PreparePrivate(
					runCtx, pin.InstallID, "0.0.20")
				unlock()
				if problem != nil {
					return nil, problem
				}
				if revision.SourceDigest != pin.ReleaseDigest {
					return nil, exit.Named(exit.Conflict, "model_production_package_changed",
						"production node %s private package changed after plan acceptance", pin.Node)
				}
				accepted = compatible
			} else {
				ref, problem := hub.ParseRef(packageName)
				if problem != nil {
					return nil, problem
				}
				hctx, cancel := productionHubContext(runCtx)
				detail, problem := resolver.catalog.PackageRelease(hctx, ref, pin.Release)
				cancel()
				if problem != nil {
					return nil, problem
				}
				accepted, problem = resolver.preflightPublished(
					ref, pin.Release, pin.ReleaseDigest, detail)
				if problem != nil {
					return nil, problem
				}
			}
			byRelease[pin.ReleaseDigest] = append([]string(nil), accepted...)
		}
		if intersection == nil {
			intersection = append([]string(nil), accepted...)
		} else {
			keep := intersection[:0]
			left, right := 0, 0
			for left < len(intersection) && right < len(accepted) {
				switch {
				case intersection[left] < accepted[right]:
					left++
				case intersection[left] > accepted[right]:
					right++
				default:
					keep = append(keep, intersection[left])
					left++
					right++
				}
			}
			intersection = keep
		}
		if len(intersection) == 0 {
			return nil, exit.Named(exit.Conflict, "model_production.no_compatible_base",
				"production packages share no active compatible WheelhouseManifest")
		}
	}
	if len(intersection) == 0 {
		return nil, exit.Named(exit.Conflict, "model_production.no_compatible_base",
			"model production has no compatible base manifest")
	}
	return intersection, nil
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

func prepareProductionSource(runCtx context.Context, local *localclient.Client, store *records.Store,
	plan modelproduction.Plan, source publishSource, rentalID string,
	progress *productionProgress, announce bool,
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
	if announce {
		progress.SourceStarting(len(plan.SourceFiles), productionSourceBytes(plan.SourceFiles))
	}
	files := make([]orchestrator.ProductionSourceFile, 0, len(plan.SourceFiles))
	for _, file := range plan.SourceFiles {
		files = append(files, orchestrator.ProductionSourceFile{Member: file.Member,
			ObjectID: "sha256:" + file.SHA256, Length: file.Length})
	}
	type sourceResult struct {
		result  api.ModelProductionActionResult
		problem *exit.Error
	}
	actionCtx, cancelAction := context.WithCancel(runCtx)
	defer cancelAction()
	answer := make(chan sourceResult, 1)
	go func() {
		result, actionProblem := local.ModelProductionActionContext(actionCtx, plan.ID(),
			api.ModelProductionAction{
				Action: "prepare_source", RentalID: rentalID,
				Source: &api.ModelProductionSourceAction{SelectionDigest: plan.SourceSelection,
					SourceURI: plan.Source, DeclaredLicense: plan.SourceLicense, Files: files,
					Profiles: plan.Production.Sources, Capabilities: source.Access},
			})
		answer <- sourceResult{result: result, problem: actionProblem}
	}()
	// The unary daemon exchange has no client-side event stream. Sample its durable
	// rows at the same cadence as node state, while Progress suppresses unchanged
	// values and all movement within an already-rendered ten-percent band.
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()
	var result api.ModelProductionActionResult
	for {
		select {
		case observed := <-answer:
			if runCtx.Err() != nil {
				return exit.New(exit.Canceled,
					"model production %s was interrupted during source preparation", plan.ID())
			}
			result, problem = observed.result, observed.problem
			goto prepared
		case <-ticker.C:
			if cancelProblem := productionCancellation(runCtx, store, plan.ID()); cancelProblem != nil {
				cancelAction()
				return cancelProblem
			}
			rows, readProblem := store.ModelProductionSourceFiles(plan.ID())
			if readProblem != nil {
				return readProblem
			}
			transferred, total := productionSourceProgress(rows)
			progress.SourceProgress(transferred, total)
		}
	}

prepared:
	if problem != nil {
		return problem
	}
	if len(result.Prepared) != len(plan.Production.Sources) {
		return exit.Named(exit.Conflict, "model_production.source_result_incomplete",
			"worker prepared %d of %d source profiles", len(result.Prepared),
			len(plan.Production.Sources))
	}
	if problem = store.AdvanceModelProduction(plan.ID(), "source_preparing", "source_prepared", 0,
		rentalID); problem != nil {
		return problem
	}
	progress.SourcePrepared(len(result.Prepared))
	return nil
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

func productionArtifactSnapshot(store *records.Store, operationID string) (map[string]bool, *exit.Error) {
	rows, problem := store.ModelProductionArtifacts(operationID)
	if problem != nil {
		return nil, problem
	}
	out := make(map[string]bool, len(rows))
	for _, row := range rows {
		out[row.NodeName+"."+row.OutputSlot] = true
	}
	return out, nil
}

func productionSourceBytes(files []modelproduction.SourceFile) int64 {
	var total int64
	for _, file := range files {
		total += file.Length
	}
	return total
}

func productionSourceProgress(files []records.ModelProductionSourceFile) (transferred, total int64) {
	for _, file := range files {
		transferred += file.TransferredBytes
		total += file.Length
	}
	return transferred, total
}

func productionResumeStage(operation records.ModelProductionOperation,
	ordered []launch.ModelProductionNode, transferred, total int64, lanes int,
) string {
	rental := operation.RentalID
	if rental == "" {
		rental = "not assigned"
	}
	switch operation.State {
	case "accepted":
		return "rental selection"
	case "source_preparing":
		if total > 0 && transferred > 0 {
			return fmt.Sprintf("source preparation has transferred %s / %s on rental %s",
				output.Bytes(transferred), output.Bytes(total), rental)
		}
		return "source preparation on rental " + rental
	case "source_prepared":
		return fmt.Sprintf("source prepared; next is node 1/%d", len(ordered))
	case "node_running":
		if len(ordered) == 0 {
			return "node execution"
		}
		index := max(0, min(len(ordered)-1, int(operation.NodeIndex)))
		node := ordered[index]
		return fmt.Sprintf("node %d/%d %s (%s)", index+1, len(ordered), node.Name, node.Callable)
	case "outputs_preparing":
		return fmt.Sprintf("all %d nodes complete; preparing %d lane publications", len(ordered), lanes)
	case "release_cut":
		return "release cut; rental cleanup remains for " + rental
	case "cleanup_pending":
		return "confirming provider absence for rental " + rental
	case "completed":
		return "already completed; no work repeated"
	case "failed", "canceled":
		return "already " + operation.State + "; no work repeated"
	default:
		return "durable state " + operation.State
	}
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
	operationID, requestID string, progress *productionProgress, nodeIndex int,
	nodeName string,
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
		stage, fraction, measured := productionRuntimeProgress(job)
		progress.NodeRuntime(nodeIndex, nodeName, stage, fraction, measured)
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

func productionRuntimeProgress(job api.JobState) (string, float64, bool) {
	stage := strings.TrimSpace(job.Stage)
	if stage == "" {
		if value, ok := job.Progress["stage"].(string); ok {
			stage = strings.TrimSpace(value)
		} else if value, ok := job.Progress["name"].(string); ok {
			stage = strings.TrimSpace(value)
		}
	}
	for _, key := range []string{"fraction", "value"} {
		if fraction, ok := number(job.Progress[key]); ok && fraction >= 0 && fraction <= 1 {
			return stage, fraction, true
		}
	}
	return stage, 0, false
}

func waitNodeArtifacts(runCtx context.Context, store *records.Store, operationID,
	nodeName string, outputs []string,
) ([]records.ModelProductionArtifact, *exit.Error) {
	want := make(map[string]bool, len(outputs))
	for _, outputSlot := range outputs {
		want[outputSlot] = true
	}
	for {
		if cancelProblem := productionCancellation(runCtx, store, operationID); cancelProblem != nil {
			return nil, cancelProblem
		}
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

func publishProductionArtifact(runCtx context.Context, ctx *Context, local *localclient.Client,
	store *records.Store, plan modelproduction.Plan, rentalID string,
	artifact records.ModelProductionArtifact, progress *productionProgress,
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
	if problem := productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return productionManifest{}, problem
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
	progress.PublicationStarting(lane)
	publicationOperation := productionPublicationOperation(plan.ID(), artifact.NodeName,
		artifact.OutputSlot)
	reason := "cozy model publish " + plan.Destination + "@" + plan.Release
	ref, parseProblem := hub.ParseRef(plan.Destination)
	if parseProblem != nil {
		return productionManifest{}, parseProblem
	}
	hctx, cancel := productionHubContext(runCtx)
	opened, problem := client(ctx).OpenPublication(hctx, ref, publicationOperation,
		plan.Release, lane, hubObjects, reason)
	cancel()
	if problem != nil {
		if runCtx.Err() != nil {
			return productionManifest{}, exit.New(exit.Canceled,
				"model production %s was interrupted while opening lane %s", plan.ID(), lane)
		}
		return productionManifest{}, problem
	}
	_ = opened
	for start := 0; start < len(objects); start += productionTransferBatch {
		if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
			return productionManifest{}, problem
		}
		end := min(start+productionTransferBatch, len(objects))
		ids := make([]string, 0, end-start)
		for _, object := range objects[start:end] {
			ids = append(ids, object.ObjectID)
		}
		hctx, cancel = productionHubContext(runCtx)
		granted, grantProblem := client(ctx).GrantKnownTransfers(hctx, ref,
			publicationOperation, ids, reason)
		cancel()
		if grantProblem != nil {
			if runCtx.Err() != nil {
				return productionManifest{}, exit.New(exit.Canceled,
					"model production %s was interrupted while granting lane %s transfers",
					plan.ID(), lane)
			}
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
		_, problem = local.ModelProductionActionContext(runCtx, plan.ID(), api.ModelProductionAction{
			Action: "transfer_artifact", RentalID: rentalID,
			Artifact: &api.ModelProductionArtifactAction{NodeName: artifact.NodeName,
				OutputSlot: artifact.OutputSlot, TransferOperationID: publicationOperation,
				Decisions: decisions},
		})
		if problem != nil {
			if runCtx.Err() != nil {
				return productionManifest{}, exit.New(exit.Canceled,
					"model production %s was interrupted while transferring lane %s", plan.ID(), lane)
			}
			return productionManifest{}, problem
		}
	}
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return productionManifest{}, problem
	}
	hctx, cancel = productionHubContext(runCtx)
	prepared, problem := client(ctx).FinalizePublication(hctx, ref, publicationOperation,
		hub.FinalizePublicationRequest{ManifestID: artifact.ManifestID,
			ManifestLength:        artifact.ManifestLength,
			ReleaseEvidenceBase64: base64.StdEncoding.EncodeToString(artifact.ReleaseEvidence)}, reason)
	cancel()
	if problem != nil {
		if runCtx.Err() != nil {
			return productionManifest{}, exit.New(exit.Canceled,
				"model production %s was interrupted while finalizing lane %s", plan.ID(), lane)
		}
		return productionManifest{}, problem
	}
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
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
	progress.PublicationPrepared(lane)
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

func cutProductionRelease(runCtx context.Context, ctx *Context, store *records.Store,
	plan modelproduction.Plan,
	progress *productionProgress,
) *exit.Error {
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
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return problem
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
	lanes := plan.Lanes()
	progress.ReleaseStarting(plan.Release, lanes)
	ref, parseProblem := hub.ParseRef(plan.Destination)
	if parseProblem != nil {
		return parseProblem
	}
	hctx, cancel := hub.LongContext()
	cut, problem := client(ctx).CutRelease(hctx, ref, plan.Release, plan.ID(), publicationIDs,
		"cozy model publish "+plan.Destination+"@"+plan.Release)
	cancel()
	if problem != nil {
		if problem.Code == exit.Unavailable || problem.Code == exit.Deadline {
			return exit.Named(problem.Code, "model_production.cut_verdict_unknown",
				"release cut verdict is unknown; replaying the same operation: %s", problem.Message)
		}
		return problem
	}
	if cut.Release != plan.Release || len(cut.Lanes) != len(plan.Production.Outputs) {
		return exit.Named(exit.Conflict, "model_production.cut_changed",
			"Tensorhub cut an incomplete or different model release")
	}
	if problem = store.AdvanceModelProduction(plan.ID(), "outputs_preparing", "release_cut",
		current.NodeIndex, current.RentalID); problem != nil {
		return problem
	}
	// Cancellation racing the unary cut is observed only after its exact response is
	// journaled. A committed cut wins and cleanup continues; it is never rewritten
	// as a failed or invisible publication.
	if cancelProblem := productionCancellation(runCtx, store, plan.ID()); cancelProblem != nil &&
		cancelProblem.Code != exit.Canceled {
		return cancelProblem
	}
	progress.ReleaseCut(plan.Release, len(lanes))
	return nil
}

func productionCancellation(runCtx context.Context, store *records.Store,
	operationID string,
) *exit.Error {
	if runCtx.Err() != nil {
		_ = store.RequestModelProductionCancel(operationID)
		return exit.New(exit.Canceled, "model production %s was interrupted", operationID)
	}
	operation, problem := store.ModelProduction(operationID)
	if problem != nil {
		return problem
	}
	if operation != nil && operation.CancelRequested {
		return exit.New(exit.Canceled, "model production %s was canceled", operationID)
	}
	return nil
}

func productionHubContext(runCtx context.Context) (context.Context, context.CancelFunc) {
	hctx, cancel := hub.LongContext()
	stop := context.AfterFunc(runCtx, cancel)
	return hctx, func() {
		stop()
		cancel()
	}
}

func failAndReleaseProduction(ctx *Context, store *records.Store, operationID, rentalID string,
	cause *exit.Error, progress *productionProgress,
) *exit.Error {
	failed := failProduction(store, operationID, cause)
	if cause.Code == exit.Canceled {
		progress.Cancellation(operationID)
	} else {
		progress.Failed(operationID)
	}
	progress.RentalReleaseStarting(rentalID)
	if release := endRentalSilently(ctx, rentalID); release != nil {
		progress.RentalReleaseUnconfirmed(rentalID)
		return release.WithRemedy("model production failed, and rental %s may still be billing; release it explicitly",
			rentalID).WithNext("cozy rental end " + rentalID)
	}
	progress.RentalReleased(rentalID)
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
