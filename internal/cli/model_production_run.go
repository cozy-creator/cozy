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
	"github.com/cozy-creator/cozy/internal/records"
)

const productionTransferBatch = 128

type productionManifest struct {
	ID, Evidence string
	Length       int64
}

func runRentedModelProduction(ctx *Context, runCtx context.Context,
	operation records.ModelProductionOperation, plan modelproduction.Plan, source publishSource,
) *exit.Error {
	if plan.Production == nil || len(plan.Jobs) == 0 {
		return exit.Named(exit.Structural, "model_production.plan_incomplete",
			"rented model production requires a reviewed graph and exact jobs")
	}
	ordered, problem := plan.Production.OrderedSteps()
	if problem != nil {
		return problem
	}
	progress := NewModelProductionProgress(ctx.Err, !ctx.Mode().JSON, len(ordered))
	layout, store, problem := rentalStores(ctx)
	if problem != nil {
		return problem
	}
	defer store.Close()
	existingArtifacts, problem := productionArtifactSnapshot(store, operation.ID)
	if problem != nil {
		return problem
	}
	files, problem := store.ModelProductionSourceFiles(operation.ID)
	if problem != nil {
		return problem
	}
	transferred, total := productionSourceProgress(files)
	progress.SeedSourceProgress(transferred, total)
	progress.Resume(operation.ID, productionResumeStage(operation, ordered, transferred, total,
		len(plan.Production.Outputs)))
	if operation.State == "completed" || operation.State == "partial" {
		return emitCompletedProduction(ctx, store, plan, &operation)
	}
	if operation.State == "failed" || operation.State == "canceled" {
		return exit.Named(exit.Conflict, "model_production.settled",
			"model production %s is already %s: %s", operation.ID, operation.State,
			operation.SafeDetail)
	}
	if operation.State == "cleanup_pending" {
		return resumeProductionCleanup(ctx, store, plan, operation, progress)
	}
	if operation.State == "outputs_preparing" {
		if problem := finishProductionOutputs(runCtx, store, plan, progress); problem != nil {
			return problem
		}
		current, problem := store.ModelProduction(operation.ID)
		if problem != nil || current == nil {
			return problem
		}
		return resumeProductionCleanup(ctx, store, plan, *current, progress)
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

	announceSource := operation.State != "source_preparing"
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
		pins[pin.Step] = pin
	}

	for index, step := range ordered {
		if cancelProblem := productionCancellation(runCtx, store, operation.ID); cancelProblem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				cancelProblem, progress)
		}
		pin, ok := pins[step.Name]
		if !ok {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
				exit.Internalf("model production step %s has no exact job pin", step.Name), progress)
		}
		row, problem := store.BeginModelProductionStep(records.ModelProductionStep{
			OperationID: operation.ID, StepIndex: int64(index), StepName: step.Name,
			State: "pending",
		})
		if problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		if row.State == "failed" || row.State == "skipped" {
			if row.State == "failed" {
				progress.StepFailed(index, step.Name)
			} else {
				progress.StepSkipped(index, step.Name)
			}
			if problem = advanceProductionStep(store, operation.ID, rentalID, int64(index+1),
				index == len(ordered)-1); problem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
			}
			continue
		}
		stepCompleted := row.State == "completed"
		if !stepCompleted {
			progress.StepStarting(index, step.Name, step.Callable, row.RequestID != "")
			packageName, function, _ := splitProductionCallable(step.Callable)
			models := make([]orchestrator.ModelRef, 0, len(step.Models))
			missingDependency := ""
			for parameter, reference := range step.Models {
				manifest, available := manifests[reference]
				if !available {
					if _, sourceSlot := plan.Production.Sources[reference]; sourceSlot {
						return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
							exit.Named(exit.Conflict, "model_production.source_input_absent",
								"step %s source input %s has no prepared Manifest", step.Name,
								reference), progress)
					}
					missingDependency = reference
					break
				}
				models = append(models, orchestrator.ModelRef{Package: packageName, Slot: parameter,
					Model: operation.ID + "/" + reference, Manifest: manifest.ID,
					ManifestLength: manifest.Length})
			}
			if missingDependency != "" {
				if problem = store.SetModelProductionStepRequest(operation.ID, int64(index),
					step.Name, row.RequestID, "skipped"); problem != nil {
					return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
				}
				progress.StepSkipped(index, step.Name)
				if problem = advanceProductionStep(store, operation.ID, rentalID, int64(index+1),
					index == len(ordered)-1); problem != nil {
					return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
				}
				continue
			}
			sort.Slice(models, func(i, j int) bool { return models[i].Slot < models[j].Slot })
			handle, submitProblem := local.SubmitJob(api.JobSubmission{
				Package: packageName, Function: function, Input: []byte("{}"),
				Release: pin.Release, ReleaseDigest: pin.ReleaseDigest,
				Rental: true, Worker: rentalID, Models: models,
				Org:                   strings.Split(plan.Destination, "/")[0],
				ProductionOperationID: operation.ID, ProductionStep: step.Name,
				ProductionStepIndex: int64(index),
			}, productionRequestKey(operation.ID, step.Name))
			if submitProblem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, submitProblem, progress)
			}
			if problem = waitProductionJob(runCtx, local, store, operation.ID, handle.JobID,
				progress, index, step.Name); problem != nil {
				if problem.Code != exit.Failed {
					return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
				}
				if recordProblem := store.SetModelProductionStepRequest(operation.ID, int64(index),
					step.Name, handle.JobID, "failed"); recordProblem != nil {
					return failAndReleaseProduction(ctx, store, operation.ID, rentalID, recordProblem,
						progress)
				}
				progress.StepFailed(index, step.Name)
				if advanceProblem := advanceProductionStep(store, operation.ID, rentalID,
					int64(index+1), index == len(ordered)-1); advanceProblem != nil {
					return failAndReleaseProduction(ctx, store, operation.ID, rentalID,
						advanceProblem, progress)
				}
				continue
			}
		}
		artifacts, problem := waitStepArtifacts(runCtx, store, operation.ID, step.Name,
			step.Outputs)
		if problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		for artifactIndex, artifact := range artifacts {
			source := artifact.StepName + "." + artifact.OutputSlot
			if !existingArtifacts[source] {
				progress.ArtifactAdopted(index, artifactIndex, len(artifacts), step.Name)
			}
			manifest, publicationProblem := publishProductionArtifact(runCtx, ctx, local,
				store, plan, rentalID, artifact, progress)
			if publicationProblem != nil {
				return failAndReleaseProduction(ctx, store, operation.ID, rentalID, publicationProblem, progress)
			}
			manifests[step.Name+"."+artifact.OutputSlot] = manifest
		}
		if problem = store.SetModelProductionStepRequest(operation.ID, int64(index), step.Name,
			artifacts[0].RequestID, "completed"); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		if problem = advanceProductionStep(store, operation.ID, rentalID, int64(index+1),
			index == len(ordered)-1); problem != nil {
			return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
		}
		if !stepCompleted {
			progress.StepCompleted(index, step.Name, step.Callable)
		}
	}

	if problem = finishProductionOutputs(runCtx, store, plan, progress); problem != nil {
		return failAndReleaseProduction(ctx, store, operation.ID, rentalID, problem, progress)
	}
	current, problem := store.ModelProduction(operation.ID)
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
	current, problem = settleProductionUpload(store, plan, current)
	if problem != nil {
		return problem
	}
	return emitCompletedProduction(ctx, store, plan, current)
}

func resumeProductionCleanup(ctx *Context, store *records.Store, plan modelproduction.Plan,
	operation records.ModelProductionOperation, progress *productionProgress,
) *exit.Error {
	if operation.RentalID != "" {
		progress.RentalReleaseStarting(operation.RentalID)
		if problem := endRentalSilently(ctx, operation.RentalID); problem != nil {
			progress.RentalReleaseUnconfirmed(operation.RentalID)
			return problem.WithRemedy("the model checkpoints are retained; confirm rental %s absence to finish cleanup",
				operation.RentalID).WithNext("cozy rental end " + operation.RentalID)
		}
		progress.RentalReleased(operation.RentalID)
	}
	settled, problem := settleProductionUpload(store, plan, &operation)
	if problem != nil {
		return problem
	}
	return emitCompletedProduction(ctx, store, plan, settled)
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
	progress.RentalSelecting(sku.Name)
	row, _, _, problem := acquireRentalContext(runCtx, ctx, layout, store, sku.Name, "",
		operationKey, "cozy model upload "+plan.Destination,
		rate, ctx.Cfg.RentalsMaxHourlySpendUSDMicros, time.Time{}, "")
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
	// rows at the same cadence as step state, while Progress suppresses unchanged
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
			Evidence: base64.StdEncoding.EncodeToString(row.CheckpointEvidence)}
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
		out[row.StepName+"."+row.OutputSlot] = true
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
	ordered []launch.ModelProductionStep, transferred, total int64, outputs int,
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
		return fmt.Sprintf("source prepared; next is step 1/%d", len(ordered))
	case "step_running":
		if len(ordered) == 0 {
			return "step execution"
		}
		index := max(0, min(len(ordered)-1, int(operation.StepIndex)))
		step := ordered[index]
		return fmt.Sprintf("step %d/%d %s (%s)", index+1, len(ordered), step.Name, step.Callable)
	case "outputs_preparing":
		return fmt.Sprintf("all %d steps complete; retaining %d output checkpoints", len(ordered), outputs)
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
	target, problem := parseProductionCallable(value)
	if problem != nil {
		return "", "", false
	}
	return target.Package, target.Function, true
}

func productionRequestKey(operationID, step string) string {
	return "model-production:" + operationID + ":" + step
}

func waitProductionJob(runCtx context.Context, local *localclient.Client, store *records.Store,
	operationID, requestID string, progress *productionProgress, stepIndex int,
	stepName string,
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
		progress.StepRuntime(stepIndex, stepName, stage, fraction, measured)
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

func waitStepArtifacts(runCtx context.Context, store *records.Store, operationID,
	stepName string, outputs []string,
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
			if row.StepName == stepName && want[row.OutputSlot] {
				found = append(found, row)
			}
		}
		if len(found) == len(want) {
			sort.Slice(found, func(i, j int) bool { return found[i].OutputSlot < found[j].OutputSlot })
			return found, nil
		}
		select {
		case <-runCtx.Done():
			return nil, exit.New(exit.Canceled, "model production interrupted waiting for step %s artifacts",
				stepName)
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func publishProductionArtifact(runCtx context.Context, ctx *Context, local *localclient.Client,
	store *records.Store, plan modelproduction.Plan, rentalID string,
	artifact records.ModelProductionArtifact, progress *productionProgress,
) (productionManifest, *exit.Error) {
	source := artifact.StepName + "." + artifact.OutputSlot
	contract, final := productionContract(plan, source)
	if !final {
		// Intermediate Manifests stay under the worker's adopted artifact root and
		// feed the next step directly. Only named outputs become owner checkpoints.
		return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
			Evidence: base64.StdEncoding.EncodeToString(artifact.CheckpointEvidence)}, nil
	}
	if artifact.PublicationID != "" && artifact.State == "prepared" {
		return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
			Evidence: base64.StdEncoding.EncodeToString(artifact.CheckpointEvidence)}, nil
	}
	if problem := productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return productionManifest{}, problem
	}
	objects, problem := store.ModelProductionObjects(plan.ID(), artifact.StepName,
		artifact.OutputSlot)
	if problem != nil {
		return productionManifest{}, problem
	}
	hubObjects := make([]hub.Object, 0, len(objects))
	for _, object := range objects {
		hubObjects = append(hubObjects, hub.Object{ID: object.ObjectID, Length: object.Length})
	}
	outputName := contract.Name
	progress.PublicationStarting(outputName)
	publicationOperation := productionPublicationOperation(plan.ID(), artifact.StepName,
		artifact.OutputSlot)
	reason := "cozy model upload " + plan.Destination
	ref, parseProblem := hub.ParseRef(plan.Destination)
	if parseProblem != nil {
		return productionManifest{}, parseProblem
	}
	hctx, cancel := productionHubContext(runCtx)
	opened, problem := client(ctx).OpenPublication(hctx, ref, publicationOperation,
		hubObjects, reason)
	cancel()
	if problem != nil {
		if runCtx.Err() != nil {
			return productionManifest{}, exit.New(exit.Canceled,
				"model upload %s was interrupted while opening output %s", plan.ID(), outputName)
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
					"model upload %s was interrupted while granting output %s transfers",
					plan.ID(), outputName)
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
			Artifact: &api.ModelProductionArtifactAction{StepName: artifact.StepName,
				OutputSlot: artifact.OutputSlot, TransferOperationID: publicationOperation,
				Decisions: decisions},
		})
		if problem != nil {
			if runCtx.Err() != nil {
				return productionManifest{}, exit.New(exit.Canceled,
					"model upload %s was interrupted while transferring output %s", plan.ID(), outputName)
			}
			return productionManifest{}, problem
		}
	}
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return productionManifest{}, problem
	}
	hctx, cancel = productionHubContext(runCtx)
	checkpoint, problem := client(ctx).FinalizePublication(hctx, ref, publicationOperation,
		hub.FinalizePublicationRequest{ManifestID: artifact.ManifestID,
			ManifestLength:           artifact.ManifestLength,
			CheckpointEvidenceBase64: base64.StdEncoding.EncodeToString(artifact.CheckpointEvidence)}, reason)
	cancel()
	if problem != nil {
		if runCtx.Err() != nil {
			return productionManifest{}, exit.New(exit.Canceled,
				"model upload %s was interrupted while finalizing output %s", plan.ID(), outputName)
		}
		return productionManifest{}, problem
	}
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return productionManifest{}, problem
	}
	if checkpoint.CheckpointID != artifact.ManifestID ||
		checkpoint.Manifest.SHA256 != strings.TrimPrefix(artifact.ManifestID, "sha256:") ||
		checkpoint.Manifest.Length != artifact.ManifestLength {
		return productionManifest{}, exit.Named(exit.Conflict, "model_production.finalize_changed",
			"Tensorhub finalized a different Manifest for %s.%s", artifact.StepName,
			artifact.OutputSlot)
	}
	if checkpoint.TopologyDigest != contract.TopologyDigest ||
		!sameStrings(checkpoint.Contract.Encoding.Set, contract.Encodings) {
		return productionManifest{}, exit.Named(exit.Conflict,
			"model_production.contract_mismatch",
			"Tensorhub-derived contract for output %s does not match the reviewed production", outputName)
	}
	if problem = store.MarkModelProductionArtifactPublished(plan.ID(), artifact.StepName,
		artifact.OutputSlot, checkpoint.PublishID); problem != nil {
		return productionManifest{}, problem
	}
	progress.PublicationPrepared(outputName)
	return productionManifest{ID: artifact.ManifestID, Length: artifact.ManifestLength,
		Evidence: base64.StdEncoding.EncodeToString(artifact.CheckpointEvidence)}, nil
}

func productionPublicationOperation(operationID, step, slot string) string {
	sum := sha256.Sum256([]byte(operationID + "\x00" + step + "\x00" + slot))
	return "model-artifact-" + hex.EncodeToString(sum[:])
}

func productionContract(plan modelproduction.Plan, source string) (
	modelproductionContract, bool,
) {
	for _, output := range plan.Production.Outputs {
		if output.Source == source {
			return modelproductionContract{Name: output.Name,
				TopologyDigest: output.RequiredContract.TopologyDigest,
				Encodings:      output.RequiredContract.Encodings}, true
		}
	}
	return modelproductionContract{}, false
}

type modelproductionContract struct {
	Name           string
	TopologyDigest string
	Encodings      []string
}

func sameStrings(left, right []string) bool {
	left, right = append([]string(nil), left...), append([]string(nil), right...)
	sort.Strings(left)
	sort.Strings(right)
	return strings.Join(left, "\x00") == strings.Join(right, "\x00")
}

func advanceProductionStep(store *records.Store, operationID, rentalID string,
	next int64, last bool,
) *exit.Error {
	current, problem := store.ModelProduction(operationID)
	if problem != nil || current == nil {
		return problem
	}
	if current.State == "step_running" && current.StepIndex >= next ||
		current.State == "outputs_preparing" ||
		current.State == "cleanup_pending" || current.State == "completed" {
		return nil
	}
	if current.State == "source_prepared" {
		if problem = store.AdvanceModelProduction(operationID, "source_prepared", "step_running",
			0, rentalID); problem != nil {
			return problem
		}
		current, problem = store.ModelProduction(operationID)
	}
	if problem != nil || current == nil {
		return problem
	}
	to := "step_running"
	if last {
		to = "outputs_preparing"
	}
	return store.AdvanceModelProduction(operationID, "step_running", to, next, rentalID)
}

func finishProductionOutputs(runCtx context.Context, store *records.Store,
	plan modelproduction.Plan, progress *productionProgress,
) *exit.Error {
	current, problem := store.ModelProduction(plan.ID())
	if problem != nil || current == nil {
		return problem
	}
	if current.State == "cleanup_pending" || current.State == "completed" {
		return nil
	}
	if current.State != "outputs_preparing" {
		return exit.Named(exit.Conflict, "model_upload.output_state_invalid",
			"model upload %s is %s before output completion", plan.ID(), current.State)
	}
	if problem = productionCancellation(runCtx, store, plan.ID()); problem != nil {
		return problem
	}
	checkpoints, problem := modelProductionCheckpoints(store, plan)
	if problem != nil {
		return problem
	}
	if problem = store.AdvanceModelProduction(plan.ID(), "outputs_preparing", "cleanup_pending",
		current.StepIndex, current.RentalID); problem != nil {
		return problem
	}
	progress.OutputsRetained(len(checkpoints))
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
		current.State != "completed" && current.State != "partial" {
		if cause.Code == exit.Canceled {
			_ = store.CancelModelProduction(operationID, current.State, cause.Message)
		} else {
			_ = store.FailModelProduction(operationID, current.State, cause.Name, cause.Message)
		}
	}
	return cause
}

func emitCompletedProduction(ctx *Context, store *records.Store, plan modelproduction.Plan,
	operation *records.ModelProductionOperation,
) *exit.Error {
	rentalID := ""
	if operation != nil {
		rentalID = operation.RentalID
	}
	checkpoints, problem := modelProductionCheckpoints(store, plan)
	if problem != nil {
		return problem
	}
	return emit(ctx, compactRecord([]output.Field{
		{K: "id", V: plan.ID()}, {K: "kind", V: "model-upload"},
		{K: "model", V: plan.Destination}, {K: "checkpoints", V: checkpoints},
		{K: "rental", V: rentalID},
		{K: "status", V: operation.State}, {K: "changed", V: false},
	}, "model", "checkpoints", "status", "changed"))
}

func settleProductionUpload(store *records.Store, plan modelproduction.Plan,
	operation *records.ModelProductionOperation,
) (*records.ModelProductionOperation, *exit.Error) {
	if operation == nil {
		return nil, exit.Internalf("model upload %s disappeared before settlement", plan.ID())
	}
	if operation.State == "completed" || operation.State == "partial" {
		return operation, nil
	}
	if operation.State != "cleanup_pending" {
		return nil, exit.Named(exit.Conflict, "model_upload.settlement_state_invalid",
			"model upload %s is %s before settlement", plan.ID(), operation.State)
	}
	checkpoints, problem := modelProductionCheckpoints(store, plan)
	if problem != nil {
		return nil, problem
	}
	if len(checkpoints) == 0 {
		if problem = store.FailModelProduction(plan.ID(), "cleanup_pending", "model_upload.no_outputs",
			"no named output checkpoint survived"); problem != nil {
			return nil, problem
		}
		failed, readProblem := store.ModelProduction(plan.ID())
		if readProblem != nil {
			return nil, readProblem
		}
		return failed, exit.Named(exit.Failed, "model_upload.no_outputs",
			"model upload %s produced no checkpoint", plan.ID())
	}
	status := "completed"
	if len(checkpoints) != len(plan.Production.Outputs) {
		status = "partial"
	}
	if problem = store.AdvanceModelProduction(plan.ID(), "cleanup_pending", status,
		operation.StepIndex, operation.RentalID); problem != nil {
		return nil, problem
	}
	return store.ModelProduction(plan.ID())
}

func modelProductionCheckpoints(store *records.Store, plan modelproduction.Plan) (map[string]string, *exit.Error) {
	artifacts, problem := store.ModelProductionArtifacts(plan.ID())
	if problem != nil {
		return nil, problem
	}
	bySource := make(map[string]records.ModelProductionArtifact, len(artifacts))
	for _, artifact := range artifacts {
		bySource[artifact.StepName+"."+artifact.OutputSlot] = artifact
	}
	checkpoints := make(map[string]string, len(plan.Production.Outputs))
	for _, output := range plan.Production.Outputs {
		artifact, ok := bySource[output.Source]
		if ok && artifact.PublicationID != "" && artifact.ManifestID != "" {
			checkpoints[output.Name] = artifact.ManifestID
		}
	}
	return checkpoints, nil
}

func firstNonempty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return "unknown"
}
