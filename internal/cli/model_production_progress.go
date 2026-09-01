package cli

import (
	"fmt"
	"io"
	"strings"

	"github.com/cozy-creator/cozy/internal/output"
)

// productionProgress is the human-only projection of one durable model production. It
// never owns lifecycle state: every line is rendered only after its caller has
// observed the corresponding record, Runtime sample, or remote acknowledgement.
type productionProgress struct {
	w          io.Writer
	enabled    bool
	totalSteps int
	sourceBand int
	stepStage  map[int]string
	stepBand   map[int]int
}

// NewModelProductionProgress is exported only across Cozy's internal product-test
// boundary; the public CLI surface remains the model publish command itself.
func NewModelProductionProgress(w io.Writer, human bool, totalSteps int) *productionProgress {
	return &productionProgress{w: w, enabled: human, totalSteps: totalSteps,
		stepStage: map[int]string{}, stepBand: map[int]int{}}
}

func (p *productionProgress) line(format string, args ...any) {
	if p == nil || !p.enabled {
		return
	}
	_ = output.Progress(p.w, fmt.Sprintf(format, args...))
}

func (p *productionProgress) Accepted(id string, outputs int) {
	p.line("Model upload %s accepted: %d steps, %d outputs.", id, p.totalSteps, outputs)
}

func (p *productionProgress) Resume(id, stage string) {
	p.line("Resuming model production %s: %s.", id, strings.TrimSuffix(stage, "."))
}

func (p *productionProgress) RentalSelecting(sku string) {
	p.line("Rental: selected %s; waiting for provider readiness.", sku)
}

func (p *productionProgress) RentalReady(id, state, sku string) {
	p.line("Rental %s: %s (%s).", id, state, sku)
}

func (p *productionProgress) WorkerWaiting(rentalID string) {
	p.line("Worker: waiting for rental %s to connect.", rentalID)
}

func (p *productionProgress) WorkerReady(rentalID string) {
	p.line("Worker: ready on rental %s.", rentalID)
}

func (p *productionProgress) SourceStarting(files int, totalBytes int64) {
	p.line("Source: preparing %d files (%s).", files, output.Bytes(totalBytes))
}

// SeedSourceProgress prevents a resumed invocation from replaying byte bands the
// previous terminal already displayed. Resume itself reports the current stage.
func (p *productionProgress) SeedSourceProgress(transferred, total int64) {
	if total > 0 && transferred > 0 {
		p.sourceBand = min(100, int(transferred*100/total)/10*10)
	}
}

// SourceProgress prints only newly observed ten-percent bands. transferred is
// Runtime's journaled byte count; this renderer never estimates movement.
func (p *productionProgress) SourceProgress(transferred, total int64) {
	if total <= 0 || transferred <= 0 {
		return
	}
	band := min(100, int(transferred*100/total)/10*10)
	if band <= p.sourceBand || band == 0 {
		return
	}
	p.sourceBand = band
	p.line("Source: downloaded %s / %s (%d%%).", output.Bytes(transferred), output.Bytes(total), band)
}

func (p *productionProgress) SourcePrepared(profiles int) {
	p.line("Source: prepared %d model profiles.", profiles)
}

func (p *productionProgress) StepStarting(index int, name, callable string, resumed bool) {
	verb := "starting"
	if resumed {
		verb = "resuming"
	}
	p.line("Step %d/%d: %s %s (%s).", index+1, p.totalSteps, verb, name, callable)
}

// StepRuntime renders only a new Runtime stage or ten-percent fraction band.
// Poll cadence and elapsed time never become user-visible progress.
func (p *productionProgress) StepRuntime(index int, name, stage string, fraction float64, measured bool) {
	stage = strings.TrimSpace(stage)
	stageChanged := stage != "" && stage != p.stepStage[index]
	if stageChanged {
		p.stepStage[index] = stage
		p.stepBand[index] = 0
	}
	band := p.stepBand[index]
	if measured && fraction >= 0 && fraction <= 1 {
		band = min(100, int(fraction*100)/10*10)
	}
	bandChanged := measured && band > p.stepBand[index] && band > 0
	if !stageChanged && !bandChanged {
		return
	}
	if bandChanged {
		p.stepBand[index] = band
	}
	detail := stage
	if detail == "" {
		detail = "running"
	}
	if bandChanged {
		detail = fmt.Sprintf("%s (%d%%)", detail, band)
	}
	p.line("Step %d/%d: %s — %s.", index+1, p.totalSteps, name, detail)
}

func (p *productionProgress) ArtifactAdopted(stepIndex, outputIndex, outputs int, stepName string) {
	p.line("Artifact: adopted output %d/%d for step %d/%d %s.", outputIndex+1, outputs,
		stepIndex+1, p.totalSteps, stepName)
}

func (p *productionProgress) PublicationStarting(outputName string) {
	p.line("Checkpoint: uploading output %s.", outputName)
}

func (p *productionProgress) PublicationPrepared(outputName string) {
	p.line("Checkpoint: output %s retained.", outputName)
}

func (p *productionProgress) StepCompleted(index int, name, callable string) {
	p.line("Step %d/%d: completed %s (%s).", index+1, p.totalSteps, name, callable)
}

func (p *productionProgress) OutputsRetained(outputs int) {
	p.line("Upload: retained %d owner-only checkpoints.", outputs)
}

func (p *productionProgress) Cancellation(id string) {
	p.line("Cancellation: model production %s stopped; cleanup is continuing.", id)
}

func (p *productionProgress) Failed(id string) {
	p.line("Failure: model production %s stopped; cleanup is continuing.", id)
}

func (p *productionProgress) RentalReleaseStarting(id string) {
	p.line("Cleanup: releasing rental %s.", id)
}

func (p *productionProgress) RentalReleased(id string) {
	p.line("Cleanup: rental %s released; provider absence confirmed.", id)
}

func (p *productionProgress) RentalReleaseUnconfirmed(id string) {
	p.line("Cleanup: rental %s release is not yet confirmed.", id)
}
