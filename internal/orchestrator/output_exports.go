package orchestrator

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/reclaim"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
	"github.com/cozy-creator/cozy/internal/units"
)

// RetryOutputExport settles one daemon-owned publication obligation — the package's
// store under outputs/ or the caller's --out. Serving files are already at that
// destination. Top-level jobs retain internal publication custody and materialize
// independent user copies of their declared media. Both settle the same export row
// after proving the accepted set matches its pre-execution contract.
// It never changes the execution terminal and never trusts a terminal path.
func (c *Orchestrator) RetryOutputExport(requestID string) {
	c.mu.Lock()
	if c.outputExporting[requestID] {
		c.mu.Unlock()
		return
	}
	c.outputExporting[requestID] = true
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.outputExporting, requestID)
		c.mu.Unlock()
	}()

	export, problem := c.opt.Store.OutputExportOf(requestID)
	if problem != nil {
		c.logf("output export %s cannot be read: %s", requestID, problem.Message)
		return
	}
	if export == nil || export.State == "published" || export.State == "skipped" {
		return
	}
	request, problem := c.opt.Store.RequestRow(requestID)
	if problem != nil || request == nil {
		c.logf("output export %s has no readable request", requestID)
		return
	}
	if request.State != "succeeded" {
		if request.State == "failed" || request.State == "canceled" || request.State == "refused" ||
			request.State == "abandoned" {
			if problem := c.opt.Store.SkipOutputExport(requestID,
				"execution ended "+request.State+" and published no successful result"); problem != nil {
				c.logf("output export %s skip could not be recorded: %s", requestID, problem.Message)
			}
		}
		return
	}
	if request.ParentRequestID != "" {
		_ = c.opt.Store.SkipOutputExport(requestID, "child results remain internal to their parent")
		return
	}
	outputs, problem := c.opt.Store.VisibleOutputs(requestID)
	if problem != nil {
		c.failOutputExport(requestID, problem)
		return
	}
	if request.IsJob() {
		outputs, problem = materializeJobMedia(*export, outputs)
		if problem != nil {
			c.failOutputExport(requestID, problem)
			return
		}
	}
	paths, problem := outputExportPaths(*export, outputs)
	if problem != nil {
		c.failOutputExport(requestID, problem)
		return
	}
	if problem := c.opt.Store.CompleteOutputExport(requestID, paths); problem != nil {
		c.logf("output export %s settlement failed: %s", requestID, problem.Message)
		return
	}
	c.logf("output export %s published %d file(s) under %s", requestID, len(paths), export.Directory)
	c.reclaimTmp(requestID)
}

func materializeJobMedia(export records.OutputExport, outputs []records.Output) ([]records.Output, *exit.Error) {
	accepted := make(map[string]records.Output, len(outputs))
	for _, output := range outputs {
		accepted[output.OutputID] = output
	}
	media := make([]records.Output, 0, len(export.Outputs))
	for _, intended := range export.Outputs {
		output, ok := accepted[intended.OutputID]
		if !ok || output.MimeType != intended.MediaType {
			return nil, exit.Named(exit.Validation, "output_export_contract_mismatch",
				"accepted output %s does not match its pre-execution media contract", intended.OutputID)
		}
		path, problem := resultfiles.Materialize(output.Path, export.Directory, output.Digest, output.MimeType, output.Length)
		if problem != nil {
			return nil, problem
		}
		output.Path = path
		media = append(media, output)
	}
	return media, nil
}

// reclaimTmp removes one settled request's `tmp/<id>/` if its writer did not — the
// backstop, run after an ack and after the export settles; either may come second, and
// both are safe to repeat.
func (c *Orchestrator) reclaimTmp(requestID string) {
	swept, problem := reclaim.Request(c.opt.Layout, c.opt.Store, requestID)
	if problem != nil {
		c.logf("tmp reclaim for %s deferred: %s", requestID, problem.Message)
	}
	if swept.Removed > 0 {
		c.logf("tmp reclaim for %s: %s", requestID, units.Bytes(swept.Bytes))
	}
}

// outputExportPaths proves the accepted outputs are the contracted set, each at the
// digest name inside the contracted directory, and returns those paths in contract order.
func outputExportPaths(export records.OutputExport, outputs []records.Output) ([]string, *exit.Error) {
	if len(export.Outputs) != len(outputs) {
		return nil, exit.Named(exit.Validation, "output_export_set_mismatch",
			"output export expects %d files and terminal accepted %d", len(export.Outputs), len(outputs))
	}
	accepted := make(map[string]records.Output, len(outputs))
	for _, output := range outputs {
		accepted[output.OutputID] = output
	}
	paths := make([]string, 0, len(export.Outputs))
	for _, intended := range export.Outputs {
		output, ok := accepted[intended.OutputID]
		if !ok || output.MimeType != intended.MediaType {
			return nil, exit.Named(exit.Validation, "output_export_contract_mismatch",
				"accepted output %s does not match its pre-execution media contract", intended.OutputID)
		}
		filename, problem := resultfiles.Filename(output.Digest, output.MimeType)
		if problem != nil {
			return nil, problem
		}
		if want := filepath.Join(export.Directory, filename); output.Path != want {
			return nil, exit.Named(exit.Conflict, "output_export_contract_mismatch",
				"accepted output %s is recorded at %s, not at its contracted %s",
				intended.OutputID, output.Path, want)
		}
		paths = append(paths, output.Path)
	}
	return paths, nil
}

func (c *Orchestrator) failOutputExport(requestID string, problem *exit.Error) {
	if persist := c.opt.Store.FailOutputExport(requestID, problem.ErrName(), problem.Message); persist != nil {
		c.logf("output export %s failed (%s) and its failure could not be recorded: %s",
			requestID, problem.ErrName(), persist.Message)
		return
	}
	c.logf("output export %s failed (%s): %s", requestID, problem.ErrName(), problem.Message)
}

// ResumeOutputExports settles every terminal obligation before the restarted daemon
// starts serving clients.
func (c *Orchestrator) ResumeOutputExports() *exit.Error {
	owed, problem := c.opt.Store.OutputExportsOwed()
	if problem != nil {
		return problem
	}
	for _, export := range owed {
		c.RetryOutputExport(export.RequestID)
	}
	return nil
}
