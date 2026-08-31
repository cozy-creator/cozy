package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/resultfiles"
)

// RetryOutputExport settles one daemon-owned --out obligation from the already-verified
// internal media. It never changes the execution terminal and never trusts a terminal path.
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
	outputs, problem := c.opt.Store.VisibleOutputs(requestID)
	if problem != nil {
		c.failOutputExport(requestID, problem)
		return
	}
	entries, problem := outputExportEntries(*export, outputs)
	if problem != nil {
		c.failOutputExport(requestID, problem)
		return
	}
	if problem := c.opt.Store.BeginOutputExport(requestID); problem != nil {
		if problem.ErrName() != "output_export_not_pending" {
			c.logf("output export %s could not begin: %s", requestID, problem.Message)
		}
		return
	}
	paths, problem := resultfiles.Publish(export.Directory, entries)
	if problem != nil {
		c.failOutputExport(requestID, problem)
		return
	}
	if problem := c.opt.Store.CompleteOutputExport(requestID, paths); problem != nil {
		c.logf("output export %s reached disk but settlement failed: %s", requestID, problem.Message)
		return
	}
	c.logf("output export %s published %d file(s) under %s", requestID, len(paths), export.Directory)
}

func outputExportEntries(export records.OutputExport, outputs []records.Output) (
	[]resultfiles.Entry, *exit.Error,
) {
	if len(export.Outputs) != len(outputs) {
		return nil, exit.Named(exit.Validation, "output_export_set_mismatch",
			"output export expects %d files and terminal accepted %d", len(export.Outputs), len(outputs))
	}
	accepted := make(map[string]records.Output, len(outputs))
	for _, output := range outputs {
		accepted[output.OutputID] = output
	}
	entries := make([]resultfiles.Entry, 0, len(export.Outputs))
	for _, intended := range export.Outputs {
		output, ok := accepted[intended.OutputID]
		if !ok || output.MimeType != intended.MediaType {
			return nil, exit.Named(exit.Validation, "output_export_contract_mismatch",
				"accepted output %s does not match its pre-execution media contract", intended.OutputID)
		}
		entries = append(entries, resultfiles.Entry{
			OutputID: output.OutputID, MediaType: output.MimeType, Filename: intended.Filename,
			Source: output.Path, Digest: output.Digest, Length: output.Length,
		})
	}
	return entries, nil
}

func (c *Orchestrator) failOutputExport(requestID string, problem *exit.Error) {
	if persist := c.opt.Store.FailOutputExport(requestID, problem.ErrName(), problem.Message); persist != nil {
		c.logf("output export %s failed (%s) and its failure could not be recorded: %s",
			requestID, problem.ErrName(), persist.Message)
		return
	}
	c.logf("output export %s failed (%s): %s", requestID, problem.ErrName(), problem.Message)
}

// ResumeOutputExports retries every terminal obligation before the restarted daemon starts
// serving clients. Exact existing files replay; changed files remain typed conflicts.
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
