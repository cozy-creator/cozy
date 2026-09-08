package orchestrator

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

func (c *Orchestrator) privateRootOutputs(req records.Request, attempt records.Attempt, doc canonical.Doc, holder *worker) ([]records.ByteOutput, []records.Output, *exit.Error) {
	resolver, ok := c.opt.Packages.(interface {
		PrivateRetainedResultFields(records.Request) (map[string]bool, *exit.Error)
	})
	if !ok {
		return nil, nil, exit.Unavailablef("native result schema verifier is absent")
	}
	retained, problem := resolver.PrivateRetainedResultFields(req)
	if problem != nil {
		return nil, nil, problem
	}
	all, problem := c.privateByteOutputs(req, attempt, doc)
	if problem != nil {
		return nil, nil, problem
	}
	var native []records.ByteOutput
	for _, b := range all {
		if retained[b.OutputID] {
			native = append(native, b)
		}
	}
	// The already authenticated outcome is projected for the existing media mirror.
	// Native File/Tree results are never interpreted as staging files.
	media := make([]canonical.Value, 0)
	var names []string
	for _, row := range doc.Sub("output_manifest").List("outputs") {
		if !retained[row.Str("output_id")] {
			media = append(media, map[string]canonical.Value(row))
			names = append(names, row.Str("output_id"))
		}
	}
	if len(media) == 0 {
		return native, nil, nil
	}
	filtered := make(canonical.Doc, len(doc))
	for key, value := range doc {
		filtered[key] = value
	}
	filtered["output_manifest"] = map[string]canonical.Value{"outputs": media}
	projected := req
	projected.Outputs = strings.Join(names, ",")
	outputs, problem := c.mirrorOutputs(projected, uint64(attempt.Attempt), filtered, holder)
	return native, outputs, problem
}

func (c *Orchestrator) retainRootByteResults(s *session, req records.Request, attempt records.Attempt) *exit.Error {
	outputs, problem := c.opt.Store.ByteOutputs(req.ID, attempt.Attempt)
	if problem != nil {
		return problem
	}
	for _, output := range outputs {
		hold, problem := c.opt.Store.ReserveByteResult(req.ID, output)
		if problem != nil {
			return problem
		}
		if hold.State != "held" {
			if problem := c.changeByteRetention(s.ctx, hold, false); problem != nil {
				return problem
			}
		}
	}
	return nil
}
