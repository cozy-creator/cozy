package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/records"
)

// The owner-side FRAME HANDLERS for one claimed stream (owner.go runs the conversation;
// #436 flipped the dial direction, so the old worker-dials hub/Register machinery is
// gone — ClaimAck/snapshot are its successors, handled in owner.go).

// ------------------------------------------------------------------- observed state

// --------------------------------------------------------------------------- accepted

// --------------------------------------------------------------------------- outcome

func (c *Orchestrator) cleanupRequestAssets(req records.Request) {
	if req.InstallID != "" && c.opt.ReclaimInstall != nil {
		if problem := c.opt.ReclaimInstall(req.InstallID); problem != nil {
			c.logf("request %s snapshot cleanup deferred: %s", req.ID, problem.Message)
		}
	}
}

// Triage is what cl-006 persisted for one attempt. An empty Subject means the outcome
// named no bundle, which is a fact, not a failure.
type Triage struct {
	Subject string
	Digest  string
	Length  int64
	Bundle  []byte // the verified bytes, kept in the attempt row (cl-116)
	Fault   string // why the bytes were not kept, when a subject was named and they were not
}
