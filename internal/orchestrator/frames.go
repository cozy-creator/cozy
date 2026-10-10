package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// The owner-side FRAME HANDLERS for one claimed stream (owner.go runs the conversation;
// #436 flipped the dial direction, so the old worker-dials hub/Register machinery is
// gone — ClaimAck/snapshot are its successors, handled in owner.go).

// ------------------------------------------------------------------- observed state

// --------------------------------------------------------------------------- accepted

// --------------------------------------------------------------------------- outcome

// outcomeRefusedCause is the record owner's OWN terminal cause: the worker's journaled
// outcome stands refused, and the request that waited on it is failed under this name.
const outcomeRefusedCause = "worker.outcome_refused"

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

// preExecution names the four causes rev-2 §6/§7 defines as worker pre-execution
// refusals. It is a CLOSED list because the protocol's is closed: a cause outside it that
// arrives with origin WORKER is a peer saying something this contract does not define,
// and settling is the conservative answer.
func preExecution(cause string) bool {
	switch cause {
	case "NO_CAPACITY", "ADMISSION_EPOCH_STALE", "UNKNOWN_PLACEMENT",
		"PLACEMENT_NOT_DISPATCHABLE":
		return true
	}
	return false
}

// outcomeError is the record owner's PROJECTION over (status, cause). Retryability is
// never a wire observation; this is the only place the neutral facts become an outcome.
func outcomeError(status, cause, message string) *exit.Error {
	switch status {
	case "SUCCEEDED":
		return nil
	case "REFUSED":
		if preExecution(cause) {
			// A PRE-EXECUTION refusal is not a judgment about the payload. Rendering it as
			// a validation failure would tell a user to change a payload that is fine.
			return exit.Named(exit.Unavailable, "attempt_not_admitted",
				"the worker did not admit this attempt (%s): %s", cause, message)
		}
		return exit.New(exit.Validation, "the attempt was refused (%s): %s", cause, message)
	case "CANCELED":
		if cause == "DEADLINE_EXPIRED" {
			return exit.New(exit.Deadline, "the attempt hit its deadline: %s", message)
		}
		return exit.New(exit.Canceled, "the attempt was canceled (%s): %s", cause, message)
	case "ABANDONED":
		return exit.New(exit.Failed, "the attempt was abandoned (%s): %s", cause, message)
	default:
		if cause == outcomeRefusedCause {
			return exit.Named(exit.Failed, cause, "%s", message)
		}
		return exit.New(exit.Failed, "the attempt failed (%s): %s", cause, message)
	}
}
