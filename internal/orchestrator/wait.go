package orchestrator

import "github.com/cozy-creator/cozy/internal/records"

// The wait vocabulary (cl-103). A queued/parked event names WHAT the queue is doing in a
// stable `wait` payload field beside the verbatim diagnostic `reason`, so a client can
// say "starting a worker" instead of parsing dispatcher vocabulary. Additive: `reason`
// is unchanged and stays the full diagnostic.
const (
	// WaitWorkerStart: no live worker serves this package yet; one is being started.
	WaitWorkerStart = "worker_start"
	// WaitWorkerWarming: a worker exists and is still loading toward dispatchable.
	WaitWorkerWarming = "worker_warming"
	// WaitSlotBusy: a dispatchable placement exists; every attempt slot is taken.
	WaitSlotBusy = "slot_busy"
	// WaitQueueAhead: capacity is spoken for by requests queued ahead of this one.
	WaitQueueAhead = "queue_ahead"
	// WaitRental: waiting for a rental machine to be acquired or assigned.
	WaitRental = "rental"
	// WaitModelTransfer: the model must land on the selected worker first.
	WaitModelTransfer = "model_transfer"
)

// waitFacts is the classification a queued/parked event carries beside the raw reason:
// the stable cause, and the machine word the wait is on when one is known.
type waitFacts struct {
	cause      string
	on         string
	waitingFor *WaitingRun
}

// WaitingRun names existing work occupying capacity. It is display metadata;
// neither the local number nor this observation authorizes routing or cancellation.
type WaitingRun struct {
	Number    int64  `json:"number"`
	RequestID string `json:"request_id"`
}

// decorate adds the wait facts and the request's package to an event payload. The
// payload's `reason` is never touched here.
func (f waitFacts) decorate(payload map[string]any, req records.Request) map[string]any {
	if f.cause != "" {
		payload["wait"] = f.cause
	}
	if f.on != "" {
		payload["waiting_on"] = f.on
	}
	if f.waitingFor != nil {
		payload["waiting_for"] = f.waitingFor
	}
	if req.Package != "" {
		payload["package"] = req.Package
	}
	return payload
}
