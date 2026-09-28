package producttest

import (
	"encoding/json"
	"time"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// outcomePage is a fake machine's run output log once its run ended: the one terminal entry
// at `sequence`, carrying the outcome, for a reader after `after`.
func outcomePage(after, sequence uint64, state string, outcome *pb.AttemptOutcome) *pb.MachineExecutionEventPage {
	page := &pb.MachineExecutionEventPage{NextAfter: max(after, sequence), HeadSequence: sequence}
	if after < sequence {
		body, _ := json.Marshal(map[string]string{"state": state, "outcome_id": outcome.OutcomeId})
		page.Events = []*pb.MachineExecutionEvent{{Sequence: sequence, AttemptOrdinal: outcome.AttemptOrdinal,
			AtMs: uint64(time.Now().UnixMilli()), Kind: "outcome", BodyCanonicalBytes: body, Outcome: outcome}}
	}
	return page
}

// outcomeEvent is the terminal entry a fake machine journals among its other events.
func outcomeEvent(sequence uint64, state string, outcome *pb.AttemptOutcome) *pb.MachineExecutionEvent {
	return outcomePage(0, sequence, state, outcome).Events[0]
}
