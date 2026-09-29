package cli

import (
	"encoding/json"
	"time"

	localapi "github.com/cozy-creator/cozy/internal/client"
)

// Call clocks are producer measurements for one attempt. Missing measurements
// remain unknown; the call's existing wall interval is preserved separately.
type callTiming struct {
	QueuedMS               *float64 `json:"queued_ms,omitempty"`
	PreparationMS          *float64 `json:"preparation_ms,omitempty"`
	ExecutionMS            *float64 `json:"execution_ms,omitempty"`
	ExecutionStartedUnixMS int64    `json:"execution_started_unix_ms,omitempty"`
}

type callPhaseEvent struct {
	Request      string `json:"request"`
	Parent       string `json:"parent"`
	Index        int    `json:"index"`
	Attempt      int64  `json:"attempt"`
	Module       string `json:"module"`
	Export       string `json:"export"`
	Label        string `json:"label"`
	Phase        string `json:"phase"`
	Status       string `json:"status,omitempty"`
	AtUnixMS     int64  `json:"at_unix_ms"`
	CalledUnixMS int64  `json:"called_unix_ms"`
	callTiming
}

func readCallPhase(e localapi.Event) (callPhaseEvent, bool) {
	raw, err := json.Marshal(e.Payload)
	if err != nil {
		return callPhaseEvent{}, false
	}
	var phase callPhaseEvent
	if json.Unmarshal(raw, &phase) != nil || phase.Request == "" || phase.AtUnixMS <= 0 {
		return phase, false
	}
	return phase, phase.valid()
}
func measuredMS(value *float64) *float64 {
	if value == nil || *value < 0 {
		return nil
	}
	return value
}
func (t callTiming) present() bool {
	return measuredMS(t.QueuedMS) != nil || measuredMS(t.PreparationMS) != nil || measuredMS(t.ExecutionMS) != nil
}
func (e callPhaseEvent) duration(phase string, at time.Time) (time.Duration, bool) {
	value := e.ExecutionMS
	switch phase {
	case "queued":
		value = e.QueuedMS
	case "preparing":
		value = e.PreparationMS
	case "running":
	default:
		return 0, false
	}
	if measuredMS(value) == nil {
		return 0, false
	}
	elapsed := time.Duration(*value * float64(time.Millisecond))
	if e.Phase == phase {
		elapsed += max(at.Sub(time.UnixMilli(e.AtUnixMS)), 0)
	}
	return elapsed, true
}

func (e callPhaseEvent) valid() bool {
	if e.AtUnixMS <= 0 {
		return false
	}
	switch e.Phase {
	case "queued", "preparing", "running", "finalizing", "paused", "terminal":
		return true
	}
	return false
}

// A terminal observation cannot reopen the same attempt. Different transitions
// may share a millisecond, so arrival order settles equal-time phase changes.
func (e callPhaseEvent) follows(attempt, at int64, phase string) bool {
	return e.Attempt > attempt || e.Attempt == attempt &&
		e.AtUnixMS >= at && (e.AtUnixMS != at || e.Phase != phase) &&
		(phase != "terminal" || e.Phase == "terminal")
}
