package orchestrator

import (
	"math"
)

// The LOSSY half of the client contract's event stream (cl-006), beside records/events.go's
// durable half. It is live-only ON PURPOSE: a durable row per denoising step would turn the
// one lifecycle authority into a log sink, and the protocol already declares this lane
// droppable (worker-protocol/01: StreamProgress is sequence-numbered, gaps visible, and
// structurally incapable of carrying terminal authority).
//
// Consequences a client can rely on:
//
//   - A frame is never replayed from a cursor, because it was never durable. What IS
//     replayed on connect is the LATEST tick per request, so a subscriber attaching mid-run
//     sees current progress instead of waiting for the next one.
//   - A flood of frames cannot delay a durable event: subscribers have bounded buffers and
//     a full buffer sheds the frame rather than blocking the protocol's own reader.
//   - Nothing here can settle anything. A progress frame that claimed success would still
//     be a progress frame.

// Frame is one live tick, exactly as the runtime encoded it. cozy-creator does not own
// this shape: `Type` and `Value` are the runtime's `{"type": kind, "payload": value}`
// (cozy-runtime session.emit_progress), and this orchestrator transports them verbatim
// rather than inventing a second vocabulary for them.
type Frame struct {
	RequestID string `json:"request_id"`
	Attempt   uint64 `json:"attempt"`
	Seq       uint64 `json:"seq"`
	Type      string `json:"type"`
	Value     any    `json:"value"`
}

// ProgressSnapshot is the latest real work coordinate Runtime reported and an optional
// whole-job estimate from sampled step intervals per overall-fraction advance. It is
// observational; the final overall fraction is copied into the attempt-end event.
type ProgressSnapshot struct {
	Stage           string
	StageFraction   *float64
	OverallFraction *float64
	Position        *int64
	Total           *int64
	StepMS          *float64
	// Unit names what Position and Total count ("bytes"); Rate is units per second
	// Runtime measured since its previous sample of the same stage.
	Unit        string
	Rate        *float64
	RemainingMS int64
	Estimated   bool
}

type progressCoordinate struct {
	stage            string
	stageFraction    float64
	hasStageFraction bool
	overallFraction  float64
	hasOverall       bool
	position         int64
	total            int64
	hasPosition      bool
	stepMS           float64
	unit             string
	rate             float64
	hasRate          bool
}

// DecodeProgressSnapshot applies the live progress contract to one already
// retained machine sample. A single sample cannot establish a whole-job ETA.
func DecodeProgressSnapshot(value any) (ProgressSnapshot, bool) {
	coordinate, ok := progressCoordinates(value)
	return coordinate.snapshot(), ok
}

func (progress progressCoordinate) snapshot() ProgressSnapshot {
	snapshot := ProgressSnapshot{Stage: progress.stage, Unit: progress.unit}
	if progress.hasRate {
		rate := progress.rate
		snapshot.Rate = &rate
	}
	if progress.stepMS > 0 {
		step := progress.stepMS
		snapshot.StepMS = &step
	}
	if progress.hasStageFraction {
		value := progress.stageFraction
		snapshot.StageFraction = &value
	}
	if progress.hasPosition {
		position, total := progress.position, progress.total
		snapshot.Position, snapshot.Total = &position, &total
	}
	if progress.hasOverall {
		value := progress.overallFraction
		snapshot.OverallFraction = &value
	}
	return snapshot
}

func finiteFraction(value any) (float64, bool) {
	number, ok := value.(float64)
	return number, ok && !math.IsNaN(number) && !math.IsInf(number, 0) && number >= 0 && number <= 1
}

func progressCoordinates(value any) (progressCoordinate, bool) {
	var out progressCoordinate
	fields, isMap := value.(map[string]any)
	if !isMap {
		return out, false
	}
	out.stage, _ = fields["stage"].(string)
	if step, ok := fields["step_ms"].(float64); ok && !math.IsNaN(step) && !math.IsInf(step, 0) && step >= 0 {
		out.stepMS = step
	}
	if out.stage == "" || len(out.stage) > 120 {
		return out, false
	}
	if unit, ok := fields["unit"].(string); ok && len(unit) <= 16 {
		out.unit = unit
	}
	if rate, ok := fields["rate"].(float64); ok && !math.IsNaN(rate) && !math.IsInf(rate, 0) && rate >= 0 {
		out.rate, out.hasRate = rate, true
	}
	if value, present := fields["stage_fraction"]; present {
		var ok bool
		out.stageFraction, ok = finiteFraction(value)
		if !ok {
			return out, false
		}
		out.hasStageFraction = true
	}
	if value, present := fields["overall_fraction"]; present {
		var ok bool
		out.overallFraction, ok = finiteFraction(value)
		if !ok {
			return out, false
		}
		out.hasOverall = true
	}
	positionValue, hasPosition := fields["position"]
	totalValue, hasTotal := fields["total"]
	if hasPosition != hasTotal {
		return out, false
	}
	if hasPosition {
		position, positionOK := positionValue.(float64)
		total, totalOK := totalValue.(float64)
		if !positionOK || !totalOK || math.Trunc(position) != position || math.Trunc(total) != total ||
			position < 0 || total <= 0 || position > total {
			return out, false
		}
		out.position, out.total, out.hasPosition = int64(position), int64(total), true
		// Counted progress has exact integer coordinates; they win over a rounded fraction.
		out.stageFraction, out.hasStageFraction = position/total, true
	}
	return out, true
}
