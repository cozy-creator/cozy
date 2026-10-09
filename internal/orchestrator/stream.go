package orchestrator

import (
	"math"
	"time"
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

// ProgressSnapshot is one real work coordinate Runtime reported. Authored overall
// fractions describe work allocation, not the time required by unknown future stages.
type ProgressSnapshot struct {
	Stage           string
	StageFraction   *float64
	OverallFraction *float64
	Position        *int64
	Total           *int64
	StepMS          *float64
	// Unit names what Position and Total count ("bytes"); Rate is units per second
	// Runtime measured since its previous sample of the same stage.
	Unit     string
	Rate     *float64
	SampleAt time.Time // Runtime event time only; zero when an older source omitted it.
}

// StageEstimate is an estimate at the latest observed pace, scoped to one counted stage.
type StageEstimate struct {
	RemainingMS int64
}

func (current ProgressSnapshot) EstimateStage(previous ProgressSnapshot) (StageEstimate, bool) {
	if current.Position == nil || current.Total == nil || *current.Position < 0 ||
		*current.Total <= 0 || *current.Position > *current.Total {
		return StageEstimate{}, false
	}
	left := *current.Total - *current.Position
	if left == 0 {
		return StageEstimate{}, true // This stage is complete; later work remains unknown.
	}
	rate := 0.0
	if current.Rate != nil {
		rate = *current.Rate
	} else if current.Stage == previous.Stage && previous.Position != nil && previous.Total != nil {
		if current.Unit != previous.Unit || *previous.Total != *current.Total ||
			*previous.Position < 0 || *current.Position <= *previous.Position {
			return StageEstimate{}, false
		}
		if !current.SampleAt.IsZero() && !previous.SampleAt.IsZero() && current.SampleAt.After(previous.SampleAt) {
			rate = float64(*current.Position-*previous.Position) / current.SampleAt.Sub(previous.SampleAt).Seconds()
		}
	} else if *current.Position == 1 && current.StepMS != nil && *current.StepMS > 0 {
		// Only the first counted unit unambiguously identifies one unit per interval.
		// A later step_ms may cover a decoded chunk or follow coalesced progress.
		rate = 1000 / *current.StepMS
	}
	if rate <= 0 || math.IsNaN(rate) || math.IsInf(rate, 0) {
		return StageEstimate{}, false
	}
	remaining := float64(left) / rate * 1000
	// Every consumer may render this as a time.Duration; never overflow that conversion.
	if math.IsNaN(remaining) || math.IsInf(remaining, 0) || remaining >= float64(math.MaxInt64/int64(time.Millisecond)) {
		return StageEstimate{}, false
	}
	return StageEstimate{RemainingMS: int64(remaining)}, true
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
	sampleAt         time.Time
}

// DecodeProgressSnapshot applies the live progress contract to one already
// retained machine sample. A single sample cannot establish a whole-job ETA.
func DecodeProgressSnapshot(value any) (ProgressSnapshot, bool) {
	coordinate, ok := progressCoordinates(value)
	return coordinate.snapshot(), ok
}

func (progress progressCoordinate) snapshot() ProgressSnapshot {
	snapshot := ProgressSnapshot{Stage: progress.stage, Unit: progress.unit, SampleAt: progress.sampleAt}
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

func progressNumber(value any) (float64, bool) {
	switch n := value.(type) {
	case float64:
		return n, true
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case uint64:
		return float64(n), true
	default:
		return 0, false
	}
}

func finiteFraction(value any) (float64, bool) {
	number, ok := progressNumber(value)
	return number, ok && !math.IsNaN(number) && !math.IsInf(number, 0) && number >= 0 && number <= 1
}

func progressCoordinates(value any) (progressCoordinate, bool) {
	var out progressCoordinate
	fields, isMap := value.(map[string]any)
	if !isMap {
		return out, false
	}
	out.stage, _ = fields["stage"].(string)
	if at, ok := progressNumber(fields["sample_unix_ms"]); ok && at > 0 && at < 1<<53 && math.Trunc(at) == at {
		out.sampleAt = time.UnixMilli(int64(at))
	}
	if step, ok := progressNumber(fields["step_ms"]); ok && !math.IsNaN(step) && !math.IsInf(step, 0) && step >= 0 {
		out.stepMS = step
	}
	if out.stage == "" || len(out.stage) > 120 {
		return out, false
	}
	if unit, ok := fields["unit"].(string); ok && len(unit) <= 16 {
		out.unit = unit
	}
	if rate, ok := progressNumber(fields["rate"]); ok && !math.IsNaN(rate) && !math.IsInf(rate, 0) && rate >= 0 {
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
		position, positionOK := progressNumber(positionValue)
		total, totalOK := progressNumber(totalValue)
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
