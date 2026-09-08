package orchestrator

import (
	"encoding/json"
	"math"
	"sync"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
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

// frameBuffer is one subscriber's depth. A slow SSE client is a slow client, never a
// slow worker: past this depth its frames are shed and the protocol reader moves on.
const frameBuffer = 64

type subscriber struct {
	requestID string // "" subscribes to every request (the multiplexed stream)
	ch        chan Frame
}

type fanout struct {
	mu       sync.Mutex
	next     uint64
	subs     map[uint64]*subscriber
	latest   map[string]Frame // the most recent progress tick per request
	progress map[string]progressAccumulator
}

func newFanout() *fanout {
	return &fanout{
		subs: map[uint64]*subscriber{}, latest: map[string]Frame{},
		progress: map[string]progressAccumulator{},
	}
}

// Subscribe opens a live frame feed. An empty requestID takes every request's frames.
// The returned function must be called; it is the only way a subscriber is forgotten.
func (c *Orchestrator) Subscribe(requestID string) (<-chan Frame, func()) {
	f := c.frames
	f.mu.Lock()
	defer f.mu.Unlock()
	f.next++
	id := f.next
	s := &subscriber{requestID: requestID, ch: make(chan Frame, frameBuffer)}
	f.subs[id] = s
	return s.ch, func() {
		f.mu.Lock()
		defer f.mu.Unlock()
		if held, ok := f.subs[id]; ok && held == s {
			delete(f.subs, id)
			close(s.ch)
		}
	}
}

// LatestFrame is the most recent tick for one request, or false when none has arrived.
// A subscriber that attaches mid-attempt gets this immediately so its first render shows
// where the attempt actually is.
func (c *Orchestrator) LatestFrame(requestID string) (Frame, bool) {
	f := c.frames
	f.mu.Lock()
	defer f.mu.Unlock()
	frame, ok := f.latest[requestID]
	return frame, ok
}

// ProgressSnapshot is the latest real work coordinate Runtime reported and an optional
// whole-job estimate from measured elapsed time per overall-fraction advance. It is
// observational; the final overall fraction is copied into the attempt-end event.
type ProgressSnapshot struct {
	Stage           string
	StageFraction   *float64
	OverallFraction *float64
	Position        *int64
	Total           *int64
	StepMS          *float64
	RemainingMS     int64
	Estimated       bool
}

type progressAccumulator struct {
	attempt          uint64
	stage            string
	stageFraction    float64
	hasStageFraction bool
	position         int64
	total            int64
	hasPosition      bool
	overallFraction  float64
	hasOverall       bool
	overallDelta     float64
	overallMSSum     float64
	stepMS           float64
}

func (c *Orchestrator) LatestProgress(requestID string, attempt uint64) (ProgressSnapshot, bool) {
	f := c.frames
	f.mu.Lock()
	defer f.mu.Unlock()
	progress, ok := f.progress[requestID]
	if !ok || progress.attempt != attempt {
		return ProgressSnapshot{}, false
	}
	snapshot := ProgressSnapshot{Stage: progress.stage}
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
	if progress.hasOverall && progress.overallDelta > 0 && progress.overallMSSum > 0 {
		remaining := max(0.0, 1-progress.overallFraction)
		snapshot.RemainingMS = int64(remaining * progress.overallMSSum / progress.overallDelta)
		snapshot.Estimated = true
	}
	return snapshot, true
}

// withProgressSummary preserves one measured fraction in the existing outcome
// transaction. Progress ticks themselves remain lossy and never write the journal.
func (c *Orchestrator) withProgressSummary(requestID string, attempt uint64, payload map[string]any) map[string]any {
	progress, ok := c.LatestProgress(requestID, attempt)
	if !ok || progress.OverallFraction == nil {
		return payload
	}
	summary := make(map[string]any, len(payload)+1)
	for key, value := range payload {
		summary[key] = value
	}
	summary["overall_fraction"] = *progress.OverallFraction
	return summary
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
	for key := range fields {
		switch key {
		case "stage", "stage_fraction", "overall_fraction", "position", "total", "step_ms":
		default:
			return out, false
		}
	}
	out.stage, _ = fields["stage"].(string)
	step, hasStep := fields["step_ms"]
	var stepOK bool
	out.stepMS, stepOK = step.(float64)
	if out.stage == "" || len(out.stage) > 120 || !hasStep || !stepOK || math.IsNaN(out.stepMS) ||
		math.IsInf(out.stepMS, 0) || out.stepMS < 0 {
		return out, false
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
		derived := position / total
		if out.hasStageFraction && math.Abs(out.stageFraction-derived) > 1e-9 {
			return out, false
		}
		if !out.hasStageFraction {
			out.stageFraction, out.hasStageFraction = derived, true
		}
	}
	return out, true
}

func (f *fanout) observeProgress(frame Frame) {
	coordinate, ok := progressCoordinates(frame.Value)
	if !ok {
		return
	}
	progress := f.progress[frame.RequestID]
	if progress.attempt != frame.Attempt {
		progress = progressAccumulator{attempt: frame.Attempt}
	}
	if coordinate.hasOverall && progress.hasOverall &&
		coordinate.overallFraction < progress.overallFraction {
		return
	}
	if coordinate.hasOverall && progress.hasOverall &&
		coordinate.overallFraction > progress.overallFraction && coordinate.stepMS > 0 {
		progress.overallDelta += coordinate.overallFraction - progress.overallFraction
		progress.overallMSSum += coordinate.stepMS
	}
	if coordinate.hasOverall {
		progress.overallFraction, progress.hasOverall = coordinate.overallFraction, true
	}
	if progress.stage != coordinate.stage {
		progress.stepMS = 0
	}
	progress.stage = coordinate.stage
	if coordinate.stepMS > 0 {
		progress.stepMS = coordinate.stepMS
	}
	progress.stageFraction, progress.hasStageFraction =
		coordinate.stageFraction, coordinate.hasStageFraction
	progress.position, progress.total, progress.hasPosition =
		coordinate.position, coordinate.total, coordinate.hasPosition
	f.progress[frame.RequestID] = progress
}

// count is how many clients are attached right now — the API's open SSE streams.
func (f *fanout) count() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return len(f.subs)
}

// publish hands a frame to every matching subscriber, shedding rather than blocking.
func (f *fanout) publish(frame Frame) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if frame.Type == "progress" {
		f.latest[frame.RequestID] = frame
		f.observeProgress(frame)
	}
	for _, s := range f.subs {
		if s.requestID != "" && s.requestID != frame.RequestID {
			continue
		}
		select {
		case s.ch <- frame:
		default: // the one loss this lane may incur, and the reason it is the lossy lane
		}
	}
}

// forget drops a settled request's live telemetry. Its last measured overall
// fraction has already committed with the attempt-end event.
func (f *fanout) forget(requestID string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.latest, requestID)
	delete(f.progress, requestID)
}

// frameOf decodes one AttemptProgress into the live shape. The runtime's payload is a
// JSON object it owns; an unreadable one becomes a frame with no value rather than a
// dropped tick, because the SEQUENCE still tells a client the attempt is alive.
func frameOf(p *pb.AttemptProgress) Frame {
	frame := Frame{RequestID: p.RequestId, Attempt: p.AttemptOrdinal, Seq: p.Seq, Type: "progress"}
	var body struct {
		Type    string `json:"type"`
		Payload any    `json:"payload"`
	}
	if err := json.Unmarshal(p.Data, &body); err == nil {
		if body.Type != "" {
			frame.Type = body.Type
		}
		frame.Value = body.Payload
	}
	return frame
}
