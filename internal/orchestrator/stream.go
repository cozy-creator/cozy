package orchestrator

import (
	"encoding/json"
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
// estimate from the mean measured step time. It is observational and live-only.
type ProgressSnapshot struct {
	Stage       string
	Fraction    float64
	RemainingMS int64
	Estimated   bool
}

type progressAccumulator struct {
	attempt   uint64
	stage     string
	fraction  float64
	position  float64
	stepMSSum float64
	samples   int64
}

func (c *Orchestrator) LatestProgress(requestID string, attempt uint64) (ProgressSnapshot, bool) {
	f := c.frames
	f.mu.Lock()
	defer f.mu.Unlock()
	progress, ok := f.progress[requestID]
	if !ok || progress.attempt != attempt {
		return ProgressSnapshot{}, false
	}
	snapshot := ProgressSnapshot{Stage: progress.stage, Fraction: progress.fraction}
	if progress.samples > 0 && progress.position > 0 && progress.fraction > 0 {
		total := progress.position / progress.fraction
		remaining := max(0, total-progress.position)
		snapshot.RemainingMS = int64(remaining * progress.stepMSSum / float64(progress.samples))
		snapshot.Estimated = true
	}
	return snapshot, true
}

func progressCoordinates(value any) (stage string, fraction, position, stepMS float64, ok bool) {
	if direct, isNumber := value.(float64); isNumber {
		return "", direct, 0, 0, direct >= 0 && direct <= 1
	}
	fields, isMap := value.(map[string]any)
	if !isMap {
		return "", 0, 0, 0, false
	}
	stage, _ = fields["name"].(string)
	fraction, ok = fields["fraction"].(float64)
	if !ok {
		fraction, ok = fields["value"].(float64)
	}
	position, _ = fields["position"].(float64)
	stepMS, _ = fields["step_ms"].(float64)
	return stage, fraction, position, stepMS, ok && fraction >= 0 && fraction <= 1
}

func (f *fanout) observeProgress(frame Frame) {
	stage, fraction, position, stepMS, ok := progressCoordinates(frame.Value)
	if !ok {
		return
	}
	progress := f.progress[frame.RequestID]
	if progress.attempt != frame.Attempt || progress.stage != stage ||
		position > 0 && progress.position > 0 && position <= progress.position {
		progress = progressAccumulator{attempt: frame.Attempt, stage: stage}
	}
	progress.fraction, progress.position = fraction, position
	if stepMS > 0 {
		progress.stepMSSum += stepMS
		progress.samples++
	}
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

// forgetFrames drops a settled request's retained tick. A terminal request's last
// progress reading is not a fact anyone needs after the terminal itself is durable.
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
