package api

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// The SSE plane. Two routes over ONE event model:
//
//	GET /v1/requests/{id}/events   one request, TERMINAL-STOP
//	GET /v1/events                 every request, one connection, never terminal
//
// The multiplexed route exists for a concrete reason and not for elegance: a browser caps
// concurrent connections per origin at six, so a gallery watching ten requests with one
// EventSource each stops receiving events for four of them, silently. The local host
// multiplexes so a UI opens exactly one.
//
// Three properties a client may rely on, in the reference client's own vocabulary
// (cozy.art's `startSSE`, which is what validated this surface):
//
//   - RESUME. Every durable event carries `id:` = its `sequence_number`. Reconnect with
//     `?starting_after=<sequence_number>` (OpenAI Responses' name) and
//     the stream replays everything after it, in order, across a server restart. The
//     cursor is the durable row's own primary key — there is no second sequence to drift.
//   - TERMINAL-STOP. A terminal event is ABSORBING on the per-request route: the server
//     closes and the client must not reconnect. A stream close WITHOUT a terminal is never
//     a verdict — it means reconnect from the cursor.
//   - LIVE PROGRESS IS NOT REPLAYED. It was never durable. What a mid-run subscriber gets
//     immediately is the LATEST tick, so its first render is current instead of blank.

const (
	// retryHint is the reconnect delay a client honours when the stream drops. Short,
	// because the peer is on this machine.
	retryHintMS = 500
	// heartbeat keeps an idle stream from being reaped by anything in between. It is a
	// COMMENT frame, so it can never be mistaken for an event.
	heartbeat = 15 * time.Second
	// pollInterval is how often the durable table is re-read. The authority is a local
	// SQLite file and a query for "rows after N" on an indexed key costs microseconds;
	// a notification plane would be a second mechanism for the same fact.
	pollInterval = 40 * time.Millisecond
)

func (s *Server) requestEvents(w http.ResponseWriter, r *http.Request) {
	reference := r.PathValue("id")
	row, e := s.store.RequestByReference(reference)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	if row == nil {
		s.refuse(w, r, http.StatusNotFound, "not_found", "no request "+reference+" on this host", "")
		return
	}
	s.stream(w, r, row.ID)
}

// stream serves one request's durable and live events.
func (s *Server) stream(w http.ResponseWriter, r *http.Request, requestID string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		s.refuse(w, r, http.StatusInternalServerError, "streaming_unsupported",
			"this transport cannot stream", "")
		return
	}
	cursor := int64(0)
	if v := r.URL.Query().Get("starting_after"); v != "" {
		n, err := strconv.ParseInt(v, 10, 64)
		if err != nil || n < 0 {
			s.refuse(w, r, http.StatusBadRequest, "invalid_starting_after",
				"starting_after must be the sequence_number of an event this stream already delivered", "")
			return
		}
		cursor = n
	} else if r.URL.Query().Get("from") == "now" {
		// A client that wants only what happens NEXT opens at the head rather than
		// replaying a request's whole history.
		head, e := s.store.LastEventSeq()
		if e != nil {
			s.refuseTyped(w, r, e)
			return
		}
		cursor = head
	}

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache, no-store")
	h.Set("Connection", "keep-alive")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, ":ok\n\nretry: %d\n\n", retryHintMS)
	flusher.Flush()

	machineOwned := false
	if requestID != "" {
		link, problem := s.store.MachineExecution(requestID)
		machineOwned = problem == nil && link != nil
	}

	// What a queued request is doing, immediately so a first render is current, and again
	// whenever it changes (a rental's boot moving) so a watcher sees the wait move. The
	// phase is live state: it is never a durable row.
	shown := ""
	phase := func() {
		if requestID == "" {
			return
		}
		frame, ok := s.waitFrame(requestID)
		if !ok {
			return
		}
		key := frameKey(frame)
		if key == shown {
			return
		}
		shown = key
		writeEvent(w, 0, liveEnvelope(frame))
		flusher.Flush()
	}
	phase()
	telemetry := map[string]uint64{}
	samples := func() {
		active := map[string]uint64{}
		for _, frame := range s.store.LiveRunEventsV1(requestID) {
			key := frame.Event.RequestID + "/" + frame.Event.Type
			active[key] = frame.Sequence
			if telemetry[key] >= frame.Sequence {
				continue
			}
			env := durableEnvelope(frame.Event)
			env.SequenceNumber = 0
			env.Payload["live"] = true
			writeEvent(w, 0, env)
			flusher.Flush()
		}
		telemetry = active // Completed runs leave no per-stream telemetry memory.
	}

	ctx := r.Context()
	beat := time.NewTicker(heartbeat)
	defer beat.Stop()
	poll := time.NewTicker(pollInterval)
	defer poll.Stop()
	observe := time.NewTicker(phaseInterval)
	defer observe.Stop()

	for {
		// DURABLE FIRST, and drained to exhaustion: a terminal must never be delivered
		// while earlier lifecycle rows are still unread, because a client that stops on
		// the terminal would then never see them.
		for {
			rows, e := s.store.EventsAfter(requestID, cursor, 100)
			if e != nil {
				s.logf("event stream read failed: %s", e.Message)
				return
			}
			if len(rows) == 0 {
				break
			}
			for _, row := range rows {
				cursor = row.Seq
				if machineOwned && records.TerminalEvent(row.Type) {
					current, problem := s.store.RequestRow(requestID)
					if problem == nil && current != nil && row.Attempt < current.Ordinal {
						continue // an explicitly resumed Runtime attempt owns the live stream
					}
				}
				writeEvent(w, row.Seq, durableEnvelope(row))
				if requestID != "" && records.TerminalEvent(row.Type) {
					// TERMINAL-STOP. The client is told to stop by the event itself,
					// and the server stops too: reconnecting to a settled request
					// would be a client polling a fact that cannot change.
					flusher.Flush()
					return
				}
			}
			flusher.Flush()
		}

		samples()
		select {
		case <-ctx.Done():
			return
		case <-beat.C:
			// A comment, not an event: it cannot be parsed as one and cannot move a
			// cursor. It exists so a dead connection is discovered by writing.
			fmt.Fprint(w, ": beat\n\n")
			flusher.Flush()
		case <-poll.C:
		case <-observe.C:
			phase()
		}
	}
}

// phaseInterval is how often a stream re-reads a queued request's phase: the cadence the
// daemon itself samples a booting rental at.
const phaseInterval = time.Second

// waitFrame is a queued request's current phase: its own preparation, or the boot of the
// rental it is pinned to.
func (s *Server) waitFrame(requestID string) (orchestrator.Frame, bool) {
	row, problem := s.store.RequestRow(requestID)
	if problem != nil || row == nil || s.publicStatusOf(*row) != "queued" {
		return orchestrator.Frame{}, false
	}
	link, problem := s.store.MachineExecution(requestID)
	if problem != nil || link == nil {
		phase, ok := s.orchestrator.QueuePhase(requestID)
		return phase.Frame(requestID), ok
	}
	phase, ok := s.orchestrator.PhaseOf(requestID)
	if !ok {
		phase, ok = s.orchestrator.RentalPhase(link.MachineID)
	}
	return phase.Frame(requestID), ok
}

// frameKey is what a phase frame says, less the clocks a watcher ages by itself.
func frameKey(frame orchestrator.Frame) string {
	value, _ := frame.Value.(map[string]any)
	kept := make(map[string]any, len(value))
	for key, field := range value {
		switch key {
		case "elapsed_ms", "sample_age_ms", "remaining_ms", "rate_bytes_per_second":
		default:
			kept[key] = field
		}
	}
	data, _ := json.Marshal(kept)
	return string(data)
}

// Envelope is the ONE event shape on the wire. Every event — durable or live — has the
// same five keys, so a client parses one thing. `sequence_number` is 0 for a live frame, which
// is exactly what "not resumable" looks like in the data rather than in prose.
type Envelope struct {
	Type           string         `json:"type"`
	RequestID      string         `json:"request_id"`
	Attempt        uint64         `json:"attempt"`
	SequenceNumber int64          `json:"sequence_number"`
	At             string         `json:"at"`
	Payload        map[string]any `json:"payload"`
}

func durableEnvelope(row records.Event) Envelope {
	payload := row.Payload
	if row.Type == records.OutputItemDelta {
		// A revision is recorded whole for this client's fold; clients see its delta.
		var product records.Product
		if json.Unmarshal(row.Raw, &product) == nil {
			raw, _ := json.Marshal(product.Delta())
			payload = nil
			_ = json.Unmarshal(raw, &payload)
		}
	}
	return Envelope{
		Type: row.Type, RequestID: row.RequestID, Attempt: uint64(row.Attempt),
		SequenceNumber: row.Seq, At: row.At, Payload: payload,
	}
}

// liveEnvelope renders one lossy frame. The runtime's own frame vocabulary is carried
// through rather than translated: `progress` becomes `request.progress` and everything
// else becomes `request.<kind>`, because inventing a second name for the runtime's
// `stage` or `metric` would be a second vocabulary for one fact.
func liveEnvelope(frame orchestrator.Frame) Envelope {
	return Envelope{
		Type: "request." + frame.Type, RequestID: frame.RequestID, Attempt: frame.Attempt,
		SequenceNumber: 0, At: time.Now().UTC().Format(time.RFC3339Nano),
		Payload: map[string]any{"seq": frame.Seq, "value": frame.Value, "live": true},
	}
}

func writeEvent(w http.ResponseWriter, id int64, env Envelope) {
	body, err := json.Marshal(env)
	if err != nil {
		return
	}
	if id > 0 {
		fmt.Fprintf(w, "id: %d\n", id)
	}
	fmt.Fprintf(w, "event: message\ndata: %s\n\n", body)
}
