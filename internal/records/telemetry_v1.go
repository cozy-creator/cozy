package records

import (
	"database/sql"
	"encoding/json"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// Telemetry carries no lifecycle authority. One first fraction plus the latest progress,
// preparation and log sample are retained per observed run. Real transitions flush those
// bounded samples in their existing FULL transaction; no telemetry event begins a write.
type runTelemetryV1 struct {
	attempt  int64
	state    string
	cursor   uint64
	flushed  uint64
	first    *telemetrySampleV1
	latest   map[string]telemetrySampleV1
	progress json.RawMessage
}
type telemetrySampleV1 struct {
	sequence uint64
	remote   uint64
	event    Event
}

// LiveRunEventV1 is a latest, nonreplayable sample. Sequence deduplicates only live samples
// within this daemon; Event.Seq stays zero and must never advance a durable SSE cursor.
type LiveRunEventV1 struct {
	Sequence uint64
	Event    Event
}

func (s *Store) seedTelemetryV1(id string, attempt int64, state string, cursor int64) {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	if s.telemetry == nil {
		s.telemetry = map[string]*runTelemetryV1{}
	}
	current := s.telemetry[id]
	if current == nil || current.attempt != attempt {
		current = &runTelemetryV1{attempt: attempt, latest: map[string]telemetrySampleV1{}}
		s.telemetry[id] = current
	}
	current.state = state
	current.cursor = max(current.cursor, uint64(max(cursor, 0)))
}

// ForgetRunTelemetryV1 detaches only this controller's lossy samples when its run observer
// ends. Accepted work and durable rows remain with their existing owners.
func (s *Store) ForgetRunTelemetryV1(id string) { s.forgetTelemetryV1(id) }

func (s *Store) forgetTelemetryV1(id string) {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	delete(s.telemetry, id)
}
func samplePayloadV1(p *v1.Progress) map[string]any {
	sample := map[string]any{"stage": p.Stage}
	if p.Fraction >= 0 {
		sample["overall_fraction"] = p.Fraction
	}
	if p.StageFraction != nil {
		sample["stage_fraction"] = *p.StageFraction
	}
	if p.Total > 0 {
		sample["position"], sample["total"] = p.Completed, p.Total
	}
	if p.BytesTotal > 0 {
		sample["bytes_done"], sample["bytes_total"] = p.BytesDone, p.BytesTotal
	}
	if p.StepMs > 0 {
		sample["step_ms"] = p.StepMs
	}
	return sample
}
func (s *Store) sameLiveStateV1(id string, state *v1.RunState) bool {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	current := s.telemetry[id]
	return current != nil && current.state == state.State && current.attempt == int64(max(state.Attempt, 1))
}

func (s *Store) sampleRunV1(id string, event *v1.RunEvent) *exit.Error {
	s.telemetryMu.Lock()
	current := s.telemetry[id]
	s.telemetryMu.Unlock()
	if current == nil {
		row, problem := s.RequestRow(id)
		if problem != nil {
			return problem
		}
		if row == nil {
			return exit.Internalf("cannot sample an unknown run %s", id)
		}
		link, problem := s.MachineExecution(id)
		if problem != nil {
			return problem
		}
		if link == nil {
			return exit.Internalf("cannot sample a run without its machine %s", id)
		}
		s.seedTelemetryV1(id, max(row.Ordinal, 1), row.State, link.RemoteCursor)
	}
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	current = s.telemetry[id]
	if current == nil {
		return nil
	}
	if event.Sequence != 0 && event.Sequence <= current.cursor {
		return nil
	}
	current.cursor = max(current.cursor, event.Sequence)
	at := time.UnixMilli(event.AtMs).UTC().Format(time.RFC3339Nano)
	if event.AtMs == 0 {
		at = now()
	}
	kind := "request.log"
	var payload map[string]any
	if p := event.GetProgress(); p != nil {
		sample := samplePayloadV1(p)
		current.progress, _ = json.Marshal(sample)
		kind = "machine.progress"
		payload = map[string]any{"type": "progress", "payload": sample}
		if current.state == "queued" || current.state == "preparing" {
			detail := p.Stage
			if p.BytesTotal > 0 {
				detail = strings.TrimSpace(detail + " " + humanBytes(p.BytesDone) + " of " + humanBytes(p.BytesTotal))
			} else if p.BytesDone > 0 {
				detail = strings.TrimSpace(detail + " · " + humanBytes(p.BytesDone) + " read")
			}
			kind, payload = "request.preparing", map[string]any{"stage": "machine", "detail": detail}
		}
	} else if log := event.GetLog(); log != nil {
		payload = map[string]any{"level": log.Level, "message": log.Text}
	}
	s.retainSampleV1(current, id, kind, payload, at, event.Sequence)
	return nil
}
func (s *Store) retainSampleV1(current *runTelemetryV1, id, kind string, payload map[string]any, at string, remote uint64) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return
	}
	s.telemetrySequence++
	sample := telemetrySampleV1{sequence: s.telemetrySequence, remote: remote, event: Event{
		RequestID: id, Type: kind, Attempt: current.attempt, Raw: raw, At: at,
	}}
	current.latest[kind] = sample
	if kind == "machine.progress" && current.first == nil {
		if fields, ok := payload["payload"].(map[string]any); ok {
			if fraction, ok := fields["overall_fraction"].(float64); ok && fraction >= 0 {
				first := sample
				current.first = &first
			}
		}
	}
}
func (s *Store) sampleLocalTelemetryV1(id, kind string, attempt int64, payload map[string]any) bool {
	if kind != "machine.progress" && kind != "request.preparing" && kind != "request.log" {
		return false
	}
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	current := s.telemetry[id]
	if current == nil {
		return false
	} // Other execution paths retain their existing contract.
	if kind == "machine.progress" {
		current.progress, _ = json.Marshal(payload["payload"])
	}
	s.retainSampleV1(current, id, kind, payload, now(), 0)
	return true
}
func decodedLiveV1(sample telemetrySampleV1) LiveRunEventV1 {
	event := sample.event
	event.Raw = append(json.RawMessage(nil), event.Raw...)
	_ = json.Unmarshal(event.Raw, &event.Payload)
	return LiveRunEventV1{Sequence: sample.sequence, Event: event}
}

// LiveRunEventsV1 returns only each run's latest bounded telemetry; an empty id multiplexes.
func (s *Store) LiveRunEventsV1(id string) []LiveRunEventV1 {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	var samples []LiveRunEventV1
	for request, current := range s.telemetry {
		if id != "" && request != id {
			continue
		}
		for _, sample := range current.latest {
			samples = append(samples, decodedLiveV1(sample))
		}
	}
	sort.Slice(samples, func(i, j int) bool { return samples[i].Sequence < samples[j].Sequence })
	return samples
}
func (s *Store) liveProgressV1(id string, attempt int64) map[string]any {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	current := s.telemetry[id]
	if current == nil || current.attempt != attempt || len(current.progress) == 0 {
		return nil
	}
	var sample map[string]any
	_ = json.Unmarshal(current.progress, &sample)
	return sample
}
func (s *Store) liveEstimateV1(id string, attempt int64) (int64, bool) {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	current := s.telemetry[id]
	if current == nil || current.attempt != attempt || current.first == nil {
		return 0, false
	}
	latest, ok := current.latest["machine.progress"]
	if !ok {
		return 0, false
	}
	fraction := func(sample telemetrySampleV1) (float64, time.Time, bool) {
		var payload struct {
			Payload struct {
				Fraction *float64 `json:"overall_fraction"`
			} `json:"payload"`
		}
		err := json.Unmarshal(sample.event.Raw, &payload)
		at, parse := time.Parse(time.RFC3339Nano, sample.event.At)
		if err != nil || parse != nil || payload.Payload.Fraction == nil {
			return 0, time.Time{}, false
		}
		return *payload.Payload.Fraction, at, true
	}
	first, began, firstOK := fraction(*current.first)
	last, ended, lastOK := fraction(latest)
	if !firstOK || !lastOK || last <= first || last > 1 || !ended.After(began) {
		return 0, false
	}
	return int64((1 - last) * float64(ended.Sub(began).Milliseconds()) / (last - first)), true
}

func (s *Store) flushTelemetryV1(tx *sql.Tx, id string, cursor int64) (uint64, error) {
	s.telemetryMu.Lock()
	current := s.telemetry[id]
	var samples []telemetrySampleV1
	if current != nil {
		if current.first != nil && current.first.sequence > current.flushed {
			samples = append(samples, *current.first)
		}
		for _, sample := range current.latest {
			if sample.sequence > current.flushed {
				samples = append(samples, sample)
			}
		}
	}
	s.telemetryMu.Unlock()
	sort.Slice(samples, func(i, j int) bool { return samples[i].sequence < samples[j].sequence })
	var through uint64
	for _, sample := range samples {
		if sample.sequence == through {
			continue
		}
		through = sample.sequence
		if sample.remote != 0 && cursor >= 0 && sample.remote <= uint64(cursor) {
			continue
		}
		event := sample.event
		if _, err := tx.Exec(`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,?,?,?,?)`,
			event.RequestID, event.Type, event.Attempt, string(event.Raw), event.At); err != nil {
			return 0, err
		}
	}
	return through, nil
}
func (s *Store) ackTelemetryV1(id string, through uint64) {
	s.telemetryMu.Lock()
	defer s.telemetryMu.Unlock()
	if current := s.telemetry[id]; current != nil {
		current.flushed = max(current.flushed, through)
	}
}
