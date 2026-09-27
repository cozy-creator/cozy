package api

import (
	"encoding/json"
	"net/http"
)

// maxEvidenceEvents bounds one run's durable history in a single read. Imported progress
// samples are not evidence and do not count against it.
const maxEvidenceEvents = 4096

// Evidence is one run's execution record read from records alone: its durable events in
// order and its last attempt's kept triage bundle (`cozy run show`).
type Evidence struct {
	RequestID string          `json:"request_id"`
	Events    []EvidenceEvent `json:"events"`
	Triage    json.RawMessage `json:"triage,omitempty"`
}

type EvidenceEvent struct {
	Type    string         `json:"type"`
	Attempt int64          `json:"attempt,omitempty"`
	At      string         `json:"at"`
	Payload map[string]any `json:"payload"`
}

func (s *Server) requestEvidence(w http.ResponseWriter, r *http.Request) {
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
	events, e := s.store.EvidenceEvents(row.ID, maxEvidenceEvents)
	if e != nil {
		s.refuseTyped(w, r, e)
		return
	}
	out := Evidence{RequestID: row.ID, Events: make([]EvidenceEvent, 0, len(events))}
	for _, event := range events {
		out.Events = append(out.Events, EvidenceEvent{Type: event.Type, Attempt: event.Attempt,
			At: event.At, Payload: event.Payload})
	}
	if attempts, problem := s.store.Attempts(row.ID); problem == nil && len(attempts) > 0 {
		last := attempts[len(attempts)-1]
		if _, bundle, problem := s.store.TriageBundle(last.AttemptKey); problem == nil && json.Valid(bundle) {
			out.Triage = bundle
		}
	} else if bundle, problem := s.store.MachineTriage(row.ID); problem == nil && bundle != nil {
		out.Triage = bundle
	}
	s.ok(w, r, http.StatusOK, out)
}
