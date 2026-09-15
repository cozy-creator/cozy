package api

import (
	"encoding/json"
	"github.com/cozy-creator/cozy/internal/records"
	"io"
	"net/http"
)

type RuntimeUpdate = records.RuntimeUpdate

func (s *Server) startRuntimeUpdate(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
	decoder.DisallowUnknownFields()
	var body *struct{}
	var trailing any
	if decoder.Decode(&body) != nil || body == nil || decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "Runtime update takes one empty object", "send {}")
		return
	}
	if s.runtimeUpdate == nil {
		s.refuse(w, r, http.StatusServiceUnavailable, "runtime_update_unavailable", "this daemon cannot update private workers", "")
		return
	}
	result, problem := s.runtimeUpdate(r.PathValue("rental_id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, result)
}

func (s *Server) readRuntimeUpdate(w http.ResponseWriter, r *http.Request) {
	result, problem := s.store.RuntimeUpdate(r.PathValue("rental_id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if result == nil {
		s.refuse(w, r, http.StatusNotFound, "runtime_update_missing", "this rental has no Runtime update", "")
		return
	}
	s.ok(w, r, http.StatusOK, result)
}
