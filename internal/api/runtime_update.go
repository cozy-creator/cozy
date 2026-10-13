package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

type RuntimeUpdate = records.RuntimeUpdate

type RuntimeUpdateRequest struct {
	RuntimeWheel    string `json:"runtime_wheel,omitempty"`
	TensorFSWheel   string `json:"tensorfs_wheel,omitempty"`
	RuntimeVersion  string `json:"runtime_version,omitempty"`
	TensorFSVersion string `json:"tensorfs_version,omitempty"`
}

// Validate refuses a request that names TensorFS alone, or a wheel and a version of one package.
func (r RuntimeUpdateRequest) Validate() *exit.Error {
	if r.TensorFSWheel != "" && r.RuntimeWheel == "" {
		return exit.New(exit.Validation, "--tensorfs-wheel requires --runtime-wheel")
	}
	if r.RuntimeWheel != "" && r.RuntimeVersion != "" || r.TensorFSWheel != "" && r.TensorFSVersion != "" {
		return exit.New(exit.Validation, "name a wheel or a version, not both")
	}
	return nil
}

func (s *Server) startRuntimeUpdate(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
	var body *RuntimeUpdateRequest
	var trailing any
	if decoder.Decode(&body) != nil || body == nil || decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "Runtime update takes one object with optional runtime_wheel and tensorfs_wheel paths", "send {} for the public release")
		return
	}
	if s.runtimeUpdate == nil {
		s.refuse(w, r, http.StatusServiceUnavailable, "runtime_update_unavailable", "this Calcifer cannot update private workers", "")
		return
	}
	result, problem := s.runtimeUpdate(r.PathValue("rental_id"), *body)
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
