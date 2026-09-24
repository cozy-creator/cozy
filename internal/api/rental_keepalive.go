package api

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/cozy-creator/cozy/internal/exit"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type RentalKeepaliveRequest struct {
	RequestID string `json:"request_id"`
}
type RentalKeepaliveResult struct {
	Rental               string `json:"rental"`
	RequestID            string `json:"request_id"`
	WorkerID             string `json:"worker_id"`
	WorkerBootID         string `json:"worker_boot_id"`
	AcknowledgedAtUnixMS int64  `json:"acknowledged_at_unix_ms"`
	IdleDeadlineUnixMS   int64  `json:"idle_deadline_unix_ms"`
}

func (s *Server) keepRentalAlive(w http.ResponseWriter, r *http.Request) {
	var body RentalKeepaliveRequest
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&body); err != nil || body.RequestID == "" || len(body.RequestID) > pb.MaxRentalKeepaliveRequestIDBytes {
		s.refuseTyped(w, r, exit.New(exit.Validation, "keepalive requires one request_id and no duration options"))
		return
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		s.refuseTyped(w, r, exit.New(exit.Validation, "keepalive accepts one JSON object"))
		return
	}
	if s.rentalKeepalive == nil {
		s.refuseTyped(w, r, exit.Unavailablef("rental keepalive is unavailable"))
		return
	}
	result, problem := s.rentalKeepalive(r.Context(), r.PathValue("rental_id"), body.RequestID)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, result)
}
