package api

import (
	"encoding/json"
	"io"
	"net/http"
)

type RentalPruneResult struct {
	Rental         string `json:"rental"`
	RemovedEntries uint32 `json:"removed_entries"`
	ReclaimedBytes uint64 `json:"reclaimed_bytes"`
	StoreBusy      bool   `json:"store_busy"`
}

func (s *Server) pruneRental(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1024))
	decoder.DisallowUnknownFields()
	var body *struct{}
	if err := decoder.Decode(&body); err != nil || body == nil {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "cache pruning takes an empty object", "send {} with no path or namespace")
		return
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "cache pruning takes one empty object", "")
		return
	}
	id := r.PathValue("rental_id")
	removed, reclaimed, busy, problem := s.orchestrator.PruneOperationCache(id)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, RentalPruneResult{Rental: id, RemovedEntries: removed, ReclaimedBytes: reclaimed, StoreBusy: busy})
}
