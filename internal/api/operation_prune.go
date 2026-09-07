package api

import (
	"encoding/json"
	"io"
	"net/http"
)

type CachePruneResult struct {
	RemovedEntries uint32 `json:"removed_entries"`
	ReclaimedBytes uint64 `json:"reclaimed_bytes"`
	StoreBusy      bool   `json:"store_busy"`
}

type RentalPruneResult struct {
	Rental string `json:"rental"`
	CachePruneResult
}

func (s *Server) pruneRental(w http.ResponseWriter, r *http.Request) {
	s.pruneOperationCache(w, r, r.PathValue("rental_id"))
}

func (s *Server) pruneCache(w http.ResponseWriter, r *http.Request) {
	s.pruneOperationCache(w, r, "")
}

func (s *Server) pruneOperationCache(w http.ResponseWriter, r *http.Request, rental string) {
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
	removed, reclaimed, busy, problem := s.orchestrator.PruneOperationCache(rental)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	result := CachePruneResult{RemovedEntries: removed, ReclaimedBytes: reclaimed, StoreBusy: busy}
	if rental != "" {
		s.ok(w, r, http.StatusOK, RentalPruneResult{Rental: rental, CachePruneResult: result})
		return
	}
	s.ok(w, r, http.StatusOK, result)
}
