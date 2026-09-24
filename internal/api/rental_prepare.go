package api

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// RentalPackagePrepareRequest is an explicit, idempotent prerequisite operation.
// The package release and optional model bindings are immutable content identity;
// Runtime's preparation ledger reuses them when a caller repeats this request.
type RentalPackagePrepareRequest struct {
	Package string                  `json:"package"`
	Release string                  `json:"release"`
	Models  []orchestrator.ModelRef `json:"models,omitempty"`
}

type RentalPackagePrepareResult struct {
	Rental  string `json:"rental"`
	Package string `json:"package"`
	Release string `json:"release"`
	Status  string `json:"status"`
}

func (s *Server) prepareRentalPackage(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	decoder.DisallowUnknownFields()
	var body RentalPackagePrepareRequest
	if err := decoder.Decode(&body); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "rental package preparation requires package and release", "send {\"package\":\"org/name\",\"release\":\"1.2.3\"}")
		return
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "rental package preparation accepts one JSON object", "remove trailing request data")
		return
	}
	body.Package = strings.TrimSpace(body.Package)
	body.Release = strings.TrimSpace(body.Release)
	if _, problem := canonicalPackageRef(body.Package); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if body.Release == "" || strings.ContainsAny(body.Release, "\r\n\x00") {
		s.refuseTyped(w, r, exit.New(exit.Validation, "rental package preparation requires one exact release"))
		return
	}
	for _, model := range body.Models {
		if model.Package != body.Package || model.Slot == "" || model.Model == "" || model.Manifest == "" {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental model preparation bindings must name the prepared package, slot, model, and manifest"))
			return
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental model preparation manifest is not a canonical digest"))
			return
		}
	}
	id := r.PathValue("rental_id")
	if s.rentalPreparation == nil {
		s.refuseTyped(w, r, exit.Unavailablef("rental preparation activity tracker is unavailable"))
		return
	}
	finish, problem := s.rentalPreparation(id)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	confirmed := false
	defer func() { finish(confirmed) }()
	instance, _, _, problem := s.orchestrator.EnsureRentalContext(r.Context(), id)
	if problem != nil {
		confirmed = problem.Code != exit.Unavailable && problem.Code != exit.Canceled
		s.refuseTyped(w, r, problem)
		return
	}
	models := orchestrator.DownloadModelRefs(body.Models)
	if len(models) != len(body.Models) {
		s.refuseTyped(w, r, exit.New(exit.Validation, "rental model preparation bindings are not downloadable checkpoint selections"))
		return
	}
	release, problem := s.orchestrator.UseRental(id)
	if problem != nil {
		confirmed = problem.Code != exit.Unavailable && problem.Code != exit.Canceled
		s.refuseTyped(w, r, problem)
		return
	}
	defer release()
	if problem := s.orchestrator.PrepareRentalPackage(r.Context(), instance,
		&pb.DownloadPackageRef{Package: body.Package, Release: body.Release}, models); problem != nil {
		confirmed = problem.Code != exit.Unavailable && problem.Code != exit.Canceled
		s.refuseTyped(w, r, problem)
		return
	}
	confirmed = true
	s.ok(w, r, http.StatusOK, RentalPackagePrepareResult{Rental: id, Package: body.Package, Release: body.Release, Status: "prepared"})
}

func canonicalPackageRef(value string) (string, *exit.Error) {
	ref, problem := hub.ParseRef(value)
	if problem != nil {
		return "", problem
	}
	return ref.String(), nil
}
