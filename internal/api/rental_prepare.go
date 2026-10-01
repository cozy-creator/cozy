package api

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
)

// RentalPackagePrepareRequest is frozen at admission and retained until the
// daemon observes a verified preparation receipt or a terminal rental failure.
type RentalPackagePrepareRequest = records.RentalInstallSelection
type RentalPackagePrepareResult = records.RentalInstall
type RentalInstallStatus = machines.InstallStatus

func (s *Server) prepareRentalPackage(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
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
	if body.Hub == "" {
		hub, problem := s.hubOf(r)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		body.Hub = hub
	}
	if body.Package != "" {
		if _, problem := canonicalPackageRef(body.Package); problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		if body.Release == "" || strings.ContainsAny(body.Release, "\r\n\x00") {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental installation requires one exact package release"))
			return
		}
	} else if body.Release != "" || len(body.Models) == 0 {
		s.refuseTyped(w, r, exit.New(exit.Validation, "rental installation requires a package release or explicitly selected models"))
		return
	}
	for _, model := range body.Models {
		if !model.Downloadable() {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental model installation requires a downloadable Hub checkpoint"))
			return
		}
		if body.Package == "" && (model.Package != "" || model.Slot != "" || model.BindingPath != "" || len(model.SharedSlots) > 0) {
			s.refuseTyped(w, r, exit.New(exit.Validation, "standalone model installation does not accept application slots"))
			return
		}

		if model.Model == "" || model.Manifest == "" || body.Package != "" && (model.Package != body.Package || model.Slot == "") {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental model installation requires exact model selections and matching package slots when a package is supplied"))
			return
		}
		if _, problem := canonicalPackageRef(model.Model); problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			s.refuseTyped(w, r, exit.New(exit.Validation, "rental model installation manifest is not a canonical digest"))
			return
		}
	}
	id := r.PathValue("rental_id")
	// The selection was resolved on the command's hub; a machine installs only what its own
	// hub serves.
	machineHub, problem := machines.InstallHub(s.store, id)
	if problem == nil && machineHub != "" {
		selected, refused := s.submissionHub(r, id)
		if problem = refused; problem == nil {
			if origin, invalid := config.HubOrigin(machineHub); invalid == nil && origin != selected {
				problem = rentalHubMismatch(id, machineHub, selected)
			}
		}
	}
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}

	if s.rentalInstall == nil {
		s.refuseTyped(w, r, exit.Unavailablef("rental installation queue is unavailable"))
		return
	}
	result, problem := s.rentalInstall(id, body)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, result)
}

func (s *Server) rentalInstallState(w http.ResponseWriter, r *http.Request) {
	if s.rentalInstallStatus == nil {
		s.refuseTyped(w, r, exit.Unavailablef("rental installation queue is unavailable"))
		return
	}
	status, problem := s.rentalInstallStatus(r.PathValue("rental_id"), r.PathValue("id"))
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, status)
}

func canonicalPackageRef(value string) (string, *exit.Error) {
	ref, problem := hub.ParseRef(value)
	if problem != nil {
		return "", problem
	}
	return ref.String(), nil
}
