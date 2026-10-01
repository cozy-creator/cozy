package api

import (
	"encoding/json"
	"io"
	"net/http"
	"slices"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// InstallRequest is a download's or installation's selection, frozen at admission.
type InstallRequest struct {
	records.InstallSelection
	IdempotencyKey string `json:"idempotency_key,omitempty"`
}

type RuntimeUpdateRequest struct {
	RuntimeWheel    string `json:"runtime_wheel,omitempty"`
	TensorFSWheel   string `json:"tensorfs_wheel,omitempty"`
	RuntimeVersion  string `json:"runtime_version,omitempty"`
	TensorFSVersion string `json:"tensorfs_version,omitempty"`
}

// updatePhases are a Runtime update's states as the phase a reader sees.
var updatePhases = map[string]string{"preparing": "checking_runtime", "updating": "updating_runtime",
	"reconciling": "reconciling_runtime", "waiting_activation": "waiting_activation"}

// operationLifecycle is a download, installation or Runtime update in the shape every
// journaled run has, with its live bytes while it holds its machine.
func (s *Server) operationLifecycle(o records.Operation) Lifecycle {
	life := Lifecycle{Number: o.Number, Kind: o.Kind, Journal: o.Kind, RequestID: o.ID, Status: o.Status(),
		Target: o.Target(), Machine: o.Machine, Hub: o.Hub, CreatedAt: o.CreatedAt, ErrorCode: o.ErrorCode,
		Error: o.Error, CanceledBy: o.CanceledBy, ResponseURL: "/v1/requests/" + o.ID}
	if o.Machine != machines.Local {
		life.RentalID = o.Machine
		if row, problem := s.store.RentalRow(o.Machine); problem == nil && row != nil && row.MachineName != "" {
			life.Machine = row.MachineName
		}
	}
	if o.State == "unusable" {
		life.ErrorCode, life.Error = "rental.unusable", o.Unusable(life.Machine).Message
	}
	if o.Kind == "upload" {
		upload := records.OutputUploadOf(o)
		life.Upload = &upload
	} else if len(o.Result) > 0 {
		life.Result = o.Result
	}
	if phase, ok := s.orchestrator.PreparationPhase(o.ID); ok && o.Active() {
		fillPhase(&life, phase)
	} else if name := updatePhases[o.State]; name != "" {
		life.Phase = name
	}
	if len(life.PhaseModels) == 0 {
		for _, model := range o.Progress {
			life.PhaseModels = append(life.PhaseModels, orchestrator.ModelDownloadProgress{Model: model.Model,
				Release: model.Release, Lane: model.Lane, Manifest: model.Manifest, Moved: model.Moved, Total: model.Total})
		}
	}
	if o.State == "queued" {
		if ahead := s.operationAhead(o); ahead != nil {
			life.WaitingFor = ahead
			life.WaitReason = ahead.String()
		}
	}
	return life
}

// operationAhead is the work a queued download or installation waits behind on its machine.
func (s *Server) operationAhead(o records.Operation) *orchestrator.WaitingRun {
	active, problem := s.store.ActiveOperations(o.Machine)
	if problem != nil {
		return nil
	}
	for _, other := range active {
		if other.Number < o.Number && other.State != "queued" {
			return &orchestrator.WaitingRun{Number: other.Number, RequestID: other.ID, Kind: other.Kind, Target: other.Target()}
		}
	}
	return nil
}

// operationBlocking is the download or installation a queued run's preparation waits on:
// one landing a model the run needs, or a package installation holding the machine's
// preparation while the run's own has not reached its models.
func (s *Server) operationBlocking(machine string, phase orchestrator.PhaseObservation) *orchestrator.WaitingRun {
	if machine == "" || (phase.Name != orchestrator.PhaseResolving && phase.Name != orchestrator.PhaseDownloading) {
		return nil
	}
	active, problem := s.store.ActiveOperations(machine)
	if problem != nil {
		return nil
	}
	for _, o := range active {
		if o.State != "installing" {
			continue
		}
		shared := phase.Name == orchestrator.PhaseResolving && o.Kind == "install"
		for _, model := range o.Install.Models {
			shared = shared || slices.ContainsFunc(phase.Models, func(m orchestrator.ModelDownloadProgress) bool {
				return m.Manifest == model.Manifest
			})
		}
		if shared {
			return &orchestrator.WaitingRun{Number: o.Number, RequestID: o.ID, Kind: o.Kind, Target: o.Target()}
		}
	}
	return nil
}

// operations is one history page of operations under the run listing's filters.
func (s *Server) operations(status, packageName, hub string, limit int, before int64) ([]Lifecycle, *exit.Error) {
	rows, problem := s.store.OperationsBefore(status, packageName, hub, s.cfg.HubURL, limit, before)
	if problem != nil {
		return nil, problem
	}
	out := make([]Lifecycle, 0, len(rows))
	for _, row := range rows {
		out = append(out, s.operationLifecycle(row))
	}
	return out, nil
}

// cancelOperation detaches its owner from a download or installation and withdraws an
// update that has not started. The machine stops a transfer only when nothing else needs it.
func (s *Server) cancelOperation(w http.ResponseWriter, r *http.Request, o records.Operation) {
	if s.operationCancel == nil {
		s.refuseTyped(w, r, exit.Unavailablef("this daemon cannot cancel operations"))
		return
	}
	if problem := s.operationCancel(o, requestActor(r)); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	current, problem := s.store.Operation(o.ID)
	if problem != nil || current == nil {
		s.refuse(w, r, http.StatusInternalServerError, "internal", "the operation was canceled and cannot be read back", "")
		return
	}
	s.ok(w, r, http.StatusOK, s.operationLifecycle(*current))
}

func (s *Server) installOnMachine(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	var body InstallRequest
	if err := decoder.Decode(&body); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "an installation names a package release or models", "send {\"package\":\"org/name\",\"release\":\"1.2.3\"}")
		return
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "an installation accepts one JSON object", "remove trailing request data")
		return
	}
	body.Package, body.Release = strings.TrimSpace(body.Package), strings.TrimSpace(body.Release)
	if body.Hub == "" {
		hub, problem := s.hubOf(r)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		body.Hub = hub
	}
	if problem := validInstall(body.InstallSelection); problem != nil {
		s.refuseTyped(w, r, problem)
		return
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
		s.refuseTyped(w, r, exit.Unavailablef("the installation queue is unavailable"))
		return
	}
	accepted, fresh, problem := s.rentalInstall(id, strings.TrimSpace(body.IdempotencyKey), body.InstallSelection)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	status := http.StatusAccepted
	if !fresh {
		status = http.StatusOK
	}
	s.ok(w, r, status, s.operationLifecycle(*accepted))
}

func validInstall(body records.InstallSelection) *exit.Error {
	if body.Package != "" {
		if _, problem := canonicalPackageRef(body.Package); problem != nil {
			return problem
		}
		if body.Release == "" || strings.ContainsAny(body.Release, "\r\n\x00") {
			return exit.New(exit.Validation, "an installation requires one exact package release")
		}
	} else if body.Release != "" || len(body.Models) == 0 {
		return exit.New(exit.Validation, "an installation requires a package release or explicitly selected models")
	}
	for _, model := range body.Models {
		if !model.Downloadable() {
			return exit.New(exit.Validation, "a model installation requires a downloadable Hub checkpoint")
		}
		if body.Package == "" && (model.Package != "" || model.Slot != "" || model.BindingPath != "" || len(model.SharedSlots) > 0) {
			return exit.New(exit.Validation, "a standalone model download does not accept application slots")
		}
		if model.Model == "" || model.Manifest == "" || body.Package != "" && (model.Package != body.Package || model.Slot == "") {
			return exit.New(exit.Validation, "a model installation requires exact model selections and matching package slots when a package is supplied")
		}
		if _, problem := canonicalPackageRef(model.Model); problem != nil {
			return problem
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			return exit.New(exit.Validation, "a model installation manifest is not a canonical digest")
		}
	}
	return nil
}

func canonicalPackageRef(value string) (string, *exit.Error) {
	ref, problem := hub.ParseRef(value)
	if problem != nil {
		return "", problem
	}
	return ref.String(), nil
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
		s.refuse(w, r, http.StatusServiceUnavailable, "runtime_update_unavailable", "this daemon cannot update private workers", "")
		return
	}
	result, problem := s.runtimeUpdate(r.PathValue("rental_id"), *body)
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusAccepted, s.operationLifecycle(*result))
}

// publicOperationStatus is the run listing's status filter for operations; a status only
// runs have matches none.
func publicOperationStatus(state string) string {
	switch state {
	case "", "queued", "in_progress", "completed", "failed", "canceled":
		return state
	}
	return "-"
}
