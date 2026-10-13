package api

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/modelsource"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// submitOperation records one operation on a machine (this computer's, `local`, or a rental by
// id) as a run under the next run number, and starts it. The same operation still unsettled on
// that machine is answered instead of a second one.
func (s *Server) submitOperation(w http.ResponseWriter, r *http.Request) {
	decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 4<<20))
	var op records.Operation
	if err := decoder.Decode(&op); err != nil {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "an operation is one JSON object", `send {"package":"org/name","release":"1.2.3"}`)
		return
	}
	var trailing any
	if decoder.Decode(&trailing) != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "an operation is one JSON object", "remove trailing request data")
		return
	}
	op.Package, op.Release = strings.TrimSpace(op.Package), strings.TrimSpace(op.Release)
	if op.Hub == "" {
		selected, problem := s.hubOf(r)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		op.Hub = selected
	}
	if problem := validOperation(op); problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	machine := r.PathValue("machine")
	payload, err := json.Marshal(op)
	if err != nil {
		s.refuseTyped(w, r, exit.Internalf("cannot encode the operation: %s", err))
		return
	}
	digest := sha256.Sum256(append([]byte(op.Kind()+"\x00"+machine+"\x00"), payload...))
	spec := orchestrator.Submission{RequestID: records.NewID("op"), IdemKey: records.NewID("operation"),
		BodyDigest: "sha256:" + hex.EncodeToString(digest[:]), Hub: op.Hub, Kind: op.Kind(),
		Package: op.Package, Entrypoint: op.Entrypoint, Release: op.Release, Payload: payload,
		MachineExecutionObserver: true}
	if machine != records.LocalMachine {
		spec.RequestedRental = machine
	}
	recorded, problem := s.store.PendingOperation(machine, spec.BodyDigest)
	fresh := false
	if problem == nil && recorded == nil {
		var row records.Request
		row, fresh, problem = s.recordSubmission(spec)
		recorded = &row
	}
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if fresh {
		s.activateRecorded(*recorded)
	}
	status := http.StatusOK
	if fresh {
		status = http.StatusAccepted
	}
	s.ok(w, r, status, s.handleOf(*recorded, 0))
}

func validOperation(op records.Operation) *exit.Error {
	if op.Package != "" {
		if _, problem := canonicalPackageRef(op.Package); problem != nil {
			return problem
		}
		// An empty release is the hub's newest, which the machine reads there.
		if strings.ContainsAny(op.Release, "\r\n\x00/@") {
			return exit.New(exit.Validation, "an operation names at most one exact package release")
		}
	} else if op.Release != "" || len(op.Models) == 0 {
		return exit.New(exit.Validation, "an operation names a package release or models")
	}
	if op.Local != "" || op.Remove != "" {
		// Local code from this computer, or one installation taken away: nothing else.
		if op.Local != "" && op.Remove != "" || len(op.Models) > 0 || op.Destination != "" || op.Write != "" || op.Warm != "" ||
			op.Local != "" && !strings.HasPrefix(op.Package, "local/") || strings.ContainsAny(op.Local+op.Remove, "\r\n\x00/") {
			return exit.New(exit.Validation, "a local installation or a removal names one package and nothing else")
		}
	}
	downloads := op.Models
	if op.Destination != "" {
		// One model put in the destination: a provider source or a written file made on the
		// machine, a checkpoint it holds or downloads, or a local/ alias it holds; the
		// destination is a Tensorhub repository or a local/ alias.
		if op.Package != "" || len(op.Models) != 1 || op.Models[0].Source == "" && op.Models[0].Manifest == "" && op.Models[0].Model == "" ||
			op.Write != "" && !strings.HasPrefix(op.Models[0].Source, "object://") {
			return exit.New(exit.Validation, "a model transfer names one model and its destination")
		}
		if _, local, problem := modelsource.LocalAlias(op.Destination); local {
			if problem != nil {
				return problem
			}
		} else if problem := hub.ValidPublicationDestination(op.Destination); problem != nil {
			return problem
		}
		downloads = nil
	}
	if op.Warm != "" {
		// A warm set member: its function's models arrive resolved (exact or exact rungs).
		if op.Package == "" || op.Entrypoint == "" || op.Destination != "" {
			return exit.New(exit.Validation, "a warm set member names one package function")
		}
		downloads = nil
	}
	for _, model := range downloads {
		if !model.Downloadable() {
			return exit.New(exit.Validation, "a model download names a downloadable Hub checkpoint")
		}
		if op.Package == "" && (model.Package != "" || model.Slot != "" || model.BindingPath != "" || len(model.SharedSlots) > 0) {
			return exit.New(exit.Validation, "a standalone model download names no package slots")
		}
		if model.Model == "" || model.Manifest == "" || op.Package != "" && (model.Package != op.Package || model.Slot == "") {
			return exit.New(exit.Validation, "a model download names exact models, and the package's slots when it names a package")
		}
		if _, problem := canonicalPackageRef(model.Model); problem != nil {
			return problem
		}
		if _, err := canonical.Raw(model.Manifest); err != nil {
			return exit.New(exit.Validation, "a model download's manifest is not a canonical digest")
		}
	}
	return nil
}

func (s *Server) machineBecameReady(w http.ResponseWriter, r *http.Request) {
	if s.machineReady == nil {
		s.refuseTyped(w, r, exit.Unavailablef("machine readiness notifications are unavailable"))
		return
	}
	s.machineReady(r.PathValue("machine"))
	s.ok(w, r, http.StatusOK, map[string]bool{"notified": true})
}

func canonicalPackageRef(value string) (string, *exit.Error) {
	ref, problem := hub.ParseRef(value)
	if problem != nil {
		return "", problem
	}
	return ref.String(), nil
}
