package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// LifecycleIdentity is one durable obligation that prevents a safe daemon down.
// Kind distinguishes ordinary invocations, run-once jobs, provider rentals, and paid
// acquisition operations no rental row stands for yet.
type LifecycleIdentity struct {
	Kind  string `json:"kind"`
	ID    string `json:"id"`
	State string `json:"state"`
}

type UnloadResult struct {
	Stopped []orchestrator.WorkerFacts `json:"stopped"`
	Count   int                        `json:"count"`
}

// DownResult is the daemon-side half of `down [--all]`. ShuttingDown is true only
// after every local invocation settled and every rental row/operation disappeared.
// Under --all, a false result tells the caller exactly what cancellation was requested
// and which paid obligations must be ended through Tensorhub before retrying.
type DownResult struct {
	ShuttingDown          bool                `json:"shutting_down"`
	CancellationRequested []LifecycleIdentity `json:"cancellation_requested"`
	Active                []LifecycleIdentity `json:"active"`
	Rentals               []LifecycleIdentity `json:"rentals"`
}

func (s *Server) unload(w http.ResponseWriter, r *http.Request) {
	stopped, problem := s.orchestrator.UnloadIdleLocalWorkers()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	s.ok(w, r, http.StatusOK, UnloadResult{Stopped: stopped, Count: len(stopped)})
}

func (s *Server) downDaemon(w http.ResponseWriter, r *http.Request) {
	s.shutdownAdmission.Lock()
	defer s.shutdownAdmission.Unlock()
	if s.shuttingDown {
		s.ok(w, r, http.StatusAccepted, DownResult{ShuttingDown: true})
		return
	}
	var body struct {
		All bool `json:"all"`
	}
	data, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	err := decoder.Decode(&body)
	var trailing any
	if err == nil {
		err = decoder.Decode(&trailing)
	}
	if err != io.EOF {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request",
			`this route takes exactly {"all":true|false}`, "send one closed down request")
		return
	}

	active, rentals, problem := s.downBlockers()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}
	if !body.All && (len(active) > 0 || len(rentals) > 0) {
		s.refuseTyped(w, r, exit.Named(exit.Conflict, "active_work",
			"daemon shutdown refused: active %s", joinLifecycleIdentities(active, rentals)).
			WithRemedy("cancel the named invocations/jobs and end the named rentals, or use explicit `cozy down --all`"))
		return
	}

	requested := []LifecycleIdentity{}
	if body.All {
		for _, identity := range active {
			row, read := s.store.RequestRow(identity.ID)
			if read != nil {
				s.refuseTyped(w, r, read)
				return
			}
			if row == nil {
				continue
			}
			changed, cancelProblem := s.cancelForDown(*row)
			if cancelProblem != nil {
				s.refuseTyped(w, r, cancelProblem.WithRemedy(
					"some cancellation requests may already be recorded; the daemon remains alive for reconciliation"))
				return
			}
			if changed {
				requested = append(requested, identity)
			}
		}
		if len(active) > 0 || len(rentals) > 0 {
			// Rental destruction belongs to Tensorhub/provider ownership. Returning the exact
			// identities while leaving this process alive lets the CLI confirm remote absence,
			// forget the local rows, and retry this same idempotent request.
			s.ok(w, r, http.StatusAccepted, DownResult{
				CancellationRequested: requested, Active: active, Rentals: rentals,
			})
			return
		}
	}

	if s.shutdown == nil {
		s.refuse(w, r, http.StatusConflict, "conflict",
			"this server was built with no shutdown hook", "")
		return
	}
	s.shuttingDown = true
	s.ok(w, r, http.StatusAccepted, DownResult{ShuttingDown: true})
	go s.shutdown()
}

// StopUnlessManaging is the idle exit's last word. It re-reads `managing` UNDER the
// shutdown-admission gate — the same boundary `down` closes — so a submission cannot
// commit between the daemon's last "nothing to manage" sample and the listener close.
// An empty answer there marks the server shutting down and fires the shutdown hook; a
// non-empty one is returned so the caller can say what held the daemon up.
func (s *Server) StopUnlessManaging(managing func() ([]string, *exit.Error)) ([]string, *exit.Error) {
	s.shutdownAdmission.Lock()
	defer s.shutdownAdmission.Unlock()
	if s.shuttingDown {
		return nil, nil
	}
	held, problem := managing()
	if problem != nil || len(held) > 0 {
		return held, problem
	}
	if s.shutdown == nil {
		return nil, exit.Internalf("this server was built with no shutdown hook")
	}
	s.shuttingDown = true
	go s.shutdown()
	return nil, nil
}

// downBlockers is the subset of the daemon's obligations `down` refuses on: the work it
// can cancel and the paid pods it must see ended. An attempt awaiting its ack and an
// export mid-copy are on their way to settlement and are drained by the close itself.
func (s *Server) downBlockers() ([]LifecycleIdentity, []LifecycleIdentity, *exit.Error) {
	obligations, problem := s.store.Obligations()
	if problem != nil {
		return nil, nil, problem
	}
	active, rentals := []LifecycleIdentity{}, []LifecycleIdentity{}
	for _, o := range obligations {
		identity := LifecycleIdentity{Kind: o.Kind, ID: o.ID, State: o.State}
		switch o.Kind {
		case "invocation", "job":
			active = append(active, identity)
		case "rental", "rental_operation":
			rentals = append(rentals, identity)
		}
	}
	return active, rentals, nil
}

// cancelForDown reuses the request authority's existing queued/live cancellation
// boundaries. A terminal awaiting acknowledgement is already on its way to settlement;
// it remains in Active and makes the caller retry rather than receiving a second verdict.
func (s *Server) cancelForDown(row records.Request) (bool, *exit.Error) {
	attempts, problem := s.store.Attempts(row.ID)
	if problem != nil {
		return false, problem
	}
	if len(attempts) == 0 {
		return true, s.orchestrator.CancelQueued(row.ID)
	}
	last := attempts[len(attempts)-1]
	switch last.State {
	case "closed", "dispatch_aborted":
		return true, s.orchestrator.CancelQueued(row.ID)
	case "terminal":
		return false, nil
	default:
		return true, s.orchestrator.CancelClient(
			row.ID, uint64(last.Attempt), orchestrator.ClientCancelGraceMS)
	}
}

func joinLifecycleIdentities(groups ...[]LifecycleIdentity) string {
	var names []string
	for _, group := range groups {
		for _, identity := range group {
			names = append(names, fmt.Sprintf("%s %s (%s)", identity.Kind, identity.ID, identity.State))
		}
	}
	return strings.Join(names, ", ")
}
