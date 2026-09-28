package api

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
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

// DownResult is the daemon-side half of client disconnect or explicit teardown.
// Normal/forced disconnect preserves requests, Runtime executions and rentals.
// Under --all, a false result tells the caller exactly what cancellation was requested
// and which paid obligations must be ended through Tensorhub before retrying.
type DownResult struct {
	ShuttingDown          bool                `json:"shutting_down"`
	CancellationRequested []LifecycleIdentity `json:"cancellation_requested"`
	Active                []LifecycleIdentity `json:"active"`
	Rentals               []LifecycleIdentity `json:"rentals"`
	// Refused names what `--all` could not close cleanly. It is reported rather than
	// returned as an error: a teardown that stops at the first problem is the failure this
	// field exists to make visible.
	Refused []string `json:"refused,omitempty"`
}

func (s *Server) downDaemon(w http.ResponseWriter, r *http.Request) {
	s.shutdownAdmission.Lock()
	defer s.shutdownAdmission.Unlock()
	if s.shuttingDown {
		s.ok(w, r, http.StatusAccepted, DownResult{ShuttingDown: true})
		return
	}
	var body struct {
		All   bool `json:"all"`
		Force bool `json:"force"`
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
			`this route takes {"all":true|false,"force":true|false}; flags are optional and mutually exclusive`, "send one closed down request")
		return
	}

	if body.All && body.Force {
		s.refuse(w, r, http.StatusBadRequest, "invalid_request", "all and force are mutually exclusive", "choose disconnect or destructive teardown")
		return
	}
	if !body.All {
		if s.shutdown == nil {
			s.refuse(w, r, http.StatusConflict, "conflict", "this server was built with no shutdown hook", "")
			return
		}
		held, problem := s.orchestrator.PrepareClientShutdown(body.Force)
		if problem != nil {
			s.refuseTyped(w, r, problem)
			return
		}
		if len(held) > 0 {
			s.refuseTyped(w, r, exit.Named(exit.Conflict, "active_work", "daemon shutdown refused: work requires this daemon online: %s", strings.Join(held, ", ")).WithRemedy("wait for the named work, or use `cozy down --force` to disconnect without canceling work or ending rentals"))
			return
		}
		s.shuttingDown = true
		s.ok(w, r, http.StatusAccepted, DownResult{ShuttingDown: true})
		go s.shutdown()
		return
	}
	active, rentals, problem := s.downBlockers()
	if problem != nil {
		s.refuseTyped(w, r, problem)
		return
	}

	requested := []LifecycleIdentity{}
	if body.All {
		// `--all` IS A GUARANTEED TEARDOWN, NOT A BEST-EFFORT ONE (owner ruling
		// 2026-09-04): "you should be able to cozy down --all or force it to shut down
		// (cancel all requests, shutdown all rentals and close)". It is the verb a person
		// reaches for BECAUSE reconciliation is not working, so it may not refuse for the
		// same reason plain `down` did — which is exactly what happened live: one request
		// whose pod had been destroyed made both verbs fail with "the stream that holds
		// req-b2df33d1…#1 is gone", and the daemon could not be stopped by any documented
		// means. It had to be killed.
		//
		// So: one failure never aborts the rest. Every request is tried, what could not be
		// closed cleanly is REPORTED rather than swallowed, and the pass still ends in a
		// shutdown decision.
		var refused []string
		for _, identity := range active {
			row, read := s.store.RequestRow(identity.ID)
			if read != nil || row == nil {
				if read != nil {
					refused = append(refused, identity.ID+": "+read.Message)
				}
				continue
			}
			changed, cancelProblem := s.cancelForDown(r.Context(), *row)
			if cancelProblem != nil {
				refused = append(refused, identity.ID+": "+cancelProblem.Message)
				continue
			}
			if changed {
				requested = append(requested, identity)
			}
		}
		// Re-read rather than trusting the pre-pass sample: what a client is told is still
		// holding the daemon has to be what IS.
		remaining, remainingRentals, problem := s.downBlockers()
		if problem != nil {
			remaining, remainingRentals = active, rentals
		}
		// THE FIXPOINT, and it always terminates. If this pass CHANGED something, there is
		// more to do and the caller gets another turn — rental destruction belongs to
		// Tensorhub/provider ownership, so the CLI ends the named pods and calls back, and
		// requests cancelled here settle through their own durable terminals.
		if len(requested) > 0 || s.newDownRentals(remainingRentals) {
			s.ok(w, r, http.StatusAccepted, DownResult{
				CancellationRequested: requested, Active: remaining,
				Rentals: remainingRentals, Refused: refused,
			})
			return
		}
		// No new cancellation remains, and accepted cleanup has finished or used
		// its shutdown grace. Report any still-unsettled rows; they remain durable
		// for the next boot. An unreachable peer cannot prevent explicit teardown.
		if s.shutdown != nil && (len(remaining) > 0 || len(remainingRentals) > 0) {
			s.shuttingDown = true
			s.ok(w, r, http.StatusAccepted, DownResult{
				ShuttingDown: true, Active: remaining,
				Rentals: remainingRentals, Refused: refused,
			})
			go s.shutdown()
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

// downBlockers is the destructive --all census. Explicit client disconnect uses
// PrepareClientShutdown instead, and never sends cancellation or rental release.
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

// newDownRentals records the paid pods this pass is handing back and answers whether the
// caller has not already been given this exact set. A set it has seen before, with nothing
// else changing, means the caller tried and could not end them — so the teardown stops
// asking and finishes instead of looping on an answer that will not move.
func (s *Server) newDownRentals(rentals []LifecycleIdentity) bool {
	ids := make([]string, 0, len(rentals))
	for _, identity := range rentals {
		ids = append(ids, identity.ID)
	}
	sort.Strings(ids)
	same := len(ids) == len(s.reportedDownRentals)
	if same {
		for i, id := range ids {
			if s.reportedDownRentals[i] != id {
				same = false
				break
			}
		}
	}
	s.reportedDownRentals = ids
	return !same && len(ids) > 0
}

// cancelForDown asks each request's owner to stop it: the machine executing it, or the
// daemon for its own model transfer.
func (s *Server) cancelForDown(ctx context.Context, row records.Request) (bool, *exit.Error) {
	if owned, problem := s.controlMachineExecution(ctx, row, "cancel", "cozy down --all"); owned {
		return problem == nil && row.State != "canceling", problem
	}
	return true, s.orchestrator.CancelQueued(row.ID, "cozy down --all")
}
