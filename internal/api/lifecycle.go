package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
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
	// Refused names what `--all` could not close cleanly. It is reported rather than
	// returned as an error: a teardown that stops at the first problem is the failure this
	// field exists to make visible.
	Refused []string `json:"refused,omitempty"`
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

	active, rentals, problem := s.downBlockers(body.All)
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
			changed, cancelProblem := s.cancelForDown(*row)
			if cancelProblem != nil {
				refused = append(refused, identity.ID+": "+cancelProblem.Message)
				continue
			}
			if changed {
				requested = append(requested, identity)
			}
		}
		if len(requested) == 0 {
			// No new request does not mean its asynchronous native cleanup has
			// finished. Give accepted cleanup the existing shutdown grace, then
			// sample its durable result. Unreachable peers still cannot veto --all.
			drain, cancel := context.WithTimeout(r.Context(), orchestrator.StopGrace)
			s.orchestrator.WaitRetainedCancellations(drain)
			cancel()
		}
		// Re-read rather than trusting the pre-pass sample: what a client is told is still
		// holding the daemon has to be what IS.
		remaining, remainingRentals, problem := s.downBlockers(true)
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

// downBlockers is the subset of the daemon's obligations `down` refuses on: the work it
// can cancel and the paid pods it must see ended. An attempt awaiting its ack and an
// export mid-copy are on their way to settlement and are drained by the close itself.
func (s *Server) downBlockers(all bool) ([]LifecycleIdentity, []LifecycleIdentity, *exit.Error) {
	read := s.store.ClientShutdownObligations
	if all {
		read = s.store.Obligations
	}
	obligations, problem := read()
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

// cancelForDown reuses the request authority's existing queued/live cancellation
// boundaries. A terminal awaiting acknowledgement is already on its way to settlement;
// it remains in Active and makes the caller retry rather than receiving a second verdict.
func (s *Server) cancelForDown(row records.Request) (bool, *exit.Error) {
	if row.RetainWork && (row.IsJob() || row.ParentRequestID != "") && (!records.Settled(row.State) || (row.State == "succeeded" && row.RetainsLocalOutputs())) && row.State != "finalizing" {
		return row.State != "canceling" && row.State != "releasing", s.orchestrator.CancelRetainedRequest(row.ID, "cozy down --all")
	}
	attempts, problem := s.store.Attempts(row.ID)
	if problem != nil {
		return false, problem
	}
	if len(attempts) == 0 {
		return true, s.orchestrator.CancelQueued(row.ID, "cozy down --all")
	}
	last := attempts[len(attempts)-1]
	switch last.State {
	case "closed", "dispatch_aborted":
		return true, s.orchestrator.CancelQueued(row.ID, "cozy down --all")
	case "terminal":
		return false, nil
	default:
		problem := s.orchestrator.CancelClient(
			row.ID, uint64(last.Attempt), orchestrator.ClientCancelGraceMS, "cozy down --all")
		if problem == nil || problem.Code != exit.Unavailable {
			return true, problem
		}
		// THE STREAM IS GONE, so there is nobody to ask and there never will be. Cancelling
		// an attempt normally means telling its worker to stop; when this process no longer
		// holds a session or a worker for it, the execution context is already lost and the
		// only honest thing left is to record that. Teardown must not depend on anything
		// remote being reachable — a pod that is gone cannot answer, and waiting for it is
		// exactly the failure this branch exists to end.
		return true, s.orchestrator.CancelLostAttempt(row.ID, last.Attempt,
			"the daemon was torn down with `cozy down --all` while this attempt's execution "+
				"context was already gone: "+problem.Message)
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
