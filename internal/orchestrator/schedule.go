package orchestrator

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// WARM FIRST (proto-061 F). A rental's queued serving work is ordered by what the worker
// holds, not by arrival alone:
//
//	(a) loaded       the placement is DISPATCHABLE and its executor holds the construction
//	(b) dispatchable the placement is DISPATCHABLE; the construction loads on arrival
//	(c) not held     no placement the worker reports binds the request's selection
//
// The oldest (a) runs, else the oldest (b). The oldest (c) issues one additive desire —
// the only preparation a rental's serving queue produces — and never holds (a) or (b)
// back. Fairness is a bound, not a FIFO: a request passed over starvationBound times
// claims the rental, and nothing younger dispatches ahead of it again. Run 868 (loran)
// is the case this replaces: a fifth-queued turbo request restaged the pod, the head
// could not use the placement that was already warm, and the GPU sat idle for 202 s.
const starvationBound = 2

type scheduleClass int

const (
	classLoaded scheduleClass = iota
	classDispatchable
	// classWarming is held by a placement that is not DISPATCHABLE for it yet.
	classWarming
	// classModeHeld waits for the rental's job mode to end before it may desire.
	classModeHeld
	classNeedsDesire
)

type scheduled struct {
	req      records.Request
	position int // in the global queue, for the parked record
}

// scheduledVenue names the rental whose scheduler owns this queued request: published
// serving work pinned to, or requested on, one rental. Jobs, captured packages, children
// and model transfers keep their own preparation paths.
func scheduledVenue(req records.Request) string {
	if req.IsJob() || req.InstallID != "" || req.ParentRequestID != "" || req.ModelTransfer != nil ||
		req.RetainWork || req.LocalPackageDigest != "" || req.Release == "" {
		return ""
	}
	if req.Worker != "" {
		return req.Worker
	}
	return req.RequestedRental
}

func logicalOf(req records.Request) LogicalPackage {
	return LogicalPackage{Package: req.Package, Release: req.Release, Function: req.Entrypoint,
		PlanID: req.PlanID, Models: append([]ModelRef(nil), req.Models...),
		NeedsAccelerator: req.NeedsAccelerator}
}

// classifyLocked reads one request against what the rental's worker reports holding: the
// class, the binding the worker authored, and the placement it holds. Callers hold c.mu.
func (c *Orchestrator) classifyLocked(w *worker, req records.Request) (scheduleClass, string, string) {
	if w != nil && !w.exited && !w.stopping && w.spec.IsJob() &&
		(!c.idleRentalWorkerLocked(w) || c.modeClaimedLocked(w, false)) {
		return classModeHeld, "", ""
	}
	if w == nil || w.exited || w.stopping || w.spec.IsJob() || !w.supportsCurrentProtocol() ||
		c.sessions[w.bootID] == nil {
		return classNeedsDesire, "", ""
	}
	placement, planID, observed, held := observedServing(w, logicalOf(req))
	switch {
	case !held:
		return classNeedsDesire, "", ""
	case observed.serving != pb.ServingState_SERVING_STATE_DISPATCHABLE || !observed.dispatchablePlanIDs[planID]:
		return classWarming, planID, placement.PlacementIDValue
	case observed.loaded(planID):
		return classLoaded, planID, placement.PlacementIDValue
	}
	return classDispatchable, planID, placement.PlacementIDValue
}

// schedule runs one rental's queue, oldest first within each class. Called by drain,
// which serializes it.
func (c *Orchestrator) schedule(venue string, queue []scheduled) {
	c.mu.Lock()
	w := c.workers[rentalInstanceID(venue)]
	classes := make([]scheduleClass, len(queue))
	plans := make([]string, len(queue))
	placements := make([]string, len(queue))
	claimant := -1
	for i, entry := range queue {
		classes[i], plans[i], placements[i] = c.classifyLocked(w, entry.req)
		if p := c.parked[entry.req.ID]; claimant < 0 && p != nil && p.overtaken >= starvationBound {
			claimant = i
		}
	}
	c.mu.Unlock()

	tried := make([]bool, len(queue))
	sent := make([]bool, len(queue))
	// seatless marks a pick whose lane had no room; full is those lanes. A seat that frees
	// on one mid-pass is the next pass's, for its oldest pick.
	seatless := make([]bool, len(queue))
	full := map[string]bool{}
	for {
		pick := -1
		for _, class := range []scheduleClass{classLoaded, classDispatchable} {
			for i := range queue {
				if !tried[i] && classes[i] == class && (claimant < 0 || i <= claimant) {
					pick = i
					break
				}
			}
			if pick >= 0 {
				break
			}
		}
		if pick < 0 {
			break
		}
		laneID, room, open := c.rentalRoom(venue, placements[pick])
		if !open {
			// The worker admits nothing: every request here draws on its seats.
			break
		}
		tried[pick] = true
		if room <= 0 || full[laneID] {
			// The placement's lane admits no more; a pick on another lane may still run.
			seatless[pick], full[laneID] = true, true
			continue
		}
		req := queue[pick].req
		if plans[pick] != req.PlanID {
			// The worker authored this function's binding under the placement it holds.
			if e := c.opt.Store.BindRequestPlan(req.ID, plans[pick]); e != nil {
				c.failQueued(req.ID, e, "")
				continue
			}
			req.PlanID = plans[pick]
		}
		attempt, e := c.dispatch(req)
		if e != nil {
			if e.Code == exit.Unavailable || e.Code == exit.Conflict {
				c.park(req, queue[pick].position, waitFacts{}, e.Message)
				continue
			}
			c.failQueued(req.ID, e, "")
			continue
		}
		sent[pick] = true
		c.forget(req.ID)
		c.logf("%s left the dispatch queue as attempt %d (%s on rental %s)", req.ID, attempt,
			classWord(classes[pick]), venue)
		var starved []string
		c.mu.Lock()
		for i := 0; i < pick; i++ {
			if sent[i] || seatless[i] {
				continue
			}
			id := queue[i].req.ID
			if !c.queued(id) {
				continue
			}
			p := c.parked[id]
			if p == nil {
				p = &parking{}
				c.parked[id] = p
			}
			p.overtaken++
			if p.overtaken == starvationBound {
				starved = append(starved, id)
			}
			if p.overtaken >= starvationBound && (claimant < 0 || i < claimant) {
				claimant = i
			}
		}
		c.mu.Unlock()
		for _, id := range starved {
			c.logf("%s was passed over %d time(s) on rental %s; nothing younger runs ahead of it",
				id, starvationBound, venue)
		}
	}

	// ONE ADDITIVE DESIRE: the oldest request no held placement serves.
	c.mu.Lock()
	desiring := c.desiring[venue]
	var launch *records.Request
	if desiring == "" {
		for i, entry := range queue {
			if classes[i] == classNeedsDesire {
				c.desiring[venue] = entry.req.ID
				desiring = entry.req.ID
				launch = &queue[i].req
				break
			}
		}
	}
	c.mu.Unlock()
	if launch != nil {
		req := *launch
		c.logf("%s: rental %s holds no placement for %s/%s; desiring it", req.ID, venue,
			req.Package, req.Entrypoint)
		go func() {
			if !c.prepare(req) {
				c.releaseDesire(venue, req.ID)
			}
		}()
	}

	for i, entry := range queue {
		if tried[i] && !seatless[i] {
			continue
		}
		req := entry.req
		facts, reason := waitFacts{}, ""
		ready := classes[i] == classLoaded || classes[i] == classDispatchable
		switch {
		case ready && claimant >= 0 && i > claimant:
			facts = waitFacts{cause: WaitQueueAhead, waitingFor: c.waitingRun(queue[claimant].req.ID)}
			reason = "an older request on this rental runs first"
		case ready:
			c.park(req, entry.position, waitFacts{}, "every attempt seat on rental "+venue+" is taken")
			continue
		case classes[i] == classModeHeld:
			reason = "the rental finishes its job work before it serves"
		case classes[i] == classWarming || desiring == req.ID:
			facts = waitFacts{cause: WaitWorkerWarming}
			reason = "its placement is being made ready on this rental"
		case classes[i] == classNeedsDesire && desiring != "":
			facts = waitFacts{cause: WaitQueueAhead, waitingFor: c.waitingRun(desiring)}
			reason = "the rental is preparing another request's selection"
		default:
			continue
		}
		c.mu.Lock()
		facts.on = c.machineWord(rentalInstanceID(venue))
		c.mu.Unlock()
		c.park(req, entry.position, facts, reason)
	}
}

// rentalRoom reads the room an offer for this placement would draw on: the placement's
// lane when the worker reports lanes, else the worker-level window. `open` is false when
// the worker admits nothing at all.
func (c *Orchestrator) rentalRoom(venue, placementID string) (laneID string, room int, open bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	w := c.workers[rentalInstanceID(venue)]
	if w == nil || !w.admissible() {
		return "", 0, false
	}
	laneID, room, _, _ = w.roomFor(placementID)
	return laneID, room, true
}

func (c *Orchestrator) releaseDesire(venue, requestID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.desiring[venue] == requestID && !c.preparing[venue] {
		delete(c.desiring, venue)
	}
}

func classWord(class scheduleClass) string {
	switch class {
	case classLoaded:
		return "loaded"
	case classDispatchable:
		return "dispatchable"
	case classWarming:
		return "warming"
	case classModeHeld:
		return "mode held"
	}
	return "not held"
}

// observedServing finds the placement the worker REPORTS that binds this request's
// function to its selection, read off the set bytes each placement was accepted under —
// never the last set this owner sent, which may not have been accepted yet. It answers
// the placement, the binding the worker authored for the function, and the observation.
// Callers hold c.mu.
func observedServing(w *worker, logical LogicalPackage) (DesiredPlacement, string, remotePlacementObservation, bool) {
	if w == nil || w.refusal != nil {
		return DesiredPlacement{}, "", remotePlacementObservation{}, false
	}
	ids := make([]string, 0, len(w.observedRemote))
	for id := range w.observedRemote {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		a, b := w.observedRemote[ids[i]], w.observedRemote[ids[j]]
		if (a.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE) != (b.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE) {
			return a.serving == pb.ServingState_SERVING_STATE_DISPATCHABLE
		}
		return ids[i] < ids[j]
	})
	for _, id := range ids {
		observed := w.observedRemote[id]
		if observed.materialization == pb.MaterializationState_MATERIALIZATION_STATE_FAILED {
			continue
		}
		digest := observed.placementSetDigest
		if digest == "" {
			digest = spellOf(w.acceptedSetDigest)
		}
		placement, row, found, problem := placementRow(w.setBytesFor(digest), digest,
			logical.Package, logical.Release, id)
		if problem != nil || !found {
			continue
		}
		if planID, ok := entrypointServes(row, logical); ok {
			return placement, planID, observed, true
		}
	}
	return DesiredPlacement{}, "", remotePlacementObservation{}, false
}
