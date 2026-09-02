package orchestrator

import (
	"fmt"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// WHICH LANE AN OFFER GOES TO (residency-aware-routing.md §3.1, cl-092). Routing is one
// ordering over facts the worker reports, never a model of its device:
//
//	score(l) = held(l) + cost(l)
//	pick     = argmin score over candidates with room(l) > 0;
//	           ties → the local worker, then lowest (instance_id, lane_id)
//
// held(l) is the attempt-equivalents already ahead on the lane: every attempt the worker
// holds on it (QUEUED, RUNNING, outcome pending ack) plus this owner's own offers the worker
// has not answered yet. cost(l) is the fill the attempt pays on arrival; until residency is
// on the wire (proto-026 `resident_placement_ids`) every lane is priced one fill — unknown
// is one fill. The constants are attempt-equivalents and live here, never in config or env
// (#1312). Nothing here is a timer: `age_ms` rides the decision log for the audit and ranks
// nothing.

const fillCost = 1

// candidate is one (worker, lane) with room for this request, scored.
type candidate struct {
	worker *worker
	laneID string // "" for a worker reporting no lanes: the worker-level window
	held   int
	cost   int
	score  int
	ageMS  int64
}

func (k candidate) local() bool { return k.worker.spec.Connection == nil }

func (k candidate) String() string {
	return fmt.Sprintf("%s/%s held=%d cost=%d score=%d age_ms=%d", k.worker.instanceID,
		orNone(k.laneID), k.held, k.cost, k.score, k.ageMS)
}

func (k candidate) event() map[string]any {
	return map[string]any{
		"worker": k.worker.instanceID, "lane": k.laneID, "held": k.held,
		"cost": k.cost, "score": k.score, "age_ms": k.ageMS,
	}
}

// laneKey names one (worker, lane) two requests compete for; lane "" is the worker-level
// window of a worker reporting no lanes, and matches every lane on that worker.
type laneKey struct{ worker, lane string }

func (k laneKey) String() string { return k.worker + "/" + orNone(k.lane) }

func (k laneKey) covers(o laneKey) bool {
	return k.worker == o.worker && (k.lane == "" || o.lane == "" || k.lane == o.lane)
}

// parking is what the queue knows about a request it skipped (cl-099): the lanes it is
// eligible for — the lanes it competes on — how many later requests were dispatched onto
// one of them past it, and the budget for that: its position in the queue when it first
// parked (gpu-hot D6 mirrored — an attempt admitted behind k others is overtaken at most
// k times; a request parked at the head is overtaken by nobody). Once the budget is spent
// the request CLAIMS its lanes: `route` skips them for everything behind it, so
// head-of-line blocking exists only among requests that compete for one lane and is
// bounded there by twice the FIFO wait. `logged` is the last reason emitted, so a drain
// that finds nothing changed says nothing.
type parking struct {
	lanes     []laneKey
	overtaken int
	budget    int
	logged    string
}

func (p *parking) claims() bool { return p.overtaken >= p.budget }

func (p *parking) competes(k laneKey) bool {
	for _, l := range p.lanes {
		if l.covers(k) {
			return true
		}
	}
	return false
}

// routing is one decision: every candidate with room, best first; the workers that
// would have been candidates but had no room, with why; every lane the request is
// eligible for (with or without room), and the lanes an earlier parked request claims.
type routing struct {
	candidates []candidate
	parked     []string
	lanes      []laneKey
	claimed    []string
}

func (r routing) pick() *candidate {
	if len(r.candidates) == 0 {
		return nil
	}
	return &r.candidates[0]
}

func (r routing) event() map[string]any {
	rows := make([]any, 0, len(r.candidates))
	for _, k := range r.candidates {
		rows = append(rows, k.event())
	}
	out := map[string]any{"candidates": rows}
	if len(r.claimed) > 0 {
		out["claimed"] = r.claimed
	}
	if pick := r.pick(); pick != nil {
		out["pick"] = map[string]any{"worker": pick.worker.instanceID, "lane": pick.laneID}
	}
	return out
}

func (r routing) String() string {
	parts := make([]string, 0, len(r.candidates))
	for _, k := range r.candidates {
		parts = append(parts, k.String())
	}
	return strings.Join(parts, "; ")
}

// route scores every claimed worker whose placement is DISPATCHABLE for the request and
// whose lane has room. Eligibility is the request's pin: a request pinned to a rental sees
// that rental's worker only, and an unpinned request sees local workers only (cl-016) —
// someone is billed for a rented card and nobody asked for it here. Callers hold c.mu.
func (c *Orchestrator) route(req records.Request) routing {
	planID := req.PlanID
	slot := pinnedPackage(req.Package, req.Worker)
	now := time.Now()
	claims := c.claimsAhead(req.ID)
	var out routing
	for _, w := range c.workers {
		if !c.eligible(w, req, slot, planID) {
			continue
		}
		laneID, room, held, why := w.roomFor(w.placementFor(slot, planID), planID)
		lane := laneKey{w.instanceID, laneID}
		out.lanes = append(out.lanes, lane)
		if room <= 0 {
			out.parked = append(out.parked, w.instanceID+": "+why)
			continue
		}
		if by, claimed := claims[lane]; claimed && room < 2 {
			out.claimed = append(out.claimed, lane.String()+" by "+by)
			continue
		}
		k := candidate{worker: w, laneID: laneID, held: held, cost: fillCost}
		k.score = k.held + k.cost
		if !w.lastReport.IsZero() {
			k.ageMS = now.Sub(w.lastReport).Milliseconds()
		}
		out.candidates = append(out.candidates, k)
	}
	sort.Slice(out.candidates, func(i, j int) bool {
		a, b := out.candidates[i], out.candidates[j]
		if a.score != b.score {
			return a.score < b.score
		}
		if a.local() != b.local() {
			return a.local()
		}
		if a.worker.instanceID != b.worker.instanceID {
			return a.worker.instanceID < b.worker.instanceID
		}
		return a.laneID < b.laneID
	})
	sort.Strings(out.parked)
	sort.Strings(out.claimed)
	return out
}

// claimsAhead is every lane claimed by a parked request queued ahead of this one (a
// request not in the queue is behind everything in it), keyed by the lane it would
// take. Callers hold c.mu.
func (c *Orchestrator) claimsAhead(requestID string) map[laneKey]string {
	claims := map[laneKey]string{}
	for _, id := range c.pending {
		if id == requestID {
			break
		}
		p := c.parked[id]
		if p == nil || !p.claims() {
			continue
		}
		for _, w := range c.workers {
			for _, lane := range w.laneKeys() {
				if p.competes(lane) {
					claims[lane] = id
				}
			}
		}
	}
	return claims
}

// overtaken records one dispatch onto `lane` past every parked request queued ahead of
// the dispatched one that competes for it. A budget spent here is a claim from the next
// routing on. Callers hold c.mu.
func (c *Orchestrator) overtaken(requestID string, lane laneKey) {
	for _, id := range c.pending {
		if id == requestID {
			return
		}
		p := c.parked[id]
		if p == nil || !p.competes(lane) {
			continue
		}
		p.overtaken++
		if p.overtaken == p.budget {
			c.logf("%s was overtaken %d time(s) on its lane(s) by %s, its budget; it claims %s",
				id, p.overtaken, requestID, laneStrings(p.lanes))
		}
	}
}

func laneStrings(lanes []laneKey) string {
	parts := make([]string, 0, len(lanes))
	for _, l := range lanes {
		parts = append(parts, l.String())
	}
	return strings.Join(parts, ",")
}

// laneKeys is every lane a dispatch to this worker can draw from: its reported lanes,
// or the worker-level window when it reports none.
func (w *worker) laneKeys() []laneKey {
	if !w.lanes.present() {
		return []laneKey{{w.instanceID, ""}}
	}
	out := make([]laneKey, 0, len(w.lanes.lanes))
	for id := range w.lanes.lanes {
		out = append(out, laneKey{w.instanceID, id})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].lane < out[j].lane })
	return out
}

// eligible is the placement half of the match: a live claimed worker in the request's
// slot whose placement advertises the plan as DISPATCHABLE. THE SLOT IS PART OF THE MATCH,
// not only the binding: matching on the plan id alone sent a request pinned to rental B
// to rental A's worker — same package, same plan digest — which made the pin advisory and
// let a request run on a pod whose credential it never presented.
func (c *Orchestrator) eligible(w *worker, req records.Request, slot, planID string) bool {
	if w.exited || w.stopping || c.sessions[w.bootID] == nil {
		return false
	}
	if req.RentalRequired && w.spec.Connection == nil {
		return false
	}
	if req.Worker != "" && !req.IsJob() && w.spec.Connection != nil {
		placement, ok := w.remotePlacements[remotePlanKey(slot, planID)]
		return w.instanceID == rentalInstanceID(req.Worker) && ok &&
			placement.Package == slot && placement.PackageRevisionDigest == remoteRevision(req) &&
			w.remoteDispatchable(placement, planID)
	}
	if w.spec.Placement.Package != slot {
		return false
	}
	if req.InstallID != "" && w.spec.Placement.InstallID != req.InstallID {
		return false
	}
	if w.spec.IsJob() {
		// A JOB worker hosts no placement: its dispatchability IS its job capacity.
		return w.dispatchable[planID]
	}
	return w.dispatchableFor(planID)
}

// roomFor is the capacity half, per worker kind (#486c generalized by proto-024): the
// worker's fence is OPEN, a worker-level seat is free, and — when the worker reports
// lanes — the placement's own lane has a free seat. It answers the lane the reservation
// draws from ("" for the worker-level window), the room on it net of this owner's
// unanswered reservations, the attempts already held ahead of a new one, and a reason
// when the room is zero.
func (w *worker) roomFor(placementID, planID string) (laneID string, room, held int, why string) {
	if w.spec.IsJob() {
		if w.jobsAvail <= 0 {
			return "", 0, 0, "no job capacity"
		}
		return "", w.jobsAvail, w.held + w.reservedJobs, ""
	}
	if w.admission != pb.AdmissionState_ADMISSION_STATE_OPEN {
		return "", 0, 0, "admission " +
			trimEnum(pb.AdmissionState_name[int32(w.admission)], "ADMISSION_STATE_")
	}
	var l *lane
	if w.lanes.present() {
		if l = w.lanes.of(placementID); l == nil {
			return "", 0, 0, fmt.Sprintf("placement %s is on none of the %d reported lane(s)",
				placementID, len(w.lanes.lanes))
		}
		laneID = l.id
		if l.outsideEnvelope {
			return laneID, 0, 0, fmt.Sprintf("lane %s is outside the granted envelope", l.id)
		}
		if l.seats.slots <= 0 {
			return laneID, 0, 0, fmt.Sprintf("lane %s has no free seat", l.id)
		}
	}
	if w.seats.slots <= 0 {
		return laneID, 0, 0, "no free attempt slot"
	}
	if l != nil {
		return laneID, l.seats.slots, l.held + l.seats.reserved, ""
	}
	return "", w.seats.slots, w.held + w.seats.reserved, ""
}

// noCapacity spells a routing with no pick as the request's queued reason.
func (r routing) noCapacity(slot, planID string) *exit.Error {
	if len(r.claimed) > 0 {
		return exit.Unavailablef("every lane with room for %s in %s is claimed by a request "+
			"queued ahead of it (%s)", planID, slot, strings.Join(r.claimed, "; "))
	}
	if len(r.parked) > 0 {
		return exit.Unavailablef(
			"no claimed worker in %s has a DISPATCHABLE placement for %s with a free attempt "+
				"slot (%s)", slot, planID, strings.Join(r.parked, "; "))
	}
	return exit.Unavailablef(
		"no claimed worker in %s has a DISPATCHABLE placement for %s with a free attempt slot",
		slot, planID)
}

// logRouting is the decision log (D8): inputs, score per candidate and the pick, as a
// durable request event and a log line, on every dispatch.
func (c *Orchestrator) logRouting(requestID string, attempt uint64, r routing) {
	pick := r.pick()
	if pick == nil {
		return
	}
	c.logf("routed %s#%d to %s/%s over %d candidate(s): %s", requestID, attempt,
		pick.worker.instanceID, orNone(pick.laneID), len(r.candidates), r)
	c.emit(requestID, "request.routed", attempt, r.event())
}
