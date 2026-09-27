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
//	           ties → p ∈ resident(l), then the local worker, then lowest (instance_id, lane_id)
//
// held(l) is the attempt-equivalents already ahead on the lane: every attempt the worker
// holds on it (QUEUED, RUNNING, outcome pending ack) plus this owner's own offers the worker
// has not answered yet. cost(l) is the load the attempt pays on arrival, priced per BINDING:
// 0 when the live executor already holds the entrypoint's construction
// (`loaded_binding_digests`), else 1 on an idle lane and 2 when the lane holds another
// tenant. Ties break on the known load time, the request's model bytes at 20 GB/s. A
// worker older than minor 61 reports no loaded set; its DISPATCHABLE placement holds its
// one construction, priced by the lane's `resident_placement_ids`. The constants are
// attempt-equivalents and live here, never in config or env (#1312). Nothing here is a
// timer: `age_ms` rides the decision log for the audit and ranks nothing.
//
// WHICH WORKERS ARE ASKED is the request's permission (D6): local lanes unless
// `--rental-only`; every attached rental's lanes with `--rental`, local winning ties; a
// request PINNED to a rental (`Worker`) sees that rental alone. The pin is routing's own
// output — written by `dispatch` when the argmin is a rental, or by the capacity decision
// when no worker holds the placement (`selectOrStart`) — never a submission's guess.

const (
	costResident = 0
	costFill     = 1
	costDisplace = 2
	// loadBytesPerMS is page cache to device: 20 GB/s.
	loadBytesPerMS = 20_000_000
)

// candidate is one (worker, lane) with room for this request, scored.
type candidate struct {
	worker      *worker
	laneID      string // "" for a worker reporting no lanes: the worker-level window
	placementID string
	held        int
	cost        int
	score       int
	loadMS      int64
	ageMS       int64
	// resident is the lane's reported resident set; manifestsMissing is how many of the
	// request's model manifests the worker's store does not hold (a DISPATCHABLE
	// placement holds all of them; the number is the audit's, §3.2).
	resident         []string
	manifestsMissing int
}

func (k candidate) local() bool { return k.worker.spec.Connection == nil }

func (k candidate) String() string {
	return fmt.Sprintf("%s/%s held=%d cost=%d score=%d load_ms=%d resident=%v missing=%d age_ms=%d",
		k.worker.instanceID, orNone(k.laneID), k.held, k.cost, k.score, k.loadMS, k.resident,
		k.manifestsMissing, k.ageMS)
}

func (k candidate) event() map[string]any {
	out := map[string]any{
		"worker": k.worker.instanceID, "lane": k.laneID, "placement": k.placementID,
		"held": k.held, "cost": k.cost, "score": k.score, "load_ms": k.loadMS, "age_ms": k.ageMS,
		"resident": k.resident, "manifests_missing": k.manifestsMissing,
	}
	if !k.local() {
		out["rental"] = k.worker.spec.Connection.RentalID
	}
	return out
}

// costOn prices the load an attempt for `planID` on `placementID` pays on `laneID`
// (§3.1): nothing when its construction is loaded, else one load of its known bytes.
func (w *worker) costOn(laneID, placementID, planID string, models []records.ModelRef) (int, int64, []string) {
	l := w.lanes.get(laneID)
	var resident []string
	if l != nil {
		resident = l.residentIDs()
	}
	observed, remote := w.observedRemote[placementID]
	if remote && observed.loadedKnown {
		if observed.loaded(planID) {
			return costResident, 0, resident
		}
	} else if l != nil && l.resident[placementID] {
		return costResident, 0, resident
	}
	var bytes int64
	for _, model := range models {
		bytes += model.Bytes
	}
	cost := costFill
	if l != nil {
		for id := range l.resident {
			if id != placementID {
				cost = costDisplace
			}
		}
	}
	return cost, bytes / loadBytesPerMS, resident
}

// missingManifests counts the request's model manifests the worker's verified store does
// not report holding.
func (w *worker) missingManifests(models []records.ModelRef) int {
	missing := 0
	for _, model := range models {
		if model.Manifest != "" && !w.heldManifests[model.Manifest] {
			missing++
		}
	}
	return missing
}

// laneKey names one (worker, lane); lane "" is the worker-level window of a worker
// reporting no lanes.
type laneKey struct{ worker, lane string }

func (k laneKey) String() string { return k.worker + "/" + orNone(k.lane) }

// parking is what the queue knows about a request it skipped: the lanes it is eligible for,
// how many later requests on its rental were dispatched past it (schedule's starvation
// bound), and the last wait it logged, so a drain that finds nothing changed says nothing.
type parking struct {
	lanes     []laneKey
	overtaken int
	logged    string
	wait      waitFacts // the queue's last typed wait, never parsed from logged text
}

// routing is one decision: every candidate with room, best first; the workers that
// would have been candidates but had no room, with why; and every lane the request is
// eligible for (with or without room).
type routing struct {
	candidates []candidate
	parked     []string
	blocked    []laneKey
	lanes      []laneKey
	// pinned is the rental this decision pinned the request to (dispatch), or "".
	pinned string
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
	if pick := r.pick(); pick != nil {
		row := map[string]any{"worker": pick.worker.instanceID, "lane": pick.laneID}
		if !pick.local() {
			row["rental"] = pick.worker.spec.Connection.RentalID
		}
		if r.pinned != "" {
			row["pinned"] = true
		}
		out["pick"] = row
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
// whose lane has room. Callers hold c.mu.
func (c *Orchestrator) route(req records.Request) routing {
	planID := req.PlanID
	now := time.Now()
	var out routing
	for _, w := range c.workers {
		if w.preparingRequest != "" && (!w.preparingReady || w.preparingRequest != req.ID) {
			continue
		}
		if !c.eligible(w, req, planID) {
			continue
		}
		placementID := w.placementID
		laneID, room, held, why := w.roomFor(placementID)
		lane := laneKey{w.instanceID, laneID}
		out.lanes = append(out.lanes, lane)
		if room <= 0 {
			out.parked = append(out.parked, w.instanceID+": "+why)
			out.blocked = append(out.blocked, lane)
			continue
		}
		k := candidate{worker: w, laneID: laneID, placementID: placementID, held: held,
			manifestsMissing: w.missingManifests(req.Models)}
		k.cost, k.loadMS, k.resident = w.costOn(laneID, placementID, planID, req.Models)
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
		if (a.cost == costResident) != (b.cost == costResident) {
			return a.cost == costResident
		}
		if a.loadMS != b.loadMS {
			return a.loadMS < b.loadMS
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
	sort.Slice(out.blocked, func(i, j int) bool { return out.blocked[i].String() < out.blocked[j].String() })
	return out
}

func laneStrings(lanes []laneKey) string {
	parts := make([]string, 0, len(lanes))
	for _, l := range lanes {
		parts = append(parts, l.String())
	}
	return strings.Join(parts, ",")
}

// eligible is the placement half of the match: a live claimed worker whose placement
// advertises the plan as DISPATCHABLE for this request. A rental's job placement lives
// under the package name pinned to that rental, so the same plan on two rentals is two
// placements, each presented its own rental's credential.
func (c *Orchestrator) eligible(w *worker, req records.Request, planID string) bool {
	if w.exited || w.stopping || !w.supportsCurrentProtocol() || c.sessions[w.bootID] == nil ||
		req.RentNew && req.Worker == "" || !c.rentalParentAllows(w, req) {
		return false
	}
	if req.RequestedRental != "" && (w.spec.Connection == nil || w.spec.Connection.RentalID != req.RequestedRental) {
		return false
	}
	if w.spec.Connection == nil {
		if req.RentalRequired || req.Worker != "" || w.spec.Placement.Package != req.Package ||
			req.InstallID != "" && w.spec.Placement.InstallID != req.InstallID {
			return false
		}
		// A JOB worker hosts no placement: its dispatchability IS its job capacity. For
		// serving the model selection is part of the match (cl-114): the plan id hashes the
		// entrypoint's interface, not its weights.
		return w.dispatchableFor(planID) && (w.spec.IsJob() || selectionServes(req.Models, w.spec.Placement.Models))
	}
	if req.Worker != "" && w.instanceID != rentalInstanceID(req.Worker) || req.Worker == "" && !req.Rental {
		return false
	}
	// Runtime executes every other rented call; a rental takes only the job it was
	// prepared for from this owner.
	return req.IsJob() && w.spec.IsJob() && w.spec.Placement.Package == pinnedPackage(req.Package, w.spec.Connection.RentalID) &&
		stagedFor(w, req) && w.dispatchableFor(planID)
}

// roomFor is the capacity half, per worker kind (#486c generalized by proto-024): the
// worker's fence is OPEN, a worker-level seat is free, and — when the worker reports
// lanes — the placement's own lane still admits an offer to its queue (a running attempt
// and a staged one behind it, proto-061 G; never assumed, always the reported count).
// It answers the lane the reservation
// draws from ("" for the worker-level window), the room on it net of this owner's
// unanswered reservations, the attempts already held ahead of a new one, and a reason
// when the room is zero.
func (w *worker) roomFor(placementID string) (laneID string, room, held int, why string) {
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

// noCapacity spells a routing with no pick as the request's queued reason, naming the
// workers the request was allowed to ask.
func (r routing) noCapacity(req records.Request) *exit.Error {
	slot, planID := pinnedPackage(req.Package, req.Worker), req.PlanID
	asked := "no claimed worker"
	switch {
	case req.Worker != "":
		asked = "no worker of the pinned rental"
	case req.RentalRequired:
		asked = "no attached rental"
	case req.Rental:
		asked = "no local worker or attached rental"
	}
	if len(r.parked) > 0 {
		return exit.Unavailablef(
			"%s in %s has a DISPATCHABLE placement for %s with a free attempt "+
				"slot (%s)", asked, slot, planID, strings.Join(r.parked, "; "))
	}
	return exit.Unavailablef(
		"%s in %s has a DISPATCHABLE placement for %s with a free attempt slot",
		asked, slot, planID)
}

// logRouting is the decision log (D8): inputs, score per candidate and the pick, as a
// durable request event and a log line, on every dispatch.
func (c *Orchestrator) logRouting(requestID string, attempt uint64, r routing) {
	pick := r.pick()
	if pick == nil {
		return
	}
	pinned := ""
	if r.pinned != "" {
		pinned = " (pinned to rental " + r.pinned + ")"
	}
	c.logf("routed %s#%d to %s/%s%s over %d candidate(s): %s", requestID, attempt,
		pick.worker.instanceID, orNone(pick.laneID), pinned, len(r.candidates), r)
	c.emit(requestID, "request.routed", attempt, r.event())
}
