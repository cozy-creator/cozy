package orchestrator

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// DEVICE LANES (proto-024, wire minor 22). The serialized resource inside one worker is a
// LANE — a set of envelope-local device ordinals — and a worker may have several. A lane's
// `available_attempt_slots` is how many more offers it will admit to its queue now (the
// device seat plus the staged one behind it, proto-061 G); this owner reads the count and
// never assumes it. An offer for placement P draws only from P's lane, so the worker-level
// count (provably Σ lanes) over-advertises on a multi-lane worker. This owner therefore
// keeps a seat ledger per lane beside the worker-level one and reserves against both. A worker that
// reports no lanes — a minor-21 runtime, a job worker — is exactly the worker-level ledger,
// which is byte for byte the behaviour before lanes were on the wire.
//
// LocalDeviceEnvelope is what this daemon GRANTS a local worker: the device names on its
// `--devices` line, and the space a reported lane's ordinals index into. Lane ordinals are
// positions in this envelope (`lane-0` over ordinal 0 is device "0"), so a lane is the
// worker's own account of which granted device it serves. One device, until case B (cl-016
// "Multi-GPU local": K independent placements on K lanes) has an owner — a wider envelope
// with no pin refuses typed at the worker rather than idling cards (group-lanes ruling 3).
func LocalDeviceEnvelope() []string { return []string{"0"} }

// RentalDeviceEnvelope is the envelope a RENTED pod's worker holds: its width, spelled as
// the ordinals its lanes index into. This daemon grants a rental nothing — the pod's cards
// are the pod's, and its worker was launched over all of them by whoever provisioned it —
// so this is not a grant but the paid WIDTH read back onto the same space a local grant
// occupies, which is what lets one lane reader and one breach check serve both. A CPU
// rental holds no device and gets no envelope.
func RentalDeviceEnvelope(count int) []string {
	out := make([]string, 0, count)
	for ordinal := 0; ordinal < count; ordinal++ {
		out = append(out, strconv.Itoa(ordinal))
	}
	return out
}

// seatLedger is one admission window as this owner sees it: the peer's last reported free
// seats, minus this owner's reservations from choosing the window until its offer is
// answered. Keeping them separate prevents a report racing an outbound offer from reopening
// a seat the offer is about to take.
type seatLedger struct {
	reported int
	reserved int
	slots    int // reported - reserved, clamped at zero
}

func (s *seatLedger) observe(n int) {
	s.reported = n
	s.slots = max(0, n-s.reserved)
}

func (s *seatLedger) reserve() {
	s.reserved++
	s.observe(s.reported)
}

// release returns a reservation whose offer was never answered by the peer.
func (s *seatLedger) release() {
	s.reserved = max(0, s.reserved-1)
	s.observe(s.reported)
}

// settle answers a reservation: consumed on Accepted or an executed outcome, returned on a
// pre-execution refusal.
func (s *seatLedger) settle(consumed bool) {
	s.reserved = max(0, s.reserved-1)
	if consumed {
		s.reported = max(0, s.reported-1)
	}
	s.observe(s.reported)
}

// lane is one reported DeviceLane with this owner's ledger over its admission seats.
type lane struct {
	id           string
	ordinals     []uint32
	placementIDs []string
	seats        seatLedger
	// held is how many attempts the worker last reported on this lane, in every state
	// from admission (QUEUED) to ack — the queue ahead of a new offer (route.go).
	held int
	// resident is the worker's last word on which placements hold device bytes on this
	// lane (`resident_placement_ids`, proto-026): the fact `cost` prices a fill by. A set,
	// never bytes; one report stale at most, and a wrong belief costs a fill, not
	// correctness (residency-aware-routing.md §3.3).
	resident map[string]bool
	// outsideEnvelope latches a lane naming an ordinal past the granted envelope. It is a
	// worker breach; the lane takes no offer and the report says why, once.
	outsideEnvelope bool
}

// laneReport is one DeviceLane as read off either wire shape: the protobuf message on an
// ObservedWorkerState, or the canonical document row in a WorkerSnapshotBody.
type laneReport struct {
	id           string
	ordinals     []uint32
	slots        int
	placementIDs []string
	resident     []string
}

func laneReportsOf(rows []*pb.DeviceLane) []laneReport {
	out := make([]laneReport, 0, len(rows))
	for _, row := range rows {
		if row == nil {
			continue
		}
		out = append(out, laneReport{
			id: row.LaneId, ordinals: append([]uint32(nil), row.DeviceOrdinals...),
			slots:        int(row.AvailableAttemptSlots),
			placementIDs: append([]string(nil), row.PlacementIds...),
			resident:     append([]string(nil), row.ResidentPlacementIds...),
		})
	}
	return out
}

func laneReportsOfDoc(rows []canonical.Doc) []laneReport {
	out := make([]laneReport, 0, len(rows))
	for _, row := range rows {
		ordinals := row.Ints("device_ordinals")
		report := laneReport{
			id: row.Str("lane_id"), ordinals: make([]uint32, 0, len(ordinals)),
			slots:        int(row.Int("available_attempt_slots")),
			placementIDs: row.Strs("placement_ids"),
			resident:     row.Strs("resident_placement_ids"),
		}
		for _, ordinal := range ordinals {
			report.ordinals = append(report.ordinals, uint32(ordinal))
		}
		out = append(out, report)
	}
	return out
}

// laneTable is a worker's lanes as last reported, keyed by lane id, with the placement →
// lane routing this owner dispatches by. Empty means the worker put no lanes on the wire.
type laneTable struct {
	lanes       map[string]*lane
	byPlacement map[string]string
}

func (t *laneTable) present() bool { return len(t.lanes) > 0 }

// observe replaces the table with the worker's latest report, carrying this owner's open
// reservations across by lane id: a report is never evidence about an offer in flight.
// `envelope` is the device set the worker was granted; an ordinal outside it is a breach.
func (t *laneTable) observe(rows []laneReport, envelope []string) (breaches []string) {
	next := make(map[string]*lane, len(rows))
	byPlacement := map[string]string{}
	for _, row := range rows {
		l := t.lanes[row.id]
		if l == nil {
			l = &lane{id: row.id}
		}
		l.ordinals, l.placementIDs = row.ordinals, row.placementIDs
		l.resident = setOf(row.resident)
		l.seats.observe(row.slots)
		outside := false
		if len(envelope) > 0 {
			for _, ordinal := range row.ordinals {
				if int(ordinal) >= len(envelope) {
					outside = true
				}
			}
		}
		if outside && !l.outsideEnvelope {
			breaches = append(breaches, fmt.Sprintf("lane %s names ordinals %v outside the "+
				"granted envelope [%s]; it takes no offer", row.id, row.ordinals,
				strings.Join(envelope, ",")))
		}
		l.outsideEnvelope = outside
		next[row.id] = l
		for _, placementID := range row.placementIDs {
			byPlacement[placementID] = row.id
		}
	}
	t.lanes, t.byPlacement = next, byPlacement
	return breaches
}

// route records the lane a placement reports itself on (`PlacementStatus.device_lane_id`),
// the per-placement fact, over the lane's own placement list.
func (t *laneTable) route(placementID, laneID string) {
	if placementID == "" || laneID == "" {
		return
	}
	if t.byPlacement == nil {
		t.byPlacement = map[string]string{}
	}
	t.byPlacement[placementID] = laneID
}

// observeHeld counts the worker's held attempts onto the lanes their placements are on,
// after `observe` and `route` have placed every placement. It answers how many were on no
// lane (a job-mode attempt names no placement).
func (t *laneTable) observeHeld(placementIDs []string) (unrouted int) {
	for _, l := range t.lanes {
		l.held = 0
	}
	for _, placementID := range placementIDs {
		if l := t.of(placementID); l != nil {
			l.held++
		} else {
			unrouted++
		}
	}
	return unrouted
}

// residentIDs is the lane's resident set, sorted, for the decision log.
func (l *lane) residentIDs() []string {
	out := make([]string, 0, len(l.resident))
	for id := range l.resident {
		out = append(out, id)
	}
	sort.Strings(out)
	return out
}

func setOf(ids []string) map[string]bool {
	if len(ids) == 0 {
		return nil
	}
	out := make(map[string]bool, len(ids))
	for _, id := range ids {
		out[id] = true
	}
	return out
}

func (t *laneTable) of(placementID string) *lane {
	if id, ok := t.byPlacement[placementID]; ok {
		return t.lanes[id]
	}
	return nil
}

func (t *laneTable) get(laneID string) *lane {
	if laneID == "" {
		return nil
	}
	return t.lanes[laneID]
}

func (t *laneTable) sorted() []*lane {
	out := make([]*lane, 0, len(t.lanes))
	for _, l := range t.lanes {
		out = append(out, l)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].id < out[j].id })
	return out
}

// devicesOf spells a lane's ordinals as the granted device names. A worker this daemon did
// not launch (a rental) was granted no envelope here, so its lanes stay ordinals only.
func devicesOf(l *lane, envelope []string) []string {
	if len(envelope) == 0 {
		return nil
	}
	out := make([]string, 0, len(l.ordinals))
	for _, ordinal := range l.ordinals {
		if int(ordinal) < len(envelope) {
			out = append(out, envelope[ordinal])
		}
	}
	return out
}

// LaneFacts is one lane as this owner reads it: the worker's ordinals and free seats, and
// the granted device names those ordinals index.
type LaneFacts struct {
	LaneID               string   `json:"lane_id"`
	DeviceOrdinals       []uint32 `json:"device_ordinals"`
	Devices              []string `json:"devices"`
	AvailableSlots       int      `json:"available_attempt_slots"`
	HeldAttempts         int      `json:"held_attempts"`
	PlacementIDs         []string `json:"placement_ids"`
	ResidentPlacementIDs []string `json:"resident_placement_ids"`
}

func laneFactsOf(w *worker) []LaneFacts {
	lanes := w.lanes.sorted()
	if len(lanes) == 0 {
		return nil
	}
	out := make([]LaneFacts, 0, len(lanes))
	for _, l := range lanes {
		out = append(out, LaneFacts{
			LaneID: l.id, DeviceOrdinals: append([]uint32(nil), l.ordinals...),
			Devices: devicesOf(l, w.spec.Devices), AvailableSlots: l.seats.slots,
			HeldAttempts: l.held, PlacementIDs: append([]string(nil), l.placementIDs...),
			ResidentPlacementIDs: l.residentIDs(),
		})
	}
	return out
}

// reserveSeat takes the seat roomFor answered: the worker-level window and, when the offer
// draws from a lane, that lane's.
func (w *worker) reserveSeat(laneID string) {
	w.seats.reserve()
	if l := w.lanes.get(laneID); l != nil {
		l.seats.reserve()
	}
}

func (w *worker) releaseSeat(laneID string) {
	w.seats.release()
	if l := w.lanes.get(laneID); l != nil {
		l.seats.release()
	}
}

// settleSeat answers a reservation. A consumed one is now an attempt the worker HOLDS, so
// it moves from this owner's reservations to the held count until the worker's next report
// says so itself — the two never both count it, and the lane never looks emptier than it is.
func (w *worker) settleSeat(laneID string, consumed bool) {
	w.seats.settle(consumed)
	if consumed {
		w.held++
	}
	if l := w.lanes.get(laneID); l != nil {
		l.seats.settle(consumed)
		if consumed {
			l.held++
		}
	}
}

// laneOf names the lane a placement is on, or "" when the worker reports none.
func (w *worker) laneOf(placementID string) string {
	if l := w.lanes.of(placementID); l != nil {
		return l.id
	}
	return ""
}

func orNone(s string) string {
	if s == "" {
		return "none"
	}
	return s
}

// laneSummary renders the lanes for the observed-state log line.
func laneSummary(lanes []*pb.DeviceLane) string {
	if len(lanes) == 0 {
		return "none"
	}
	parts := make([]string, 0, len(lanes))
	for _, l := range lanes {
		parts = append(parts, fmt.Sprintf("%s%v:%d resident=%v", l.LaneId, l.DeviceOrdinals, l.AvailableAttemptSlots, l.ResidentPlacementIds))
	}
	return strings.Join(parts, " ")
}
