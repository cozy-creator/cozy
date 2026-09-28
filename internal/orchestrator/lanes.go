package orchestrator

import (
	"strconv"
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
