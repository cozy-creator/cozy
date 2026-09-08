package producttest

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// TestDispatchReadsLanes is proto-024's owner-side arm. The fake worker reports TWO device
// lanes with one seat each and every placement on lane-0, so its worker-level
// `available_attempt_slots` is 2 — the honest sum — while the placement can draw exactly
// ONE seat. An owner that reads the sum offers two attempts to that placement and gets the
// second refused CAUSE_CODE_NO_CAPACITY, journaled (the fake worker does exactly that). The
// owner under test reads lanes: the second request parks until the first attempt's outcome
// is acked and the lane reports its seat free, and no refusal is ever journaled.
func TestDispatchReadsLanes(t *testing.T) {
	o := hostOwner(t, "lanes")
	// A two-device envelope: the worker's lanes are ordinals into it.
	spec := fakeSpec("lanes", "10,11", "--arm", "lanes")
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID, ""))

	// THE LANES ARE READ, and spelled back as the devices this daemon granted.
	facts := o.c.Worker(instance)
	if facts == nil || len(facts.Lanes) != 2 {
		t.Fatalf("the worker's two lanes were not read: %+v", facts)
	}
	if facts.AvailableSlots != 2 {
		t.Errorf("worker-level slots = %d, want the sum 2", facts.AvailableSlots)
	}
	if facts.PlacementLane != "lane-0" {
		t.Errorf("placement lane = %q, want lane-0 (PlacementStatus.device_lane_id)", facts.PlacementLane)
	}
	for i, want := range []orchestrator.LaneFacts{
		{LaneID: "lane-0", DeviceOrdinals: []uint32{0}, Devices: []string{"10"}, AvailableSlots: 1,
			PlacementIDs: []string{facts.PlacementID}},
		{LaneID: "lane-1", DeviceOrdinals: []uint32{1}, Devices: []string{"11"}, AvailableSlots: 1},
	} {
		got := facts.Lanes[i]
		if got.LaneID != want.LaneID || got.AvailableSlots != want.AvailableSlots ||
			strings.Join(got.Devices, ",") != strings.Join(want.Devices, ",") ||
			len(got.DeviceOrdinals) != 1 || got.DeviceOrdinals[0] != want.DeviceOrdinals[0] ||
			strings.Join(got.PlacementIDs, ",") != strings.Join(want.PlacementIDs, ",") {
			t.Errorf("lane %d = %+v, want %+v", i, got, want)
		}
	}

	// TWO ATTEMPTS FOR ONE PLACEMENT. The first takes lane-0's seat; the second must
	// park, not be offered against a sum that lane-0 cannot honour.
	requestA, attemptA, e := o.c.Submit(submission(planID, "fake/lanes", "lanes-a",
		map[string]any{"n": 1}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestA, attemptA, 15*time.Second))
	requestB, _, e := o.c.Submit(submission(planID, "fake/lanes", "lanes-b",
		map[string]any{"n": 2}))
	fatal(t, e)
	// Give the queue every chance to over-offer: a reader of the sum dispatches B at once.
	time.Sleep(600 * time.Millisecond)
	if n := countEvents(o, "AttemptOffer "+requestB+"#"); n != 0 {
		t.Errorf("B was offered %d time(s) while A held the placement's only lane seat", n)
	}
	if position, _ := o.c.QueueState(requestB); position == 0 {
		t.Error("B is not parked in the queue while lane-0 is full")
	}
	held := o.c.Worker(instance)
	if held.Lanes[0].AvailableSlots != 0 || held.Lanes[1].AvailableSlots != 1 || held.AvailableSlots != 1 {
		t.Errorf("seats while A runs: lane-0=%d lane-1=%d worker=%d, want 0/1/1",
			held.Lanes[0].AvailableSlots, held.Lanes[1].AvailableSlots, held.AvailableSlots)
	}

	// A's outcome is acked, the lane reports its seat free, B dispatches — in that order.
	if _, e := o.c.AwaitSettled(requestA, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
		t.Fatalf("A did not settle on its outcome: %s", briefly(e))
	}
	if _, e := o.c.AwaitSettled(requestB, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
		t.Fatalf("B did not settle after A freed the lane: %s", briefly(e))
	}
	events := o.c.Events()
	appliedA, offeredB := -1, -1
	for i, line := range events {
		if appliedA < 0 && strings.Contains(line, "AttemptOutcome "+requestA+"#") && strings.Contains(line, "applied in") {
			appliedA = i
		}
		if offeredB < 0 && strings.Contains(line, "AttemptOffer "+requestB+"#") {
			offeredB = i
		}
	}
	if appliedA < 0 || offeredB < 0 || offeredB < appliedA {
		t.Errorf("B's offer (line %d) did not follow A's applied outcome (line %d)", offeredB, appliedA)
	}
	if n := countEvents(o, "NO_CAPACITY"); n != 0 {
		t.Errorf("%d NO_CAPACITY refusal(s) were journaled; the lane reader must never over-offer", n)
	}
	if n := countEvents(o, "AttemptOffer "+requestA+"#1") + countEvents(o, "AttemptOffer "+requestB+"#1"); n != 2 {
		t.Errorf("%d offers for two requests, want exactly 2 (one each, no requeue)", n)
	}
	// THE DURABLE RECORD names the lane and the granted device the attempt ran on.
	rows, e := o.store.EventsAfter(requestA, 0, 100)
	fatal(t, e)
	dispatched := false
	for _, row := range rows {
		if row.Type != "request.dispatched" {
			continue
		}
		dispatched = true
		devices, _ := row.Payload["devices"].([]any)
		if row.Payload["device_lane"] != "lane-0" || len(devices) != 1 || devices[0] != "10" {
			t.Errorf("request.dispatched payload = %v, want device_lane lane-0 on device 10", row.Payload)
		}
	}
	if !dispatched {
		t.Error("no request.dispatched event for A")
	}
	// The worker's own words: the seat was taken once per attempt and never refused.
	log, _ := os.ReadFile(filepath.Join(o.root, "workers", instance, "worker.log"))
	if !bytes.Contains(log, []byte("lane lane-0 seat taken by "+requestA)) ||
		!bytes.Contains(log, []byte("lane lane-0 seat taken by "+requestB)) {
		t.Errorf("the fake worker did not record both seats:\n%s", log)
	}
	if bytes.Contains(log, []byte("REFUSED NO_CAPACITY")) {
		t.Errorf("the fake worker refused an over-offer:\n%s", log)
	}
}

// TestOneLaneSnapshotByteIdentity is the compatibility arm. A 1-device worker's snapshot at
// minor 22 is the minor-21 document plus exactly two keys — `lanes` and the placement's
// `device_lane_id` — and minor 23 (proto-026) adds the residency sets (`held_manifests`, the
// lane's `resident_placement_ids`) and the held attempt's `lane_id` / `plan_digest`. This
// reader over a document WITHOUT them sees no lanes, which is the worker-level window: the
// behaviour before proto-024.
//
// Until minor 23 this arm proved the claim by DIGEST: stripping the new keys reproduced
// worker-protocol 9a1e3ee's frozen minor-21 bytes (sha256:e7c53ff2…). The minor-23 corpus
// re-authored the vector — one held attempt is QUEUED, whose `executor_epoch` is 0 and
// therefore absent — so no key-stripping reproduces those bytes any more, and the arm
// proves the SHAPE instead: what remains is a lane-less, residency-less worker document that
// this binding reads as such.
func TestOneLaneSnapshotByteIdentity(t *testing.T) {
	frozen, err := os.ReadFile(filepath.Join(corpusDir, "canonical", "worker_snapshot_body.json"))
	must(t, err)
	doc, err := canonical.Read(frozen, &pb.WorkerSnapshotBody{})
	must(t, err)
	lanes := doc.List("lanes")
	if len(lanes) != 1 || lanes[0].Str("lane_id") != "lane-0" ||
		len(lanes[0].Ints("device_ordinals")) != 1 || lanes[0].Ints("device_ordinals")[0] != 0 ||
		strings.Join(lanes[0].Strs("placement_ids"), ",") != "plc-h3-t2v-1" {
		t.Fatalf("the frozen 1-lane snapshot did not read as one lane over ordinal 0: %v", lanes)
	}
	if doc.List("placements")[0].Str("device_lane_id") != "lane-0" {
		t.Fatalf("the frozen placement names no lane: %v", doc.List("placements")[0])
	}
	if len(doc.Strs("held_manifests")) == 0 || len(lanes[0].Strs("resident_placement_ids")) == 0 {
		t.Fatalf("the frozen 1-lane snapshot carries no residency sets: %v", doc)
	}

	// Strip the minor-22 keys and the minor-23 keys and re-emit.
	raw := map[string]canonical.Value(doc)
	delete(raw, "lanes")
	delete(raw, "held_manifests")
	for _, p := range raw["placements"].([]canonical.Value) {
		delete(p.(map[string]canonical.Value), "device_lane_id")
	}
	for _, h := range raw["held_attempts"].([]canonical.Value) {
		delete(h.(map[string]canonical.Value), "lane_id")
		delete(h.(map[string]canonical.Value), "plan_digest")
	}
	stripped, err := canonical.Write(raw)
	must(t, err)
	// The pre-lane document reads under the minor-24 binding, and reads as NO lanes and no
	// residency: the worker-level window.
	old, err := canonical.Read(stripped, &pb.WorkerSnapshotBody{})
	must(t, err)
	if n := len(old.List("lanes")); n != 0 {
		t.Errorf("a pre-lane document read %d lane(s)", n)
	}
	if n := len(old.Strs("held_manifests")); n != 0 {
		t.Errorf("a pre-residency document read %d held manifest(s)", n)
	}
	if old.List("placements")[0].Str("device_lane_id") != "" {
		t.Errorf("a pre-lane placement named a lane: %v", old.List("placements")[0])
	}
	for _, h := range old.List("held_attempts") {
		if h.Str("lane_id") != "" || h.Str("plan_digest") != "" {
			t.Errorf("a pre-queue held attempt carried a lane or plan: %v", h)
		}
	}
	// And the writer is stable over the 1-lane body: unmarshal the wire bytes, re-identify.
	wire, err := os.ReadFile(filepath.Join(corpusDir, "canonical", "worker_snapshot_body.bin"))
	must(t, err)
	var body pb.WorkerSnapshotBody
	must(t, proto.Unmarshal(wire, &body))
	again, _, err := canonical.Identity(&body)
	must(t, err)
	if !bytes.Equal(again, frozen) {
		t.Errorf("the 1-lane snapshot did not round-trip: %d B written, %d B frozen", len(again), len(frozen))
	}
	if len(body.Lanes) != 1 || body.Lanes[0].LaneId != "lane-0" || body.Placements[0].DeviceLaneId != "lane-0" {
		t.Errorf("the binding read the lane wrong: %v / %q", body.Lanes, body.Placements[0].DeviceLaneId)
	}
}
