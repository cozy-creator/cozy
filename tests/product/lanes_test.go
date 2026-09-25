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

// TestDispatchReadsLanes is proto-024's owner-side arm at proto-061 G's queue depth. The
// fake worker reports TWO device lanes, each admitting a running attempt and one staged
// behind it, with every placement on lane-0 — so its worker-level
// `available_attempt_slots` is 4, the honest sum, while the placement can draw exactly TWO.
// An owner that reads the sum offers a third attempt to that placement and gets it refused
// CAUSE_CODE_NO_CAPACITY, journaled (the fake worker does exactly that). The owner under
// test reads lanes: the third request parks until the running attempt releases the device,
// and no refusal is ever journaled.
func TestDispatchReadsLanes(t *testing.T) {
	o := hostOwner(t, "lanes")
	gate := t.TempDir()
	// A two-device envelope: the worker's lanes are ordinals into it.
	spec := fakeSpec("lanes", "10,11", "--arm", "lanes", "--gate", gate)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID, ""))

	// THE LANES ARE READ, and spelled back as the devices this daemon granted.
	facts := o.c.Worker(instance)
	if facts == nil || len(facts.Lanes) != 2 {
		t.Fatalf("the worker's two lanes were not read: %+v", facts)
	}
	if facts.AvailableSlots != 4 {
		t.Errorf("worker-level slots = %d, want the sum 4", facts.AvailableSlots)
	}
	if facts.PlacementLane != "lane-0" {
		t.Errorf("placement lane = %q, want lane-0 (PlacementStatus.device_lane_id)", facts.PlacementLane)
	}
	for i, want := range []orchestrator.LaneFacts{
		{LaneID: "lane-0", DeviceOrdinals: []uint32{0}, Devices: []string{"10"}, AvailableSlots: 2,
			PlacementIDs: []string{facts.PlacementID}},
		{LaneID: "lane-1", DeviceOrdinals: []uint32{1}, Devices: []string{"11"}, AvailableSlots: 2},
	} {
		got := facts.Lanes[i]
		if got.LaneID != want.LaneID || got.AvailableSlots != want.AvailableSlots ||
			strings.Join(got.Devices, ",") != strings.Join(want.Devices, ",") ||
			len(got.DeviceOrdinals) != 1 || got.DeviceOrdinals[0] != want.DeviceOrdinals[0] ||
			strings.Join(got.PlacementIDs, ",") != strings.Join(want.PlacementIDs, ",") {
			t.Errorf("lane %d = %+v, want %+v", i, got, want)
		}
	}

	// THREE ATTEMPTS FOR ONE PLACEMENT. A runs, B is staged behind it; C must park, not
	// be offered against a sum that lane-0 cannot honour.
	requestA, attemptA, e := o.c.Submit(submission(planID, "fake/lanes", "lanes-a",
		map[string]any{"n": 1}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestA, attemptA, 15*time.Second))
	requestB, _, e := o.c.Submit(submission(planID, "fake/lanes", "lanes-b",
		map[string]any{"n": 2}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(requestB, 1, 15*time.Second))
	requestC, _, e := o.c.Submit(submission(planID, "fake/lanes", "lanes-c",
		map[string]any{"n": 3}))
	fatal(t, e)
	// A reader of the sum offers C at once; the lane reader parks it.
	awaitDurable(t, o, requestC, "request.parked")
	if n := countEvents(o, "AttemptOffer "+requestC+"#"); n != 0 {
		t.Errorf("C was offered %d time(s) while A and B held both of lane-0's seats", n)
	}
	if position, _ := o.c.QueueState(requestC); position == 0 {
		t.Error("C is not parked in the queue while lane-0 is full")
	}
	waitUntil(t, "the worker reports lane-0 full", func() bool {
		held := o.c.Worker(instance)
		return held.Lanes[0].AvailableSlots == 0 && held.Lanes[0].HeldAttempts == 2
	})
	if held := o.c.Worker(instance); held.Lanes[1].AvailableSlots != 2 || held.AvailableSlots != 2 {
		t.Errorf("seats while A runs and B is staged: lane-1=%d worker=%d, want 2/2",
			held.Lanes[1].AvailableSlots, held.AvailableSlots)
	}

	// A releases the device, B enters it, C is offered into the freed seat — in that order.
	releaseGate(t, gate, requestA)
	if _, e := o.c.AwaitSettled(requestA, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
		t.Fatalf("A did not settle on its outcome: %s", briefly(e))
	}
	waitUntil(t, "C is offered after A's release", func() bool {
		return countEvents(o, "AttemptOffer "+requestC+"#") == 1
	})
	releaseGate(t, gate, requestB)
	releaseGate(t, gate, requestC)
	for _, id := range []string{requestB, requestC} {
		if _, e := o.c.AwaitSettled(id, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
			t.Fatalf("%s did not settle after A freed the lane: %s", id, briefly(e))
		}
	}
	events := o.c.Events()
	appliedA, offeredC := -1, -1
	for i, line := range events {
		if appliedA < 0 && strings.Contains(line, "AttemptOutcome "+requestA+"#") && strings.Contains(line, "applied in") {
			appliedA = i
		}
		if offeredC < 0 && strings.Contains(line, "AttemptOffer "+requestC+"#") {
			offeredC = i
		}
	}
	if appliedA < 0 || offeredC < 0 || offeredC < appliedA {
		t.Errorf("C's offer (line %d) did not follow A's applied outcome (line %d)", offeredC, appliedA)
	}
	if n := countEvents(o, "NO_CAPACITY"); n != 0 {
		t.Errorf("%d NO_CAPACITY refusal(s) were journaled; the lane reader must never over-offer", n)
	}
	offers := 0
	for _, id := range []string{requestA, requestB, requestC} {
		offers += countEvents(o, "AttemptOffer "+id+"#1")
	}
	if offers != 3 {
		t.Errorf("%d offers for three requests, want exactly 3 (one each, no requeue)", offers)
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
	// The worker's own words: each seat was taken once per attempt and never refused.
	log, _ := os.ReadFile(filepath.Join(o.root, "workers", instance, "worker.log"))
	for _, id := range []string{requestA, requestB, requestC} {
		if !bytes.Contains(log, []byte("lane lane-0 seat taken by "+id)) {
			t.Errorf("the fake worker did not record %s's seat:\n%s", id, log)
		}
	}
	if bytes.Contains(log, []byte("REFUSED NO_CAPACITY")) {
		t.Errorf("the fake worker refused an over-offer:\n%s", log)
	}
}

// TestSecondRequestOfferedWhileFirstRuns is proto-061 G's queue depth: while attempt N runs
// on a lane, the owner offers N+1 to that lane, the worker admits it QUEUED, and it enters
// the device the moment N releases it — no round trip through the owner in between.
func TestSecondRequestOfferedWhileFirstRuns(t *testing.T) {
	o := hostOwner(t, "staged")
	gate := t.TempDir()
	spec := fakeSpec("staged", "10,11", "--arm", "lanes", "--gate", gate)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID, ""))
	log := func() []byte {
		out, _ := os.ReadFile(filepath.Join(o.root, "workers", instance, "worker.log"))
		return out
	}

	first, firstAttempt, e := o.c.Submit(submission(planID, "fake/staged", "staged-1",
		map[string]any{"n": 1}))
	fatal(t, e)
	fatal(t, o.c.AwaitAccepted(first, firstAttempt, 15*time.Second))
	waitUntil(t, "the first attempt enters the device", func() bool {
		return bytes.Contains(log(), []byte(first+"#1 entered the device on lane-0"))
	})
	second, _, e := o.c.Submit(submission(planID, "fake/staged", "staged-2",
		map[string]any{"n": 2}))
	fatal(t, e)

	// The first attempt is held on the device by the gate, so everything below happens
	// while it is RUNNING.
	fatal(t, o.c.AwaitAccepted(second, 1, 15*time.Second))
	if !bytes.Contains(log(), []byte("lane lane-0 seat taken by "+second+"#1 (queued=true)")) {
		t.Fatalf("the worker did not stage the second attempt behind the first:\n%s", log())
	}
	if n := countEvents(o, "AttemptOutcome "+first+"#"); n != 0 {
		t.Fatalf("the first attempt has an outcome (%d) before the second was offered", n)
	}
	waitUntil(t, "the worker reports both attempts held on lane-0", func() bool {
		lane := o.c.Worker(instance).Lanes[0]
		return lane.HeldAttempts == 2 && lane.AvailableSlots == 0
	})
	if bytes.Contains(log(), []byte(second+"#1 entered the device")) {
		t.Fatal("the staged attempt entered the device while the first still held it")
	}

	releaseGate(t, gate, first)
	waitUntil(t, "the staged attempt enters the device on the first's release", func() bool {
		return bytes.Contains(log(), []byte(second+"#1 entered the device on lane-0"))
	})
	releaseGate(t, gate, second)
	for _, id := range []string{first, second} {
		if _, e := o.c.AwaitSettled(id, 30*time.Second); e == nil || !strings.Contains(e.Message, "no GPU") {
			t.Fatalf("%s did not settle on its own outcome: %s", id, briefly(e))
		}
	}
	if n := countEvents(o, "NO_CAPACITY"); n != 0 {
		t.Errorf("%d NO_CAPACITY refusal(s); the second offer had a staged seat", n)
	}
}

// releaseGate opens the fake worker's device gate for one request's attempt.
func releaseGate(t *testing.T, gate, requestID string) {
	t.Helper()
	must(t, os.WriteFile(filepath.Join(gate, requestID), nil, 0o644))
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
