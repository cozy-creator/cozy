package producttest

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

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
