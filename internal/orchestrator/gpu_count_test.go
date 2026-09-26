package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"testing"
)

func TestSelectedGroupPinsSubsetWithoutShrinkingMachineInventory(t *testing.T) {
	set := &pb.PlacementSet{Placements: []*pb.Placement{{PlacementId: "h3", Entrypoints: []*pb.Entrypoint{{Name: "run", Slots: []*pb.Slot{{Slot: "model"}}}}}}}
	raw, _, err := canonical.Identity(set)
	if err != nil {
		t.Fatal(err)
	}
	pool := []string{"0", "1", "2", "3"}
	for _, count := range []int{0, 1, 2, 4} {
		pins, problem := devicePins(raw, pool, []ModelRef{{GPUs: count}})
		if problem != nil {
			t.Fatal(problem)
		}
		want := count
		if want == 0 {
			want = 4
		}
		if len(pins) != 1 || len(pins[0].DeviceOrdinals) != want {
			t.Fatalf("group%d: %v", count, pins)
		}
	}
	if _, problem := devicePins(raw, pool, []ModelRef{{GPUs: 5}}); problem == nil {
		t.Fatal("oversized group accepted")
	}
	if len(pool) != 4 {
		t.Fatal("machine inventory changed")
	}
}
