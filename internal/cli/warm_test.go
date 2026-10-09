package cli

import (
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// `--warm` sends the machine's whole set back with one member changed: others stay as sent
// (without what Status added), the member is replaced, `off` removes it, and an unknown level
// is refused before anything is sent. A member's models travel by name: its binding's rungs, lanes only.
func TestAWarmSetChangesOneMemberAndKeepsTheRest(t *testing.T) {
	member := func(pkg, fn string, level v1.WarmLevel) *v1.WarmItem {
		return &v1.WarmItem{Source: &v1.WarmItem_Release{Release: &v1.Release{Package: pkg, Release: "1.0.0"}},
			Entrypoint: fn, Level: level, Holds: "imported", HeldBack: "a request took its room"}
	}
	current := []*v1.WarmItem{member("paul/sdxl", "generate", v1.WarmLevel_WARM_LEVEL_IMPORTED),
		member("paul/anima", "generate", v1.WarmLevel_WARM_LEVEL_HOST)}
	selection := records.RentalInstallSelection{Package: "paul/sdxl", Release: "2.6.0", Entrypoint: "generate", Warm: "host",
		Models: []records.ModelRef{{Choice: true, Package: "paul/sdxl", Slot: "model", BindingPath: "model", Model: "paul/sdxl", Release: "1.0.0",
			Ladder: []records.ModelRung{{GPU: "*", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("ab", 32)}}}}}
	set, problem := warmSetV1(current, selection)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(set.Items) != 2 {
		t.Fatalf("want 2 members, got %v", set.Items)
	}
	kept, changed := set.Items[0], set.Items[1]
	if kept.GetRelease().GetPackage() != "paul/anima" || kept.Holds != "" || kept.HeldBack != "" || kept.Level != v1.WarmLevel_WARM_LEVEL_HOST {
		t.Fatalf("the other member was not sent back as asked: %v", kept)
	}
	if changed.GetRelease().GetRelease() != "2.6.0" || changed.Level != v1.WarmLevel_WARM_LEVEL_HOST ||
		len(changed.Models) != 1 || changed.Models[0].Parameter != "model" || changed.Models[0].Repository != "paul/sdxl" ||
		len(changed.Models[0].Rungs) != 1 || changed.Models[0].Rungs[0].Lane != "bf16" || changed.Models[0].Manifest != "" {
		t.Fatalf("the member was not replaced with its new level and models: %v", changed)
	}
	selection.Warm = "off"
	if set, _ = warmSetV1(current, selection); len(set.Items) != 1 || set.Items[0].GetRelease().GetPackage() != "paul/anima" {
		t.Fatalf("off did not remove the member: %v", set.Items)
	}
	selection.Warm = "hot"
	if _, problem := warmSetV1(current, selection); problem == nil {
		t.Fatal("an unknown level was accepted")
	}
}
