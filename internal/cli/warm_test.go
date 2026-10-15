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
	selection := records.Operation{Package: "paul/sdxl", Release: "2.6.0", Entrypoint: "generate", Warm: "host",
		Models: []records.ModelRef{{Choice: true, Package: "paul/sdxl", Slot: "model", BindingPath: "model", Model: "paul/sdxl", Release: "1.0.0",
			Ladder: []records.ModelRung{{GPU: "*", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("ab", 32)}}}}}
	set, problem := warmSetV1(current, nil, selection)
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
	if set, _ = warmSetV1(current, nil, selection); len(set.Items) != 1 || set.Items[0].GetRelease().GetPackage() != "paul/anima" {
		t.Fatalf("off did not remove the member: %v", set.Items)
	}
	selection.Warm = "hot"
	if _, problem := warmSetV1(current, nil, selection); problem == nil {
		t.Fatal("an unknown level was accepted")
	}
}

// Local code is warmed as the installation its machine holds of it: the member names that
// installation, replaces the one warmed before, and is refused while the machine holds none.
func TestALocalPackageIsWarmedAsItsInstallation(t *testing.T) {
	current := []*v1.WarmItem{{Source: &v1.WarmItem_Installation{Installation: "local-old"}, Entrypoint: "main", Level: v1.WarmLevel_WARM_LEVEL_HOST}}
	environments := []*v1.Environment{{Installation: "local-old", Package: "local/probe"}, {Installation: "paul-sdxl", Package: "paul/sdxl", Hub: "https://tensorhub.com"}}
	selection := records.Operation{Package: "local/probe", Release: "0.1.0", Entrypoint: "main", Warm: "gpu"}
	set, problem := warmSetV1(current, environments, selection)
	if problem != nil {
		t.Fatal(problem)
	}
	if len(set.Items) != 1 || set.Items[0].GetInstallation() != "local-old" || set.Items[0].Level != v1.WarmLevel_WARM_LEVEL_GPU {
		t.Fatalf("the local member is not its installation at the asked level: %v", set.Items)
	}
	if _, problem := warmSetV1(nil, environments[1:], selection); problem == nil || problem.ErrName() != "warm.local_not_installed" {
		t.Fatalf("a local package the machine holds no installation of was warmed: %v", problem)
	}
}
