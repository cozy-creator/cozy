package producttest

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
)

// Tensorhub removes obsolete default bindings. The ordinary list reports only
// the current rows, without exposing an operator cleanup state to invokers.
func TestBindingsListOnlyCurrentDefaults(t *testing.T) {
	h := newLadderHub(t)
	root := ladderRoot(t, h)
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{goodLadder()}
	h.mu.Unlock()
	for _, flags := range [][]string{nil, {"--json"}} {
		args := append([]string{"package", "bindings", ladderPackage}, flags...)
		code, out := runCozy(t, root, args...)
		if code != 0 || !strings.Contains(out, ladderSlot) {
			t.Fatalf("current binding is not listed [exit %d]: %s", code, out)
		}
		if strings.Contains(strings.ToLower(out), "standing") || strings.Contains(out, "orphaned") {
			t.Fatalf("operator cleanup state leaked into binding list: %s", out)
		}
	}
}

// `cozy package bindings` answers "what does this slot run?" from the release itself:
// each declared slot's authored default ladder beside the owner's override.
func TestBindingsShowReleaseDefaultsBesideOverrides(t *testing.T) {
	h := newLadderHub(t, authoredH3(ladderLane))
	root := ladderRoot(t, h)
	type row struct {
		Slot, Source, Model, Release, Ladder, Default string
		PackageRelease                                string `json:"package_release"`
	}
	list := func(args ...string) []row {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"--json", "package", "bindings", ladderPackage, "--full"}, args...)...)
		var doc struct{ Bindings []row }
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
			t.Fatalf("bindings %v: %d %s", args, code, out)
		}
		for i := range doc.Bindings {
			if doc.Bindings[i].PackageRelease != "1.0.0" {
				t.Fatalf("listing did not name its release: %s", out)
			}
		}
		return doc.Bindings
	}
	authored := ladderModel + "@" + ladderRelease + " H100=" + ladderLane
	got := list()
	want := []row{
		{Slot: ladderSlot, Source: "default", Model: ladderModel, Release: ladderRelease, Ladder: "H100=" + ladderLane, Default: authored, PackageRelease: "1.0.0"},
		{Slot: "lane.models.pruned", Source: "none", PackageRelease: "1.0.0"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("defaults view:\n got %+v\nwant %+v", got, want)
	}
	// A job slot takes its model from the caller, so its missing default asks for nothing.
	if code, out := runCozy(t, root, "package", "bindings", ladderPackage); code != 0 ||
		!strings.Contains(out, "proof/h3@1.0.0") || !strings.Contains(out, authored) ||
		strings.Contains(out, "cozy package bind") {
		t.Fatalf("human defaults view: %d %s", code, out)
	}
	bare := newLadderHub(t)
	if code, out := runCozy(t, ladderRoot(t, bare), "package", "bindings", ladderPackage); code != 0 ||
		!strings.Contains(out, "cozy package bind proof/h3 "+ladderSlot) {
		t.Fatalf("a serving slot with no default did not say how to bind it: %d %s", code, out)
	}
	owner := goodLadder()
	stale := hub.PackageBindingRow{Slot: "retired.models.model", Model: ladderModel, Release: ladderRelease,
		Ladder: []hub.BindingRung{{GPU: "*", Lane: "bf16-full"}}, Revision: 1}
	h.mu.Lock()
	h.bindings = []hub.PackageBindingRow{owner, stale}
	h.mu.Unlock()
	got = list("--version", "1.0.0")
	if len(got) != 3 || got[0].Source != "override" || got[0].Ladder != hub.LadderText(owner.Ladder) ||
		got[0].Default != authored || got[1].Source != "none" ||
		got[2].Slot != stale.Slot || got[2].Source != "override (undeclared)" || got[2].Default != "" {
		t.Fatalf("override view: %+v", got)
	}
	if code, out := runCozy(t, root, "package", "bindings", ladderPackage, "--version", "9.9.9"); code == 0 {
		t.Fatalf("an absent release listed defaults: %s", out)
	}
}
