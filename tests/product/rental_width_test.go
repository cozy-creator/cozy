package producttest

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/rental"
)

// cl-179: WIDTH. A rental is bought at a width and the pod must deliver it; Runtime alone
// decides which of those cards each call uses.

// TestWideProductsAreOnlyBoughtForAPackageThatDeclaresTheDegree is the PRE-SPEND half.
// Width is not capacity — every GPU of a group holds the full weights — so the only thing
// that uses a second card is a group of that degree, and only the package's author can say
// the construction can be built at one. A wide product is therefore excluded from the
// ladder unless the degree is declared, which keeps a typed worker refusal from costing an
// hour's rent. A REUSE is not held to it: the pod is paid for, and the worker runs it at
// the largest declared degree that fits.
func TestWideProductsAreOnlyBoughtForAPackageThatDeclaresTheDegree(t *testing.T) {
	skus := []hub.RentalSKU{
		{Name: "h100", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 1,
			VRAMGB: 80, PriceUSDMicrosPerHour: 3_490_000},
		{Name: "h100", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 2,
			VRAMGB: 80, PriceUSDMicrosPerHour: 6_980_000},
		{Name: "h100", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 4,
			VRAMGB: 80, PriceUSDMicrosPerHour: 13_960_000},
	}
	verdicts := func(constraints rental.Constraints, job bool) map[string]string {
		out := map[string]string{}
		for _, c := range rental.Purchases(skus, nil, true, job, constraints) {
			out[orchestrator.MachineLabel(c.SKU, c.GPUs)] = c.Verdict
		}
		return out
	}
	// Nothing declared: the ladder is one-card products, which is every package today.
	undeclared := verdicts(rental.Constraints{}, false)
	if undeclared["h100"] != "" {
		t.Fatalf("the one-card product was excluded: %q", undeclared["h100"])
	}
	for _, name := range []string{"2x h100", "4x h100"} {
		if !strings.Contains(undeclared[name], "width_undeclared") ||
			!strings.Contains(undeclared[name], "no sequence-parallel degree") {
			t.Fatalf("%s verdict = %q, want a width refusal naming the missing degree",
				name, undeclared[name])
		}
	}
	// `@sequence_parallel(degrees=(2, 4))`: both widths become buyable, and the one-card
	// product stays buyable beside them — a declared degree is a capability, not a demand.
	declared := verdicts(rental.Constraints{Degrees: []int{2, 4}}, false)
	for _, name := range []string{"h100", "2x h100", "4x h100"} {
		if declared[name] != "" {
			t.Fatalf("%s was excluded from a package declaring degrees 2 and 4: %q",
				name, declared[name])
		}
	}
	// A wider offer may run the largest package-supported subset, which is two here.
	partial := verdicts(rental.Constraints{Degrees: []int{2}}, false)
	for _, name := range []string{"2x h100", "4x h100"} {
		if partial[name] != "" {
			t.Fatalf("supported subset was excluded on %s: %q", name, partial[name])
		}
	}
	// A JOB shards nothing: one bounded attempt on a wide pod idles every card but one for
	// the whole hour, whatever the package declares.
	jobs := verdicts(rental.Constraints{Degrees: []int{2, 4}}, true)
	for _, name := range []string{"2x h100", "4x h100"} {
		if !strings.Contains(jobs[name], "width_undeclared") ||
			!strings.Contains(jobs[name], "a job shards none of them") {
			t.Fatalf("%s was buyable for a job: %q", name, jobs[name])
		}
	}
}

// A group shards the selected construction. Intersect all its model slots, while
// keeping a separate sibling's unsupported slots out of that decision.
func TestDeclaredDegreesAreTheIntersectionOverEveryModelSlot(t *testing.T) {
	raw, err := os.ReadFile("testdata/h3/package-interface-1.1.2.json")
	must(t, err)
	declare := func(perEntrypoint ...string) *launch.PackageInterface {
		var document map[string]any
		must(t, json.Unmarshal(raw, &document))
		entrypoints := document["entrypoints"].([]any)
		for i, spelled := range perEntrypoint {
			if spelled == "" {
				continue
			}
			var degrees any
			must(t, json.Unmarshal([]byte(spelled), &degrees))
			slot := entrypoints[i].(map[string]any)["models"].([]any)[0].(map[string]any)
			slot["sequence_parallel"] = map[string]any{"degrees": degrees}
		}
		edited, err := json.Marshal(document)
		must(t, err)
		iface, problem := launch.DecodePackageInterface(edited)
		fatal(t, problem)
		return iface
	}
	spell := func(degrees []int) string {
		out := make([]string, 0, len(degrees))
		for _, degree := range degrees {
			out = append(out, strconv.Itoa(degree))
		}
		return strings.Join(out, ",")
	}
	for _, arm := range []struct {
		name     string
		declared []string
		want     []int
	}{
		{"the published interface declares none", []string{"", ""}, nil},
		{"both slots declare 2 and 4", []string{"[2,4]", "[2,4]"}, []int{2, 4}},
		{"only the overlap survives", []string{"[2,4]", "[4,8]"}, []int{4}},
		{"one silent slot makes the construction unshardable", []string{"[2,4]", ""}, nil},
		{"no overlap declares nothing", []string{"[2]", "[4]"}, nil},
	} {
		t.Run(arm.name, func(t *testing.T) {
			iface := declare(arm.declared...)
			selected := launch.Entrypoint{}
			for _, endpoint := range iface.Entrypoints {
				selected.Models = append(selected.Models, endpoint.Models...)
			}
			got := selected.SequenceParallelDegrees()
			if spell(got) != spell(arm.want) {
				t.Fatalf("degrees = %v, want %v", got, arm.want)
			}
		})
	}
	iface := declare("[2,4]", "")
	if got := spell(iface.Entrypoints[0].SequenceParallelDegrees()); got != "2,4" {
		t.Fatalf("an unselected unshardable sibling restricted the normal function: %s", got)
	}
	if got := iface.Entrypoints[1].SequenceParallelDegrees(); len(got) != 0 {
		t.Fatalf("the unshardable function inherited its sibling's degrees: %v", got)
	}
}
