package producttest

import (
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// Run531 bound exactly one H3 slot, but rental preflight added the install's
// captured segment job and refused two H100s at 95.9 GiB per card. These are the
// actual selected checkpoint bytes and package interface from that request.
func TestH3ServingRentalIgnoresCapturedSiblingResidency(t *testing.T) {
	fixture := filepath.Join("testdata", "h3-rental-residency")
	raw, err := os.ReadFile(filepath.Join(fixture, "request.json"))
	must(t, err)
	var selected struct{ Models []records.ModelRef }
	must(t, json.Unmarshal(raw, &selected))
	if len(selected.Models) != 1 || selected.Models[0].Slot != "fl2va.models.model" {
		t.Fatal("failure fixture no longer records one actual bound slot")
	}
	model := selected.Models[0]
	const textBytes = int64(51_506_191_840)
	if model.ComponentBytes["text_encoder"] != textBytes {
		t.Fatal("failure fixture lost measured text-encoder bytes")
	}
	card := hub.ModelCard{Model: hub.Resource{Org: "paul", Name: "minimax-h3"},
		Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0-rc.2"},
			Lanes: []hub.ModelLaneSummary{{Lane: "fp8-pruned", ManifestID: model.Manifest, Bytes: model.Bytes,
				Components: slices.Sorted(maps.Keys(model.ComponentBytes)), ComponentBytes: model.ComponentBytes}}}}}
	catalog := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/models/paul/minimax-h3" {
			t.Errorf("unexpected catalog operation: %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(card)
	}))
	defer catalog.Close()
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	installed := cleanupTestInstall(layout, "1111111111111111", "1.14.2")
	installed.Package = "local/minimax-h3"
	raw, err = os.ReadFile(filepath.Join(fixture, "package-interface.json"))
	must(t, err)
	raw, err = canonical.NormalizeJCS(raw)
	must(t, err)

	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), raw, 0o444))
	fatal(t, store.RecordInstall(installed))
	fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: installed.ID,
		ChildInstallID: installed.ID,
		Module:         "h3", Export: "segment", Entrypoint: "segment"}}))
	resolver := cli.NewResolver(store, config.Config{Home: layout.Root, HubURL: catalog.URL})
	request := records.Request{InstallID: installed.ID, Package: installed.Package,
		Entrypoint: "fl2va", Kind: "serving", NeedsAccelerator: true, Models: selected.Models}
	children, problem := resolver.UnpublishedChildModels(request)
	fatal(t, problem)
	models := append(slices.Clone(request.Models), children...)
	sku := hub.RentalSKU{Name: "h100-sxm5-80gb", AcceleratorModel: h100SXM,
		AcceleratorCount: 2, VRAMGB: 80}
	constraints := rental.Constraints{Degrees: []int{2, 4}}
	candidate := rental.Purchases([]hub.RentalSKU{sku}, models, true, false, constraints)[0]
	if candidate.Verdict != "" || len(children) != 0 {
		t.Fatalf("one H3 serving slot acquired %d unrelated captured slots: %s", len(children), candidate.Verdict)
	}
	if need := records.Resident(candidate.Models, h100SXM, false); need.Bytes != textBytes {
		t.Fatalf("actual one-slot H3 need = %+v", need)
	}

	// Two independent slots still require two copies even for identical checkpoint
	// bytes. Correct invocation scoping must not turn into manifest deduplication.
	independent := model
	independent.Slot = "fl2va.models.independent_model"
	models = []records.ModelRef{model, independent}
	if need := records.Resident(models, h100SXM, false); need.Bytes != 2*textBytes {
		t.Fatalf("independent resident slots were collapsed: %+v", need)
	}
	candidate = rental.Purchases([]hub.RentalSKU{sku}, models, true, false, constraints)[0]
	if !strings.HasPrefix(candidate.Verdict, "excluded:vram_short:") {
		t.Fatalf("two actual slots bypassed the per-card floor: %+v", candidate)
	}

	// The same captured segment remains available to the CPU long_form parent;
	// its model defaults still participate in choosing a suitable rental.
	request.Entrypoint, request.Kind = "long_form", "job"
	request.NeedsAccelerator, request.Models = false, nil
	children, problem = resolver.UnpublishedChildModels(request)
	fatal(t, problem)
	if len(children) != 1 || children[0].BindingSlot() != "segment.models.model" || len(children[0].Ladder) != 3 {
		t.Fatalf("CPU composition lost its captured model defaults: %+v", children)
	}
}
