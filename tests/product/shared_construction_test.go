package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// h3a-018: the H3 package serves two entrypoints over ONE construction, and the measured
// cost of switching between them on a pod was a second package prepare (28 s for zero new
// bytes), a rebuilt executor (22 s) and a cold read of the other DiT. The selection now
// rides under every slot that shares its bytes, so the pod prepares one placement carrying
// both bindings; the second entrypoint is a dispatch onto it.

const (
	sharedFirst  = "first_last_frame_to_video"
	sharedSecond = "reference_media_to_video"
)

// A request for the second entrypoint routes to the placement the first one prepared:
// one prepare whose download set names both slots, one placement advertising both
// bindings, and the second offer carrying its own binding with no prepare in between.
func TestSharedConstructionEntrypointsSwitchWithoutAPrepare(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, slots: 2,
		// The Runtime's own rule (package_prepare._entrypoints): an entrypoint is bound
		// when every slot it declares is selected, so one binding per selected slot.
		preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
			placement := podPlacement(download, pkg, release, "")
			doc, err := canonical.Read(download, &pb.DownloadDelegation{})
			must(t, err)
			placement.Entrypoints = nil
			for _, row := range doc.List("models") {
				manifest, err := canonical.Raw(row.Str("manifest"))
				must(t, err)
				id := "model-" + row.Str("slot")
				placement.Models = append(placement.Models, &pb.Model{Id: id, Repo: row.Str("model"),
					Version: row.Str("release"), Lane: row.Str("lane"),
					Manifest: &pb.Ref{Digest: manifest, Length: 164}})
				name := strings.TrimSuffix(row.Str("slot"), ".models.model")
				placement.Entrypoints = append(placement.Entrypoints,
					&pb.Entrypoint{Name: name,
						Slots: []*pb.Slot{{Slot: "model", ReferenceModelId: id,
							Components: []*pb.Component{{Component: "dit", ModelId: id}}}}})
			}
			return placement
		}}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "shared-construction", rentalWiring(connection, private))
	submit := func(function, sibling string) {
		_, _, problem := o.c.Submit(orchestrator.Submission{
			IdemKey: "shared-" + function, Package: "cozy/h3-package", Entrypoint: function,
			Release: "1.1.2", Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"},
			Worker: podRental, Rental: true, RentalRequired: true,
			Models: []orchestrator.ModelRef{{Package: "cozy/h3-package", Slot: function + ".models.model",
				SharedSlots: []string{sibling + ".models.model"},
				Model:       "source/h3", Release: "1.0.0", Lane: "fp8-adaln-pruned",
				Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}},
		})
		fatal(t, problem)
	}
	offer := func(n int) *pb.AttemptOffer {
		deadline := time.Now().Add(10 * time.Second)
		for {
			pod.mu.Lock()
			offers := append([]*pb.AttemptOffer(nil), pod.offers...)
			pod.mu.Unlock()
			if len(offers) >= n {
				return offers[n-1]
			}
			if time.Now().After(deadline) {
				t.Fatalf("offer %d never arrived: %v", n, o.c.Events())
			}
			time.Sleep(10 * time.Millisecond)
		}
	}
	binding := func(offer *pb.AttemptOffer) string {
		spec, err := canonical.Read(offer.InvocationSpecCanonicalBytes, &pb.InvocationSpec{})
		must(t, err)
		return spec.Sub("serving").Str("entrypoint_binding_digest")
	}
	spelled := func(name string) string {
		pod.mu.Lock()
		raw := append([]byte(nil), pod.preparedSet...)
		pod.mu.Unlock()
		set, err := canonical.Read(raw, &pb.PlacementSet{})
		must(t, err)
		for _, entry := range set.List("placements")[0].List("entrypoints") {
			if entry.Str("name") == name {
				return entry.Str("entrypoint_binding_digest")
			}
		}
		t.Fatalf("prepared set omitted %s", name)
		return ""
	}

	submit(sharedFirst, sharedSecond)
	first := offer(1)
	if got := binding(first); got != spelled(sharedFirst) {
		t.Fatalf("the first request dispatched binding %s, not %s", got, sharedFirst)
	}
	pod.mu.Lock()
	prepares := append([]*pb.PreparePackageSetCall(nil), pod.prepares...)
	desired := len(pod.desired)
	pod.mu.Unlock()
	if len(prepares) != 1 {
		t.Fatalf("the first request issued %d prepare(s), want one", len(prepares))
	}
	doc, err := canonical.Read(prepares[0].PackageSet.DownloadDelegation, &pb.DownloadDelegation{})
	must(t, err)
	var slots []string
	for _, row := range doc.List("models") {
		slots = append(slots, row.Str("slot"))
	}
	want := []string{sharedFirst + ".models.model", sharedSecond + ".models.model"}
	if !slices.Equal(slots, want) {
		t.Fatalf("the download set selects %v, want the construction's every slot %v", slots, want)
	}

	submit(sharedSecond, sharedFirst)
	second := offer(2)
	if got := binding(second); got != spelled(sharedSecond) {
		t.Fatalf("the second request dispatched binding %s, not %s", got, sharedSecond)
	}
	if first.PlacementId == "" || second.PlacementId != first.PlacementId {
		t.Fatalf("the switch left placement %q for %q", first.PlacementId, second.PlacementId)
	}
	pod.mu.Lock()
	again, desiredAgain := len(pod.prepares), len(pod.desired)
	pod.mu.Unlock()
	if again != 1 || desiredAgain != desired {
		t.Fatalf("the switch issued %d prepare(s) and %d desired state(s) after the first %d: "+
			"a second entrypoint of one construction is a dispatch, not a prepare", again-1, desiredAgain-desired, desired)
	}
}

// The daemon derives the shared slots from what the owner bound: the sibling slot's class
// is the selected slot's, and its hub default names the same model release under the same
// ladder (or offers the pinned lane). Another class, another release or another ladder is
// another construction.
func TestRemoteReleaseSharesSlotsOfOneConstruction(t *testing.T) {
	iface := []byte(`{"application":"h3:app","entrypoints":[` +
		`{"name":"first_last_frame_to_video","models":[{"class":"H3Model","component_use":{"sample":["dit"]},"path":"first_last_frame_to_video.models.model"}],"request":{"fields":[]},"result":{"fields":[]}},` +
		`{"name":"reference_media_to_video","models":[{"class":"H3Model","component_use":{"sample":["dit"]},"path":"reference_media_to_video.models.model"}],"request":{"fields":[]},"result":{"fields":[]}},` +
		`{"name":"upscale","models":[{"class":"Upscaler","component_use":{"run":["net"]},"path":"upscale.models.model"}],"request":{"fields":[]},"result":{"fields":[]}}` +
		`],"format":"cozy.package.interface/1","jobs":[]}`)
	contract, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25", "torch<3,>=2.13"}
	ladder := []hub.BindingRung{{GPU: "H100", Lane: "fp8-adaln-pruned"}, {GPU: "*", Lane: "bf16-full"}}
	bindings := []hub.PackageBindingRow{
		{Slot: sharedFirst + ".models.model", Model: "proof/minimax", Release: "1.0.0", Ladder: ladder, Revision: 1},
		{Slot: sharedSecond + ".models.model", Model: "proof/minimax", Release: "1.0.0", Ladder: ladder, Revision: 1},
		{Slot: "upscale.models.model", Model: "proof/upscaler", Release: "2.0.0", Ladder: ladder, Revision: 1},
	}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/h3/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(detail)
	})
	mux.HandleFunc("GET /v1/packages/proof/h3/bindings", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(map[string]any{"bindings": bindings})
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root := t.TempDir()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	resolver := cli.NewResolver(store, config.Config{HubURL: server.URL, Home: root}, nil)
	manifest := "sha256:" + strings.Repeat("a", 64)
	rungs := []records.ModelRung{{GPU: "H100", Lane: "fp8-adaln-pruned", Manifest: manifest},
		{GPU: "*", Lane: "bf16-full", Manifest: "sha256:" + strings.Repeat("b", 64)}}
	shared := func(function string, model orchestrator.ModelRef) []string {
		t.Helper()
		model.Package, model.Slot = "proof/h3", function+".models.model"
		logical, _, problem := resolver.ResolveRemoteRelease("proof/h3", "1.0.0", function, []orchestrator.ModelRef{model})
		fatal(t, problem)
		return logical.Models[0].SharedSlots
	}
	unpinned := orchestrator.ModelRef{Model: "proof/minimax", Release: "1.0.0", Ladder: rungs}
	if got := shared(sharedFirst, unpinned); !slices.Equal(got, []string{sharedSecond + ".models.model"}) {
		t.Fatalf("the laddered selection shares %v, want the sibling H3Model slot alone", got)
	}
	if got := shared(sharedSecond, unpinned); !slices.Equal(got, []string{sharedFirst + ".models.model"}) {
		t.Fatalf("the sibling shares %v back, want the first slot", got)
	}
	if got := shared("upscale", orchestrator.ModelRef{Model: "proof/upscaler", Release: "2.0.0", Ladder: rungs}); len(got) != 0 {
		t.Fatalf("another class shares %v, want nothing", got)
	}
	pinned := orchestrator.ModelRef{Model: "proof/minimax", Release: "1.0.0", Lane: "fp8-adaln-pruned", Manifest: manifest}
	if got := shared(sharedFirst, pinned); !slices.Equal(got, []string{sharedSecond + ".models.model"}) {
		t.Fatalf("an explicit lane the sibling's ladder offers shares %v, want the sibling slot", got)
	}
	foreign := orchestrator.ModelRef{Model: "proof/minimax", Release: "1.0.0", Lane: "int4", Manifest: manifest}
	if got := shared(sharedFirst, foreign); len(got) != 0 {
		t.Fatalf("a lane the sibling's ladder never offered shares %v, want nothing", got)
	}
	// A sibling may share the base class yet require an additional model. Neither
	// an absent default nor a default for bytes this request did not select may
	// leave its base slot in the preparation's selected set.
	var document map[string]any
	must(t, json.Unmarshal(iface, &document))
	sibling := document["entrypoints"].([]any)[1].(map[string]any)
	sibling["models"] = append(sibling["models"].([]any), map[string]any{
		"class": "Adapter", "component_use": map[string]any{}, "path": sharedSecond + ".models.adapter",
	})
	setInterface := func() {
		t.Helper()
		changed, err := json.Marshal(document)
		must(t, err)
		parsed, problem := launch.DecodePackageInterface(changed)
		fatal(t, problem)
		detail.PackageInterface = changed
		detail.Release.PackageInterfaceDigest = assessmentDigest(parsed.Raw)
		detail.Release.PackageInterfaceLength = int64(len(changed))
	}
	setInterface()
	if got := shared(sharedFirst, unpinned); len(got) != 0 {
		t.Fatalf("an unbound sibling leaked partial shared slots: %v", got)
	}
	bindings = append(bindings, hub.PackageBindingRow{Slot: sharedSecond + ".models.adapter", Model: "proof/adapter", Release: "1.0.0", Ladder: ladder, Revision: 1})
	if got := shared(sharedFirst, unpinned); len(got) != 0 {
		t.Fatalf("an unselected sibling adapter leaked partial shared slots: %v", got)
	}
	// When one already-selected construction supplies every sibling slot, retain
	// the original sharing optimization for all of those slots together.
	adapter := sibling["models"].([]any)[1].(map[string]any)
	adapter["class"] = "H3Model"
	bindings[len(bindings)-1].Model = "proof/minimax"
	setInterface()
	if got := shared(sharedFirst, unpinned); !slices.Equal(got, []string{sharedSecond + ".models.adapter", sharedSecond + ".models.model"}) {
		t.Fatalf("a fully covered sibling was not shared as a complete set: %v", got)
	}
	// Restore the original interface for the ladder/release mismatch arms.
	detail.PackageInterface = iface
	detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	bindings[1].Ladder = []hub.BindingRung{{GPU: "*", Lane: "bf16-full"}}
	if got := shared(sharedFirst, unpinned); len(got) != 0 {
		t.Fatalf("a sibling bound under another ladder shares %v, want nothing", got)
	}
	bindings[1] = hub.PackageBindingRow{Slot: sharedSecond + ".models.model", Model: "proof/minimax", Release: "1.1.0", Ladder: ladder, Revision: 2}
	if got := shared(sharedFirst, unpinned); len(got) != 0 {
		t.Fatalf("a sibling bound to another release shares %v, want nothing", got)
	}
}
