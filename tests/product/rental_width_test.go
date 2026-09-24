package producttest

import (
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// cl-179: WIDTH, carried end to end. A rental is bought at a width, the pod delivers that
// many cards, and a model-bearing placement on it is pinned to ALL of them — one group
// lane of degree K, which is what makes a sequence-parallel run possible at all.
//
// Everything below the test is the product: the records store's own rental row, the
// production `rental.Resolver` / `ObserveWorker` / `ClaimProof` wiring the daemon uses, the
// real control stream, and `tests/support`'s independent pod peer answering it. Nothing is
// stubbed between the paid width and the `device_pins` the worker receives.

// rentalWidthWiring is daemon_serve's own rental wiring over a rental this host bought at
// `width`, attached with real credentials. The two facts under test — the persisted width
// and the pod's reported one — are the arguments; everything else is production code.
func rentalWidthWiring(t *testing.T, pod *fakePod, connection *orchestrator.WorkerConnection,
	certPath string, width int) func(*orchestrator.Options) {
	t.Helper()
	return func(o *orchestrator.Options) {
		identity, problem := rental.PendingCreatorIdentity(o.Layout, "width-proof")
		fatal(t, problem)
		public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
		must(t, err)
		pod.controlKey = ed25519.PublicKey(public)
		certificate, err := os.ReadFile(certPath)
		must(t, err)
		fatal(t, rental.Attach(o.Layout, o.Store, records.Rental{
			ID: podRental, MachineName: "wide-machine", SKU: "fake-x", Hub: "https://hub.invalid",
			AcceleratorModel: "fake-4090", AcceleratorCount: width,
			HourlyRateUSDMicros: 1, State: "ready",
			Address:          connection.Addr,
			MediaAddress:     connection.Media.Addr,
			ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID,
		}, string(certificate), connection.Media.Token, identity))
		o.Rentals = rental.Resolver(o.Layout, o.Store)
		o.ObserveRental = rental.ObserveWorker(o.Store)
		o.RentalClaimProof = rental.ClaimProof(o.Layout)
		o.RentalPackageSet = func(packages []*pb.DownloadPackageRef,
			models []*pb.DownloadModelRef) ([]byte, *exit.Error) {
			body, err := canonical.Bytes(&pb.DownloadDelegation{Models: models, Packages: packages})
			if err != nil {
				return nil, exit.Internalf("%v", err)
			}
			return body, nil
		}
		o.RentalPrepareFacts = func(_ context.Context, _ *orchestrator.WorkerConnection,
			ref *pb.DownloadPackageRef) (orchestrator.PrepareFacts, *exit.Error) {
			return testPrepareFacts(ref.Package, ref.Release), nil
		}
	}
}

// modelBearingPlacement is a prepared placement that HOLDS WEIGHTS: one model, one
// entrypoint binding it. Weight is the whole question a pin turns on — a group shards a
// model's attention, so a weightless placement is never pinned to one.
func modelBearingPlacement(t *testing.T) func([]byte, string, string) *pb.Placement {
	return func(download []byte, pkg, release string) *pb.Placement {
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
				&pb.Entrypoint{Name: name, EntrypointBindingDigest: sha256Of([]byte("entrypoint:" + pkg)),
					Slots: []*pb.Slot{{Slot: "model", ReferenceModelId: id,
						Components: []*pb.Component{{Component: "dit", ModelId: id}}}}})
		}
		return placement
	}
}

func submitToWideRental(t *testing.T, o *owner, idem string) {
	t.Helper()
	_, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: idem, Package: "cozy/h3-package", Entrypoint: "tile", Release: "1.1.2",
		Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"},
		Worker: podRental, Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{{Package: "cozy/h3-package", Slot: "tile.models.model",
			Model: "source/h3", Release: "1.0.0", Lane: "bf16",
			Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}},
	})
	fatal(t, problem)
}

// awaitPlacementSet returns the last desired state the pod received that actually names a
// placement, and that placement's id. An empty set is a drain, not the convergence the pin
// rides on.
func awaitPlacementSet(t *testing.T, o *owner, pod *fakePod) (*pb.DesiredPlacementSet, string) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for {
		pod.mu.Lock()
		states := append([]*pb.DesiredWorkerState(nil), pod.desired...)
		pod.mu.Unlock()
		for i := len(states) - 1; i >= 0; i-- {
			set := states[i].GetPlacementSet()
			if set == nil {
				continue
			}
			doc, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
			if err != nil || len(doc.List("placements")) != 1 {
				continue
			}
			return set, doc.List("placements")[0].Str("placement_id")
		}
		if time.Now().After(deadline) {
			t.Fatalf("no serving desired state reached the pod: %v", o.c.Events())
		}
		time.Sleep(10 * time.Millisecond)
	}
}

// TestRentalWidthPinsThePlacementToEveryPaidCard is the whole plumbing in one line of
// evidence: what the pod is told to do with its cards is the width the rental was bought
// at. At one card nothing is pinned — one envelope device is one lane and the worker
// assigns it by measured fit, exactly as before wide products existed. At two and four the
// placement carries ONE pin over ALL the ordinals, which is what fuses a group lane of
// that degree; a pin over fewer would idle paid cards without saying so.
func TestRentalWidthPinsThePlacementToEveryPaidCard(t *testing.T) {
	for _, arm := range []struct {
		name  string
		width int
		pin   []uint32
	}{
		{"one card pins nothing", 1, nil},
		{"degree two", 2, []uint32{0, 1}},
		{"degree four", 4, []uint32{0, 1, 2, 3}},
	} {
		t.Run(arm.name, func(t *testing.T) {
			pod := &fakePod{serve: true, deviceCount: uint32(arm.width),
				preparedPlacement: modelBearingPlacement(t)}
			root := t.TempDir()
			connection, certPath := startFakePod(t, root, pod)
			o := hostOwner(t, "rental-width-"+arm.name,
				rentalWidthWiring(t, pod, connection, certPath, arm.width))
			submitToWideRental(t, o, "width-"+arm.name)
			set, placementID := awaitPlacementSet(t, o, pod)
			if len(arm.pin) == 0 {
				if len(set.DevicePins) != 0 {
					t.Fatalf("a %d-card rental pinned %v; one device is one lane",
						arm.width, set.DevicePins)
				}
				return
			}
			if len(set.DevicePins) != 1 {
				t.Fatalf("a %d-card rental sent %d pin(s), want exactly one for its one placement",
					arm.width, len(set.DevicePins))
			}
			pin := set.DevicePins[0]
			if pin.PlacementId != placementID {
				t.Errorf("the pin names placement %q, not the set's %q", pin.PlacementId, placementID)
			}
			if len(pin.DeviceOrdinals) != len(arm.pin) {
				t.Fatalf("the pin covers %v, want every paid ordinal %v", pin.DeviceOrdinals, arm.pin)
			}
			for i, ordinal := range arm.pin {
				if pin.DeviceOrdinals[i] != ordinal {
					t.Fatalf("the pin covers %v, want every paid ordinal %v", pin.DeviceOrdinals, arm.pin)
				}
			}
		})
	}
}

// TestWeightlessPlacementIsNeverPinnedToAGroup is the other half of the rule. A group
// shards a model's attention; a placement holding no weights has nothing to shard, and the
// worker refuses `device_group_unsupported` for a weightless pin. So the owner authors
// none and the worker places it on its own least-loaded lane — on a four-card pod as on
// a one-card one.
func TestWeightlessPlacementIsNeverPinnedToAGroup(t *testing.T) {
	pod := &fakePod{serve: true, deviceCount: 4}
	root := t.TempDir()
	connection, certPath := startFakePod(t, root, pod)
	o := hostOwner(t, "rental-width-weightless",
		rentalWidthWiring(t, pod, connection, certPath, 4))
	submitToWideRental(t, o, "width-weightless")
	set, _ := awaitPlacementSet(t, o, pod)
	if len(set.DevicePins) != 0 {
		t.Fatalf("a weightless placement was pinned to %v", set.DevicePins)
	}
}

// TestPodDeliveringFewerCardsThanPaidForIsRefused is the money arm. The width readback is
// the same class of fence as the accelerator model's: a pod that does not deliver what is
// being billed cannot be dispatched to, and the refusal names the two numbers so the
// operator can act on it. Before cl-179 this check was `count == 1`, which is why a pod
// that delivered exactly what was bought — four cards — was refused as well.
func TestPodDeliveringFewerCardsThanPaidForIsRefused(t *testing.T) {
	pod := &fakePod{serve: true, deviceCount: 2}
	root := t.TempDir()
	connection, certPath := startFakePod(t, root, pod)
	o := hostOwner(t, "rental-width-short", rentalWidthWiring(t, pod, connection, certPath, 4))
	_, _, _, problem := o.c.EnsureRental(podRental)
	if problem == nil || problem.ErrName() != "rental.accelerator_count_mismatch" {
		t.Fatalf("a pod delivering 2 of 4 paid cards was accepted: %v", problem)
	}
	if !strings.Contains(problem.Message, "2 accelerator(s)") ||
		!strings.Contains(problem.Message, "bought 4") {
		t.Fatalf("the refusal does not name both widths: %s", problem.Message)
	}
	pod.mu.Lock()
	defer pod.mu.Unlock()
	if len(pod.desired) != 0 {
		t.Fatalf("a refused-width pod was sent %d desired state(s)", len(pod.desired))
	}
}

// TestWideProductsAreOnlyBoughtForAPackageThatDeclaresTheDegree is the PRE-SPEND half.
// Width is not capacity — every rank of a group holds the full weights — so the only thing
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
	// A degree the author did not declare stays out however wide the machine is.
	partial := verdicts(rental.Constraints{Degrees: []int{2}}, false)
	if partial["2x h100"] != "" {
		t.Fatalf("the declared width was excluded: %q", partial["2x h100"])
	}
	if !strings.Contains(partial["4x h100"], "width_undeclared") ||
		!strings.Contains(partial["4x h100"], "degrees 2") {
		t.Fatalf("4x h100 verdict = %q, want a refusal naming the declared degrees",
			partial["4x h100"])
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

func TestJobInputModelsNeverPinWeightlessPreparationToAGroup(t *testing.T) {
	pod := &fakePod{serve: true, jobReady: true, deviceCount: 4, preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
		placement := podPlacement(download, pkg, release, "")
		placement.Entrypoints = nil
		placement.Models = []*pb.Model{{Id: "job-source", Repo: "source/h3", Version: "1.0.0", Lane: "bf16", Manifest: &pb.Ref{Digest: sha256Of([]byte("job-source")), Length: 164}}}
		return placement
	}}
	root := t.TempDir()
	connection, certPath := startFakePod(t, root, pod)
	o := hostOwner(t, "job-input-wide-prepare", rentalWidthWiring(t, pod, connection, certPath, 4))
	submitPublishedRentalJob(t, o, "cozy/h3-package", "1.0.7", "sha256:"+strings.Repeat("35", 32), "job-input-wide-prepare")
	waitUntil(t, "weightless package preparation followed by ordinary job directive", func() bool { pod.mu.Lock(); defer pod.mu.Unlock(); return len(pod.jobDirectives) > 0 })
	pod.mu.Lock()
	defer pod.mu.Unlock()
	prepared := false
	for _, desired := range pod.desired {
		set := desired.GetPlacementSet()
		if set == nil {
			continue
		}
		document, err := canonical.Read(set.PlacementSetCanonicalBytes, &pb.PlacementSet{})
		must(t, err)
		if len(document.List("placements")) == 0 {
			continue
		}
		placement := document.List("placements")[0]
		if len(placement.List("models")) != 1 || len(placement.List("entrypoints")) != 0 {
			t.Fatal("fixture omitted the actual job-only model metadata shape")
		}
		if len(set.DevicePins) > 0 {
			t.Fatalf("job input inventory became a CP serving group: %+v", set.DevicePins)
		}
		prepared = true
	}
	if !prepared || pod.jobDirectives[0].DeviceCount > 1 {
		t.Fatal("ordinary job did not follow ungrouped package preparation")
	}
}
