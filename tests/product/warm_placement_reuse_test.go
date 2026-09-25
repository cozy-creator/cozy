package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A WARM PLACEMENT SERVES EVERY REQUEST IT BINDS (run 864). One rental ran fl2va, then a
// queued turbo request replaced the package's whole selection, the device was handed to
// the turbo placement, and the next fl2va request restaged the first selection from a
// standing start — each alternation a ~100 GB prepare for zero new bytes. The package's
// selection is now extended per slot: a repeated request is a dispatch, another entrypoint
// with other models adds them to the one placement, and the entrypoints already bound
// stay bound.
func TestWarmPlacementServesRepeatsAndExtendsForOtherEntrypoints(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, slots: 8, preparedPlacement: runtimePreparedPlacement(t)}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "warm-placement-reuse", rentalWiring(connection, private))

	const pkg = "paul/minimax-h3"
	h3 := func(function, lane, digit string, shared ...string) orchestrator.ModelRef {
		return orchestrator.ModelRef{Package: pkg, Slot: function + ".models.model", SharedSlots: shared,
			Model: pkg, Release: "1.0.0", Lane: lane, Manifest: "sha256:" + strings.Repeat(digit, 64),
			ManifestLength: 164}
	}
	turbo := func(function, sibling string) []orchestrator.ModelRef {
		return []orchestrator.ModelRef{
			{Package: pkg, Slot: function + ".models.base_model", SharedSlots: []string{sibling + ".models.base_model"},
				Model: pkg, Release: "1.0.1", Lane: "fp8-adaln-pruned",
				Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 164},
			{Package: pkg, Slot: function + ".models.turbo_lora", SharedSlots: []string{sibling + ".models.turbo_lora"},
				Model: pkg + "-turbo-lora", Release: "1.0.0", Lane: "pdd8",
				Manifest: "sha256:" + strings.Repeat("3", 64), ManifestLength: 164},
		}
	}
	submitted := 0
	submit := func(function string, models []orchestrator.ModelRef) {
		submitted++
		_, _, problem := o.c.Submit(orchestrator.Submission{
			IdemKey: fmt.Sprintf("warm-%d", submitted), Package: pkg, Entrypoint: function,
			Release: "1.0.0", Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"},
			Worker: podRental, Rental: true, RentalRequired: true, Models: models,
		})
		fatal(t, problem)
	}
	offer := func(n int) *pb.AttemptOffer {
		deadline := time.Now().Add(20 * time.Second)
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
	counts := func() (int, int) {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.prepares), len(pod.desired)
	}
	entrypoints := func() map[string]string {
		pod.mu.Lock()
		d := pod.desired[len(pod.desired)-1]
		pod.mu.Unlock()
		set, err := canonical.Read(d.GetPlacementSet().PlacementSetCanonicalBytes, &pb.PlacementSet{})
		must(t, err)
		out := map[string]string{}
		for _, placement := range set.List("placements") {
			for _, entrypoint := range placement.List("entrypoints") {
				out[entrypoint.Str("name")] = placement.Str("placement_id")
			}
		}
		return out
	}
	fl2va := []orchestrator.ModelRef{h3("fl2va", "fp8-pruned", "1", "ref2va.models.model")}

	submit("fl2va", fl2va)
	first := offer(1)
	prepares, desired := counts()
	if prepares != 1 || desired != 1 {
		t.Fatalf("the first request issued %d prepare(s) and %d desired state(s), want one each", prepares, desired)
	}

	// An identical request is a dispatch onto the warm placement: no prepare, no revision.
	submit("fl2va", fl2va)
	second := offer(2)
	if again, desiredAgain := counts(); again != prepares || desiredAgain != desired {
		t.Fatalf("an identical request issued %d prepare(s) and %d desired state(s)",
			again-prepares, desiredAgain-desired)
	}
	if second.PlacementId != first.PlacementId {
		t.Fatalf("the identical request left warm placement %s for %s", first.PlacementId, second.PlacementId)
	}

	// Another entrypoint with other models EXTENDS the package's selection: the one
	// placement the pod prepares binds the turbo entrypoints beside every entrypoint the
	// rental already served.
	submit("ref2va_turbo", turbo("ref2va_turbo", "fl2va_turbo"))
	third := offer(3)
	prepares, desired = counts()
	if prepares != 2 {
		t.Fatalf("the turbo request issued %d prepare(s) in total, want one more", prepares)
	}
	bound := entrypoints()
	for _, name := range []string{"fl2va", "ref2va", "fl2va_turbo", "ref2va_turbo"} {
		if bound[name] == "" {
			t.Fatalf("the rental's set dropped %s when the turbo selection joined: %v", name, bound)
		}
	}
	if bound["fl2va"] != third.PlacementId {
		t.Fatalf("fl2va is bound on %s, not the placement serving turbo (%s)", bound["fl2va"], third.PlacementId)
	}

	// Alternating back is a dispatch onto the extended placement — the thrash is gone —
	// and so is the sibling turbo entrypoint the selection already binds.
	submit("fl2va", fl2va)
	fourth := offer(4)
	submit("fl2va_turbo", turbo("fl2va_turbo", "ref2va_turbo"))
	fifth := offer(5)
	if again, desiredAgain := counts(); again != prepares || desiredAgain != desired {
		t.Fatalf("returning to fl2va and its turbo sibling issued %d prepare(s) and %d desired state(s); "+
			"the extended placement already binds both", again-prepares, desiredAgain-desired)
	}
	if fourth.PlacementId != third.PlacementId || fifth.PlacementId != third.PlacementId {
		t.Fatalf("the extended placement %s did not serve fl2va (%s) and fl2va_turbo (%s)",
			third.PlacementId, fourth.PlacementId, fifth.PlacementId)
	}
}

// runtimePreparedPlacement is the Runtime's own derivation (package_prepare._entrypoints):
// model ids number the SORTED selected slots, an entrypoint binds when all its slots are
// selected, and its binding digest covers those model ids.
func runtimePreparedPlacement(t *testing.T) func([]byte, string, string) *pb.Placement {
	return func(download []byte, pkg, release string) *pb.Placement {
		placement := podPlacement(download, pkg, release, "")
		doc, err := canonical.Read(download, &pb.DownloadDelegation{})
		must(t, err)
		rows := doc.List("models")
		sort.Slice(rows, func(i, j int) bool { return rows[i].Str("slot") < rows[j].Str("slot") })
		placement.Entrypoints = nil
		bound := map[string]*pb.Entrypoint{}
		for index, row := range rows {
			manifest, err := canonical.Raw(row.Str("manifest"))
			must(t, err)
			id := fmt.Sprintf("model-%04d", index)
			placement.Models = append(placement.Models, &pb.Model{Id: id, Repo: row.Str("model"),
				Version: row.Str("release"), Lane: row.Str("lane"),
				Manifest: &pb.Ref{Digest: manifest, Length: 164}})
			name, slot, _ := strings.Cut(row.Str("slot"), ".models.")
			if bound[name] == nil {
				bound[name] = &pb.Entrypoint{Name: name}
				placement.Entrypoints = append(placement.Entrypoints, bound[name])
			}
			bound[name].Slots = append(bound[name].Slots, &pb.Slot{Slot: slot, ReferenceModelId: id,
				Components: []*pb.Component{{Component: "dit", ModelId: id}}})
		}
		return placement
	}
}

// A LATER REQUEST DOES NOT RESTAGE THE HEAD'S WARM PLACEMENT (run 868). The rental's one
// seat is busy, an identical request waits for it at the head of the rental's queue, and a
// turbo request queued behind it would change the selection. It prepares nothing — no
// PodHost prepare, no desired revision — while the head is queued: that head runs next on
// the placement already serving.
func TestQueuedDifferentSelectionWaitsBehindHeadOnWarmPlacement(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, slots: 1, preparedPlacement: runtimePreparedPlacement(t)}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "warm-placement-head", rentalWiring(connection, private))

	const pkg = "paul/minimax-h3"
	fl2va := []orchestrator.ModelRef{{Package: pkg, Slot: "fl2va.models.model",
		SharedSlots: []string{"ref2va.models.model"}, Model: pkg, Release: "1.0.0", Lane: "fp8-pruned",
		Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164}}
	turbo := []orchestrator.ModelRef{
		{Package: pkg, Slot: "ref2va_turbo.models.base_model", Model: pkg, Release: "1.0.1",
			Lane: "fp8-adaln-pruned", Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 164},
		{Package: pkg, Slot: "ref2va_turbo.models.turbo_lora", Model: pkg + "-turbo-lora", Release: "1.0.0",
			Lane: "pdd8", Manifest: "sha256:" + strings.Repeat("3", 64), ManifestLength: 164},
	}
	submit := func(idem, function string, models []orchestrator.ModelRef) string {
		id, _, problem := o.c.Submit(orchestrator.Submission{
			IdemKey: idem, Package: pkg, Entrypoint: function,
			Release: "1.0.0", Payload: []byte(`{"prompt":"a fox"}`), Outputs: []string{"video"},
			Worker: podRental, Rental: true, RentalRequired: true, Models: models,
		})
		fatal(t, problem)
		return id
	}
	counts := func() (int, int, int) {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.prepares), len(pod.desired), len(pod.offers)
	}

	submit("head-running", "fl2va", fl2va)
	waitUntil(t, "the first request's offer", func() bool { _, _, offers := counts(); return offers == 1 })
	head := submit("head-queued", "fl2va", fl2va)
	if _, ok := waitEvent(o, head+" PARKED", 10*time.Second); !ok {
		t.Fatalf("the identical request did not queue behind the busy seat: %v", o.c.Events())
	}
	submit("later-turbo", "ref2va_turbo", turbo)
	time.Sleep(3 * time.Second)
	if prepares, desired, offers := counts(); prepares != 1 || desired != 1 || offers != 1 {
		t.Fatalf("with the head queued on the warm placement the later request caused %d prepare(s), "+
			"%d desired state(s) and %d offer(s); want 1, 1, 1", prepares, desired, offers)
	}
}
