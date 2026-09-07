package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// cl-132. On 2026-09-04 auto-placement bought an rtx-4090 at $0.953504/hr twice while an
// rtx-a4000 at $0.463504/hr was believed to be on offer. Two explanations fit the
// evidence — a compatibility filter that wrongly excluded the cheap card, or a card that
// was simply out of stock in that minute — and the recorded decision was IDENTICAL under
// both, because only the winner was written down. Settling it took an hour of reading
// provider inventory generations out of the hub's Postgres.
//
// These prices are the live figures from that night, written as literals on purpose: a
// test that recomputed them from the catalog under test would assert the code equals
// itself (decisions.md row 698).
const (
	liveA4000TotalUSDMicros = 463_504 // $0.25 gpu + $0.213504 storage
	live4090TotalUSDMicros  = 953_504 // $0.74 gpu + $0.213504 storage
)

// marketWithA4000 is what RunPod published at 03:00:11 on 2026-09-04, when an explicit
// `cozy rental new rtx-a4000` succeeded and acquired `cisqua`.
func marketWithA4000() []hub.RentalSKU {
	return append(marketWithoutA4000(), hub.RentalSKU{
		Name: "rtx-a4000", AcceleratorModel: "NVIDIA RTX A4000",
		PriceUSDMicrosPerHour: 250_000, StorageUSDMicrosPerHour: 213_504,
		BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"})
}

// marketWithoutA4000 is the SAME provider, 32 minutes later: generations 14612 through
// 14715 (06:48:08Z to 07:20:27Z) carried no A4000 offer at all, and both disputed pods
// were bought inside that window.
func marketWithoutA4000() []hub.RentalSKU {
	return []hub.RentalSKU{
		{Name: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090",
			PriceUSDMicrosPerHour: 740_000, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
		{Name: "rtx-5090", AcceleratorModel: "NVIDIA GeForce RTX 5090",
			PriceUSDMicrosPerHour: 990_000, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
		{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 70_000,
			BaseWorkerProfile: "python3.12-cpu-linux-x86"},
	}
}

// choose is the modelless buy: Plan walked to its first fitting product, the way an
// unmodeled request buys. It keeps the pre-ladder proofs below on the real chooser.
func choose(skus []hub.RentalSKU, needsAccelerator bool, constraints rental.Constraints) (
	hub.RentalSKU, orchestrator.SKUDecision, string, bool,
) {
	steps, decision := rental.Plan(skus, nil, needsAccelerator, constraints)
	if len(steps) == 0 {
		return hub.RentalSKU{}, decision, decision.Mismatch, false
	}
	rental.Conclude(&decision, steps, nil, 0)
	return steps[0].SKU, decision, "", true
}

func find(t *testing.T, decision orchestrator.SKUDecision, name string) orchestrator.SKUCandidate {
	t.Helper()
	for _, candidate := range decision.Offered {
		if candidate.Name == name {
			return candidate
		}
	}
	t.Fatalf("%q is not in the recorded offer set %+v", name, decision.Offered)
	return orchestrator.SKUCandidate{}
}

func absent(t *testing.T, decision orchestrator.SKUDecision, name string) {
	t.Helper()
	for _, candidate := range decision.Offered {
		if candidate.Name == name {
			t.Fatalf("%q is recorded as offered, but this market did not carry it: %+v",
				name, decision.Offered)
		}
	}
}

// TestTheChoiceRecordsWhatItChoseOver is the fix's whole point: the two markets above
// must produce DIFFERENT records, so a reader can tell an out-of-stock cheap card from a
// wrongly-excluded one without a database.
func TestTheChoiceRecordsWhatItChoseOver(t *testing.T) {
	stocked, withA4000, _, ok := choose(marketWithA4000(), true, rental.Constraints{})
	if !ok || stocked.Name != "rtx-a4000" {
		t.Fatalf("with the A4000 in stock the cheapest compatible pick must be rtx-a4000; got %q (ok=%v)",
			stocked.Name, ok)
	}
	if got := find(t, withA4000, "rtx-a4000").TotalUSDMicrosPerHour; got != liveA4000TotalUSDMicros {
		t.Fatalf("recorded A4000 total is %d micros/hour; the live figure was %d",
			got, liveA4000TotalUSDMicros)
	}
	// The dearer card is on the record WITH the reason it lost. This is the row whose
	// absence made the overpay unauditable.
	if verdict := find(t, withA4000, "rtx-4090").Verdict; verdict != orchestrator.VerdictDearer {
		t.Fatalf("rtx-4090 lost to a cheaper card but is recorded with verdict %q; want %q",
			verdict, orchestrator.VerdictDearer)
	}

	short, withoutA4000, _, ok := choose(marketWithoutA4000(), true, rental.Constraints{})
	if !ok || short.Name != "rtx-4090" {
		t.Fatalf("with no A4000 offered the pick must be rtx-4090; got %q (ok=%v)", short.Name, ok)
	}
	if got := find(t, withoutA4000, "rtx-4090").TotalUSDMicrosPerHour; got != live4090TotalUSDMicros {
		t.Fatalf("recorded 4090 total is %d micros/hour; the live figure was %d",
			got, live4090TotalUSDMicros)
	}
	// THE FACT THAT WAS MISSING LIVE. A stock-out is an absence from the offer set, and
	// an absence is readable. Without this the two markets are indistinguishable after
	// the fact, which is exactly how th-151 came to be filed as a placement defect.
	absent(t, withoutA4000, "rtx-a4000")

	if len(withA4000.Offered) == len(withoutA4000.Offered) {
		t.Fatalf("the two markets produced offer sets of the same size (%d); the record "+
			"is not distinguishing them", len(withA4000.Offered))
	}
}

// TestCheapestOfferedIsAlwaysExplained is the invariant, and the red arm proves the
// predicate can actually fire. A dearer buy is allowed — the cheap card is often gone —
// but a cheaper compatible product passed over with NO stated reason is a chooser defect.
func TestCheapestOfferedIsAlwaysExplained(t *testing.T) {
	for _, market := range [][]hub.RentalSKU{marketWithA4000(), marketWithoutA4000()} {
		_, decision, _, ok := choose(market, true, rental.Constraints{})
		if !ok {
			t.Fatalf("a GPU market with compatible products chose nothing: %+v", decision)
		}
		if unexplained := decision.UnexplainedPick(); unexplained != "" {
			t.Fatalf("real choice passed over %q with no stated reason: %+v",
				unexplained, decision.Offered)
		}
	}

	// RED ARM — plant the violation. This is the shape a broken chooser would emit:
	// the cheapest product sits in the offer set carrying no verdict, and something
	// dearer was bought anyway. If UnexplainedPick cannot see this, it cannot see a
	// real overpay either and the invariant is decorative.
	planted := orchestrator.SKUDecision{
		Chosen: "rtx-4090",
		Offered: []orchestrator.SKUCandidate{
			{Name: "rtx-a4000", TotalUSDMicrosPerHour: liveA4000TotalUSDMicros},
			{Name: "rtx-4090", TotalUSDMicrosPerHour: live4090TotalUSDMicros},
		},
	}
	if unexplained := planted.UnexplainedPick(); unexplained != "rtx-a4000" {
		t.Fatalf("planted silent overpay was not detected: UnexplainedPick()=%q, want rtx-a4000",
			unexplained)
	}

	// A cheaper card that lost for a STATED reason is not an overpay: that is the
	// out-of-stock and base-mismatch case, and it must stay quiet.
	explained := planted
	explained.Offered = []orchestrator.SKUCandidate{
		{Name: "rtx-a4000", TotalUSDMicrosPerHour: liveA4000TotalUSDMicros,
			Verdict: orchestrator.VerdictBaseMismatch + ": torch 2.13 needs sm_89"},
		{Name: "rtx-4090", TotalUSDMicrosPerHour: live4090TotalUSDMicros},
	}
	if unexplained := explained.UnexplainedPick(); unexplained != "" {
		t.Fatalf("a cheaper card that lost for a stated reason was reported as unexplained: %q",
			unexplained)
	}
}

// TestRentalProvenanceSurvivesToTheRecord runs the real store. The managed path used to
// write "" as the reason, so no rental could be attributed to the command that bought
// it; that is what let a REFUSED explicit ask be credited with two pods auto-placement
// had bought minutes either side of it.
func TestRentalProvenanceSurvivesToTheRecord(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	if problem != nil {
		t.Fatalf("cannot open the records store: %v", problem)
	}
	defer store.Close()

	explicit := "cozy rental new rtx-a4000"
	managed := rental.AcquisitionReason(records.Request{
		ID: "req-59119a5207ab58898ac5d626", Package: "paul/sdxl"})
	job := rental.AcquisitionReason(records.Request{
		ID: "job-177b899c60b525988346f2fe", Package: "paul/minimax-h3-tools", Kind: "job"})

	// The managed reason must actually SAY something. The old behaviour was "", and an
	// empty reason is the defect: assert non-empty and that it names its request.
	if managed == "" || !strings.Contains(managed, "req-59119a5207ab58898ac5d626") {
		t.Fatalf("a managed acquisition recorded %q; it must name the request that bought the pod", managed)
	}
	if job == "" || !strings.Contains(job, "job-177b899c60b525988346f2fe") {
		t.Fatalf("a job acquisition recorded %q; it must name the job that bought the pod", job)
	}
	// A job pod and a serving pod are the two machines that were confused for one
	// explicit ask. They must not read the same.
	if managed == job {
		t.Fatalf("a serving buy and a job buy record the same provenance %q", managed)
	}
	for _, reason := range []string{managed, job} {
		if strings.Contains(reason, explicit) {
			t.Fatalf("an auto-placement buy reads as an explicit ask: %q", reason)
		}
	}

	// Round-trip all three through the real store, exactly as an acquisition does:
	// begin the paid operation carrying its reason, then bind the rental id the hub
	// answered with.
	write := func(key, reason, rentalID string) {
		t.Helper()
		body := []byte(`{"sku":"rtx-4090"}`)
		author := func(string) ([]byte, string, *exit.Error) {
			return body, "sha256:" + strings.Repeat("a", 64), nil
		}
		_, _, problem := store.BeginRentalOperation(records.RentalOperation{
			Key: key, Hub: "http://127.0.0.1:8819", Reason: reason,
			HourlyRateUSDMicros: 740_000, State: "pending",
		}, 10_000_000, 213_504, author)
		if problem != nil {
			t.Fatalf("cannot begin rental operation %s: %v", key, problem)
		}
		if problem := store.AdvanceRentalOperation(key, rentalID, "acquiring"); problem != nil {
			t.Fatalf("cannot bind rental %s to operation %s: %v", rentalID, key, problem)
		}
	}
	write("idem-explicit", explicit, "pr-explicit")
	write("managed-rental-req-59119a5207ab58898ac5d626", managed, "pr-9098f0a5e1362d842123")
	write("managed-rental-job-177b899c60b525988346f2fe", job, "pr-d9cd0d8882a20fe989a5")

	provenance, problem := store.RentalProvenance()
	if problem != nil {
		t.Fatalf("cannot read rental provenance: %v", problem)
	}
	// The three pods that were confused for one must now be individually attributable.
	for id, want := range map[string]string{
		"pr-explicit":             explicit,
		"pr-9098f0a5e1362d842123": managed,
		"pr-d9cd0d8882a20fe989a5": job,
	} {
		if got := provenance[id]; got != want {
			t.Fatalf("rental %s reports provenance %q; want %q", id, got, want)
		}
	}
}
