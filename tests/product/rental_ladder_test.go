package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// cl-166. On 2026-09-07 `cozy run paul/minimax-h3 … --rental-only` bought an RTX 5090
// (32 GB) for the fp8 lane (103 GB), and the price ordering would have bought an RTX
// 4090 (24 GB) first: the choice read price alone and never the card's memory. The
// binding is now a FIT MAP — which lane belongs on which GPU class — and the buy walks it
// rung by rung, keeping only the machines the lane fits.

const (
	gib          = int64(1) << 30
	fp8Manifest  = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	mxfpManifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	bf16Manifest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
)

// h3Ladder is the owner's fit map for the H3 release: fp8 on H100-class cards, mxfp8 on
// B200, bf16 anywhere else that can hold it.
func h3Ladder(fp8Bytes int64) []records.ModelRef {
	return []records.ModelRef{{Package: "paul/minimax-h3", Slot: "generate.models.model",
		Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Ladder: []records.ModelRung{
			{GPU: "H100", Lane: "fp8-adaln-pruned", Manifest: fp8Manifest, Bytes: fp8Bytes},
			{GPU: "B200", Lane: "mxfp8-adaln-pruned", Manifest: mxfpManifest, Bytes: 55 * gib},
			{GPU: "*", Lane: "bf16-full", Manifest: bf16Manifest, Bytes: 130 * gib},
		}}}
}

// market20260907 is the catalog shape on the day of the defect, with the memory figures
// the hub publishes for each product.
func market20260907() []hub.RentalSKU {
	gpu := func(name, model string, vram int64, price int64) hub.RentalSKU {
		return hub.RentalSKU{Name: name, AcceleratorModel: model, VRAMGB: vram,
			ComputeCapability: "9.0", MinimumRAMPerGPUGB: 64,
			PriceUSDMicrosPerHour: price, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"}
	}
	return []hub.RentalSKU{
		gpu("rtx-4090", "NVIDIA GeForce RTX 4090", 24, 740_000),
		gpu("rtx-5090", "NVIDIA GeForce RTX 5090", 32, 990_000),
		gpu("h100-80", "NVIDIA H100 80GB HBM3", 80, 2_490_000),
		gpu("h100-nvl", "NVIDIA H100 NVL", 94, 2_790_000),
		gpu("h200", "NVIDIA H200", 141, 3_590_000),
		gpu("b200", "NVIDIA B200", 180, 5_990_000),
		{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 70_000,
			BaseWorkerProfile: "python3.12-cpu-linux-x86"},
	}
}

func stepNames(steps []rental.Step) []string {
	out := make([]string, 0, len(steps))
	for _, step := range steps {
		out = append(out, step.SKU.Name)
	}
	return out
}

func TestLadderWalkNeversBuysWhatTheLaneCannotFit(t *testing.T) {
	// The defect as it happened: a 103 GB lane against a market whose only cards were
	// 24 GB and 32 GB. Nothing fits, nothing is bought, and the record says why per card.
	steps, decision := rental.Plan(market20260907()[:2], h3Ladder(103*gib), true, rental.Constraints{})
	if len(steps) != 0 {
		t.Fatalf("a 103 GB lane was given a machine to buy: %v", stepNames(steps))
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		if verdict := find(t, decision, name).Verdict; !strings.HasPrefix(verdict, orchestrator.VerdictVRAMShort) {
			t.Fatalf("%s recorded %q; want a %s verdict naming the shortfall", name, verdict, orchestrator.VerdictVRAMShort)
		}
	}

	// The whole market, a 60 GB fp8 lane: the walk is the OWNER'S rung order, cheapest
	// fitting product first within a rung, and price never lifts a later rung over an
	// earlier one — the $5.99 B200 (rung 2) precedes the $3.59 H200 (rung 3).
	steps, decision = rental.Plan(market20260907(), h3Ladder(60*gib), true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h100-80 h100-nvl b200 h200" {
		t.Fatalf("walk order %q; want h100-80 h100-nvl b200 h200", got)
	}
	if steps[0].Rung != 1 || steps[0].Models[0].Lane != "fp8-adaln-pruned" || steps[0].Models[0].Manifest != fp8Manifest {
		t.Fatalf("the H100 step pins %+v; want rung 1 fp8-adaln-pruned", steps[0].Models[0])
	}
	if steps[2].Models[0].Lane != "mxfp8-adaln-pruned" || steps[3].Models[0].Lane != "bf16-full" {
		t.Fatalf("B200 and H200 steps pin %q and %q; want mxfp8-adaln-pruned and bf16-full",
			steps[2].Models[0].Lane, steps[3].Models[0].Lane)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if row.Rung != 3 || row.Lane != "bf16-full" || !strings.HasPrefix(row.Verdict, orchestrator.VerdictVRAMShort) {
			t.Fatalf("%s recorded %+v; want rung 3 bf16-full %s", name, row, orchestrator.VerdictVRAMShort)
		}
	}
	absent(t, decision, "cpu")
	if len(decision.Ladder) != 1 || decision.Ladder[0] != "H100=fp8-adaln-pruned > B200=mxfp8-adaln-pruned > *=bf16-full" {
		t.Fatalf("the record does not carry the ladder it walked: %v", decision.Ladder)
	}

	// The hub refuses the first H100 for inventory; the walk continues within the rung
	// and the record explains every row — no_inventory, chosen, dearer, later_rung.
	rental.Conclude(&decision, steps, []int{0}, 1)
	want := map[string]string{"h100-80": orchestrator.VerdictNoInventory, "h100-nvl": "",
		"b200": orchestrator.VerdictLaterRung + ": rung 2", "h200": orchestrator.VerdictLaterRung + ": rung 3"}
	for name, verdict := range want {
		if got := find(t, decision, name).Verdict; got != verdict {
			t.Fatalf("%s recorded verdict %q; want %q", name, got, verdict)
		}
	}
	if decision.Chosen != "h100-nvl" || decision.UnexplainedPick() != "" {
		t.Fatalf("chosen=%q unexplained=%q; want h100-nvl with every other row explained",
			decision.Chosen, decision.UnexplainedPick())
	}
}

func TestExplicitLaneStillRefusesShortVRAM(t *testing.T) {
	// `model.<param>=org/model@release/lane` pins one lane; the buy still refuses every
	// card the lane does not fit and takes the cheapest that does.
	exact := []records.ModelRef{{Package: "paul/minimax-h3", Slot: "generate.models.model",
		Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: "bf16-full", Manifest: bf16Manifest, Bytes: 130 * gib}}
	steps, decision := rental.Plan(market20260907(), exact, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h200 b200" {
		t.Fatalf("an explicit 130 GiB lane may buy %q; want h200 b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090", "h100-80", "h100-nvl"} {
		if verdict := find(t, decision, name).Verdict; !strings.HasPrefix(verdict, orchestrator.VerdictVRAMShort) {
			t.Fatalf("%s recorded %q; want %s", name, verdict, orchestrator.VerdictVRAMShort)
		}
	}
	rental.Conclude(&decision, steps, nil, 0)
	if decision.Chosen != "h200" || find(t, decision, "b200").Verdict != orchestrator.VerdictDearer {
		t.Fatalf("chosen %q, b200 %q; want h200 with b200 dearer", decision.Chosen, find(t, decision, "b200").Verdict)
	}
}

func TestLadderWithoutCatchAllExcludesUnnamedClasses(t *testing.T) {
	models := h3Ladder(60 * gib)
	models[0].Ladder = models[0].Ladder[:2] // H100 and B200 only
	steps, decision := rental.Plan(market20260907(), models, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h100-80 h100-nvl b200" {
		t.Fatalf("walk %q; want h100-80 h100-nvl b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090", "h200"} {
		if row := find(t, decision, name); row.Verdict != orchestrator.VerdictGPUMismatch || row.Rung != 0 {
			t.Fatalf("%s recorded %+v; want %s on no rung", name, row, orchestrator.VerdictGPUMismatch)
		}
	}
	// A CPU-class request fits only through the catch-all, and has no device to check.
	steps, _ = rental.Plan(market20260907(), models, false, rental.Constraints{})
	if len(steps) != 0 {
		t.Fatalf("a CPU box fits an H100/B200-only ladder: %v", stepNames(steps))
	}
	steps, _ = rental.Plan(market20260907(), h3Ladder(60*gib), false, rental.Constraints{})
	if len(steps) != 1 || steps[0].SKU.Name != "cpu" || steps[0].Models[0].Lane != "bf16-full" {
		t.Fatalf("CPU class walked %v; want cpu alone on the catch-all lane", stepNames(steps))
	}
}

func TestRungMatchingIsTheAcceleratorSubsequence(t *testing.T) {
	for _, test := range []struct {
		gpu, accelerator string
		want             bool
	}{
		{"H100", "NVIDIA H100 80GB HBM3", true},
		{"H100", "NVIDIA H100 NVL", true},
		{"h100", "NVIDIA H100 NVL", true},
		{"H100", "NVIDIA H200", false},
		{"RTX 4090", "NVIDIA GeForce RTX 4090", true},
		{"4090", "NVIDIA GeForce RTX 5090", false},
		{"*", "NVIDIA GeForce RTX 5090", true},
		{"*", "", true},
		{"H100", "", false},
	} {
		if got := records.RungMatches(test.gpu, test.accelerator); got != test.want {
			t.Fatalf("RungMatches(%q, %q)=%v; want %v", test.gpu, test.accelerator, got, test.want)
		}
	}
	models := h3Ladder(60 * gib)
	rung, index, ok := models[0].RungFor("NVIDIA GeForce RTX 5090")
	if !ok || index != 2 || rung.Lane != "bf16-full" {
		t.Fatalf("a 5090 fits rung %d %+v; want the catch-all bf16-full", index, rung)
	}
	pinned, missing := records.PinModels(models, "NVIDIA H100 NVL")
	if missing != "" || !pinned[0].Pinned() || pinned[0].Lane != "fp8-adaln-pruned" || pinned[0].Ladder != nil {
		t.Fatalf("pinning to an H100 NVL gave %+v (missing %q)", pinned, missing)
	}
	models[0].Ladder = models[0].Ladder[:1]
	if _, missing := records.PinModels(models, "NVIDIA H200"); missing != "generate.models.model" {
		t.Fatalf("an H200 against an H100-only ladder should name the unfit slot; got %q", missing)
	}
}

// ladderRequest is one queued --rental serving request carrying no selection yet.
func ladderRequest(t *testing.T, store *records.Store, id string) {
	t.Helper()
	_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("8", 64),
		Package: "paul/minimax-h3", Entrypoint: "generate", Release: "1.0.0", Rental: true,
		Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]"})
	fatal(t, problem)
}

// The pin and the lane are one write: a request's selection lands with its rental, and a
// refused paid ask frees the request for the next rung under a fresh operation.
func TestRentalPinCarriesTheLaneAndARejectedBuyFreesTheNextRung(t *testing.T) {
	store, problem := records.Open(filepath.Join(t.TempDir(), "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	const request = "req-ladder-pin"
	ladderRequest(t, store, request)
	fatal(t, store.PinRequestModels(request, h3Ladder(60*gib)))
	row, problem := store.RequestRow(request)
	fatal(t, problem)
	if len(row.Models) != 1 || row.Models[0].Pinned() || len(row.Models[0].Ladder) != 3 {
		t.Fatalf("the unpinned ladder did not survive the row: %+v", row.Models)
	}
	pinned, _ := records.PinModels(h3Ladder(60*gib), "NVIDIA H100 80GB HBM3")
	fatal(t, store.RecordRental(records.Rental{ID: "pr-ladder-h100", MachineName: "ladder-h100",
		SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3", HourlyRateUSDMicros: 2_490_000,
		State: "ready", Hub: "http://hub.example"}))
	assigned, problem := store.PinRental(request, "pr-ladder-h100", pinned)
	fatal(t, problem)
	row, problem = store.RequestRow(request)
	fatal(t, problem)
	if !assigned || row.Worker != "pr-ladder-h100" || row.Models[0].Lane != "fp8-adaln-pruned" ||
		row.Models[0].Manifest != fp8Manifest || row.Models[0].Ladder != nil {
		t.Fatalf("the pin did not carry the rung's lane: worker=%q models=%+v", row.Worker, row.Models)
	}

	const walker = "req-ladder-walk"
	ladderRequest(t, store, walker)
	first, problem := store.ManagedRentalOperationKey(walker)
	fatal(t, problem)
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: first, Hub: "http://hub.example",
		HourlyRateUSDMicros: 2_490_000, ManagedRequestID: walker}, 20_000_000, 0, replacementAuthor)
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, "", "rejected"))
	next, problem := store.ManagedRentalOperationKey(walker)
	fatal(t, problem)
	if next == first {
		t.Fatalf("a rejected buy replays its operation key %q; the next rung needs a fresh one", next)
	}
}
