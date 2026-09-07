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
//
// cl-168. The first fit rule held the WHOLE lane against the card and refused the exact
// production case: that same 103 GB lane runs on an 80 GB H100, because cozy-runtime
// stages components per method and no method holds more than the 51.5 GiB text encoder.
// The card now publishes each component's bytes (th-181) and the device is held to the
// entrypoint's largest resident group; a lane the card sizes no components for fits by
// the owner's rung alone.

const (
	gib          = int64(1) << 30
	fp8Manifest  = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	mxfpManifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	bf16Manifest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	// textEncoderNeed is the largest group any H3 method holds resident: the text
	// conditioning's 51.5 GiB encoder. Every DiT sample holds one ~20 GiB DiT.
	textEncoderNeed    = 103 * gib / 2
	textEncoderVerdict = "vram_short: needs 51.5 GiB resident (condition_text: text_encoder)"
)

// h3Components is the fp8-adaln-pruned lane as the card publishes it: 103 GB in five
// components. h3BF16Components is the bf16-full lane, 130 GiB, same text encoder.
func h3Components() map[string]int64 {
	return map[string]int64{"text_encoder": textEncoderNeed, "fl2va_dit": 20 * gib,
		"ref2va_dit": 20 * gib, "video_vae": 52 * gib / 5, "audio_vae": 3 * gib / 5}
}

func h3BF16Components() map[string]int64 {
	return map[string]int64{"text_encoder": textEncoderNeed, "fl2va_dit": 135 * gib / 4,
		"ref2va_dit": 135 * gib / 4, "video_vae": 52 * gib / 5, "audio_vae": 3 * gib / 5}
}

// h3ComponentUse is the H3 entrypoint slot's component_use exactly as the package
// interface publishes it: each method names the components it stages.
func h3ComponentUse() map[string][]string {
	return map[string][]string{"condition_text": {"text_encoder"}, "condition_fl2va_media": {"video_vae"},
		"condition_ref2va_media": {"audio_vae", "video_vae"}, "sample_fl2va": {"fl2va_dit"},
		"sample_ref2va": {"ref2va_dit"}, "decode_video": {"video_vae"}, "decode_audio": {"audio_vae"}}
}

// h3Ladder is the owner's fit map for the H3 release: fp8 on H100-class cards, mxfp8 on
// B200 (a lane the card sizes no components for), bf16 anywhere else that can hold it.
func h3Ladder() []records.ModelRef {
	return []records.ModelRef{{Package: "paul/minimax-h3", Slot: "generate.models.model",
		Model: "paul/minimax-h3", Release: "1.0.0-rc.1", ComponentUse: h3ComponentUse(),
		Ladder: []records.ModelRung{
			{GPU: "H100", Lane: "fp8-adaln-pruned", Manifest: fp8Manifest, Bytes: 103 * gib, ComponentBytes: h3Components()},
			{GPU: "B200", Lane: "mxfp8-adaln-pruned", Manifest: mxfpManifest, Bytes: 55 * gib},
			{GPU: "*", Lane: "bf16-full", Manifest: bf16Manifest, Bytes: 130 * gib, ComponentBytes: h3BF16Components()},
		}}}
}

// fp8Exact is the explicit `model.<param>=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned`.
func fp8Exact() records.ModelRef {
	return records.ModelRef{Package: "paul/minimax-h3", Slot: "generate.models.model",
		Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: "fp8-adaln-pruned", Manifest: fp8Manifest,
		Bytes: 103 * gib, ComponentBytes: h3Components(), ComponentUse: h3ComponentUse()}
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

// sized asserts one audit row says what it compared.
func sized(t *testing.T, row orchestrator.SKUCandidate, fit string, resident, vramGB int64) {
	t.Helper()
	if row.Fit != fit || row.ResidentBytes != resident || row.VRAMBytes != vramGB<<30 {
		t.Fatalf("%s recorded fit=%q resident=%d vram=%d; want %s %d %d GB", row.Name, row.Fit,
			row.ResidentBytes, row.VRAMBytes, fit, resident, vramGB)
	}
}

func TestLadderWalkNeversBuysWhatTheLaneCannotFit(t *testing.T) {
	// The defect as it happened: the H3 ladder against a market whose only cards were
	// 24 GB and 32 GB. Both land on the catch-all bf16 rung, whose text encoder alone
	// outweighs them: nothing is bought, and the record says the need per card.
	steps, decision := rental.Plan(market20260907()[:2], h3Ladder(), true, rental.Constraints{})
	if len(steps) != 0 {
		t.Fatalf("a lane whose text encoder is 51.5 GiB was given a machine to buy: %v", stepNames(steps))
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		if verdict := find(t, decision, name).Verdict; !strings.HasPrefix(verdict, textEncoderVerdict) {
			t.Fatalf("%s recorded %q; want a %s verdict naming the resident need", name, verdict, orchestrator.VerdictVRAMShort)
		}
	}

	// The whole market: the walk is the OWNER'S rung order, cheapest fitting product first
	// within a rung, and price never lifts a later rung over an earlier one — the $5.99
	// B200 (rung 2) precedes the $3.59 H200 (rung 3). The 103 GB fp8 lane fits both H100s
	// because no method holds more than its 51.5 GiB text encoder (the production case).
	steps, decision = rental.Plan(market20260907(), h3Ladder(), true, rental.Constraints{})
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
	sized(t, find(t, decision, "h100-80"), records.FitComponents, textEncoderNeed, 80)
	sized(t, find(t, decision, "h100-nvl"), records.FitComponents, textEncoderNeed, 94)
	// The mxfp8 lane publishes no component bytes: the owner's B200 rung is the fit, and
	// the record says so rather than pretending a figure was compared.
	sized(t, find(t, decision, "b200"), records.FitRungAsserted, 0, 180)
	sized(t, find(t, decision, "h200"), records.FitComponents, textEncoderNeed, 141)
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if row.Rung != 3 || row.Lane != "bf16-full" || !strings.HasPrefix(row.Verdict, textEncoderVerdict) {
			t.Fatalf("%s recorded %+v; want rung 3 bf16-full %s", name, row, orchestrator.VerdictVRAMShort)
		}
	}
	sized(t, find(t, decision, "rtx-4090"), records.FitComponents, textEncoderNeed, 24)
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

func TestExplicitLaneIsHeldToTheSameFloor(t *testing.T) {
	// `model.<param>=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` pins the 103 GB lane; the
	// buy takes the cheapest card that holds its largest resident group — the 80 GB H100
	// — and refuses the 24 GB and 32 GB cards naming the 51.5 GiB text encoder.
	steps, decision := rental.Plan(market20260907(), []records.ModelRef{fp8Exact()}, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h100-80 h100-nvl h200 b200" {
		t.Fatalf("an explicit fp8 lane may buy %q; want h100-80 h100-nvl h200 b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if !strings.HasPrefix(row.Verdict, textEncoderVerdict) || !strings.HasSuffix(row.Verdict, name+" has "+
			map[string]string{"rtx-4090": "24", "rtx-5090": "32"}[name]+" GB") {
			t.Fatalf("%s recorded %q; want the need and the card's memory", name, row.Verdict)
		}
	}
	sized(t, find(t, decision, "rtx-4090"), records.FitComponents, textEncoderNeed, 24)
	sized(t, find(t, decision, "h100-80"), records.FitComponents, textEncoderNeed, 80)
	rental.Conclude(&decision, steps, nil, 0)
	if decision.Chosen != "h100-80" || find(t, decision, "h100-nvl").Verdict != orchestrator.VerdictDearer {
		t.Fatalf("chosen %q, h100-nvl %q; want h100-80 with h100-nvl dearer", decision.Chosen, find(t, decision, "h100-nvl").Verdict)
	}
}

func TestComponentFitHoldsTheOwnersRung(t *testing.T) {
	// A slot without component_use holds its largest single component.
	plain := fp8Exact()
	plain.ComponentUse = nil
	need := records.Resident([]records.ModelRef{plain})
	if need.Fit != records.FitComponents || need.Bytes != textEncoderNeed || need.Need != "text_encoder" {
		t.Fatalf("no component_use sized %+v; want the 51.5 GiB text encoder", need)
	}
	// A group is the SUM of the components its method stages, and the largest group wins.
	pair := fp8Exact()
	pair.ComponentUse = map[string][]string{"sample_pair": {"fl2va_dit", "ref2va_dit"}, "decode": {"audio_vae"}}
	if need := records.Resident([]records.ModelRef{pair}); need.Bytes != 40*gib || need.Need != "sample_pair: fl2va_dit+ref2va_dit" {
		t.Fatalf("a two-DiT group sized %+v; want 40 GiB", need)
	}
	// A placement holds every slot at once: two H3 slots need two text encoders.
	two := []records.ModelRef{fp8Exact(), fp8Exact()}
	two[1].Slot = "generate.models.refiner"
	if need := records.Resident(two); need.Bytes != 103*gib || need.Fit != records.FitComponents ||
		need.Need != "condition_text: text_encoder; condition_text: text_encoder" {
		t.Fatalf("two slots sized %+v; want 103 GiB", need)
	}
	steps, decision := rental.Plan(market20260907(), two, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h200 b200" {
		t.Fatalf("two text encoders may buy %q; want h200 b200", got)
	}
	if verdict := find(t, decision, "h100-80").Verdict; !strings.HasPrefix(verdict, "vram_short: needs 103.0 GiB resident (condition_text: text_encoder; condition_text: text_encoder), h100-80 has 80 GB") {
		t.Fatalf("h100-80 recorded %q; want both slots' needs", verdict)
	}

	// No component bytes on the card: the rung (or the explicit lane) is the owner's
	// assertion. Nothing is excluded, and the record says the device was not sized.
	asserted := fp8Exact()
	asserted.ComponentBytes = nil
	if need := records.Resident([]records.ModelRef{asserted}); need.Fit != records.FitRungAsserted || need.Bytes != 0 {
		t.Fatalf("a lane without component bytes sized %+v; want rung_asserted", need)
	}
	steps, decision = rental.Plan(market20260907(), []records.ModelRef{asserted}, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "rtx-4090 rtx-5090 h100-80 h100-nvl h200 b200" {
		t.Fatalf("an asserted lane walked %q; want every card in price order", got)
	}
	sized(t, find(t, decision, "rtx-4090"), records.FitRungAsserted, 0, 24)
	if row := find(t, decision, "rtx-4090"); row.Verdict != "" {
		t.Fatalf("an asserted lane was refused a card: %q", row.Verdict)
	}
	if verdict := rental.Fit(records.Residency{Fit: records.FitRungAsserted}, 24, "rtx-4090"); verdict != "" {
		t.Fatalf("the floor overruled the owner's rung: %q", verdict)
	}
	// One slot without bytes makes the whole selection asserted: there is no figure to
	// hold the device to.
	if need := records.Resident([]records.ModelRef{fp8Exact(), asserted}); need.Fit != records.FitRungAsserted {
		t.Fatalf("a half-sized selection was held to %+v", need)
	}
	if need := records.Resident(nil); need.Fit != "" {
		t.Fatalf("an empty selection was sized %+v", need)
	}
}

func TestLadderWithoutCatchAllExcludesUnnamedClasses(t *testing.T) {
	models := h3Ladder()
	models[0].Ladder = models[0].Ladder[:2] // H100 and B200 only
	steps, decision := rental.Plan(market20260907(), models, true, rental.Constraints{})
	if got := strings.Join(stepNames(steps), " "); got != "h100-80 h100-nvl b200" {
		t.Fatalf("walk %q; want h100-80 h100-nvl b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090", "h200"} {
		if row := find(t, decision, name); row.Verdict != orchestrator.VerdictGPUMismatch || row.Rung != 0 || row.Fit != "" {
			t.Fatalf("%s recorded %+v; want %s on no rung, unsized", name, row, orchestrator.VerdictGPUMismatch)
		}
	}
	// A CPU-class request fits only through the catch-all, and has no device to check.
	steps, _ = rental.Plan(market20260907(), models, false, rental.Constraints{})
	if len(steps) != 0 {
		t.Fatalf("a CPU box fits an H100/B200-only ladder: %v", stepNames(steps))
	}
	steps, decision = rental.Plan(market20260907(), h3Ladder(), false, rental.Constraints{})
	if len(steps) != 1 || steps[0].SKU.Name != "cpu" || steps[0].Models[0].Lane != "bf16-full" || find(t, decision, "cpu").Fit != "" {
		t.Fatalf("CPU class walked %v; want cpu alone on the catch-all lane, unsized", stepNames(steps))
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
	models := h3Ladder()
	rung, index, ok := models[0].RungFor("NVIDIA GeForce RTX 5090")
	if !ok || index != 2 || rung.Lane != "bf16-full" {
		t.Fatalf("a 5090 fits rung %d %+v; want the catch-all bf16-full", index, rung)
	}
	pinned, missing := records.PinModels(models, "NVIDIA H100 NVL")
	if missing != "" || !pinned[0].Pinned() || pinned[0].Lane != "fp8-adaln-pruned" || pinned[0].Ladder != nil ||
		pinned[0].ComponentBytes["text_encoder"] != textEncoderNeed || len(pinned[0].ComponentUse) != len(h3ComponentUse()) {
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
	fatal(t, store.PinRequestModels(request, h3Ladder()))
	row, problem := store.RequestRow(request)
	fatal(t, problem)
	if len(row.Models) != 1 || row.Models[0].Pinned() || len(row.Models[0].Ladder) != 3 ||
		row.Models[0].Ladder[0].ComponentBytes["text_encoder"] != textEncoderNeed ||
		row.Models[0].ComponentUse["sample_fl2va"][0] != "fl2va_dit" {
		t.Fatalf("the unpinned ladder and its sizing facts did not survive the row: %+v", row.Models)
	}
	pinned, _ := records.PinModels(h3Ladder(), "NVIDIA H100 80GB HBM3")
	fatal(t, store.RecordRental(records.Rental{ID: "pr-ladder-h100", MachineName: "ladder-h100",
		SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3", HourlyRateUSDMicros: 2_490_000,
		State: "ready", Hub: "http://hub.example"}))
	assigned, problem := store.PinRental(request, "pr-ladder-h100", pinned)
	fatal(t, problem)
	row, problem = store.RequestRow(request)
	fatal(t, problem)
	if !assigned || row.Worker != "pr-ladder-h100" || row.Models[0].Lane != "fp8-adaln-pruned" ||
		row.Models[0].Manifest != fp8Manifest || row.Models[0].Ladder != nil ||
		row.Models[0].ComponentBytes["fl2va_dit"] != 20*gib {
		t.Fatalf("the pin did not carry the rung's lane and bytes: worker=%q models=%+v", row.Worker, row.Models)
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
