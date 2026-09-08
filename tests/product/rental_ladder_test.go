package producttest

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
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
	h100SXM      = "NVIDIA H100 80GB HBM3"
	mxfpManifest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	bf16Manifest = "sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"
	// fp8LaterManifest is the fp8 lane of the sized 1.0.0-rc.2 release (cl-174).
	fp8LaterManifest = "sha256:dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
	// textEncoderNeed is the largest group any H3 method holds resident: the text
	// conditioning's 51.5 GiB encoder. Every DiT sample holds one ~20 GiB DiT.
	textEncoderNeed    = 103 * gib / 2
	textEncoderVerdict = "excluded:vram_short: needs 51.5 GiB resident (condition_text: text_encoder)"
)

// fp8ManifestBody is the fp8 lane's manifest document, and fp8Manifest its real digest: a
// JOB binds EXACT manifest bytes and verifies their digest before it is submitted
// (jobManifestInputs), so the stand-in hub has to be able to serve them.
var (
	fp8ManifestBody = []byte(`{"cozytensors":"proof","lane":"fp8-adaln-pruned"}`)
	fp8Manifest     = mustSpell(fp8ManifestBody)
)

func mustSpell(raw []byte) string {
	id, err := canonical.Spell(canonical.Digest(raw))
	if err != nil {
		panic(err)
	}
	return id
}

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

// walk is the order the choice would buy in if the hub refused every product for stock in
// turn: each pick is marked no_stock and the choice repeats on a copy of the candidates.
func walk(candidates []orchestrator.PlacementCandidate) []string {
	rows := append([]orchestrator.PlacementCandidate(nil), candidates...)
	var out []string
	for i := rental.Place("balanced", rows); i >= 0; i = rental.Place("balanced", rows) {
		out = append(out, rows[i].Name())
		rows[i].Verdict = orchestrator.VerdictNoStock
	}
	return out
}

// sized asserts one audit row says what it compared.
func sized(t *testing.T, row orchestrator.PlacementCandidate, fit string) {
	t.Helper()
	if row.Fit != fit {
		t.Fatalf("%s recorded fit=%q; want %q", row.Name(), row.Fit, fit)
	}
}

func mark(candidates []orchestrator.PlacementCandidate, name, verdict string) {
	for i := range candidates {
		if candidates[i].Name() == name {
			candidates[i].Verdict = verdict
		}
	}
}

func TestLadderWalkNeversBuysWhatTheLaneCannotFit(t *testing.T) {
	// The defect as it happened: the H3 ladder against a market whose only cards were
	// 24 GB and 32 GB. Both land on the catch-all bf16 rung, whose text encoder alone
	// outweighs them: nothing is bought, and the record says the need per card.
	decision := rental.Purchases(market20260907()[:2], h3Ladder(), true, false, rental.Constraints{})
	if bought := walk(decision); len(bought) != 0 {
		t.Fatalf("a lane whose text encoder is 51.5 GiB was given a machine to buy: %v", bought)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		if verdict := find(t, decision, name).Verdict; !strings.HasPrefix(verdict, textEncoderVerdict) {
			t.Fatalf("%s recorded %q; want a %s verdict naming the resident need", name, verdict, orchestrator.ExcludedVRAMShort)
		}
	}

	// The whole market: the walk is the OWNER'S rung order, cheapest fitting product first
	// within a rung, and price never lifts a later rung over an earlier one — the $5.99
	// B200 (rung 2) precedes the $3.59 H200 (rung 3). The 103 GB fp8 lane fits both H100s
	// because no method holds more than its 51.5 GiB text encoder (the production case).
	decision = rental.Purchases(market20260907(), h3Ladder(), true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h100-80 h100-nvl b200 h200" {
		t.Fatalf("walk order %q; want h100-80 h100-nvl b200 h200", got)
	}
	if h100 := find(t, decision, "h100-80"); h100.Rung != 1 || h100.Lane != "fp8-adaln-pruned" || h100.Models[0].Manifest != fp8Manifest {
		t.Fatalf("the H100 candidate pins %+v; want rung 1 fp8-adaln-pruned", h100)
	}
	if find(t, decision, "b200").Lane != "mxfp8-adaln-pruned" || find(t, decision, "h200").Lane != "bf16-full" {
		t.Fatalf("B200 and H200 pin %q and %q; want mxfp8-adaln-pruned and bf16-full",
			find(t, decision, "b200").Lane, find(t, decision, "h200").Lane)
	}
	sized(t, find(t, decision, "h100-80"), "components 51.5 GiB of 80 GB")
	sized(t, find(t, decision, "h100-nvl"), "components 51.5 GiB of 94 GB")
	// The mxfp8 lane publishes no component bytes: the owner's B200 rung is the fit, and
	// the record says so rather than pretending a figure was compared.
	sized(t, find(t, decision, "b200"), records.FitRungAsserted)
	sized(t, find(t, decision, "h200"), "components 51.5 GiB of 141 GB")
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if row.Rung != 3 || row.Lane != "bf16-full" || !strings.HasPrefix(row.Verdict, textEncoderVerdict) {
			t.Fatalf("%s recorded %+v; want rung 3 bf16-full %s", name, row, orchestrator.ExcludedVRAMShort)
		}
	}
	sized(t, find(t, decision, "rtx-4090"), "components 51.5 GiB of 24 GB")
	absent(t, decision, "cpu")
	if ladder := rental.Ladder(h3Ladder()); len(ladder) != 1 || ladder[0] != "H100=fp8-adaln-pruned > B200=mxfp8-adaln-pruned > *=bf16-full" {
		t.Fatalf("the record does not carry the ladder it walked: %v", ladder)
	}

	// The hub refuses the first H100 for inventory; the choice repeats within the rung
	// and the record explains every row — no_stock, chosen, unmeasured.
	mark(decision, "h100-80", orchestrator.VerdictNoStock)
	rental.Conclude(decision, rental.Place("balanced", decision))
	want := map[string]string{"h100-80": orchestrator.VerdictNoStock, "h100-nvl": orchestrator.VerdictChosen,
		"b200": orchestrator.VerdictUnmeasured, "h200": orchestrator.VerdictUnmeasured}
	for name, verdict := range want {
		if got := find(t, decision, name).Verdict; got != verdict {
			t.Fatalf("%s recorded verdict %q; want %q", name, got, verdict)
		}
	}
	if record := (orchestrator.PlacementDecision{Candidates: decision}); record.Unexplained() != "" {
		t.Fatalf("unexplained=%q; want every row explained", record.Unexplained())
	}
}

func TestExplicitLaneIsHeldToTheSameFloor(t *testing.T) {
	// `model.<param>=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` pins the 103 GB lane; the
	// buy takes the cheapest card that holds its largest resident group — the 80 GB H100
	// — and refuses the 24 GB and 32 GB cards naming the 51.5 GiB text encoder.
	decision := rental.Purchases(market20260907(), []records.ModelRef{fp8Exact()}, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h100-80 h100-nvl h200 b200" {
		t.Fatalf("an explicit fp8 lane may buy %q; want h100-80 h100-nvl h200 b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if !strings.HasPrefix(row.Verdict, textEncoderVerdict) || !strings.HasSuffix(row.Verdict, name+" has "+
			map[string]string{"rtx-4090": "24", "rtx-5090": "32"}[name]+" GB") {
			t.Fatalf("%s recorded %q; want the need and the card's memory", name, row.Verdict)
		}
	}
	sized(t, find(t, decision, "rtx-4090"), "components 51.5 GiB of 24 GB")
	sized(t, find(t, decision, "h100-80"), "components 51.5 GiB of 80 GB")
	rental.Conclude(decision, rental.Place("balanced", decision))
	if find(t, decision, "h100-80").Verdict != orchestrator.VerdictChosen || find(t, decision, "h100-nvl").Verdict != orchestrator.VerdictUnmeasured {
		t.Fatalf("h100-80 %q, h100-nvl %q; want h100-80 chosen with h100-nvl unmeasured",
			find(t, decision, "h100-80").Verdict, find(t, decision, "h100-nvl").Verdict)
	}
}

func TestComponentFitHoldsTheOwnersRung(t *testing.T) {
	// A slot without component_use holds its largest single component.
	plain := fp8Exact()
	plain.ComponentUse = nil
	need := records.Resident([]records.ModelRef{plain}, h100SXM, false)
	if need.Fit != records.FitComponents || need.Bytes != textEncoderNeed || need.Need != "text_encoder" {
		t.Fatalf("no component_use sized %+v; want the 51.5 GiB text encoder", need)
	}
	// A group is the SUM of the components its method stages, and the largest group wins.
	pair := fp8Exact()
	pair.ComponentUse = map[string][]string{"sample_pair": {"fl2va_dit", "ref2va_dit"}, "decode": {"audio_vae"}}
	if need := records.Resident([]records.ModelRef{pair}, h100SXM, false); need.Bytes != 40*gib || need.Need != "sample_pair: fl2va_dit+ref2va_dit" {
		t.Fatalf("a two-DiT group sized %+v; want 40 GiB", need)
	}
	// A placement holds every slot at once: two H3 slots need two text encoders.
	two := []records.ModelRef{fp8Exact(), fp8Exact()}
	two[1].Slot = "generate.models.refiner"
	if need := records.Resident(two, h100SXM, false); need.Bytes != 103*gib || need.Fit != records.FitComponents ||
		need.Need != "condition_text: text_encoder; condition_text: text_encoder" {
		t.Fatalf("two slots sized %+v; want 103 GiB", need)
	}
	decision := rental.Purchases(market20260907(), two, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h200 b200" {
		t.Fatalf("two text encoders may buy %q; want h200 b200", got)
	}
	if verdict := find(t, decision, "h100-80").Verdict; !strings.HasPrefix(verdict, "excluded:vram_short: needs 103.0 GiB resident (condition_text: text_encoder; condition_text: text_encoder), h100-80 has 80 GB") {
		t.Fatalf("h100-80 recorded %q; want both slots' needs", verdict)
	}

	// No component bytes on the card and no ladder: the whole lane is the need (cl-170).
	// Nothing asserts a 103 GB lane onto a 24 GB card.
	unsized := fp8Exact()
	unsized.ComponentBytes = nil
	if need := records.Resident([]records.ModelRef{unsized}, h100SXM, false); need.Fit != records.FitLaneBytes ||
		need.Bytes != 103*gib || need.Need != "lane fp8-adaln-pruned" {
		t.Fatalf("a lane without component bytes sized %+v; want its whole bytes", need)
	}
	decision = rental.Purchases(market20260907(), []records.ModelRef{unsized}, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h200 b200" {
		t.Fatalf("an unsized lane walked %q; want only the cards that hold 103 GiB whole", got)
	}
	sized(t, find(t, decision, "rtx-4090"), "lane_bytes 103.0 GiB of 24 GB")
	// A rung-asserted slot needs no figure, and the floor never overrules it.
	if verdict := rental.Fit(records.Residency{Fit: records.FitRungAsserted}, 24, "rtx-4090"); verdict != "" {
		t.Fatalf("the floor overruled the owner's rung: %q", verdict)
	}
	// Mixed slots sum what they can and name every rule.
	mixed := []records.ModelRef{fp8Exact(), unsized}
	mixed[1].Slot = "generate.models.refiner"
	if need := records.Resident(mixed, h100SXM, false); need.Fit != "components+lane_bytes" || need.Bytes != textEncoderNeed+103*gib ||
		need.Need != "condition_text: text_encoder; lane fp8-adaln-pruned" {
		t.Fatalf("mixed slots sized %+v", need)
	}
	if need := records.Resident(nil, h100SXM, false); need.Fit != "" || need.Bytes != 0 {
		t.Fatalf("an empty selection was sized %+v", need)
	}
}

// cl-170. On 2026-09-07 `model.model=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` on a card
// publishing no component bytes read as "rung 1, rung_asserted" on EVERY product: the
// override had become a one-lane ladder, and a 24 GB RTX 4090 was bought for the 103 GB
// lane. An explicit lane is not a rung. Under it a card fits by the components when the
// card sizes them; else by the owner's rung naming this card AND this lane; else by the
// whole lane's bytes, conservatively.
func TestExplicitLaneIsNotARung(t *testing.T) {
	explicit := fp8Exact()
	explicit.ComponentBytes = nil
	explicit.Ladder = h3Ladder()[0].Ladder
	models := []records.ModelRef{explicit}
	decision := rental.Purchases(market20260907(), models, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h100-80 h100-nvl h200 b200" {
		t.Fatalf("an explicit unsized fp8 lane may buy %q; want h100-80 h100-nvl h200 b200", got)
	}
	// The H100 rung names fp8: asserted, no figure compared. The B200 rung names mxfp8 and
	// the catch-all bf16: fp8 is off the ladder there, so its 103 GiB are held to the card.
	sized(t, find(t, decision, "h100-80"), records.FitRungAsserted)
	sized(t, find(t, decision, "h100-nvl"), records.FitRungAsserted)
	sized(t, find(t, decision, "b200"), "lane_bytes 103.0 GiB of 180 GB")
	sized(t, find(t, decision, "h200"), "lane_bytes 103.0 GiB of 141 GB")
	sized(t, find(t, decision, "rtx-4090"), "lane_bytes 103.0 GiB of 24 GB")
	for _, name := range []string{"rtx-4090", "rtx-5090"} {
		row := find(t, decision, name)
		if row.Rung != 0 || !strings.HasPrefix(row.Verdict, "excluded:vram_short: needs 103.0 GiB resident (lane fp8-adaln-pruned), "+name+" has ") {
			t.Fatalf("%s recorded %+v; want no rung and the whole lane's need", name, row)
		}
	}
	if h100 := find(t, decision, "h100-80"); h100.Rung != 0 || h100.Lane != "fp8-adaln-pruned" || len(h100.Models[0].Ladder) != 3 {
		t.Fatalf("the H100 candidate pins %+v; want no rung, the lane, the ladder kept", h100)
	}
	// The record says which lane was pinned and shows the real ladder, never one made of it.
	if ladder := rental.Ladder(models); len(ladder) != 1 || ladder[0] != "H100=fp8-adaln-pruned > B200=mxfp8-adaln-pruned > *=bf16-full" {
		t.Fatalf("the ladder rendered %v", ladder)
	}
	if rental.Override(models) != "fp8-adaln-pruned" || rental.Override(h3Ladder()) != "" {
		t.Fatalf("override rendered %q and %q", rental.Override(models), rental.Override(h3Ladder()))
	}
	rental.Conclude(decision, rental.Place("balanced", decision))
	if line := (orchestrator.PlacementDecision{Tier: "balanced", Candidates: decision}).Line(); line != "placement: buy h100-80 (fp8-adaln-pruned) — balanced, unmeasured" {
		t.Fatalf("an override names a rung: %q", line)
	}
	// A rung naming the card and the pinned lane asserts it wherever it sits: the
	// catch-all's bf16 on an H100.
	bf16 := explicit
	bf16.Lane, bf16.Manifest, bf16.Bytes = "bf16-full", bf16Manifest, 130*gib
	if need := records.Resident([]records.ModelRef{bf16}, h100SXM, false); need.Fit != records.FitRungAsserted || need.Bytes != 0 {
		t.Fatalf("the catch-all did not assert bf16 on an H100: %+v", need)
	}
	// No ladder bound at all: only the components or the whole lane size a card.
	explicit.Ladder = nil
	decision = rental.Purchases(market20260907(), []records.ModelRef{explicit}, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h200 b200" {
		t.Fatalf("an unsized lane with no ladder may buy %q; want h200 b200 alone", got)
	}
	if verdict := find(t, decision, "h100-80").Verdict; !strings.HasPrefix(verdict, "excluded:vram_short: needs 103.0 GiB resident (lane fp8-adaln-pruned), h100-80 has 80 GB") {
		t.Fatalf("with no ladder the H100 recorded %q", verdict)
	}
	sized(t, find(t, decision, "h100-80"), "lane_bytes 103.0 GiB of 80 GB")
	if ladder := rental.Ladder([]records.ModelRef{explicit}); ladder != nil {
		t.Fatalf("no ladder rendered as %v", ladder)
	}
	// The card's own figures beat both rules: a sized override is held to its components (cl-168).
	if need := records.Resident([]records.ModelRef{fp8Exact()}, "NVIDIA GeForce RTX 4090", false); need.Fit != records.FitComponents || need.Bytes != textEncoderNeed {
		t.Fatalf("a sized override was not held to its components: %+v", need)
	}
}

func TestLadderWithoutCatchAllExcludesUnnamedClasses(t *testing.T) {
	models := h3Ladder()
	models[0].Ladder = models[0].Ladder[:2] // H100 and B200 only
	decision := rental.Purchases(market20260907(), models, true, false, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h100-80 h100-nvl b200" {
		t.Fatalf("walk %q; want h100-80 h100-nvl b200", got)
	}
	for _, name := range []string{"rtx-4090", "rtx-5090", "h200"} {
		if row := find(t, decision, name); row.Verdict != orchestrator.VerdictNoRung || row.Rung != 0 || row.Fit != "" {
			t.Fatalf("%s recorded %+v; want %s on no rung, unsized", name, row, orchestrator.VerdictNoRung)
		}
	}
	// A CPU-class request fits only through the catch-all, and has no device to check.
	if bought := walk(rental.Purchases(market20260907(), models, false, false, rental.Constraints{})); len(bought) != 0 {
		t.Fatalf("a CPU box fits an H100/B200-only ladder: %v", bought)
	}
	decision = rental.Purchases(market20260907(), h3Ladder(), false, false, rental.Constraints{})
	if got := walk(decision); len(got) != 1 || got[0] != "cpu" || find(t, decision, "cpu").Lane != "bf16-full" || find(t, decision, "cpu").Fit != "" {
		t.Fatalf("CPU class walked %v; want cpu alone on the catch-all lane, unsized", got)
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
	pinned, first, fits := rental.Pin(models, "NVIDIA H100 NVL")
	if !fits || first != 1 || !pinned[0].Pinned() || pinned[0].Lane != "fp8-adaln-pruned" || len(pinned[0].Ladder) != 3 ||
		pinned[0].ComponentBytes["text_encoder"] != textEncoderNeed || len(pinned[0].ComponentUse) != len(h3ComponentUse()) {
		t.Fatalf("pinning to an H100 NVL gave %+v (rung %d, fits %t)", pinned, first, fits)
	}
	models[0].Ladder = models[0].Ladder[:1]
	if _, _, fits := rental.Pin(models, "NVIDIA H200"); fits {
		t.Fatal("an H200 fits an H100-only ladder")
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
	pinned, _, _ := rental.Pin(h3Ladder(), "NVIDIA H100 80GB HBM3")
	fatal(t, store.RecordRental(records.Rental{ID: "pr-ladder-h100", MachineName: "ladder-h100",
		SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3", HourlyRateUSDMicros: 2_490_000,
		State: "ready", Hub: "http://hub.example"}))
	assigned, problem := store.PinRental(request, "pr-ladder-h100", pinned)
	fatal(t, problem)
	row, problem = store.RequestRow(request)
	fatal(t, problem)
	if !assigned || row.Worker != "pr-ladder-h100" || row.Models[0].Lane != "fp8-adaln-pruned" ||
		row.Models[0].Manifest != fp8Manifest || len(row.Models[0].Ladder) != 3 ||
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

// cl-180. `attention-lane` (se-037) reads its source's HEADER, inherits every tensor by
// reference and adds 550 bytes, and was sized "fit components 48.0 GiB of 48 GB" — the
// components a SERVING construction of that lane would stage. cozy-runtime hands a JOB a
// derive-only view of the Manifest and refuses load and component access on it, so no
// component of a job's model is ever on the device, whatever the closure weighs. The
// figure is dropped, never lowered: the same selection under a serving request is still
// held to its 51.5 GiB text encoder.
func TestAJobsModelIsNeverResident(t *testing.T) {
	models := []records.ModelRef{fp8Exact()}
	if need := records.Resident(models, h100SXM, true); need.Fit != records.FitDeriveOnly ||
		need.Bytes != 0 || need.Need != "" {
		t.Fatalf("a job's model sized %+v; want derive_only with no figure", need)
	}
	if need := records.Resident(models, h100SXM, false); need.Fit != records.FitComponents || need.Bytes != textEncoderNeed {
		t.Fatalf("the same selection served sized %+v; want its 51.5 GiB text encoder", need)
	}
	// The whole market is admissible for the job and the cheapest card wins; the serving
	// request still refuses both small cards naming the need.
	job := rental.Purchases(market20260907(), models, true, true, rental.Constraints{})
	if got := strings.Join(walk(job), " "); got != "rtx-4090 rtx-5090 h100-80 h100-nvl h200 b200" {
		t.Fatalf("a job walked %q; want every card, cheapest first", got)
	}
	sized(t, find(t, job, "rtx-4090"), records.FitDeriveOnly)
	serving := rental.Purchases(market20260907(), models, true, false, rental.Constraints{})
	if got := strings.Join(walk(serving), " "); got != "h100-80 h100-nvl h200 b200" {
		t.Fatalf("serving walked %q; want the cards that hold the text encoder", got)
	}

	// A job's ladder still QUALIFIES the machine: a rung nobody names is still no_rung,
	// and the rung it lands on still pins the lane the job reads.
	unpinned := h3Ladder()
	unpinned[0].Ladder = unpinned[0].Ladder[:1] // H100 only
	decision := rental.Purchases(market20260907(), unpinned, true, true, rental.Constraints{})
	if got := strings.Join(walk(decision), " "); got != "h100-80 h100-nvl" {
		t.Fatalf("a job on an H100-only ladder walked %q; want the H100s alone", got)
	}
	if row := find(t, decision, "rtx-4090"); row.Verdict != orchestrator.VerdictNoRung {
		t.Fatalf("rtx-4090 recorded %q; want %s", row.Verdict, orchestrator.VerdictNoRung)
	}
	if row := find(t, decision, "h100-80"); row.Lane != "fp8-adaln-pruned" || row.Rung != 1 {
		t.Fatalf("a job's rung pinned %+v; want rung 1 fp8-adaln-pruned", row)
	}
	// A whole-lane figure and a rung assertion are equally beside the point for a job.
	unsized := fp8Exact()
	unsized.ComponentBytes = nil
	for _, model := range []records.ModelRef{unsized, fp8Exact()} {
		if need := records.Resident([]records.ModelRef{model}, "NVIDIA GeForce RTX 4090", true); need.Bytes != 0 {
			t.Fatalf("a job's model sized %+v against a 24 GB card", need)
		}
	}
	if need := records.Resident(nil, h100SXM, true); need.Fit != "" || need.Bytes != 0 {
		t.Fatalf("a job binding no model sized %+v", need)
	}
}
