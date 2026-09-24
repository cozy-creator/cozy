package producttest

import (
	"crypto/sha256"
	"encoding/hex"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// cl-165, placement-economics.md. Creator used to buy the cheapest fitting card after
// reusing any ready rental; nothing knew how fast a lane runs on a card, or whether a warm
// pod with work queued beats a cold one. The model's published throughput rows now give
// every candidate — attached rental or purchase — an expected time and cost, and
// `placement.prefer` picks among them. These arms are the design's worked example on the
// real chooser, then the real daemon against a stand-in hub serving the rows.

const fp8Lane = "fp8-adaln-pruned"

// exampleMarket is the worked example's two cards at today's rates including storage:
// H100 SXM $3.52/h, RTX 5090 $0.99/h.
func exampleMarket() []hub.RentalSKU {
	return []hub.RentalSKU{
		{Name: "h100-sxm5-80gb", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 1, VRAMGB: 80, ComputeCapability: "9.0",
			MinimumRAMPerGPUGB: 64, PriceUSDMicrosPerHour: 3_306_496, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
		{Name: "rtx-5090", AcceleratorModel: "NVIDIA GeForce RTX 5090", AcceleratorCount: 1, VRAMGB: 32, ComputeCapability: "12.0",
			MinimumRAMPerGPUGB: 64, PriceUSDMicrosPerHour: 776_496, StorageUSDMicrosPerHour: 213_504,
			BaseWorkerProfile: "torch2.13.0-cu130-cp312-linux-x86"},
	}
}

// exampleLadder runs the fp8 lane on the H100 rung and, staged, on anything else; the card
// sizes no components, so the owner's rung is the fit.
func exampleLadder() []records.ModelRef {
	return []records.ModelRef{{Package: "paul/minimax-h3", Slot: "generate.models.model",
		Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Ladder: []records.ModelRung{
			{GPU: "H100", Lane: fp8Lane, Manifest: fp8Manifest, Bytes: 103 * gib},
			{GPU: "*", Lane: fp8Lane, Manifest: fp8Manifest, Bytes: 103 * gib}}}}
}

// exampleRows is the measured H100 (33 s/step × 29 + 40 s ≈ 1016 s) and the illustrative
// 5090, both with a 390 s prepare.
func exampleRows() []hub.ModelThroughput {
	return []hub.ModelThroughput{
		{Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: fp8Lane, SKU: "h100-sxm5-80gb", Runs: 3, MedianS: 1016, MinS: 1000, MaxS: 1030, PrepareS: 390},
		{Model: "paul/minimax-h3", Release: "1.0.0-rc.1", Lane: fp8Lane, SKU: "rtx-5090", Runs: 3, MedianS: 4200, MinS: 4100, MaxS: 4300, PrepareS: 390},
	}
}

// exampleCandidates is the worked example's table: buy the H100, buy the 5090, or wait on
// the attached H100 `morgiana` with `ahead` attempts queued before this one.
func exampleCandidates(t *testing.T, ahead int) []orchestrator.PlacementCandidate {
	t.Helper()
	pinned, rung, ok := rental.Pin(exampleLadder(), "NVIDIA H100 80GB HBM3")
	if !ok || rung != 1 {
		t.Fatalf("the H100 does not fit rung 1: %v %d", pinned, rung)
	}
	attached := orchestrator.PlacementCandidate{Rental: "pr-morgiana", Machine: "morgiana", SKU: "h100-sxm5-80gb",
		Rung: rung, Lane: fp8Lane, Ahead: ahead, RateUSDMicrosPerHour: 3_520_000, Models: pinned}
	candidates := append([]orchestrator.PlacementCandidate{attached},
		rental.Purchases(exampleMarket(), exampleLadder(), true, false, rental.Constraints{})...)
	if used := rental.Measure(candidates, exampleRows(), exampleLadder()); len(used) != 2 {
		t.Fatalf("the two rows were not both used: %+v", used)
	}
	return candidates
}

func near(a, b float64) bool { return math.Abs(a-b) < 0.01 }

func TestPlacementWorkedExample(t *testing.T) {
	candidates := exampleCandidates(t, 1)
	// The table: 1016 + 390 = 1406 s at $1.37; 4200 + 390 = 4590 s at $1.26; 1016 + 1016 =
	// 2032 s at $0.99 (only the run bills to this request).
	for name, want := range map[string]struct {
		seconds float64
		cost    int64
	}{"h100-sxm5-80gb": {1406, 1_374_756}, "rtx-5090": {4590, 1_262_250}, "morgiana": {2032, 993_422}} {
		if row := find(t, candidates, name); !row.Measured || row.TimeS != want.seconds || row.CostUSDMicros != want.cost {
			t.Fatalf("%s measured %+v; want %.0f s at %d micros", name, row, want.seconds, want.cost)
		}
	}
	// fast buys the H100 (1406 s); cheap waits on the attached H100 ($0.99); balanced buys
	// the H100 (1.38 < 1.45). The 5090 is worse on both axes and wins nothing.
	for tier, want := range map[string]string{"fast": "h100-sxm5-80gb", "cheap": "morgiana", "balanced": "h100-sxm5-80gb"} {
		rows := append([]orchestrator.PlacementCandidate(nil), candidates...)
		i := rental.Place(tier, rows)
		if i < 0 || rows[i].Name() != want {
			t.Fatalf("%s placed on %q; want %s", tier, rows[i].Name(), want)
		}
		rental.Conclude(rows, i)
		if !near(find(t, rows, "h100-sxm5-80gb").Score, 1.38) || !near(find(t, rows, "rtx-5090").Score, 4.15) ||
			!near(find(t, rows, "morgiana").Score, 1.45) {
			t.Fatalf("%s scores %+v; want 1.38, 4.15, 1.45", tier, rows)
		}
		record := orchestrator.PlacementDecision{Tier: tier, Candidates: rows}
		if record.Unexplained() != "" || find(t, rows, "rtx-5090").Verdict != orchestrator.VerdictSlower {
			t.Fatalf("%s left %q unexplained, 5090 %q", tier, record.Unexplained(), find(t, rows, "rtx-5090").Verdict)
		}
		loser, verdict := "morgiana", orchestrator.VerdictSlower
		if tier == "cheap" {
			loser, verdict = "h100-sxm5-80gb", orchestrator.VerdictDearer
		}
		if find(t, rows, loser).Verdict != verdict {
			t.Fatalf("%s recorded %s as %q; want %s", tier, loser, find(t, rows, loser).Verdict, verdict)
		}
		if line := record.Line(); tier == "cheap" && line != "placement: reuse morgiana (h100-sxm5-80gb, fp8-adaln-pruned) — cheap, 2032 s, $0.99" ||
			tier == "balanced" && line != "placement: buy h100-sxm5-80gb (fp8-adaln-pruned) — balanced, 1406 s, $1.37" {
			t.Fatalf("%s line %q", tier, line)
		}
	}
	// With nothing ahead the attached H100 scores 1.00 and wins every tier.
	candidates = exampleCandidates(t, 0)
	for _, tier := range []string{"fast", "cheap", "balanced"} {
		rows := append([]orchestrator.PlacementCandidate(nil), candidates...)
		i := rental.Place(tier, rows)
		if i < 0 || rows[i].Name() != "morgiana" || rows[i].TimeS != 1016 || rows[i].Score != 1 {
			t.Fatalf("%s with nothing ahead placed on %+v; want morgiana at 1016 s, score 1", tier, rows[i])
		}
	}
}

func TestPlacementUnmeasuredFallsToTheLadder(t *testing.T) {
	// No row anywhere: an attached rental on a LATER rung still beats every buy — today's
	// reuse-first behaviour — and the buys fall to rung order, cheapest within a rung.
	candidates := exampleCandidates(t, 3)
	for i := range candidates {
		candidates[i].Measured, candidates[i].TimeS, candidates[i].CostUSDMicros = false, 0, 0
	}
	candidates[0].Rung = 2
	if got := walk(candidates); strings.Join(got, " ") != "morgiana h100-sxm5-80gb rtx-5090" {
		t.Fatalf("unmeasured order %v; want the attached rental first, then rung order", got)
	}
	// One measured buy beside an unmeasured attached rental: measured wins.
	candidates = exampleCandidates(t, 0)
	candidates[0].Measured = false
	if i := rental.Place("cheap", candidates); candidates[i].Name() != "rtx-5090" {
		t.Fatalf("cheap placed on %q with only buys measured; want the cheaper measured buy", candidates[i].Name())
	}
}

// placementRoot is ladderRoot with the tier set, returning the config's sha256 too.
func placementRoot(t *testing.T, h *ladderHub, prefer string) (string, string) {
	t.Helper()
	root := t.TempDir()
	body := []byte("tensorhub_url: " + h.server.URL + "\ntensorhub_token: ladder-test\nrentals:\n  max_hourly_spend_usd: 20\n" +
		"daemon:\n  idle_shutdown_s: 0\nplacement:\n  prefer: " + prefer + "\n")
	must(t, os.WriteFile(filepath.Join(root, config.FileName), body, 0600))
	sum := sha256.Sum256(body)
	return root, hex.EncodeToString(sum[:])
}

// proofRows measures the fp8 lane on both H100 products and the bf16 lane on the H200.
func proofRows() []hub.ModelThroughput {
	row := func(lane, sku string, median float64) hub.ModelThroughput {
		return hub.ModelThroughput{Model: ladderModel, Release: "1.0.0-rc.1", Lane: lane, SKU: sku,
			Runs: 3, MedianS: median, MinS: median, MaxS: median, PrepareS: 390}
	}
	return []hub.ModelThroughput{row(fp8Lane, "h100-80", 1016), row(fp8Lane, "h100-nvl", 1016), row("bf16-full", "h200", 2000)}
}

func TestPlacementReusesTheMeasuredCard(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.throughput = proofRows()
	root, digest := placementRoot(t, h, "balanced")
	cert := filepath.Join(root, "morgiana.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	seed := records.Rental{AcceleratorCount: 1, ID: "pr-morgiana", MachineName: "morgiana", SKU: "h100-80", AcceleratorModel: "NVIDIA H100 80GB HBM3",
		HourlyRateUSDMicros: 2_490_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL}
	fatal(t, store.RecordRental(seed))
	h.addReady(seed.ID, seed.MachineName, seed.AcceleratorModel, seed.HourlyRateUSDMicros)
	store.Close()
	startDaemonProcess(t, root)
	// The attached H100 with nothing ahead: 1016 s at $2.70/h × 1016 s = $0.76, against a
	// bought H100 at 1406 s. It wins, and the run says so in one line.
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--idempotency-key", "placement-reuse")
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("placement-reuse")
	fatal(t, problem)
	if row == nil || row.Worker != "pr-morgiana" || row.Models[0].Lane != fp8Lane {
		t.Fatalf("the run was not placed on morgiana: %+v\n%s", row, out)
	}
	const line = "placement: reuse morgiana (h100-80, fp8-adaln-pruned) — balanced, 1016 s, $0.76"
	if !strings.Contains(out, line) {
		t.Fatalf("cozy run did not print %q:\n%s\nDaemon log:\n%s", line, out, tail(filepath.Join(root, "daemon.log")))
	}
	placement := placementEvent(t, store, row.ID)
	if placement["tier"] != "balanced" || placement["config_digest"] != digest || placement["line"] != line ||
		placement["rental"] != "pr-morgiana" || placement["bought"] != false {
		t.Fatalf("the record does not cite the tier, config and choice: %v", placement)
	}
	// Every row that measured a candidate is cited once: the H100 row serves both the
	// attached morgiana and the buy.
	used := placement["throughput"].([]any)
	if len(used) != 3 || used[0].(map[string]any)["sku"] != "h100-80" || used[0].(map[string]any)["median_s"] != 1016.0 {
		t.Fatalf("the record does not cite the rows used: %v", used)
	}
	verdicts := candidateVerdicts(placement)
	if verdicts["morgiana"] != "chosen" || verdicts["h100-80"] != "slower" || verdicts["h100-nvl"] != "slower" ||
		verdicts["h200"] != "slower" || verdicts["b200"] != "unmeasured" || !strings.HasPrefix(verdicts["rtx-4090"], "excluded:vram_short") {
		t.Fatalf("verdicts %v", verdicts)
	}
	for _, raw := range placement["candidates"].([]any) {
		c := raw.(map[string]any)
		if c["machine"] == "morgiana" && (c["time_s"] != 1016.0 || c["cost_usd_micros"] != 762_989.0 ||
			c["rate_usd_micros_per_hour"] != 2_703_504.0 || c["score"] != 1.0 || c["ahead"] != nil) {
			t.Fatalf("the chosen candidate's figures: %v", c)
		}
		if c["sku"] == "h100-80" && c["machine"] == nil && (c["time_s"] != 1406.0 || c["cost_usd_micros"] != 1_055_869.0) {
			t.Fatalf("the H100 buy's figures: %v", c)
		}
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a measured live rental was passed over for a buy: %v", asks)
	}
}

func TestPlacementBuysAroundAStockOut(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.throughput = proofRows()
	h.provisions = true
	h.soldOut["h100-80"] = true
	root, _ := placementRoot(t, h, "fast")
	startDaemonProcess(t, root)
	// fast: the two H100s tie at 1406 s and the cheaper is asked first; the hub has none,
	// so it is dropped and the choice repeats onto the H100 NVL over the slower H200.
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--idempotency-key", "placement-stock")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("placement-stock")
	fatal(t, problem)
	if row == nil || row.Worker != "pr-ladder-h100-nvl" || row.Models[0].Lane != fp8Lane {
		t.Fatalf("the run was not placed on the bought H100 NVL: %+v\n%s", row, out)
	}
	if got := strings.Join(h.postedSKUs(), " "); got != "h100-80/fp8-adaln-pruned h100-nvl/fp8-adaln-pruned" {
		t.Fatalf("paid asks %q; want the cheaper H100 refused, then the NVL", got)
	}
	const line = "placement: buy h100-nvl (fp8-adaln-pruned) — fast, 1406 s, $1.17"
	if !strings.Contains(out, line) {
		t.Fatalf("cozy run did not print %q:\n%s", line, out)
	}
	placement := placementEvent(t, store, row.ID)
	verdicts := candidateVerdicts(placement)
	if placement["bought"] != true || verdicts["h100-80"] != "no_stock" || verdicts["h100-nvl"] != "chosen" ||
		verdicts["h200"] != "slower" || verdicts["b200"] != "unmeasured" {
		t.Fatalf("verdicts %v", verdicts)
	}
}

func TestPlacementPreferIsValidatedAtLoad(t *testing.T) {
	h := newLadderHub(t)
	root, _ := placementRoot(t, h, "fastest")
	code, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only")
	if code == 0 || !strings.Contains(out, `placement.prefer "fastest" is not fast, balanced or cheap`) {
		t.Fatalf("an unknown tier was admitted [exit %d]: %s", code, out)
	}
}
