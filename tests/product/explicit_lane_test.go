package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// cl-170, the run that filed it (2026-09-07 22:27Z, req-91ca03bbf302e11e7ecb1ae9). An explicit
// `model.model=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` on a card publishing no component
// bytes made EVERY product "rung 1, rung_asserted" — the override had become a one-lane
// ladder — and the user's ready H100 was `excluded:unattached` for the seconds before its
// worker's control stream opened, so a 24 GB RTX 4090 was bought for the 103 GB lane and
// refused on the pod (device_shortfall, short by 35.8 GB). The next request reused another
// 4090 over the attached H100 on the rung tie. These arms drive the real binary and daemon
// against the stand-in hub with the card's lanes unsized, as the card stood.

const (
	explicitFP8 = "model.model=proof/minimax@1.0.0-rc.1/fp8-adaln-pruned"
	fourKShort  = "excluded:vram_short: needs 103.0 GiB resident (lane fp8-adaln-pruned), rtx-4090 has 24 GB"
)

// placementEvents is every durable `request.placement` record of the request, oldest first.
func placementEvents(t *testing.T, store *records.Store, requestID string) []map[string]any {
	t.Helper()
	events, problem := store.EventsAfter(requestID, 0, 200)
	fatal(t, problem)
	var out []map[string]any
	for _, event := range events {
		if event.Type == "request.placement" {
			out = append(out, event.Payload)
		}
	}
	return out
}

// candidateFits reads the record's candidates as name -> fit.
func candidateFits(placement map[string]any) map[string]string {
	out := map[string]string{}
	for _, raw := range placement["candidates"].([]any) {
		c := raw.(map[string]any)
		name, _ := c["machine"].(string)
		if name == "" {
			name, _ = c["sku"].(string)
		}
		out[name], _ = c["fit"].(string)
	}
	return out
}

func TestExplicitLaneWaitsForTheAttachingRentalAndNeverBuysAround(t *testing.T) {
	h := newLadderHub(t)
	h.unsized = true
	h.bind(goodLadder())
	root := ladderRoot(t, h)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	// The user's H100, ready seconds ago and not yet attached: no address, no certificate.
	yumichika := records.Rental{AcceleratorCount: 1, ID: "pr-yumichika", MachineName: "yumichika", SKU: "h100-80",
		AcceleratorModel: h100SXM, HourlyRateUSDMicros: 2_490_000, State: "ready", Hub: h.server.URL}
	fatal(t, store.RecordRental(yumichika))
	h.addReady(yumichika.ID, yumichika.MachineName, yumichika.AcceleratorModel, yumichika.HourlyRateUSDMicros)
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", explicitFP8, "--rental-only", "--json",
		"--idempotency-key", "explicit-wait")
	row, problem := store.RequestByIdempotencyKey("explicit-wait")
	fatal(t, problem)
	if row == nil {
		t.Fatalf("the run was not submitted: %s", out)
	}
	waitFor(t, root, "the placement waiting on the H100", func() bool {
		return len(placementEvents(t, store, row.ID)) > 0
	})
	placement := placementEvent(t, store, row.ID)
	verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
	if verdicts["yumichika"] != "attaching" || verdicts["rtx-4090"] != fourKShort ||
		verdicts["h100-80"] != "excluded:attaching: yumichika" || verdicts["b200"] != "excluded:attaching: yumichika" {
		t.Fatalf("verdicts %v", verdicts)
	}
	if fits["yumichika"] != "rung_asserted" || fits["rtx-4090"] != "lane_bytes 103.0 GiB of 24 GB" ||
		fits["b200"] != "lane_bytes 103.0 GiB of 180 GB" {
		t.Fatalf("fits %v", fits)
	}
	if placement["override"] != "fp8-adaln-pruned" || placement["rental"] != nil || placement["bought"] != false ||
		placement["line"] != "placement: wait for yumichika (h100-80, fp8-adaln-pruned) to attach — balanced" {
		t.Fatalf("the wait record: %v", placement)
	}
	if ladder, _ := placement["ladder"].([]any); len(ladder) != 1 || ladder[0] != "H100=fp8-adaln-pruned > B200=mxfp8-adaln-pruned > *=bf16-full" {
		t.Fatalf("the record does not carry the owner's ladder: %v", placement["ladder"])
	}
	for _, raw := range placement["candidates"].([]any) {
		if c := raw.(map[string]any); c["rung"] != nil {
			t.Fatalf("a candidate under an override carries a rung: %v", c)
		}
	}
	if queued, problem := store.RequestByIdempotencyKey("explicit-wait"); problem != nil || queued.Worker != "" || settled(queued.State) {
		t.Fatalf("the waiting request is %+v; want queued and unpinned", queued)
	}
	// The worker attaches — address and certificate land on the row — and the fleet's next
	// observation pins the waiting request to the H100. The daemon's own reconcile can
	// rewrite the row from a read taken before the attach, so it is re-asserted until seen.
	cert := filepath.Join(root, "yumichika.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	attached := yumichika
	attached.Address, attached.CertPath = "127.0.0.1:1", cert
	fatal(t, store.RecordRental(attached))
	var placements []map[string]any
	waitFor(t, root, "the request pinned to the attached H100 and the reuse recorded", func() bool {
		if current, problem := store.RentalRow(yumichika.ID); problem == nil && current != nil && current.CertPath == "" {
			_ = store.RecordRental(attached)
		}
		placements = placementEvents(t, store, row.ID)
		current, problem := store.RequestByIdempotencyKey("explicit-wait")
		return problem == nil && current != nil && current.Worker == yumichika.ID &&
			candidateVerdicts(placements[len(placements)-1])["yumichika"] == "chosen"
	})
	pinned, problem := store.RequestByIdempotencyKey("explicit-wait")
	fatal(t, problem)
	if pinned.Models[0].Lane != "fp8-adaln-pruned" || pinned.Models[0].Manifest != fp8Manifest {
		t.Fatalf("pinned %+v; want the explicit fp8 lane", pinned.Models[0])
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a fitting rental was bought around: %v", asks)
	}
	if last := candidateVerdicts(placements[len(placements)-1]); len(placements) < 2 || last["rtx-4090"] != fourKShort {
		t.Fatalf("%d placement record(s), last %v; want the wait and then the reuse", len(placements), last)
	}
	log, err := os.ReadFile(filepath.Join(root, "daemon.log"))
	must(t, err)
	for _, want := range []string{"placement: wait for yumichika (h100-80, fp8-adaln-pruned) to attach — balanced",
		"placement: reuse yumichika (h100-80, fp8-adaln-pruned) — balanced, unmeasured;"} {
		if !strings.Contains(string(log), want) {
			t.Fatalf("daemon.log does not say %q:\n%s", want, tail(filepath.Join(root, "daemon.log")))
		}
	}
}

func TestExplicitLaneReusesTheFittingRentalOverTheCheaperShortOne(t *testing.T) {
	h := newLadderHub(t)
	h.unsized = true
	h.bind(goodLadder())
	root := ladderRoot(t, h)
	cert := filepath.Join(root, "fixture.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// Two attached rentals: the cheaper 4090 sorts first and used to win the rung tie.
	for _, seed := range []records.Rental{
		{AcceleratorCount: 1, ID: "pr-hairu", MachineName: "hairu", SKU: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090",
			HourlyRateUSDMicros: 740_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL},
		{AcceleratorCount: 1, ID: "pr-yumichika", MachineName: "yumichika", SKU: "h100-80", AcceleratorModel: h100SXM,
			HourlyRateUSDMicros: 2_490_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL},
	} {
		fatal(t, store.RecordRental(seed))
		h.addReady(seed.ID, seed.MachineName, seed.AcceleratorModel, seed.HourlyRateUSDMicros)
	}
	store.Close()
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", explicitFP8, "--rental-only", "--json",
		"--idempotency-key", "explicit-reuse")
	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "the request pinned", func() bool {
		row, problem := store.RequestByIdempotencyKey("explicit-reuse")
		return problem == nil && row != nil && row.Worker != ""
	})
	row, problem := store.RequestByIdempotencyKey("explicit-reuse")
	fatal(t, problem)
	if row == nil {
		t.Fatalf("the run was not submitted: %s", out)
	}
	if row.Worker != "pr-yumichika" || row.Models[0].Lane != "fp8-adaln-pruned" || row.Models[0].Manifest != fp8Manifest {
		t.Fatalf("pinned to %q with %+v; want pr-yumichika on fp8-adaln-pruned", row.Worker, row.Models[0])
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("two attached rentals and the fleet still bought: %v", asks)
	}
	placement := placementEvent(t, store, row.ID)
	verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
	if verdicts["yumichika"] != "chosen" || verdicts["hairu"] != fourKShort || verdicts["rtx-4090"] != fourKShort ||
		verdicts["h100-80"] != "unmeasured" {
		t.Fatalf("verdicts %v", verdicts)
	}
	if fits["yumichika"] != "rung_asserted" || fits["hairu"] != "lane_bytes 103.0 GiB of 24 GB" {
		t.Fatalf("fits %v", fits)
	}
	if placement["override"] != "fp8-adaln-pruned" || placement["rental"] != "pr-yumichika" ||
		placement["line"] != "placement: reuse yumichika (h100-80, fp8-adaln-pruned) — balanced, unmeasured" {
		t.Fatalf("the record: %v", placement)
	}
}
