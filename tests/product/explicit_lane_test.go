package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/records"
)

// cl-170, the run that filed it (2026-09-07 22:27Z, req-91ca03bbf302e11e7ecb1ae9). An explicit
// `model.model=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` was placed as if every product
// fit it, and the user's ready H100 was `excluded:unattached` for the seconds before its
// worker's control stream opened, so a 24 GB RTX 4090 was bought for the 103 GB lane and
// refused on the pod (device_shortfall, short by 35.8 GB). The next request reused another
// 4090 over the attached H100. These arms drive the real binary and Calcifer against the
// stand-in hub: every product is held to the components the explicit lane stages.

const (
	explicitFP8 = "model.model=proof/minimax@1.0.0-rc.1/fp8"
	fourKShort  = textEncoderVerdict + ", rtx-4090 has 24 GB"
	h100Fit     = "components 51.5 GiB (total memory unmeasured for this exact workload/SKU) of 80 GB"
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
	h := newStandInHub(t)
	root := standInRoot(t, h)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	// The user's H100, ready seconds ago and not yet attached: no address, no certificate.
	yumichika := records.Rental{AcceleratorCount: 1, ID: "pr-yumichika", MachineName: "yumichika", SKU: "h100-80",
		AcceleratorModel: h100SXM, HourlyRateUSDMicros: 2_490_000, State: "ready", Hub: h.server.URL}
	fatal(t, store.RecordRental(yumichika))
	h.addReady(yumichika.ID, yumichika.MachineName, yumichika.AcceleratorModel, yumichika.HourlyRateUSDMicros)
	startCalciferProcess(t, root)
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
	// The H100 is ready but not yet attached: the request is placed on it at once and waits
	// there for its machine, and nothing is bought around it.
	placement := placementEvent(t, store, row.ID)
	verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
	if verdicts["yumichika"] != "chosen" || verdicts["rtx-4090"] != fourKShort {
		t.Fatalf("verdicts %v", verdicts)
	}
	if fits["yumichika"] != h100Fit ||
		fits["b200"] != "components 51.5 GiB (total memory unmeasured for this exact workload/SKU) of 180 GB" {
		t.Fatalf("fits %v", fits)
	}
	if placement["override"] != "fp8" || placement["rental"] != yumichika.ID || placement["bought"] != false ||
		placement["line"] != "placement: reuse yumichika (h100-80, fp8) — balanced, unmeasured" {
		t.Fatalf("the placement record: %v", placement)
	}
	waitFor(t, root, "the request pinned to the H100", func() bool {
		current, problem := store.RequestByIdempotencyKey("explicit-wait")
		return problem == nil && current != nil && current.Worker == yumichika.ID
	})
	pinned, problem := store.RequestByIdempotencyKey("explicit-wait")
	fatal(t, problem)
	if pinned.Models[0].Lane != "fp8" || pinned.Models[0].Manifest != fp8Manifest || !pinned.Models[0].Choice {
		t.Fatalf("pinned %+v; want the explicit fp8 lane", pinned.Models[0])
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a fitting rental was bought around: %v", asks)
	}
}

func TestExplicitLaneReusesTheFittingRentalOverTheCheaperShortOne(t *testing.T) {
	h := newStandInHub(t)
	root := standInRoot(t, h)
	cert := filepath.Join(root, "fixture.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// Two attached rentals: the cheaper 4090 sorts first.
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
	startCalciferProcess(t, root)
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
	if row.Worker != "pr-yumichika" || row.Models[0].Lane != "fp8" || row.Models[0].Manifest != fp8Manifest {
		t.Fatalf("pinned to %q with %+v; want pr-yumichika on fp8", row.Worker, row.Models[0])
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
	if fits["yumichika"] != h100Fit {
		t.Fatalf("fits %v", fits)
	}
	if placement["override"] != "fp8" || placement["rental"] != "pr-yumichika" ||
		placement["line"] != "placement: reuse yumichika (h100-80, fp8) — balanced, unmeasured" {
		t.Fatalf("the record: %v", placement)
	}
}
