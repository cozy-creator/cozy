package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// cl-174, the runs that filed it (2026-09-08 00:06–00:25Z, req-829410d83cf9a3d7996184dc,
// req-2e366a367e2050667b25ad35 and siblings). An explicit
// `model.model=paul/minimax-h3@1.0.0-rc.1/fp8-adaln-pruned` under a package whose ladder binds
// rc.2 carried no ladder, so the unsized rc.1 lane was held to whole-lane bytes (96.2 GiB) and
// BOTH attached H100s — one running that very lane, one idle — read excluded:vram_short; the
// H200 and B200 fit and the spend cap refused them; the refusal FAILED the request with no
// request.placement record. At 00:05:40Z a bare run had also taken the busy H100 over the idle
// one on the name tie. These arms drive the real binary and daemon against the stand-in hub
// with the card as it stood: rc.1 unsized, rc.2 sized and bound.

const fleetCapVerdict = "excluded:rental.fleet_spend_cap"

// fleetHub is the card and binding of 2026-09-08: the package's ladder names rc.2.
func fleetHub(t *testing.T) *ladderHub {
	h := newLadderHub(t)
	h.unsized, h.later = true, true
	ladder := goodLadder()
	ladder.Release = "1.0.0-rc.2"
	h.bind(ladder)
	return h
}

// fleetRoot is ladderRoot under a fleet spend cap of `capUSD` per hour.
func fleetRoot(t *testing.T, h *ladderHub, capUSD string) string {
	t.Helper()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+
		"\ntensorhub_token: ladder-test\nrentals:\n  max_hourly_spend_usd: "+capUSD+"\n  idle_release_s: 0\n"+
		"daemon:\n  idle_shutdown_s: 0\n"), 0600))
	return root
}

// plantH100 records one H100 rental the fleet holds, with the media bearer and creator key
// an attached one carries; `attached` gives it its worker address and certificate.
func plantH100(t *testing.T, root string, h *ladderHub, store *records.Store, id, machine string, attached bool) records.Rental {
	t.Helper()
	rentals := filepath.Join(root, "rentals")
	must(t, os.MkdirAll(rentals, 0o700))
	must(t, os.WriteFile(filepath.Join(rentals, id+".media-token"), []byte("media-"+id), 0o600))
	must(t, os.WriteFile(filepath.Join(rentals, id+".pem"), []byte("-----BEGIN CERTIFICATE-----\n"), 0o600))
	_, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	key, err := x509.MarshalPKCS8PrivateKey(private)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(rentals, id+".creator.pem"),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0o600))
	row := records.Rental{ID: id, MachineName: machine, SKU: "h100-80", AcceleratorModel: h100SXM,
		HourlyRateUSDMicros: 2_490_000, State: "ready", Hub: h.server.URL}
	if attached {
		row.Address, row.CertPath = "127.0.0.1:1", filepath.Join(rentals, id+".pem")
	}
	fatal(t, store.RecordRental(row))
	h.addReady(id, machine, h100SXM, row.HourlyRateUSDMicros)
	return row
}

// holdQueued pins one queued request to the rental, so a new request waits behind it.
func holdQueued(t *testing.T, store *records.Store, requestID, rentalID string) {
	t.Helper()
	if _, _, problem := store.Submit(records.Request{ID: requestID, IdemKey: "idem-" + requestID,
		BodyDigest: "sha256:" + strings.Repeat("ab", 32), Package: ladderPackage, Entrypoint: "generate",
		Payload: []byte("{}"), Rental: true, Worker: rentalID}); problem != nil {
		t.Fatal(problem.Message)
	}
}

func submitExplicit(t *testing.T, root, key string) {
	t.Helper()
	runCozy(t, root, "run", "proof/h3/generate", "steps=1", explicitFP8, "--rental-only", "--json",
		"--idempotency-key", key)
}

func lastPlacement(t *testing.T, store *records.Store, requestID string) map[string]any {
	t.Helper()
	placements := placementEvents(t, store, requestID)
	if len(placements) == 0 {
		t.Fatalf("%s has no request.placement record", requestID)
	}
	return placements[len(placements)-1]
}

// The 00:25:31Z arm: two attached idle H100s, the explicit rc.1 lane, every purchase over
// the cap. The owner's rung names the card and the lane, so the lane fits; the cheaper-named
// idle machine is reused, the record is written, nothing is bought.
func TestExplicitLaneOfAnotherReleaseReusesTheIdleAttachedRental(t *testing.T) {
	h := fleetHub(t)
	root := fleetRoot(t, h, "5")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantH100(t, root, h, store, "pr-guchuko", "guchuko", true)
	plantH100(t, root, h, store, "pr-lumachina", "lumachina", true)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "fleet-reuse")
	var row *records.Request
	waitFor(t, root, "the request pinned", func() bool {
		row, problem = store.RequestByIdempotencyKey("fleet-reuse")
		return problem == nil && row != nil && row.Worker != ""
	})
	if row.Worker != "pr-guchuko" || row.Models[0].Lane != "fp8-adaln-pruned" || row.Models[0].Manifest != fp8Manifest {
		t.Fatalf("pinned to %q with %+v; want pr-guchuko on the explicit rc.1 fp8 lane", row.Worker, row.Models[0])
	}
	placement := lastPlacement(t, store, row.ID)
	verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
	if verdicts["guchuko"] != "chosen" || verdicts["lumachina"] != "unmeasured" || verdicts["h100-80"] != fleetCapVerdict ||
		verdicts["h200"] != fleetCapVerdict || verdicts["rtx-4090"] != fourKShort {
		t.Fatalf("verdicts %v", verdicts)
	}
	if fits["guchuko"] != "rung_asserted" || fits["lumachina"] != "rung_asserted" || fits["h100-80"] != "rung_asserted" ||
		fits["h200"] != "lane_bytes 103.0 GiB of 141 GB" {
		t.Fatalf("fits %v", fits)
	}
	if ladder, _ := placement["ladder"].([]any); len(ladder) != 1 || ladder[0] != "H100=fp8-adaln-pruned > B200=mxfp8-adaln-pruned > *=bf16-full" ||
		placement["override"] != "fp8-adaln-pruned" || placement["bought"] != false {
		t.Fatalf("the record: %v", placement)
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("two idle H100s and the fleet still bought: %v", asks)
	}
}

// The 00:06–00:08Z arm: one H100 busy, one attaching, every purchase over the cap. The
// request waits (record written), takes the busy one when it frees, and a second request
// takes the attaching one — idle — once it attaches, over the busy one.
func TestExplicitLaneWaitsForTheAttachingRentalOverQueueingBehindTheBusyOne(t *testing.T) {
	h := fleetHub(t)
	root := fleetRoot(t, h, "5")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantH100(t, root, h, store, "pr-guchuko", "guchuko", true)
	holdQueued(t, store, "req-held", "pr-guchuko")
	lumachina := plantH100(t, root, h, store, "pr-lumachina", "lumachina", false)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "fleet-wait")
	var first *records.Request
	waitFor(t, root, "the placement waiting on the attaching H100", func() bool {
		first, problem = store.RequestByIdempotencyKey("fleet-wait")
		return problem == nil && first != nil && len(placementEvents(t, store, first.ID)) > 0
	})
	placement := lastPlacement(t, store, first.ID)
	verdicts := candidateVerdicts(placement)
	if verdicts["guchuko"] != "excluded:attaching: lumachina" || verdicts["lumachina"] != "attaching" ||
		verdicts["h100-80"] != fleetCapVerdict || verdicts["h200"] != fleetCapVerdict || verdicts["b200"] != fleetCapVerdict ||
		placement["line"] != "placement: wait for lumachina (h100-80, fp8-adaln-pruned) to attach — balanced" {
		t.Fatalf("the wait record: %v", placement)
	}
	if current, problem := store.RequestByIdempotencyKey("fleet-wait"); problem != nil || current.Worker != "" || settled(current.State) {
		t.Fatalf("the waiting request is %+v; want queued and unpinned", current)
	}
	// The busy H100 frees: its held request settles, and the next observation takes it.
	fatal(t, store.SettleRequest("req-held", "canceled"))
	waitFor(t, root, "the request pinned to the freed H100 and the reuse recorded", func() bool {
		current, problem := store.RequestByIdempotencyKey("fleet-wait")
		return problem == nil && current != nil && current.Worker == "pr-guchuko" &&
			candidateVerdicts(lastPlacement(t, store, first.ID))["guchuko"] == "chosen"
	})
	if verdicts := candidateVerdicts(lastPlacement(t, store, first.ID)); verdicts["lumachina"] != "attaching" {
		t.Fatalf("the reuse record: %v", verdicts)
	}
	// A second request finds the H100 busy again and waits; the other attaches — address
	// and certificate land on the row — and, idle, wins over the busy one.
	submitExplicit(t, root, "fleet-second")
	var second *records.Request
	waitFor(t, root, "the second request waiting", func() bool {
		second, problem = store.RequestByIdempotencyKey("fleet-second")
		return problem == nil && second != nil && len(placementEvents(t, store, second.ID)) > 0
	})
	if verdicts := candidateVerdicts(lastPlacement(t, store, second.ID)); verdicts["guchuko"] != "excluded:attaching: lumachina" {
		t.Fatalf("the second wait record: %v", verdicts)
	}
	attached := lumachina
	attached.Address, attached.CertPath = "127.0.0.1:1", filepath.Join(root, "rentals", "pr-lumachina.pem")
	fatal(t, store.RecordRental(attached))
	waitFor(t, root, "the second request pinned to the attached idle H100", func() bool {
		if current, problem := store.RentalRow(lumachina.ID); problem == nil && current != nil && current.CertPath == "" {
			_ = store.RecordRental(attached)
		}
		current, problem := store.RequestByIdempotencyKey("fleet-second")
		return problem == nil && current != nil && current.Worker == "pr-lumachina" &&
			candidateVerdicts(lastPlacement(t, store, second.ID))["lumachina"] == "chosen"
	})
	record := lastPlacement(t, store, second.ID)
	if verdicts := candidateVerdicts(record); verdicts["lumachina"] != "chosen" || verdicts["guchuko"] != "unmeasured" {
		t.Fatalf("the idle machine did not win over the busy one: %v", verdicts)
	}
	for _, raw := range record["candidates"].([]any) {
		if c := raw.(map[string]any); c["machine"] == "guchuko" && c["ahead"] != 1.0 {
			t.Fatalf("the busy H100 does not carry its queue: %v", c)
		}
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a fitting rental was bought around: %v", asks)
	}
}

// The true refusal: nothing the fleet holds fits and the cap refuses every purchase. The
// request fails with the cap error, and the record says why each candidate was passed.
func TestExplicitLaneFailsOnTheCapOnlyWhenNoRentalFits(t *testing.T) {
	h := fleetHub(t)
	root := fleetRoot(t, h, "1")
	cert := filepath.Join(root, "hairu.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	hairu := records.Rental{ID: "pr-hairu", MachineName: "hairu", SKU: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090",
		HourlyRateUSDMicros: 740_000, State: "ready", Address: "127.0.0.1:1", CertPath: cert, Hub: h.server.URL}
	fatal(t, store.RecordRental(hairu))
	h.addReady(hairu.ID, hairu.MachineName, hairu.AcceleratorModel, hairu.HourlyRateUSDMicros)
	startDaemonProcess(t, root)
	_, out := runCozy(t, root, "run", "proof/h3/generate", "steps=1", explicitFP8, "--rental-only", "--json",
		"--idempotency-key", "fleet-refusal")
	var row *records.Request
	waitFor(t, root, "the request settled", func() bool {
		row, problem = store.RequestByIdempotencyKey("fleet-refusal")
		return problem == nil && row != nil && settled(row.State)
	})
	if row.State != "failed" || !strings.Contains(out, "rental.fleet_spend_cap") {
		t.Fatalf("the request is %s: %s", row.State, out)
	}
	placement := lastPlacement(t, store, row.ID)
	if verdicts := candidateVerdicts(placement); verdicts["hairu"] != fourKShort || verdicts["h100-80"] != fleetCapVerdict ||
		verdicts["h200"] != fleetCapVerdict || placement["line"] != "placement: nothing chosen" {
		t.Fatalf("the refusal record: %v", placement)
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("a refused run reached a paid ask: %v", asks)
	}
}
