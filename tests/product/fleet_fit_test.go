package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/pem"
	"math/big"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

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

// fleetRoot is ladderRoot with the stand-in hub's owner spend cap at `capUSD` per hour.
func fleetRoot(t *testing.T, h *ladderHub, capUSD float64) string {
	t.Helper()
	h.mu.Lock()
	h.spendCap = int64(capUSD * 1_000_000)
	h.mu.Unlock()
	return ladderRoot(t, h)
}

// plantH100 records one H100 rental the fleet holds, with the media bearer and creator key
// an attached one carries; `attached` gives it its worker address and certificate.
func plantH100(t *testing.T, root string, h *ladderHub, store *records.Store, id, machine string, attached bool) records.Rental {
	t.Helper()
	return plantAttached(t, root, h, store, records.Rental{AcceleratorCount: 1, ID: id, MachineName: machine, SKU: "h100-80",
		AcceleratorModel: h100SXM, HourlyRateUSDMicros: 2_490_000, State: "ready", Hub: h.server.URL}, attached)
}

// plantAttached records one ready rental of any class the fleet holds.
func plantAttached(t *testing.T, root string, h *ladderHub, store *records.Store, row records.Rental, attached bool) records.Rental {
	t.Helper()
	id := row.ID
	rentals := filepath.Join(root, "rentals")
	must(t, os.MkdirAll(rentals, 0o700))
	must(t, os.WriteFile(filepath.Join(rentals, id+".media-token"), []byte("media-"+id), 0o600))
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	// A readable pin for a pod that never answers: a run handed to it waits on the machine.
	der, err := x509.CreateCertificate(rand.Reader, &x509.Certificate{SerialNumber: big.NewInt(1),
		NotAfter: time.Now().Add(time.Hour)}, &x509.Certificate{SerialNumber: big.NewInt(1)}, public, private)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(rentals, id+".pem"), pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), 0o600))
	key, err := x509.MarshalPKCS8PrivateKey(private)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(rentals, id+".creator.pem"),
		pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: key}), 0o600))
	if attached {
		row.Address, row.CertPath = "127.0.0.1:1", filepath.Join(rentals, id+".pem")
	}
	fatal(t, store.RecordRental(row))
	h.addReady(id, row.MachineName, row.AcceleratorModel, row.HourlyRateUSDMicros)
	return row
}

// holdQueued pins one rented run, already handed to the rental's Runtime and not yet
// accepted, to the rental, so a new request waits behind it.
func holdQueued(t *testing.T, store *records.Store, requestID, rentalID string) {
	t.Helper()
	if _, _, problem := store.Submit(records.Request{ID: requestID, IdemKey: "idem-" + requestID,
		BodyDigest: "sha256:" + strings.Repeat("ab", 32), Package: ladderPackage, Entrypoint: "generate",
		Payload: []byte("{}"), Rental: true, Worker: rentalID, MachineExecutionObserver: true}); problem != nil {
		t.Fatal(problem.Message)
	}
	fatal(t, store.LinkMachineExecution(requestID, rentalID))
	_, problem := store.MarkRunV1Sent(requestID)
	fatal(t, problem)

}

// submitExplicit submits the explicit-lane run and leaves once it is recorded: a client that
// goes away never cancels its run (cl-108), so nothing waits out the CLI's optimistic
// observation of a run that is meant to stay queued.
func submitExplicit(t *testing.T, root, key string) {
	t.Helper()
	cmd := exec.Command(cozyBin, "run", "proof/h3/generate", "steps=1", explicitFP8, "--rental-only", "--json",
		"--idempotency-key", key)
	cmd.Env = childEnv(t, root)
	must(t, cmd.Start())
	exited := make(chan struct{})
	go func() { _ = cmd.Wait(); close(exited) }()
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	deadline := time.After(20 * time.Second)
	for {
		if row, problem := store.RequestByIdempotencyKey(key); problem == nil && row != nil {
			break
		}
		select {
		case <-exited:
			return
		case <-deadline:
			_ = cmd.Process.Kill()
			t.Fatalf("the run %s was never recorded", key)
		case <-time.After(20 * time.Millisecond):
		}
	}
	_ = cmd.Process.Kill()
	<-exited
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
// the hub's cap (never asked: an attached machine fits). The owner's rung names the card and the lane, so the lane fits; the cheaper-named
// idle machine is reused, the record is written, nothing is bought.
func TestExplicitLaneOfAnotherReleaseReusesTheIdleAttachedRental(t *testing.T) {
	h := fleetHub(t)
	root := fleetRoot(t, h, 5)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantH100(t, root, h, store, "pr-guchuko", "guchuko", true)
	plantH100(t, root, h, store, "pr-lumachina", "lumachina", true)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "fleet-reuse")
	var row *records.Request
	waitFor(t, root, "the request pinned and its placement recorded", func() bool {
		row, problem = store.RequestByIdempotencyKey("fleet-reuse")
		return problem == nil && row != nil && row.Worker != "" && len(placementEvents(t, store, row.ID)) > 0
	})
	if row.Worker != "pr-guchuko" || row.Models[0].Lane != "fp8-adaln-pruned" || row.Models[0].Manifest != fp8Manifest {
		t.Fatalf("pinned to %q with %+v; want pr-guchuko on the explicit rc.1 fp8 lane", row.Worker, row.Models[0])
	}
	placement := lastPlacement(t, store, row.ID)
	verdicts, fits := candidateVerdicts(placement), candidateFits(placement)
	if verdicts["guchuko"] != "chosen" || verdicts["lumachina"] != "unmeasured" || verdicts["h100-80"] == "chosen" ||
		verdicts["h200"] == "chosen" || verdicts["rtx-4090"] != fourKShort {
		t.Fatalf("verdicts %v", verdicts)
	}
	if fits["guchuko"] != "rung_asserted" || fits["lumachina"] != "rung_asserted" || fits["h100-80"] != "rung_asserted" ||
		fits["h200"] != "lane_bytes 103.0 GiB (working memory unmeasured) of 141 GB" {
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
	root := fleetRoot(t, h, 5)
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
		verdicts["h100-80"] == "chosen" || verdicts["h200"] == "chosen" || verdicts["b200"] == "chosen" ||
		placement["line"] != "placement: wait for lumachina (h100-80, fp8-adaln-pruned) to attach — balanced" {
		held, _ := store.RequestRow("req-held")
		t.Fatalf("the wait record: %v; held request: %+v\n%s", placement, held, tail(filepath.Join(root, "daemon.log")))
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
	root := fleetRoot(t, h, 1)
	cert := filepath.Join(root, "hairu.pem")
	must(t, os.WriteFile(cert, []byte("fixture"), 0600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	hairu := records.Rental{AcceleratorCount: 1, ID: "pr-hairu", MachineName: "hairu", SKU: "rtx-4090", AcceleratorModel: "NVIDIA GeForce RTX 4090",
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
	if rentals, problem := store.Rentals(); problem != nil || len(rentals) != 1 {
		t.Fatalf("a refused run bought a rental: %v %v", rentals, problem)
	}
}
