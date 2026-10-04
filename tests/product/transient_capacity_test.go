package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// cl-185, the run that filed it. 2026-09-08 11:04:22Z, req-395c87fc062d407199601cc1
// (`paul/minimax-h3/ref2va`) settled FAILED with
//
//	rental.no_fitting_sku — no rental SKU on offer fits paul/minimax-h3:
//	  karam excluded:not_ready; gyokuyou excluded:not_ready
//
// Both machines were rentals this daemon had already bought. `karam` was bought at
// 10:56:45Z and the hub reported it `ready` at 11:08:19Z — the refusal landed four
// minutes INSIDE its own acquisition window, and the recorded decision carried no
// purchase candidate at all, because Tensorhub's catalog answered with nothing of the
// request's class in that instant. Neither fact was permanent. Both reached the owner as
// a sentence about the ladder.
//
// These arms drive the REAL binary and the REAL daemon against the stand-in hub, with the
// fleet and the market in the exact shapes of that minute.

// plantRental records one H100 rental the fleet holds in the hub's own lifecycle word.
// `attached` gives it the worker triple a serving pod carries; a machine still coming up
// has none, which is what makes it indistinguishable from a dead one until the state is
// read.
func plantRental(t *testing.T, root string, h *ladderHub, store *records.Store,
	id, machine, state string, attached bool,
) records.Rental {
	t.Helper()
	rentals := filepath.Join(root, "rentals")
	must(t, os.MkdirAll(rentals, 0o700))
	must(t, os.WriteFile(filepath.Join(rentals, id+".media-token"), []byte("media-"+id), 0o600))
	must(t, os.WriteFile(filepath.Join(rentals, id+".pem"), []byte("-----BEGIN CERTIFICATE-----\n"), 0o600))
	row := records.Rental{ID: id, MachineName: machine, SKU: "h100-80", AcceleratorModel: h100SXM,
		AcceleratorCount: 1, HourlyRateUSDMicros: 2_490_000, State: state, Hub: h.server.URL}
	if attached {
		row.Address, row.CertPath = "127.0.0.1:1", filepath.Join(rentals, id+".pem")
		h.addReady(id, machine, h100SXM, row.HourlyRateUSDMicros)
	} else {
		h.addState(id, machine, h100SXM, state, row.HourlyRateUSDMicros)
	}
	fatal(t, store.RecordRental(row))
	return row
}

// THE RUN 412 SHAPE. Two H100s mid-acquisition, nothing on offer. Every input is
// transient, so the request must WAIT — and when one machine comes up it must take it.
func TestAFleetStillComingUpIsWaitedForAndNeverRefused(t *testing.T) {
	h := fleetHub(t)
	h.sell()
	root := fleetRoot(t, h, 20)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantRental(t, root, h, store, "pr-karam", "karam", "acquiring", false)
	plantRental(t, root, h, store, "pr-gyokuyou", "gyokuyou", "pending_acquisition", false)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "coming-up")
	var row *records.Request
	waitFor(t, root, "the placement waiting on the fleet that is coming up", func() bool {
		row, problem = store.RequestByIdempotencyKey("coming-up")
		return problem == nil && row != nil && len(placementEvents(t, store, row.ID)) > 0
	})
	placement := lastPlacement(t, store, row.ID)
	verdicts := candidateVerdicts(placement)
	if verdicts["karam"] != "attaching" || verdicts["gyokuyou"] != "attaching" {
		t.Fatalf("verdicts %v; both machines are coming up and NEITHER may be excluded", verdicts)
	}
	if line, _ := placement["line"].(string); !strings.HasPrefix(line, "placement: wait for ") {
		t.Fatalf("the record says %q; a fleet that is coming up is waited for, not refused", line)
	}
	if current, problem := store.RequestByIdempotencyKey("coming-up"); problem != nil ||
		current.Worker != "" || settled(current.State) {
		t.Fatalf("the request is %+v; want queued and unpinned, never settled on a fleet that is booting", current)
	}
	// It self-heals: the machine finishes coming up and the request lands on it, with
	// nothing bought.
	attached := plantRental(t, root, h, store, "pr-karam", "karam", "ready", true)
	waitFor(t, root, "the request pinned to the machine that finished coming up", func() bool {
		if current, problem := store.RentalRow(attached.ID); problem == nil && current != nil && current.CertPath == "" {
			_ = store.RecordRental(attached)
		}
		current, problem := store.RequestByIdempotencyKey("coming-up")
		return problem == nil && current != nil && current.Worker == "pr-karam"
	})
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("two machines were already coming up and the fleet still bought: %v", asks)
	}
}

// A catalog that offers nothing is WEATHER, not a bad ladder — even when every machine
// the fleet holds is genuinely dead. The request waits; the refusal never claims a SKU
// failed to fit, because no SKU was on offer to fit.
func TestAnEmptyCatalogBesideADeadFleetWaitsAndSaysWhichIsWhich(t *testing.T) {
	h := fleetHub(t)
	h.sell()
	root := fleetRoot(t, h, 20)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantRental(t, root, h, store, "pr-karam", "karam", "failed", false)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "empty-catalog")
	var row *records.Request
	waitFor(t, root, "the placement over an empty catalog and its park", func() bool {
		row, problem = store.RequestByIdempotencyKey("empty-catalog")
		return problem == nil && row != nil && len(placementEvents(t, store, row.ID)) > 0 &&
			len(parkReasons(t, store, row.ID)) > 0
	})
	if verdicts := candidateVerdicts(lastPlacement(t, store, row.ID)); verdicts["karam"] != "excluded:not_ready: failed" {
		t.Fatalf("verdicts %v; a finished rental is excluded and NAMES the state that finished it", verdicts)
	}
	current, problem := store.RequestByIdempotencyKey("empty-catalog")
	fatal(t, problem)
	if settled(current.State) {
		t.Fatalf("the request settled %q; an empty catalog is not a refusal", current.State)
	}
	reasons := parkReasons(t, store, row.ID)
	found := false
	for _, reason := range reasons {
		found = found || containsAll(reason,
			"offered no accelerator product", "karam excluded:not_ready: failed")
	}
	if !found {
		t.Fatalf("the park reasons are %q; one must name the empty catalog AND what the fleet held", reasons)
	}
}

// The permanent arm, unchanged: a market that DOES offer machines, none of which any rung
// of the ladder names, still fails terminally — and now it is the only thing that does.
func TestAMarketNoRungNamesStillFailsTerminally(t *testing.T) {
	h := fleetHub(t)
	h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: "1.0.0-rc.2",
		Ladder: []hub.BindingRung{{GPU: "MI300X", Lane: "fp8-adaln-pruned"}}, Revision: 4})
	root := fleetRoot(t, h, 20)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantRental(t, root, h, store, "pr-karam", "karam", "failed", false)
	startDaemonProcess(t, root)
	runCozy(t, root, "run", "proof/h3/generate", "steps=1", "--rental-only", "--json",
		"--idempotency-key", "no-rung")
	var row *records.Request
	waitFor(t, root, "the request failing on a ladder no offered machine fits", func() bool {
		row, problem = store.RequestByIdempotencyKey("no-rung")
		return problem == nil && row != nil && settled(row.State)
	})
	if row.State != "failed" {
		t.Fatalf("the request settled %q; a ladder no machine fits is terminal", row.State)
	}
	cause := lastEventField(t, store, row.ID, "run.failed", "error_type")
	if reason := failureReason(t, store, row.ID); cause != "rental.no_fitting_sku" ||
		!containsAll(reason, "no_rung") {
		t.Fatalf("the failure is %s / %q; want rental.no_fitting_sku over no_rung verdicts", cause, reason)
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("nothing fit and the fleet still bought: %v", asks)
	}
}

// cl-132/cl-174, the head-of-line half. A pod this fleet has already bought, attached and
// left IDLE is taken now, even when the tier's own arithmetic ranks a purchase above it
// and another machine is still coming up. Reading the tier's winner alone made the
// request wait and the idle pod bill for nothing.
func TestAnIdleAttachedMachineOutranksWaitingForOneComingUp(t *testing.T) {
	h := fleetHub(t)
	// Only the h200 purchase is measured, so the tier's winner is a BUY: an unmeasured
	// candidate does not compete with a measured one (placement-economics.md).
	h.throughput = []hub.ModelThroughput{{Release: "1.0.0-rc.1", Lane: "fp8-adaln-pruned",
		SKU: "h200", MedianS: 10, PrepareS: 10}}
	root := fleetRoot(t, h, 20)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	plantRental(t, root, h, store, "pr-guchuko", "guchuko", "ready", true)
	plantRental(t, root, h, store, "pr-lumachina", "lumachina", "acquiring", false)
	startDaemonProcess(t, root)
	submitExplicit(t, root, "idle-wins")
	var row *records.Request
	waitFor(t, root, "the request pinned to the idle attached machine and its placement recorded", func() bool {
		row, problem = store.RequestByIdempotencyKey("idle-wins")
		return problem == nil && row != nil && row.Worker != "" && len(placementEvents(t, store, row.ID)) > 0
	})
	if row.Worker != "pr-guchuko" {
		t.Fatalf("pinned to %q; the idle attached machine is taken over waiting", row.Worker)
	}
	if verdicts := candidateVerdicts(lastPlacement(t, store, row.ID)); verdicts["guchuko"] != "chosen" ||
		verdicts["lumachina"] != "attaching" {
		t.Fatalf("verdicts %v; want guchuko chosen beside lumachina still attaching", verdicts)
	}
	if asks := h.postedSKUs(); len(asks) != 0 {
		t.Fatalf("an idle attached H100 was open and the fleet still bought: %v", asks)
	}
}

// parkReason is the newest `request.parked` reason, and failureReason the `run.failed`
// error — the two ends a placement refusal can reach, read from the durable record rather
// than from stdout.
func parkReasons(t *testing.T, store *records.Store, requestID string) []string {
	t.Helper()
	events, problem := store.EventsAfter(requestID, 0, 500)
	fatal(t, problem)
	var out []string
	for _, event := range events {
		if event.Type == "request.parked" {
			reason, _ := event.Payload["reason"].(string)
			out = append(out, reason)
		}
	}
	return out
}

func failureReason(t *testing.T, store *records.Store, requestID string) string {
	t.Helper()
	return lastEventField(t, store, requestID, "run.failed", "error")
}

func lastEventField(t *testing.T, store *records.Store, requestID, eventType, field string) string {
	t.Helper()
	events, problem := store.EventsAfter(requestID, 0, 500)
	fatal(t, problem)
	out := ""
	for _, event := range events {
		if event.Type == eventType {
			out, _ = event.Payload[field].(string)
		}
	}
	return out
}

func containsAll(haystack string, needles ...string) bool {
	for _, needle := range needles {
		if !strings.Contains(haystack, needle) {
			return false
		}
	}
	return true
}
