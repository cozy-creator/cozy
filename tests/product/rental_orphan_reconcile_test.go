package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// THE MONEY-LEAK HALF (cl-199, on hub th-199).
//
// 2026-09-07: six H100 NVLs at $3.19/hour were created hub-side and reached
// `booting`. Creator's width fence refused the create answer, so this host recorded
// none of them. `cozy rental list` — the one command a person types to ask what they
// are paying for — printed an empty fleet; `cozy rental end <name>` reported success
// against every one; the $10/hour spend ceiling admitted all six because the ceiling
// was computed from local rows and none of the six had one. Roughly $19/hour ran
// unattached until someone read the hub's table by hand.
//
// The board reconciled only in ONE direction: local row → hub. That can correct a row
// and can never notice a pod that has no row. These arms hold the other direction
// open, through the real CLI process against a hub answering the real routes.

// orphanHub is a stand-in hub that publishes th-199's account listing and holds one
// live rental at `rate`, which this host has never recorded.
func orphanHub(t *testing.T, name string) (string, *fakeRentalHub) {
	t.Helper()
	root, _, stand := rentalEndRoot(t, name)
	stand.publishListing()
	return root, stand
}

// orphan plants one hub-side rental with no local record: a machine word, an
// accelerator, a rate, and the moment its billing began.
func orphan(stand *fakeRentalHub, id, machine string, rate int64) {
	stand.add(id, machine)
	stand.set(id, "state", "booting")
	stand.set(id, "requested_accelerator_model", "NVIDIA H100 NVL")
	stand.set(id, "hourly_rate_usd_micros", rate)
	stand.set(id, "created_at", "2026-09-07T21:18:49Z")
}

// TestRentalListNamesMachinesThisHostNeverRecorded is the incident as behaviour: the
// account is billed for a machine this host has no row for, and the ordinary board
// says so — by machine word, with the rate, counted in the burn.
func TestRentalListNamesMachinesThisHostNeverRecorded(t *testing.T) {
	root, stand := orphanHub(t, "rental-orphan-list")
	orphan(stand, "pr-1111111111111111stiy", "stiyl", 3_190_000)

	code, out := runCozy(t, root, "rental", "list")
	if code != 0 {
		t.Fatalf("rental list failed [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "stiyl") {
		t.Fatalf("the board does not name the machine the account is billed for\n%s", out)
	}
	if !strings.Contains(out, "Remote machines running: 1") {
		t.Fatalf("an unrecorded machine is not counted in the fleet\n%s", out)
	}
	if !strings.Contains(out, "Current spend per hour: $3.19") {
		t.Fatalf("the burn does not carry the unrecorded machine's rate\n%s", out)
	}
	if strings.Contains(out, "RECORDED") || strings.Contains(out, "this host has no record") || strings.Contains(out, "cozy rental end") {
		t.Fatalf("ordinary fleet display exposes controller bookkeeping or suggests ending unseen work\n%s", out)
	}
	if code, selected := runCozy(t, root, "rental", "list", "--fields=machine,running,queued"); code != 0 || !regexp.MustCompile(`(?m)^stiyl\s+—\s+—\s*$`).MatchString(selected) {
		t.Fatalf("unknown activity became zero in the table [exit %d]\n%s", code, selected)
	}
	if code, selected := runCozy(t, root, "rental", "list", "--fields=machine,$/hour"); code != 0 || !regexp.MustCompile(`(?m)^stiyl\s+\$3\.19\s*$`).MatchString(selected) {
		t.Fatalf("hub rental lost its hourly rate [exit %d]\n%s", code, selected)
	}

	// The machine document carries it as a fact a program can branch on, not a spelling.
	code, out = runCozy(t, root, "--json", "--full", "rental", "list")
	if code != 0 {
		t.Fatalf("JSON snapshot failed [exit %d]\n%s", code, out)
	}
	var document struct {
		Rentals []map[string]any `json:"rentals"`
		Running int              `json:"machines_running"`
		Burn    int64            `json:"hourly_spend_usd_micros"`
		Orphans int              `json:"unrecorded_rentals"`
		Leaked  int64            `json:"unrecorded_hourly_spend_usd_micros"`
	}
	must(t, json.Unmarshal([]byte(out), &document))
	if document.Running != 1 || document.Burn != 3_190_000 ||
		document.Orphans != 1 || document.Leaked != 3_190_000 {
		t.Fatalf("the JSON totals do not carry the unrecorded machine: %s", out)
	}
	if len(document.Rentals) != 1 {
		t.Fatalf("the JSON fleet is not the account's: %s", out)
	}
	row := document.Rentals[0]
	if row["machine"] != "stiyl" || row["rental_id"] != "pr-1111111111111111stiy" ||
		row["state"] != "booting" || row["recorded"] != false ||
		row["hourly_rate_usd_micros"] != float64(3_190_000) {
		t.Fatalf("the unrecorded row is not the hub's truth: %s", out)
	}
	for _, field := range []string{"running", "queued", "idle_s", "idle_since_at", "release_due_at"} {
		if _, present := row[field]; present {
			t.Fatalf("unobserved %s was reported as a fact: %s", field, out)
		}
	}
	if stand.releases("pr-1111111111111111stiy") != 0 {
		t.Fatal("listing released a rental owned outside this controller")
	}
	// Uptime for a pod with no local `rented_at` can only come from the hub's own
	// clock; without it nobody can say how long the machine has been costing money.
	if row["rented_at"] != "2026-09-07T21:18:49Z" {
		t.Fatalf("the unrecorded row lost the moment its billing began: %s", out)
	}
	stand.setRate("pr-1111111111111111stiy", 0)
	code, out = runCozy(t, root, "rental", "list", "--json")
	if code == 0 || !strings.Contains(out, `"code":"rental.rate_unknown"`) ||
		strings.Contains(out, `"hourly_spend_usd_micros"`) {
		t.Fatalf("an unknown rate produced an account total [exit %d]\n%s", code, out)
	}
}

// TestEndingAnUnrecordedMachineByItsHubNameReleasesIt is the actionable half: the
// board shows a machine word, and that word is enough to stop the money. Before the
// listing route the word was Creator's own alias and the hub could not look it up at
// all, so the only recovery was an admin token against the hub's database.
func TestEndingAnUnrecordedMachineByItsHubNameReleasesIt(t *testing.T) {
	root, stand := orphanHub(t, "rental-orphan-end")
	orphan(stand, "pr-2222222222222222obam", "obama", 3_190_000)

	code, out := runCozy(t, root, "rental", "end", "obama", "--json")
	if code != 0 {
		t.Fatalf("ending an unrecorded machine by name failed [exit %d]\n%s", code, out)
	}
	if stand.releases("pr-2222222222222222obam") != 1 {
		t.Fatalf("no DELETE reached the hub for the machine that was billing\n%s", out)
	}
	var answer struct {
		Machine string `json:"machine"`
		Rental  string `json:"rental"`
		State   string `json:"state"`
		Changed bool   `json:"changed"`
	}
	must(t, json.Unmarshal([]byte(out), &answer))
	// `changed` means this command found a live pod and the hub destroyed it. An
	// orphan released through a name must never answer the way the six H100s did.
	if answer.State != "ended" || !answer.Changed ||
		answer.Rental != "pr-2222222222222222obam" || answer.Machine != "obama" {
		t.Fatalf("the release did not report a real teardown: %s", out)
	}
	if code, listed := runCozy(t, root, "rental", "list"); code != 0 ||
		strings.Contains(listed, "obama") {
		t.Fatalf("the released machine is still on the board [exit %d]\n%s", code, listed)
	}
}

// TestUnrecordedSpendRefusesTheNextPurchase is the ceiling that replaces an automatic
// shutdown nobody should want for a machine this host cannot prove it owns. The cap
// is a SPEND cap; a cap computed from local rows admitted six pods it should have
// refused, so the burn it compares against is the account's, not the filing cabinet's.
func TestUnrecordedSpendRefusesTheNextPurchase(t *testing.T) {
	root, stand := orphanHub(t, "rental-orphan-cap")
	orphan(stand, "pr-3333333333333333sem", "semiu", 3_190_000)
	orphan(stand, "pr-4444444444444444ursu", "ursula", 3_190_000)
	stand.setSKUs(map[string]any{
		"name": "h100-nvl", "accelerator_model": "NVIDIA H100 NVL", "accelerator_count": 1,
		"base_worker_profile": "proof", "compute_capability": "9.0", "vram_gb": 94,
		"minimum_ram_per_gpu_gb": 64, "price_usd_micros_per_hour": 3_190_000,
		"storage_usd_micros_per_hour": 30_000,
	})

	// The root's ceiling is $5.00/hour and the account is already burning $6.38 of it.
	code, out := runCozy(t, root, "rental", "new", "h100-nvl", "--json")
	if code == 0 {
		t.Fatalf("a purchase was admitted over a ceiling the account has already breached\n%s", out)
	}
	if !strings.Contains(out, "rental.fleet_spend_cap") {
		t.Fatalf("the refusal is not the spend ceiling: %s", out)
	}
	if !strings.Contains(out, "$5.00/hour account limit; current spend is $6.38/hour") ||
		!strings.Contains(out, "cozy rental list") {
		t.Fatalf("the ceiling refusal does not explain account spend and its limit: %s", out)
	}
	if strings.Contains(out, "holds no record") || strings.Contains(out, "recorded only at the hub") {
		t.Fatalf("the ceiling refusal exposes controller bookkeeping: %s", out)
	}
	if stand.releases("pr-3333333333333333sem") != 0 {
		t.Fatalf("a spend refusal ended a machine on its own")
	}
}

// TestAHubWithNoListingIsNotAnEmptyFleet is the honesty arm. A hub older than th-199
// answers this GET from net/http's own router, and reading that as "the account owns
// nothing" would rebuild the exact silence this work exists to remove.
func TestAHubWithNoListingIsNotAnEmptyFleet(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-orphan-unpublished")
	stand.mu.Lock()
	stand.publishes = false
	stand.mu.Unlock()
	orphan(stand, "pr-5555555555555555punp", "punpun", 3_190_000)

	code, out := runCozy(t, root, "rental", "list")
	if code == 0 || !strings.Contains(out, "account rental census unavailable") || strings.Contains(out, "Current spend") {
		t.Fatalf("unavailable census claimed account totals [exit %d]\n%s", code, out)
	}
	if strings.Contains(out, "punpun") {
		t.Fatalf("a hub that publishes no listing somehow named a machine\n%s", out)
	}
	if !strings.Contains(out, "publishes no rental listing") {
		t.Fatalf("the board claims an empty fleet it cannot prove\n%s", out)
	}
	// And the release verb refuses the same way rather than pretending to a lookup.
	code, out = runCozy(t, root, "rental", "end", "punpun", "--json")
	if code == 0 || !strings.Contains(out, "rental.unknown") {
		t.Fatalf("ending an unlistable name did not refuse [exit %d]\n%s", code, out)
	}
	if stand.releases("pr-5555555555555555punp") != 0 {
		t.Fatalf("a refusal still sent a DELETE")
	}
	code, out = runCozy(t, root, "rental", "end", "pr-5555555555555555punp", "--json")
	if code != 0 || !strings.Contains(out, `"changed":true`) || stand.releases("pr-5555555555555555punp") != 1 {
		t.Fatalf("unavailable census blocked release by exact rental ID [exit %d]\n%s", code, out)
	}
}

// TestTheDaemonSaysUnrecordedSpendAndDoesNotEndIt is the idle ruling as behaviour.
//
// The daemon HAS an automatic shutdown — a fixed fifteen minutes — and it applies to the machines this host owns. It must not apply here. A
// rental the hub bills this account for that this host has no record of may be
// another host's live machine: this daemon holds none of its credentials, cannot see
// its work, and cannot tell an abandoned pod from one somebody is using. Ending it on
// that evidence is worse than the leak.
//
// So the daemon's answer is an ALARM, not a reaper: it says the machine, its rate and
// its unknown activity, once, and leaves control with the person paying.
func TestTheDaemonSaysUnrecordedSpendAndDoesNotEndIt(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-orphan-daemon")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	// An overdue owned witness proves the sweep ran; the unrecorded pod survives it.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 10.00\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")

	stand := newFakeRentalHub(t, port)
	stand.publishListing()
	orphan(stand, "pr-6666666666666666take", "takemikazuchi", 3_190_000)

	// A rental this host DOES own, so the reaper is provably running in this arm.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	stand.add("rental-orphan-owned", "heron")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ReadyAt: time.Now().Add(-time.Hour).UTC().Format(time.RFC3339Nano),
		ID:      "rental-orphan-owned", MachineName: "heron", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-orphan-owned.pem"),
	}))

	startDaemonProcess(t, root)
	awaitRentalGone(t, store, "rental-orphan-owned", 20*time.Second, logPath)
	awaitLog(t, logPath, "this host holds no record of it; activity is unknown to this controller",
		20*time.Second)

	// The reaper ran, took the machine it owns, and left the one it cannot account for.
	if stand.releases("rental-orphan-owned") != 1 {
		t.Fatalf("the idle release did not run in this arm\n%s", tail(logPath))
	}
	if stand.releases("pr-6666666666666666take") != 0 {
		t.Fatalf("the daemon ended a machine it holds no record of and cannot judge\n%s", tail(logPath))
	}
	// Said ONCE, however many sweeps pass over it.
	log, _ := os.ReadFile(logPath)
	if strings.Count(string(log), "this host holds no record of it; activity is unknown to this controller") != 1 {
		t.Fatalf("the unrecorded-spend alarm repeats on every sweep\n%s", tail(logPath))
	}
}
