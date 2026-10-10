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
	for _, field := range []string{"running", "queued", "release_due_at"} {
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

// TestTheDaemonSaysUnrecordedSpendAndDoesNotEndIt is the idle ruling as behaviour.
//
// Rentals end themselves after fifteen idle minutes; this daemon ends none on its own. A
// rental the hub bills this account for that this host has no record of may be
// another host's live machine: this daemon holds none of its credentials, cannot see
// its work, and cannot tell an abandoned pod from one somebody is using. Ending it on
// that evidence is worse than the leak.
//
// So the daemon's answer is an ALARM, not a reaper: it says the machine, its rate and
// its unknown activity, once, and leaves control with the person paying.
func TestTheDaemonSaysUnrecordedSpendAndDoesNotEndIt(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-orphan-daemon")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	stand := newFakeRentalHub(t, 0)
	port := stand.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	logPath := filepath.Join(root, "daemon.log")

	stand.publishListing()
	orphan(stand, "pr-6666666666666666take", "takemikazuchi", 3_190_000)

	// A rental this host DOES own puts the hub on its reconcile list.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	stand.add("rental-orphan-owned", "heron")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ReadyAt: time.Now().UTC().Format(time.RFC3339Nano),
		ID:      "rental-orphan-owned", MachineName: "heron", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-orphan-owned.pem"),
	}))

	startDaemonProcess(t, root)
	awaitLog(t, logPath, "this host holds no record of it; activity is unknown to this controller",
		20*time.Second)
	// Several fleet ticks pass over it: it is said ONCE and never ended.
	time.Sleep(6 * time.Second)
	if stand.releases("pr-6666666666666666take") != 0 {
		t.Fatalf("the daemon ended a machine it holds no record of and cannot judge\n%s", tail(logPath))
	}
	log, _ := os.ReadFile(logPath)
	if strings.Count(string(log), "this host holds no record of it; activity is unknown to this controller") != 1 {
		t.Fatalf("the unrecorded-spend alarm repeats on every tick\n%s", tail(logPath))
	}
}

// A machine billed to this account but attached to another Creator home is a row of
// the ordinary board, marked as such, with its rate: the header and the table agree.
func TestRentalListShowsOtherClientsMachinesBesideItsOwn(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-other-client")
	stand.publishListing()
	const own = "pr-2b2b2b2b2b2b2b2b2b2b"
	stand.add(own, "loran")
	stand.set(own, "state", "ready")
	stand.setRate(own, 7_021_700)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{
		ID: own, MachineName: "loran", SKU: "h100-sxm5-80gb", AcceleratorModel: "NVIDIA H100 80GB HBM3", AcceleratorCount: 2,
		HourlyRateUSDMicros: 7_021_700, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, own+".pem"),
	}))
	store.Close()
	orphan(stand, "pr-3c3c3c3c3c3c3c3conar", "onara", 1_637_399)
	stand.set("pr-3c3c3c3c3c3c3c3conar", "state", "ready")
	stand.set("pr-3c3c3c3c3c3c3c3conar", "requested_accelerator_model", "NVIDIA A100-SXM4-80GB")
	stand.set("pr-3c3c3c3c3c3c3c3conar", "accelerator_count", 1)

	code, out := runCozy(t, root, "rental", "list", "--no-watch")
	if code != 0 {
		t.Fatalf("rental list failed [exit %d]\n%s", code, out)
	}
	if !strings.Contains(out, "Remote machines running: 2") || !strings.Contains(out, "Current spend per hour: $8.66") {
		t.Fatalf("the header does not count both machines\n%s", out)
	}
	if !regexp.MustCompile(`(?m)^MACHINE\s+SKU\s+GPUS\s+STATE\s`).MatchString(out) {
		t.Fatalf("the GPUS column does not follow SKU\n%s", out)
	}
	if !regexp.MustCompile(`(?m)^loran\s+h100-sxm5-80gb\s+2\s+ready\s+\$7\.02\s`).MatchString(out) {
		t.Fatalf("this host's own machine is not a plain row\n%s", out)
	}
	if !regexp.MustCompile(`(?m)^onara\s+NVIDIA A100-SXM4-80GB\s+1\s+ready \(other client\)\s+\$1\.64\s`).MatchString(out) {
		t.Fatalf("the other client's machine is not a marked row with its rate\n%s", out)
	}
	if rows := regexp.MustCompile(`(?m)^(loran|onara)\s`).FindAllString(out, -1); len(rows) != 2 {
		t.Fatalf("the table has %d machine rows for a header of 2\n%s", len(rows), out)
	}
	code, out = runCozy(t, root, "rental", "list", "--json")
	var document struct {
		Rentals []map[string]any `json:"rentals"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Rentals) != 2 {
		t.Fatalf("rental list --json [%d]: %s", code, out)
	}
	for _, row := range document.Rentals {
		want := map[string]float64{"loran": 2, "onara": 1}[row["machine"].(string)]
		if row["gpus"] != want {
			t.Fatalf("%s gpus = %v, want %v: %s", row["machine"], row["gpus"], want, out)
		}
	}
}
