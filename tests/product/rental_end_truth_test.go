package producttest

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
)

// The rental END verbs against a hub that is the authority (cl-193).
//
// Measured live 2026-09-07: six H100 NVLs sat in `booting`/`pending_acquisition` on the
// provider, provisioned and billing, with NO local record on the host that bought them —
// tensorhub's create answer had omitted `accelerator_count` (th-198, hub PR #560), the
// width fence refused it, and Creator filed the ask as having bought nothing. Asked to end
// each machine by name, the CLI answered
//
//	state:   ended
//	changed: false
//
// and sent no DELETE at all. Releasing them took an admin token against the hub.
//
// Two things had to be true for that. The command read the answer to a lookup it could not
// perform — a machine word is Creator's own alias, so `GET /v1/rentals/aeirik` 404s for a
// key nothing is filed under — as proof of absence. And the ask that bought the pod had
// already been destroyed locally, because a create answer this client could not USE was
// treated as a create the hub had REFUSED.
//
// These arms hold both closed, through the real CLI process, the real records authority
// and a hub answering the real routes.

// rentalEndRoot writes one product root pointed at a stand-in hub.
func rentalEndRoot(t *testing.T, name string) (string, string, *fakeRentalHub) {
	t.Helper()
	base := filepath.Join(os.TempDir(), "cozy-product-test")
	must(t, os.MkdirAll(base, 0o755))
	root, err := os.MkdirTemp(base, name+"-")
	must(t, err)
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 5.00\n  idle_release_s: 0\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	return root, hubURL, newFakeRentalHub(t, port)
}

// A complete account listing proves that a mistyped machine name is absent; it
// must not release a different live rental or claim that a deletion occurred.
func TestEndingAnUnknownMachineDoesNotClaimItEnded(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-end-unrecorded")
	stand.add("pr-0badc0de0badc0de0bad", "koharu")

	code, out := runCozy(t, root, "rental", "end", "unknown-name", "--json")
	if code == 0 {
		t.Fatalf("ending a machine absent from the account exited 0\n%s", out)
	}
	if strings.Contains(out, `"state":"ended"`) {
		t.Fatalf("the CLI reported an end it did not perform\n%s", out)
	}
	if !strings.Contains(out, "rental.unknown") {
		t.Fatalf("the refusal is not the typed one that names what happened\n%s", out)
	}
	// The refusal must distinguish an unknown name from a successful deletion.
	if !strings.Contains(out, "none of them") {
		t.Fatalf("the refusal does not report the account lookup\n%s", out)
	}
	if n := stand.releases("pr-0badc0de0badc0de0bad"); n != 0 {
		t.Fatalf("the hub saw %d release(s) from a command that refused", n)
	}
	// And the machine is still there, still billing — which is the point of refusing.
	if code, out := runCozy(t, root, "rental", "end", "pr-0badc0de0badc0de0bad", "--json"); code != 0 ||
		!strings.Contains(out, `"state":"ended"`) || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("naming the rental id did not release the live pod [exit %d]\n%s", code, out)
	}
	if n := stand.releases("pr-0badc0de0badc0de0bad"); n != 1 {
		t.Fatalf("the hub saw %d release(s) of the named rental, wanted 1", n)
	}
}

// TestEndingAReleasedRentalIsAnHonestNoOp keeps the legitimate idempotent arm reachable.
// `state: ended, changed: false` is the right answer for a rental this host released
// earlier — and only for that. The record that entitles it to say so is the paid
// operation, which outlives the rental row on purpose.
func TestEndingAReleasedRentalIsAnHonestNoOp(t *testing.T) {
	root, hubURL, stand := rentalEndRoot(t, "rental-end-idempotent")
	const id = "pr-1a1a1a1a1a1a1a1a1a1a"
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// A real acquisition, authored the way the paid path authors one: the store mints the
	// machine word and the ask is recorded BEFORE the pod exists.
	machine := ""
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{
		Key: "idempotent-proof", Hub: hubURL, Reason: "cozy rental new cpu",
		HourlyRateUSDMicros: 100_000,
	}, 5_000_000, 10_000, func(name string) ([]byte, string, *exit.Error) {
		machine = name
		body, problem := hub.RentalRequestBytes(name, "cpu", strings.Repeat("ab", 32),
			base64.RawURLEncoding.EncodeToString(make([]byte, 32)), hub.DeclaredWorkload{}, nil)
		return body, "digest-" + name, problem
	}, nil)

	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, id, "attached"))
	stand.add(id, machine)
	fatal(t, store.RecordRental(records.Rental{
		ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
	}))
	store.Close()

	code, out := runCozy(t, root, "rental", "end", machine, "--json")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("ending a recorded live rental did not destroy the pod [exit %d]\n%s", code, out)
	}
	if n := stand.releases(id); n != 1 {
		t.Fatalf("the hub saw %d release(s), wanted 1", n)
	}
	// Second time: the rental row is forgotten, the hub says released, and the PAID ASK is
	// what entitles this host to answer at all — it outlives the row on purpose.
	code, out = runCozy(t, root, "rental", "end", machine, "--json")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) || !strings.Contains(out, `"changed":false`) {
		t.Fatalf("the second end is not an idempotent no-op [exit %d]\n%s", code, out)
	}
	if n := stand.releases(id); n != 1 {
		t.Fatalf("the idempotent no-op sent another DELETE: %d", n)
	}
	if strings.Contains(out, "already released") {
		t.Fatalf("the no-op still explains itself by local absence\n%s", out)
	}
	if !strings.Contains(out, "idempotent-proof") {
		t.Fatalf("the no-op does not name the record that proves this host owned it\n%s", out)
	}
}

// TestAnAcceptedAskSurvivesAnUnusableCreateAnswer is the incident's ROOT, reproduced
// through the paid path: tensorhub answers 202 for a pod it really created, with the
// create answer th-198 was missing a field from. Creator cannot attach it — that refusal
// is correct — but the ask bought a machine, so the machine must remain endable by name.
//
// On the code this was written against, `rental new` filed the operation as `rejected`,
// deleted its pending credential, and left nothing on this host that named the pod.
func TestAnAcceptedAskSurvivesAnUnusableCreateAnswer(t *testing.T) {
	root, _, stand := rentalEndRoot(t, "rental-end-unusable-answer")
	stand.publishListing()
	stand.setSKUs(map[string]any{
		"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"price_usd_micros_per_hour": 100_000, "storage_usd_micros_per_hour": 10_000,
		"base_worker_profile": "torch2.13.0-cu130-cp312-linux-x86",
	})
	var mu sync.Mutex
	var asked string
	// The hub's own th-198 answer, verbatim in shape: a real accepted rental whose create
	// response omits the width.
	stand.rent = func(request map[string]any) map[string]any {
		mu.Lock()
		defer mu.Unlock()
		asked, _ = request["name"].(string)
		return map[string]any{
			"rental_id": "pr-th198th198th198th19", "name": asked, "state": "pending_acquisition",
			"requested_accelerator_model": "CPU", "hourly_rate_usd_micros": 100_000,
		}
	}
	startDaemonProcess(t, root)

	code, out := runCozy(t, root, "rental", "new", "cpu", "--json")
	if code == 0 {
		t.Fatalf("Creator attached a pod whose width the hub never stated\n%s", out)
	}
	if !strings.Contains(out, "accelerator_count") && !strings.Contains(out, "accelerator count") {
		t.Fatalf("the refusal is not the width fence\n%s", out)
	}
	mu.Lock()
	machine := asked
	mu.Unlock()
	if machine == "" {
		t.Fatal("the hub never saw a create ask")
	}

	// THE ASK IS NOT A REFUSAL. The operation stays open and names the pod the hub made.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	operations, problem := store.RentalOperations()
	fatal(t, problem)
	store.Close()
	if len(operations) != 1 {
		t.Fatalf("wanted one recorded paid ask, have %d", len(operations))
	}
	if operations[0].State == "rejected" {
		t.Fatalf("an ask the hub ACCEPTED was filed as having bought nothing: %+v", operations[0])
	}
	if operations[0].RentalID != "pr-th198th198th198th19" {
		t.Fatalf("the identity the hub did give was thrown away: %+v", operations[0])
	}

	// The board shows the paid ask rather than an empty fleet.
	code, board := runCozy(t, root, "rental", "list", "--json", "--full")
	if code != 0 {
		t.Fatalf("rental list failed [exit %d]\n%s", code, board)
	}
	var listed struct {
		Rentals                    []map[string]any `json:"rentals"`
		UnattachedRentalOperations int              `json:"unattached_rental_operations"`
		MachinesRunning            int              `json:"machines_running"`
		HourlySpendUSDMicros       int64            `json:"hourly_spend_usd_micros"`
	}
	if err := json.Unmarshal([]byte(board), &listed); err != nil {
		t.Fatalf("rental list is not JSON: %v\n%s", err, board)
	}
	if listed.UnattachedRentalOperations != 0 || len(listed.Rentals) != 1 ||
		listed.Rentals[0]["machine"] != machine || listed.MachinesRunning != 1 ||
		listed.HourlySpendUSDMicros != 110_000 {
		t.Fatalf("a paid pod with no local record is invisible on the board\n%s", board)
	}

	if code, rates := runCozy(t, root, "rental", "list", "--fields=machine,$/hour"); code != 0 || !strings.Contains(rates, fmt.Sprintf("$%.2f", float64(*operations[0].EstimatedHourlyRateUSDMicros)/1_000_000)) {
		t.Fatalf("unattached paid ask has no hourly rate [exit %d]\n%s", code, rates)
	}

	// And the machine word ends the machine, which is the whole ask.
	code, out = runCozy(t, root, "rental", "end", machine, "--json")
	if code != 0 || !strings.Contains(out, `"state":"ended"`) || !strings.Contains(out, `"changed":true`) {
		t.Fatalf("the pod bought by an unusable answer could not be ended by name [exit %d]\n%s", code, out)
	}
	if n := stand.releases("pr-th198th198th198th19"); n != 1 {
		t.Fatalf("the hub saw %d release(s) of the orphaned pod, wanted 1", n)
	}
}
