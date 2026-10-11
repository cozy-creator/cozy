package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Provider facts are the operator's (th-254): a renter's Hub view omits them, and every rental
// view (the listing in full and as JSON, `rental show`, the history, the boot's words) then has
// no provider column, field or datacenter. An operator's view carries them, and they show. The
// operator's own verbs are hidden or say they are an operator's.
func TestRentalViewsShowProviderFactsOnlyWhereTheHubGivesThem(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-operator-facts")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	hub.add("pr-renter", "otter")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: "pr-renter", MachineName: "otter",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
	store.Close()
	hub.set("pr-renter", "spend_usd_micros", 1_230_000)
	hub.set("pr-renter", "spend_basis", "unmetered")
	// Another client's rental, booting its second attempt after a replan.
	hub.add("pr-booting", "heron")
	hub.set("pr-booting", "state", "acquiring")
	started := time.Now().UTC().Add(-10 * time.Minute)
	boot := func(where ...string) map[string]any {
		b := map[string]any{"attempt": 2, "state": "booting", "activity": "pulling_image", "started_at": started.Add(5 * time.Minute),
			"replanned_from": map[string]any{"attempt": 1, "started_at": started, "ended_at": started.Add(5 * time.Minute), "failure_code": "boot_stalled"}}
		if len(where) == 2 {
			b["datacenter"], b["replanned_from"].(map[string]any)["datacenter"] = where[1], where[0]
		}
		return b
	}
	hub.set("pr-booting", "boot", boot())
	if code, out := runCozy(t, root, "rental", "--help"); code != 0 || strings.Contains(out, "end-external") ||
		!strings.Contains(out, "operators only") {
		t.Fatalf("rental help shows operator tooling as a renter's [exit %d]:\n%s", code, out)
	}

	code, out := runCozy(t, root, "rental", "list", "--no-watch", "--full")
	if code != 0 || strings.Contains(out, "PROVIDER") || !strings.Contains(out, "$1.23 unmetered") ||
		!strings.Contains(out, "replanned after 5m (boot stalled)") || strings.Contains(out, "replanned from") {
		t.Fatalf("a renter's full listing [exit %d]:\n%s", code, out)
	}
	var document struct {
		Rentals []map[string]any `json:"rentals"`
	}
	code, out = runCozy(t, root, "rental", "list", "--json", "--full")
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Rentals) != 2 {
		t.Fatalf("rental list --json --full [exit %d]:\n%s", code, out)
	}
	for _, row := range document.Rentals {
		for key := range row {
			if strings.HasPrefix(key, "provider") {
				t.Fatalf("a renter's JSON row names %s: %v", key, row)
			}
		}
		if row["machine"] == "heron" && strings.Contains(fmt.Sprint(row["boot"]), "datacenter") {
			t.Fatalf("a renter's boot names a datacenter: %v", row["boot"])
		}
	}
	if code, out := runCozy(t, root, "rental", "show", "otter", "--full"); code != 0 || strings.Contains(out, "provider") ||
		!strings.Contains(out, "$1.23 unmetered") {
		t.Fatalf("a renter's rental show [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "list", "--all", "--full"); code != 0 || strings.Contains(out, "PROVIDER") ||
		!strings.Contains(out, "otter") {
		t.Fatalf("a renter's rental history [exit %d]:\n%s", code, out)
	}

	// An operator's view of the same rentals.
	hub.set("pr-renter", "provider", "vast")
	hub.set("pr-renter", "provider_machine_id", "144163")
	hub.set("pr-booting", "boot", boot("AP-IN-1", "EU-RO-1"))
	code, out = runCozy(t, root, "rental", "list", "--no-watch", "--full")
	if code != 0 || !strings.Contains(out, "PROVIDER MACHINE") || !strings.Contains(out, "144163") ||
		strings.Contains(out, "PROVIDER RESOURCE") || !strings.Contains(out, "now EU-RO-1") ||
		!strings.Contains(out, "replanned from AP-IN-1 after 5m") {
		t.Fatalf("an operator's full listing [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "show", "otter", "--full"); code != 0 || !strings.Contains(strings.Join(strings.Fields(out), " "), "provider: vast provider machine: 144163") {
		t.Fatalf("an operator's rental show [exit %d]:\n%s", code, out)
	}
}
