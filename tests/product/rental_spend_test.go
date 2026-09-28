package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Tensorhub says what each rental has cost (spend_usd_micros, spend_basis). The listing and
// `rental show` put it beside the rate, "est." until the provider's charges settle; a
// rental its Hub does not describe (an older Hub) shows "-" and fails nothing.
func TestRentalListingShowsAccruedSpend(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-spend")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	for id, machine := range map[string]string{"pr-estimate": "otter", "pr-billed": "heron"} {
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: id, MachineName: machine,
			SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
	}
	store.Close()
	hub.set("pr-estimate", "spend_usd_micros", 470_000)
	hub.set("pr-estimate", "spend_basis", "estimate")
	hub.set("pr-billed", "spend_usd_micros", 3_700_000)
	hub.set("pr-billed", "spend_basis", "provider_billed")

	board := func(want ...string) {
		t.Helper()
		code, out := runCozy(t, root, "rental", "list", "--no-watch")
		lines := map[string]string{}
		for _, line := range strings.Split(out, "\n") {
			for _, machine := range []string{"otter", "heron"} {
				if strings.HasPrefix(line, machine+" ") {
					lines[machine] = strings.Join(strings.Fields(line), " ")
				}
			}
		}
		if code != 0 || !strings.Contains(out, "SPENT") || !strings.Contains(out, want[0]) ||
			!strings.Contains(lines["otter"], want[1]) || !strings.Contains(lines["heron"], want[2]) {
			t.Fatalf("board does not show %q [exit %d]:\n%s", want, code, out)
		}
	}
	type document struct {
		Rentals []map[string]any `json:"rentals"`
		Spend   *int64           `json:"spend_usd_micros"`
		Basis   string           `json:"spend_basis"`
	}
	listed := func() (document, map[string]map[string]any) {
		t.Helper()
		code, out := runCozy(t, root, "rental", "list", "--json")
		var doc document
		if code != 0 || json.Unmarshal([]byte(out), &doc) != nil {
			t.Fatalf("rental list --json failed [exit %d]:\n%s", code, out)
		}
		rows := map[string]map[string]any{}
		for _, row := range doc.Rentals {
			rows[row["machine"].(string)] = row
		}
		return doc, rows
	}

	board("Current spend per hour: $0.20 · accrued $4.17 est.", "$0.10 $0.47 est.", "$0.10 $3.70 ")
	doc, rows := listed()
	if doc.Spend == nil || *doc.Spend != 4_170_000 || doc.Basis != "estimate" ||
		rows["otter"]["spend_usd_micros"] != float64(470_000) || rows["otter"]["spend_basis"] != "estimate" ||
		rows["heron"]["spend_usd_micros"] != float64(3_700_000) || rows["heron"]["spend_basis"] != "provider_billed" {
		t.Fatalf("JSON lost the spend facts: %+v", doc)
	}
	if code, out := runCozy(t, root, "rental", "show", "heron"); code != 0 ||
		!strings.Contains(out, "pr-billed") || !strings.Contains(out, "spent:") ||
		!strings.Contains(out, "$3.70\n") {
		t.Fatalf("rental show does not show the billed spend [exit %d]:\n%s", code, out)
	}
	var shown map[string]any
	code, out := runCozy(t, root, "rental", "show", "pr-estimate", "--json")
	if code != 0 || json.Unmarshal([]byte(out), &shown) != nil || shown["machine"] != "otter" ||
		shown["spend_usd_micros"] != float64(470_000) || shown["spend_basis"] != "estimate" {
		t.Fatalf("rental show --json lost the spend facts [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "show", "nobody", "--json"); code == 0 || !strings.Contains(out, `"rental.unknown"`) {
		t.Fatalf("rental show of an unknown rental did not refuse [exit %d]:\n%s", code, out)
	}

	// A Hub older than the fields: that rental's spend is unknown, never $0.00.
	hub.mu.Lock()
	delete(hub.rentals["pr-billed"], "spend_usd_micros")
	delete(hub.rentals["pr-billed"], "spend_basis")
	hub.mu.Unlock()
	board("accrued at least $0.47 est.", "$0.47 est.", "$0.10 - ")
	if doc, rows := listed(); doc.Spend != nil || rows["heron"]["spend_basis"] != nil || rows["otter"]["spend_basis"] != "estimate" {
		t.Fatalf("JSON invented spend an older Hub never said: %+v", doc)
	}
	hub.mu.Lock()
	delete(hub.rentals["pr-estimate"], "spend_usd_micros")
	delete(hub.rentals["pr-estimate"], "spend_basis")
	hub.mu.Unlock()
	board("Current spend per hour: $0.20 · accrued -\n", "$0.10 - ", "$0.10 - ")
}
