//go:build !windows

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

func TestStableMachineNameCLIAndOriginalAlias(t *testing.T) {
	root := filepath.Join(scratchBase, "stable-machine-name")
	must(t, os.MkdirAll(root, 0o755))
	h := newFakeRentalHub(t, 0)
	origin := fmt.Sprintf("http://127.0.0.1:%d", h.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+origin+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	h.add("pr-stable", "julian")
	h.set("pr-stable", "state", "acquiring")
	h.set("pr-stable", "machine_name", "mikatan-356e23")
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	fatal(t, st.RecordRental(records.Rental{ID: "pr-stable", MachineName: "julian", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, State: "acquiring", Hub: origin}))
	st.Close()
	code, out := runCozy(t, root, "rental", "list", "--json", "--full")
	if code != 0 || !strings.Contains(out, "mikatan-356e23") {
		t.Fatalf("stable name missing from list [%d]: %s", code, out)
	}
	for _, name := range []string{"julian", "mikatan-356e23", "pr-stable"} {
		code, out := runCozy(t, root, "rental", "show", name, "--json")
		var got map[string]any
		if code != 0 || json.Unmarshal([]byte(out), &got) != nil || got["machine"] != "mikatan-356e23" {
			t.Fatalf("show %s [%d]: %s", name, code, out)
		}
	}
	st, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	row, problem := st.RentalByMachine("mikatan-356e23")
	fatal(t, problem)
	if row == nil || row.MachineName != "julian" || row.ID != "pr-stable" {
		t.Fatalf("alias or ID changed: %+v", row)
	}
	st.Close()

	// The same physical host can back another live rental owned by another client.
	// A host label must never pick an arbitrary DELETE target.
	h.add("pr-collision", "egami")
	h.set("pr-collision", "machine_name", "mikatan-356e23")
	if code, out := runCozy(t, root, "rental", "list"); code != 0 || !strings.Contains(out, "pr-stable") || !strings.Contains(out, "pr-collision") {
		t.Fatalf("two rentals on one host are indistinguishable rows [%d]: %s", code, out)
	}
	code, out = runCozy(t, root, "rental", "end", "mikatan-356e23", "--json")
	if code == 0 || !strings.Contains(out, "rental.machine_name_ambiguous") {
		t.Fatalf("ambiguous host was accepted [%d]: %s", code, out)
	}
	h.mu.Lock()
	if len(h.released) != 0 {
		h.mu.Unlock()
		t.Fatal("ambiguous host mutated a rental")
	}
	delete(h.rentals, "pr-collision")
	h.mu.Unlock()
	code, out = runCozy(t, root, "rental", "end", "mikatan-356e23", "--json")
	if code != 0 {
		t.Fatalf("end stable machine [%d]: %s", code, out)
	}
	h.mu.Lock()
	released := h.released["pr-stable"]
	h.mu.Unlock()
	if released == 0 {
		t.Fatal("stable name did not resolve to the correct provider rental")
	}

	code, out = runCozy(t, root, "rental", "show", "mikatan-356e23", "--json")
	if code != 0 || !strings.Contains(out, "pr-stable") {
		t.Fatalf("ended host history missing [%d]: %s", code, out)
	}
	h.add("pr-rerented", "shanks")
	h.set("pr-rerented", "state", "acquiring")
	h.set("pr-rerented", "machine_name", "mikatan-356e23")
	code, out = runCozy(t, root, "rental", "show", "mikatan-356e23", "--json")
	if code != 0 || !strings.Contains(out, "pr-rerented") {
		t.Fatalf("re-rented host did not reuse its name [%d]: %s", code, out)
	}
	code, out = runCozy(t, root, "rental", "list", "--all", "--json")
	if code != 0 || strings.Count(out, `"machine":"mikatan-356e23"`) != 2 {
		t.Fatalf("re-rental history lost original IDs [%d]: %s", code, out)
	}

	// Once both have ended, the host name names neither: history never picks one.
	h.setState("pr-rerented", "released", "")
	code, out = runCozy(t, root, "rental", "show", "mikatan-356e23", "--json")
	if code == 0 || !strings.Contains(out, "rental.machine_name_ambiguous") || !strings.Contains(out, "pr-stable") || !strings.Contains(out, "pr-rerented") {
		t.Fatalf("an ended host name chose one of its rentals [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "show", "pr-stable", "--json"); code != 0 || !strings.Contains(out, `"machine":"mikatan-356e23"`) {
		t.Fatalf("an ended rental's ID did not show it [%d]: %s", code, out)
	}
}
