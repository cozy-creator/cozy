package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
)

// `cozy rental show` answers for a rental that has ended, by its id or by a name several rentals
// bore (the newest), from any hub this host is signed in to: who ended it, when, and what it
// cost, on the Hub's clock. A name no hub knows is unknown.
func TestRentalShowSaysHowAnEndedRentalEnded(t *testing.T) {
	ended := map[string]map[string]any{
		"pr-kanna-older": {"rental_id": "pr-kanna-older", "name": "kanna", "state": "released",
			"requested_accelerator_model": "RTX PRO 6000", "accelerator_count": 4, "hourly_rate_usd_micros": 8_400_000,
			"created_at": "2026-10-05T09:00:00Z", "ended_at": "2026-10-05T10:00:00Z", "release_cause": "owner_stop",
			"spend_usd_micros": 8_400_000, "spend_basis": "provider_billed"},
		"pr-kanna-newest": {"rental_id": "pr-kanna-newest", "name": "kanna", "state": "released",
			"requested_accelerator_model": "RTX PRO 6000", "accelerator_count": 4, "hourly_rate_usd_micros": 8_400_000,
			"created_at": "2026-10-06T09:21:05Z", "ended_at": "2026-10-06T17:37:50Z", "release_cause": "idle_unreached",
			"spend_usd_micros": 69_500_000, "spend_basis": "estimate"},
	}
	other := newAccountHubWith(t, func(mux *http.ServeMux) {
		mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
			rows := []map[string]any{}
			for _, row := range ended {
				if r.URL.Query().Get("state") == "all" && row["name"] == r.URL.Query().Get("name") {
					rows = append(rows, row)
				}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
		})
		mux.HandleFunc("GET /v1/rentals/{id}", func(w http.ResponseWriter, r *http.Request) {
			row, ok := ended[r.PathValue("id")]
			if !ok {
				w.WriteHeader(http.StatusNotFound)
				_, _ = w.Write([]byte(`{"error":{"code":"rental.not_found","message":"no rental"}}`))
				return
			}
			_ = json.NewEncoder(w).Encode(row)
		})
	})
	current := newFakeRentalHub(t, 0)
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(fmt.Sprintf(
		"tensorhub_url: http://127.0.0.1:%d\ntensorhub_token: rental-idle-test\nhubs:\n  other: %s\n", current.port(), other.URL)), 0o600))
	login := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "auth", "login", "proof@example.test", "--tensorhub", "other")
	login.Env = childEnv(t, root)
	login.Stdin = strings.NewReader("123456\nproof\n") //cozy:stdin-value test login code and account name
	var out bytes.Buffer
	login.Stdout, login.Stderr = &out, &out
	if err := login.Run(); err != nil {
		t.Fatalf("login: %v\n%s", err, out.String())
	}

	show := func(subject string) map[string]any {
		t.Helper()
		code, out := runCozy(t, root, "rental", "show", subject, "--json")
		var record map[string]any
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &record) != nil {
			t.Fatalf("rental show %s [exit %d]\n%s", subject, code, out)
		}
		return record
	}
	newest := show("kanna")
	if newest["rental_id"] != "pr-kanna-newest" || newest["hub"] != "other" || newest["state"] != "released" ||
		newest["release_cause"] != "idle_unreached" || newest["ended_at"] != "2026-10-06T17:37:50Z" ||
		newest["spend_usd_micros"] != float64(69_500_000) || newest["spend_basis"] != "estimate" {
		t.Fatalf("by name, not the newest kanna and how it ended: %v", newest)
	}
	if older := show("pr-kanna-older"); older["rental_id"] != "pr-kanna-older" || older["release_cause"] != "owner_stop" ||
		older["spend_basis"] != "provider_billed" {
		t.Fatalf("by id: %v", older)
	}
	if code, human := runCozy(t, root, "rental", "show", "kanna"); code != 0 ||
		!strings.Contains(human, "idle_unreached") || !strings.Contains(human, "$69.50 est.") {
		t.Fatalf("rental show kanna [exit %d]\n%s", code, human)
	}
	if code, out := runCozy(t, root, "rental", "show", "nobody"); code == 0 || !strings.Contains(out, `no rental is named "nobody"`) {
		t.Fatalf("an unknown name [exit %d]\n%s", code, out)
	}
}
