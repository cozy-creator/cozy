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

// `cozy rental list --ended` lists the ended rentals of every hub this host is signed in to,
// the latest ended first: why each ended, when, and what it cost. The default list leaves
// them out and says where they are; --tensorhub preserves every hub; a long history says how
// many rows it hid and how to see them.
func TestRentalListEndedSaysWhyWhenAndWhatEachCost(t *testing.T) {
	other := newAccountHubWith(t, func(mux *http.ServeMux) {
		mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
			rows := []map[string]any{}
			if r.URL.Query().Get("state") == "all" {
				rows = append(rows, map[string]any{"rental_id": "pr-kanna-newest", "name": "kanna", "state": "released",
					"requested_accelerator_model": "RTX PRO 6000", "accelerator_count": 4, "hourly_rate_usd_micros": 8_400_000,
					"created_at": "2026-10-06T09:21:05Z", "ended_at": "2026-10-06T17:37:50Z", "release_cause": "idle_unreached",
					"spend_usd_micros": 69_500_000, "spend_basis": "estimate"})
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"rentals": rows})
		})
	})
	current := newFakeRentalHub(t, 0)
	current.add("pr-otter-live", "otter")
	ended := func(id, name, endedAt, cause string, spend int64) {
		current.mu.Lock()
		defer current.mu.Unlock()
		current.rentals[id] = map[string]any{"rental_id": id, "name": name, "state": "released",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000,
			"created_at": "2026-10-05T08:00:00Z", "ended_at": endedAt, "release_cause": cause,
			"spend_usd_micros": spend, "spend_basis": "provider_billed"}
	}
	ended("pr-heron-ended", "heron", "2026-10-06T12:00:00Z", "released_by_pod", 1_250_000)
	ended("pr-wren-ended", "wren", "2026-10-05T09:00:00Z", "owner_stop", 100_000)
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

	if code, live := runCozy(t, root, "rental", "list", "--no-watch"); code != 0 || !strings.Contains(live, "otter") ||
		strings.Contains(live, "heron") || strings.Contains(live, "kanna") || !strings.Contains(live, "Next: cozy rental list --ended") {
		t.Fatalf("the default list shows ended rentals or hides where they are [exit %d]\n%s", code, live)
	}
	type endedRow struct {
		Machine      string `json:"machine"`
		RentalID     string `json:"rental_id"`
		Hub          string `json:"hub"`
		State        string `json:"state"`
		ReleaseCause string `json:"release_cause"`
		EndedAt      string `json:"ended_at"`
		Spend        *int64 `json:"spend_usd_micros"`
		SpendBasis   string `json:"spend_basis"`
	}
	list := func(args ...string) []endedRow {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"--json", "rental", "list", "--ended"}, args...)...)
		var document struct {
			Ended []endedRow `json:"ended_rentals"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
			t.Fatalf("rental list --ended %v [exit %d]\n%s", args, code, out)
		}
		return document.Ended
	}
	rows := list()
	if len(rows) != 3 || rows[0].Machine != "kanna" || rows[1].Machine != "heron" || rows[2].Machine != "wren" {
		t.Fatalf("not every hub's ended rentals, latest ended first: %+v", rows)
	}
	kanna, heron := rows[0], rows[1]
	if kanna.Hub != "other" || kanna.ReleaseCause != "idle_unreached" || kanna.EndedAt != "2026-10-06T17:37:50Z" ||
		kanna.Spend == nil || *kanna.Spend != 69_500_000 || kanna.SpendBasis != "estimate" || kanna.State != "released" {
		t.Fatalf("kanna does not say why, when and what it cost: %+v", kanna)
	}
	if heron.RentalID != "pr-heron-ended" || heron.ReleaseCause != "released_by_pod" || heron.Spend == nil ||
		*heron.Spend != 1_250_000 || heron.SpendBasis != "provider_billed" {
		t.Fatalf("heron: %+v", heron)
	}
	if scoped := list("--tensorhub", "other"); len(scoped) != 3 || scoped[0].Machine != "kanna" {
		t.Fatalf("--tensorhub=other hides another hub's ended rentals: %+v", scoped)
	}
	code, human := runCozy(t, root, "rental", "list", "--ended")
	for _, want := range []string{"Ended rentals: 3 · accrued $70.85 est.", "ENDED BY", "idle_unreached", "released_by_pod",
		"owner_stop", "2026-10-06T17:37:50Z", "$69.50 est.", "$1.25", "HUB"} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("the human ended list lacks %q [exit %d]\n%s", want, code, human)
		}
	}
	if strings.Contains(human, "otter") {
		t.Fatalf("a live rental is listed as ended\n%s", human)
	}

	for index := 1; index <= 22; index++ {
		ended(fmt.Sprintf("pr-old-%02d", index), fmt.Sprintf("old%02d", index), fmt.Sprintf("2026-10-01T%02d:00:00Z", index), "owner_stop", 0)
	}
	if code, human := runCozy(t, root, "rental", "list", "--ended"); code != 0 ||
		!strings.Contains(human, "5 more not shown. Use --full to show all.") || strings.Contains(human, "old01") {
		t.Fatalf("a long history does not say what it hid [exit %d]\n%s", code, human)
	}
	if code, human := runCozy(t, root, "rental", "list", "--ended", "--full"); code != 0 ||
		!strings.Contains(human, "old01") || strings.Contains(human, "not shown") {
		t.Fatalf("--full does not show every ended rental [exit %d]\n%s", code, human)
	}
	if code, out := runCozy(t, root, "rental", "list", "--ended", "--watch"); code == 0 || !strings.Contains(out, "cannot --watch") {
		t.Fatalf("--ended --watch [exit %d]\n%s", code, out)
	}
}
