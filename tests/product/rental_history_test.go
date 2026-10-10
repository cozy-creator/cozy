//go:build !windows

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
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
)

// `cozy rental list --all` lists every rental of every hub this host is signed in to, live
// first and then the latest ended: its SKU, when it started and ended, how long it lived,
// the runs this computer sent it, what it cost, and why it ended. `--ended` is the same view
// of the ended ones. Runs are counted from this computer's records in one grouped read; a
// rental this computer never recorded says so. The default list leaves history out and says
// where it is; --tensorhub preserves every hub; a long history says what it hid.
func TestRentalHistoryShowsEachRentalsLifetimeRunsAndCost(t *testing.T) {
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
	current.mu.Lock()
	current.rentals["pr-otter-live"]["created_at"] = time.Now().Add(-90 * time.Minute).UTC().Format(time.RFC3339Nano)
	current.mu.Unlock()
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

	// This computer bought heron (its SKU is in the purchase record) and sent runs to heron
	// and otter, settled as they settle; it never recorded wren or kanna.
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "heron-op", Hub: "http://127.0.0.1", Reason: "rental", HourlyRateUSDMicros: 100_000},
		func(name string) ([]byte, string, *exit.Error) {
			return []byte(`{"name":"` + name + `","sku":"cpu","accelerator_count":1}`), "proof", nil
		})
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, "pr-heron-ended", "released"))
	run := func(id, rental, state string, dispatched bool) {
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, Package: "proof/history", Entrypoint: "main", Kind: "job",
			Payload: []byte(`{}`), BodyDigest: childDigest("1"), Rental: true, RequestedRental: rental})
		fatal(t, problem)
		if dispatched {
			pinned, problem := store.PinRental(id, rental, nil)
			fatal(t, problem)
			if !pinned {
				t.Fatalf("run %s was not placed on %s", id, rental)
			}
		}
		fatal(t, store.SettleRequest(id, state))
	}
	for i := range 3 {
		run(fmt.Sprintf("heron-ok-%d", i), "pr-heron-ended", "succeeded", true)
	}
	run("heron-failed", "pr-heron-ended", "failed", true)
	run("heron-never-placed", "pr-heron-ended", "failed", false)
	run("heron-canceled", "pr-heron-ended", "canceled", true)
	run("otter-ok-0", "pr-otter-live", "succeeded", true)
	run("otter-ok-1", "pr-otter-live", "succeeded", true)

	if code, live := runCozy(t, root, "rental", "list", "--no-watch"); code != 0 || !strings.Contains(live, "otter") ||
		strings.Contains(live, "heron") || strings.Contains(live, "kanna") || !strings.Contains(live, "Next: cozy rental list --all") {
		t.Fatalf("the default list shows ended rentals or hides where they are [exit %d]\n%s", code, live)
	}
	type historyRow struct {
		Machine      string `json:"machine"`
		RentalID     string `json:"rental_id"`
		SKU          string `json:"sku"`
		Hub          string `json:"hub"`
		State        string `json:"state"`
		ReleaseCause string `json:"release_cause"`
		RentedAt     string `json:"rented_at"`
		EndedAt      string `json:"ended_at"`
		Lifetime     *int64 `json:"lifetime_s"`
		Succeeded    *int   `json:"runs_succeeded"`
		Failed       *int   `json:"runs_failed"`
		Canceled     *int   `json:"runs_canceled"`
		Spend        *int64 `json:"spend_usd_micros"`
		SpendBasis   string `json:"spend_basis"`
	}
	list := func(view string, args ...string) []historyRow {
		t.Helper()
		code, out := runCozy(t, root, append([]string{"--json", "rental", "list", view}, args...)...)
		var document struct {
			All   []historyRow `json:"rental_history"`
			Ended []historyRow `json:"ended_rentals"`
			Runs  int          `json:"runs_succeeded"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil {
			t.Fatalf("rental list %s %v [exit %d]\n%s", view, args, code, out)
		}
		return append(document.All, document.Ended...)
	}
	rows := list("--all", "--full")
	if len(rows) != 4 || rows[0].Machine != "otter" || rows[1].Machine != "kanna" || rows[2].Machine != "heron" || rows[3].Machine != "wren" {
		t.Fatalf("not every hub's rentals, live first and then the latest ended: %+v", rows)
	}
	otter, kanna, heron, wren := rows[0], rows[1], rows[2], rows[3]
	if otter.State != "ready" || otter.EndedAt != "" || otter.Lifetime == nil || *otter.Lifetime < 89*60 || *otter.Lifetime > 120*60 ||
		otter.Succeeded == nil || *otter.Succeeded != 2 || *otter.Failed != 0 || otter.SKU != "" {
		t.Fatalf("live otter: %+v", otter)
	}
	if kanna.Hub != "other" || kanna.ReleaseCause != "idle_unreached" || kanna.EndedAt != "2026-10-06T17:37:50Z" ||
		kanna.Lifetime == nil || *kanna.Lifetime != 8*3600+16*60+45 || kanna.Succeeded != nil ||
		kanna.Spend == nil || *kanna.Spend != 69_500_000 || kanna.SpendBasis != "estimate" || kanna.State != "released" {
		t.Fatalf("kanna does not say when, how long, why and what it cost: %+v", kanna)
	}
	if heron.RentalID != "pr-heron-ended" || heron.SKU != "cpu" || heron.ReleaseCause != "released_by_pod" ||
		heron.Lifetime == nil || *heron.Lifetime != 28*3600 || heron.Succeeded == nil || *heron.Succeeded != 3 ||
		*heron.Failed != 2 || *heron.Canceled != 1 || heron.Spend == nil || *heron.Spend != 1_250_000 || heron.SpendBasis != "provider_billed" {
		t.Fatalf("heron: %+v", heron)
	}
	if wren.Succeeded != nil || wren.SKU != "" || wren.Lifetime == nil || *wren.Lifetime != 3600 {
		t.Fatalf("wren, which this computer never recorded: %+v", wren)
	}
	if scoped := list("--ended", "--tensorhub", "other"); len(scoped) != 3 || scoped[0].Machine != "kanna" {
		t.Fatalf("--ended --tensorhub=other is not every hub's ended rentals: %+v", scoped)
	}
	code, human := runCozy(t, root, "rental", "list", "--all")
	for _, want := range []string{"Rentals: 4 (1 live) · lifetime 38h", " · 5 runs from this computer · accrued at least $70.85 est.",
		"MACHINE", "SKU", "GPUS", "HUB", "STARTED", "ENDED", "LIFETIME", "RUNS", "SPENT", "ENDED BY",
		"live", "8h16m", "28h0m", "idle_unreached", "released_by_pod", "$69.50 est.", "$1.25", "RUNS counts the runs this computer sent"} {
		if code != 0 || !strings.Contains(human, want) {
			t.Fatalf("the human history lacks %q [exit %d]\n%s", want, code, human)
		}
	}
	code, human = runCozy(t, root, "rental", "list", "--ended")
	if code != 0 || !strings.Contains(human, "Ended rentals: 3 · lifetime 37h16m · 3 runs from this computer · accrued $70.85 est.") ||
		strings.Contains(human, "otter") {
		t.Fatalf("the ended view [exit %d]\n%s", code, human)
	}
	// On a terminal it is a live board, like the fleet's: a run that settles on otter is
	// counted while it watches.
	code, tty := ptyDrive(t, root, 30, 2, func(step int, drawn string) []byte {
		switch {
		case step == 0 && strings.Contains(drawn, "· 5 runs from this computer"):
			run("otter-ok-2", "pr-otter-live", "succeeded", true)
			return []byte{}
		case step == 1 && strings.Contains(drawn, "· 6 runs from this computer"):
			return []byte("q")
		}
		return nil
	}, "rental", "list", "--all")
	if code != 0 || !strings.Contains(tty, "\x1b[?1049h") || !strings.Contains(tty, "STARTED") {
		t.Fatalf("the history board did not watch the new run [exit %d]\n%q", code, tty)
	}

	for index := 1; index <= 22; index++ {
		ended(fmt.Sprintf("pr-old-%02d", index), fmt.Sprintf("old%02d", index), fmt.Sprintf("2026-10-01T%02d:00:00Z", index), "owner_stop", 0)
	}
	if code, human := runCozy(t, root, "rental", "list", "--ended"); code != 0 ||
		!strings.Contains(human, "5 more not shown. Use --full to show all.") || strings.Contains(human, "old01") {
		t.Fatalf("a long history does not say what it hid [exit %d]\n%s", code, human)
	}
	if code, human := runCozy(t, root, "rental", "list", "--all", "--full"); code != 0 ||
		!strings.Contains(human, "old01") || strings.Contains(human, "not shown") {
		t.Fatalf("--full does not show every rental [exit %d]\n%s", code, human)
	}
	if code, out := runCozy(t, root, "rental", "list", "--all", "--watch"); code == 0 || !strings.Contains(out, "--watch requires interactive terminal output") {
		t.Fatalf("--all --watch off a terminal [exit %d]\n%s", code, out)
	}
}
