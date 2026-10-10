//go:build !windows

package producttest

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// set moves one field of a fake hub rental, the way Tensorhub's own state machine does
// while a pod provisions.
func (h *fakeRentalHub) set(id, key string, value any) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.rentals[id][key] = value
}

// TestRentalListLiveBoard is cl-114 as behaviour: `cozy rental list` on a terminal is
// the fleet as a live board — MACHINE SKU STATE $/HOUR SPENT UPTIME RUNNING QUEUED IDLE, redrawn in
// place every second — and the same verb piped or --json is one plain snapshot. The
// board is watched through a real pseudo-terminal across planted transitions: the pod
// acquiring, then ready with observed idle time, then held by queued work, then counting
// elapsed idle time again once the work settles.
func TestRentalListLiveBoard(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-list-tui")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	hub := newFakeRentalHub(t, 0)
	port := hub.port()
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	// The board reports observed inactivity without inventing a worker deadline.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"), 0o600))

	hub.add("rental-tui", "sparrow")
	hub.set("rental-tui", "state", "acquiring")
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
		ID: "rental-tui", MachineName: "sparrow", SKU: "cpu", AcceleratorModel: "CPU",
		HourlyRateUSDMicros: 100_000, State: "acquiring", Hub: hubURL,
		Address: "127.0.0.1:1", CertPath: filepath.Join(root, "rental-tui.pem"),
	}))

	// Each transition is planted once the board has drawn the one before it: the hub moves
	// the pod to ready, queued work holds it, the work settles, and the idle clock moves on.
	countdown := regexp.MustCompile(`sparrow\s+cpu\s+—\s+ready\s+\$0\.10\s+-\s+\S+\s+0\s+0\s+(\d+)s`)
	firstIdle := ""
	code, tty := ptyDrive(t, root, 24, 4, func(step int, drawn string) []byte {
		switch step {
		case 0:
			if !regexp.MustCompile(`sparrow\s+cpu\s+—\s+acquiring`).MatchString(drawn) {
				return nil
			}
			hub.set("rental-tui", "state", "ready")
		case 1:
			idle := countdown.FindStringSubmatch(drawn)
			if idle == nil || !regexp.MustCompile(`sparrow\s+cpu\s+—\s+ready`).MatchString(drawn) {
				return nil
			}
			firstIdle = idle[1]
			_, _, problem := store.Submit(records.Request{
				ID: "req-rental-tui", IdemKey: "idem-rental-tui",
				BodyDigest: "sha256:" + strings.Repeat("ef", 32),
				Package:    "fake/tui", Entrypoint: "generate", Payload: []byte("{}"),
				Rental: true, Worker: "rental-tui",
			})
			fatal(t, problem)
		case 2:
			if !regexp.MustCompile(`sparrow\s+cpu\s+—\s+ready\s+\$0\.10\s+-\s+\S+\s+0\s+1\s+-`).MatchString(drawn) {
				return nil
			}
			fatal(t, store.SettleRequest("req-rental-tui", "canceled"))
		case 3:
			idle := countdown.FindAllStringSubmatch(drawn, -1)
			if len(idle) == 0 || idle[len(idle)-1][1] == firstIdle {
				return nil
			}
			return []byte("q")
		}
		return []byte{}
	}, "rental", "list")
	if code != 0 {
		t.Fatalf("q did not exit the live rental board cleanly [exit %d]\n%q", code, tty)
	}
	for _, control := range []string{
		"\x1b[?1049h", "\x1b[?1049l", "\x1b[?25l", "\x1b[?25h",
		"\x1b[?1000h", "\x1b[?1000l", "\x1b[?1006h", "\x1b[?1006l",
	} {
		if !strings.Contains(tty, control) {
			t.Fatalf("live board did not emit terminal restoration %q\n%q", control, tty)
		}
	}
	if !regexp.MustCompile(`MACHINE\s+SKU\s+GPUS\s+STATE\s+\$/HOUR\s+SPENT\s+UPTIME\s+RUNNING\s+QUEUED\s+IDLE`).MatchString(tty) {
		t.Fatalf("the board does not carry the ruled columns\n%q", tty)
	}
	if strings.Count(tty, "\x1b[H\x1b[J") < 5 {
		t.Fatalf("the board did not redraw in place\n%q", tty)
	}
	if !strings.Contains(tty, "Remote machines running: 1") ||
		!strings.Contains(tty, "Current spend per hour: $0.1") {
		t.Fatalf("the board lost the fleet spend header\n%q", tty)
	}
	acquiring := regexp.MustCompile(`sparrow\s+cpu\s+—\s+acquiring`).FindStringIndex(tty)
	ready := regexp.MustCompile(`sparrow\s+cpu\s+—\s+ready`).FindStringIndex(tty)
	if acquiring == nil || ready == nil || acquiring[0] >= ready[0] {
		t.Fatalf("the board did not redraw acquiring→ready in order\n%q", tty)
	}
	busy := regexp.MustCompile(`sparrow\s+cpu\s+—\s+ready\s+\$0\.10\s+-\s+\S+\s+0\s+1\s+-`).FindStringIndex(tty)
	if busy == nil {
		t.Fatalf("queued work did not blank the idle countdown\n%q", tty)
	}
	elapsed := countdown.FindAllStringSubmatchIndex(tty, -1)
	if len(elapsed) < 2 {
		t.Fatalf("the IDLE cell did not count observed elapsed time\n%q", tty)
	}
	first := tty[elapsed[0][2]:elapsed[0][3]]
	last := tty[elapsed[len(elapsed)-1][2]:elapsed[len(elapsed)-1][3]]
	if first == last {
		t.Fatalf("the IDLE elapsed clock never moved: always %ss\n%q", first, tty)
	}
	if elapsed[len(elapsed)-1][0] < busy[0] {
		t.Fatalf("the idle countdown did not resume after the work settled\n%q", tty)
	}

	// PIPED: one plain snapshot — no terminal control bytes, the same columns, the spend
	// header — and bare `cozy rental` is byte-for-byte the same verb (the elapsed idle
	// second is the one moving part).
	code, listed := runCozy(t, root, "rental", "list")
	if code != 0 || strings.ContainsAny(listed, "\r\x1b") {
		t.Fatalf("piped snapshot carries terminal control bytes [exit %d]\n%q", code, listed)
	}
	if !regexp.MustCompile(`MACHINE\s+SKU\s+GPUS\s+STATE\s+\$/HOUR\s+SPENT\s+UPTIME\s+RUNNING\s+QUEUED\s+IDLE`).MatchString(listed) ||
		!strings.Contains(listed, "Remote machines running: 1") ||
		!strings.Contains(listed, "Unused rentals time out after 15 minutes.") {
		t.Fatalf("piped snapshot lost the ruled surface\n%s", listed)
	}
	// Bare `cozy rental` names the group's VERBS; it is not one of them. The live table
	// lives at `cozy rental list` and nowhere else — the same shape bare `cozy package`
	// and `cozy model` already have.
	code, bare := runCozy(t, root, "rental")
	if code != 0 {
		t.Fatalf("bare `cozy rental` did not print its verbs [exit %d]\n%s", code, bare)
	}
	for _, verb := range []string{"rental list", "rental end"} {
		if !strings.Contains(bare, verb) {
			t.Fatalf("bare `cozy rental` does not offer `cozy %s`\n%s", verb, bare)
		}
	}
	for _, ran := range []string{"MACHINE", "Remote machines running:"} {
		if strings.Contains(bare, ran) {
			t.Fatalf("bare `cozy rental` still RUNS the list verb instead of naming it\n%s", bare)
		}
	}
	if code, helped := runCozy(t, root, "help", "rental"); code != 0 || helped != bare {
		t.Fatalf("bare `cozy rental` is not `cozy help rental` [exit %d]\nbare:\n%s\nhelp:\n%s",
			code, bare, helped)
	}

	// A watch that cannot be a terminal refuses instead of degrading.
	if code, out := runCozy(t, root, "rental", "list", "--watch"); code == 0 ||
		!strings.Contains(out, "--watch requires interactive terminal output") {
		t.Fatalf("piped watch did not refuse clearly [exit %d]\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "list", "--watch", "--json"); code == 0 ||
		!strings.Contains(out, "--watch requires interactive terminal output") {
		t.Fatalf("JSON watch did not refuse clearly [exit %d]\n%s", code, out)
	}

	// JSON: field-complete typed machine rows — the underlying facts (counts as numbers,
	// moments as timestamps), never the table's spellings — plus the reconciled fleet
	// totals in the document itself.
	code, out := runCozy(t, root, "--json", "--full", "rental", "list")
	if code != 0 {
		t.Fatalf("JSON snapshot failed [exit %d]\n%s", code, out)
	}
	var document struct {
		Rentals              []map[string]any `json:"rentals"`
		MachinesRunning      int              `json:"machines_running"`
		HourlySpendUSDMicros int64            `json:"hourly_spend_usd_micros"`
	}
	must(t, json.Unmarshal([]byte(out), &document))
	if len(document.Rentals) != 1 || document.MachinesRunning != 1 ||
		document.HourlySpendUSDMicros != 100_000 {
		t.Fatalf("JSON lost the fleet totals: %s", out)
	}
	row := document.Rentals[0]
	for _, field := range []string{"machine", "sku", "state", "rental_id", "accelerator",
		"accelerator_count", "address", "hub", "rented_at", "ready_at", "running", "queued",
		"idle_s", "idle_since_at", "hourly_rate_usd_micros"} {
		if _, ok := row[field]; !ok {
			t.Fatalf("JSON row lost field %q: %s", field, out)
		}
	}
	if row["machine"] != "sparrow" || row["state"] != "ready" || row["rental_id"] != "rental-tui" ||
		row["running"] != float64(0) || row["queued"] != float64(0) || row["hourly_rate_usd_micros"] != float64(100_000) ||
		row["accelerator_count"] != float64(1) ||
		row["idle_s"] == nil || row["release_due_at"] != nil {
		t.Fatalf("JSON row is not the live idle truth: %s", out)
	}
	for _, spelling := range []string{`"idle":`, `"uptime":`, `"rented":`} {
		if strings.Contains(out, spelling) {
			t.Fatalf("JSON carries the table spelling %s: %s", spelling, out)
		}
	}

	// The spend header stays on th-120's billed totals: when the hub's reconciled billed
	// rate moves, the next snapshot's burn line says the billed figure, not the quote.
	hub.setRate("rental-tui", 720_000)
	if code, out := runCozy(t, root, "rental", "list"); code != 0 ||
		!strings.Contains(out, "Current spend per hour: $0.72") {
		t.Fatalf("the spend header does not follow the billed rate [exit %d]\n%s", code, out)
	}
}
