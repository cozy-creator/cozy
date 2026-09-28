package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// `cozy rental logs` reads a rental's provider boot log from its Hub through the daemon: an
// older Hub that keeps none says so; a newer one's pages print in order with each line's time
// and step, --json is one document, -f follows onto a replanned attempt and waits for new
// lines until the rental is no longer booting, and the log still reads once the rental ended.
func TestRentalLogsPrintsTheProviderBootLog(t *testing.T) {
	root := filepath.Join(scratchBase, "rental-logs")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	hub := newFakeRentalHub(t, 0)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", hub.port())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	hub.add("pr-bootlog", "otter")
	fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1, ID: "pr-bootlog", MachineName: "otter",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL}))
	store.Close()

	if code, out := runCozy(t, root, "rental", "logs", "otter"); code == 0 || !strings.Contains(out, "this hub doesn't keep boot logs") {
		t.Fatalf("an older Hub's missing route was not named [exit %d]:\n%s", code, out)
	}

	line := func(second int, step, text string) map[string]any {
		return map[string]any{"at": fmt.Sprintf("2026-09-25T20:50:%02dZ", second), "step": step, "line": text}
	}
	attempts := [][]map[string]any{
		{line(1, "", "create 20GB network volume"), line(1, "creating_container", "create container index.docker.io/tensorhub/worker"),
			line(2, "pulling_image", "edd1ed89f0d4 Already exists")},
		{line(40, "creating_container", "create container index.docker.io/tensorhub/worker"),
			line(41, "pulling_image", "9134e987953c Pulling fs layer")},
	}
	booting, armed, grown := true, false, false
	hub.mux.HandleFunc("GET /v1/rentals/{id}/boot-log", func(w http.ResponseWriter, r *http.Request) {
		hub.mu.Lock()
		defer hub.mu.Unlock()
		attempt, _ := strconv.Atoi(r.URL.Query().Get("attempt"))
		after, _ := strconv.Atoi(r.URL.Query().Get("after"))
		if attempt == 0 {
			attempt = len(attempts)
		}
		lines := attempts[attempt-1][min(after, len(attempts[attempt-1])):]
		lines = lines[:min(2, len(lines))] // a small page, so the CLI pages
		if len(lines) == 0 && attempt == 2 && armed && !grown {
			grown = true // the boot writes one more line while the CLI waits
			attempts[1] = append(attempts[1], line(43, "starting_container", "start container for index.docker.io/tensorhub/worker: begin"))
		} else if len(lines) == 0 && attempt == 2 && grown {
			booting = false
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"attempt": attempt, "attempts": len(attempts),
			"booting": booting, "lines": lines, "next": after + len(lines)})
	})

	code, out := runCozy(t, root, "rental", "logs", "otter", "--attempt", "1")
	want := "2026-09-25T20:50:01Z                      create 20GB network volume\n" +
		"2026-09-25T20:50:01Z  creating_container  create container index.docker.io/tensorhub/worker\n" +
		"2026-09-25T20:50:02Z  pulling_image       edd1ed89f0d4 Already exists\n"
	if code != 0 || !strings.Contains(out, want) {
		t.Fatalf("attempt 1's log, across two pages [exit %d]:\n%s", code, out)
	}
	var document struct {
		Rental   string `json:"rental"`
		Attempt  int    `json:"attempt"`
		Attempts int    `json:"attempts"`
		Lines    []struct {
			Step string `json:"step"`
		} `json:"lines"`
	}
	code, out = runCozy(t, root, "rental", "logs", "pr-bootlog", "--json")
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || document.Rental != "pr-bootlog" ||
		document.Attempt != 2 || document.Attempts != 2 || len(document.Lines) != 2 || document.Lines[1].Step != "pulling_image" {
		t.Fatalf("the latest attempt as one document [exit %d]:\n%s", code, out)
	}
	hub.mu.Lock()
	armed = true
	hub.mu.Unlock()
	code, out = runCozy(t, root, "rental", "logs", "otter", "--attempt", "1", "-f")
	if code != 0 || !strings.Contains(out, want+"attempt 2 of 2\n") || !strings.HasSuffix(out,
		"2026-09-25T20:50:43Z  starting_container  start container for index.docker.io/tensorhub/worker: begin\n") {
		t.Fatalf("-f did not follow onto the replanned attempt and its new line [exit %d]:\n%s", code, out)
	}
	// The log outlives the pod: an ended rental no Hub lists any more still reads by name.
	hub.setState("pr-bootlog", "released", "")
	if code, out := runCozy(t, root, "rental", "list", "--no-watch"); code != 0 || strings.Contains(out, "otter") {
		t.Fatalf("the ended rental is still listed [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "rental", "logs", "otter", "--attempt", "1"); code != 0 || !strings.Contains(out, want) {
		t.Fatalf("an ended rental's log did not read by name [exit %d]:\n%s", code, out)
	}
}
