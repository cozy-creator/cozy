//go:build !windows

package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// Esc leaves a live board exactly as q does. Esc is ALSO the first byte of every arrow,
// page and wheel report a terminal sends, so the two halves are one proof and neither
// stands alone: pressing Esc must exit, and pressing an arrow must scroll and NOT exit.
// Both boards are driven through a real pseudo-terminal — the keys travel the same
// termios path a person's do, and nothing here reads a keymap.

var boardLocation = regexp.MustCompile(`rows (\d+)-(\d+)/(\d+)`)

// boardKeys is the input the proof sends, in order: a CSI arrow, a second CSI arrow, an
// SS3 arrow (the same key from a terminal in application cursor mode), then a bare Esc.
// Each is one write, which is how a terminal emits one keypress.
func boardKeys() [][]byte {
	return [][]byte{[]byte("\x1b[B"), []byte("\x1b[B"), []byte("\x1bOB"), []byte("\x1b")}
}

// scrolledThenExited reads one board's transcript for both halves at once.
func scrolledThenExited(t *testing.T, board string, code int, tty string) {
	t.Helper()
	if code != 0 {
		t.Fatalf("Esc did not leave the %s board the way q does [exit %d]\n%q", board, code, tty)
	}
	for _, control := range []string{"\x1b[?1049h", "\x1b[?1049l", "\x1b[?25l", "\x1b[?25h"} {
		if !strings.Contains(tty, control) {
			t.Fatalf("the %s board did not restore the terminal on Esc: missing %q\n%q",
				board, control, tty)
		}
	}
	frames := boardLocation.FindAllStringSubmatch(tty, -1)
	if len(frames) == 0 {
		t.Fatalf("the %s board drew no located frame\n%q", board, tty)
	}
	first, last := frames[0], frames[len(frames)-1]
	top, _ := strconv.Atoi(first[1])
	end, _ := strconv.Atoi(first[2])
	total, _ := strconv.Atoi(first[3])
	if total <= end {
		t.Fatalf("every %s row fits one page (%q), so a scroll cannot be told from a quit — plant more rows\n%q",
			board, first[0], tty)
	}
	moved, _ := strconv.Atoi(last[1])
	if moved <= top {
		t.Fatalf("the arrow keys did not scroll the %s board: %q then %q — Esc's leading byte was taken for the whole key\n%q",
			board, first[0], last[0], tty)
	}
}

// TestRentalBoardEscExits is `cozy rental list` on a terminal.
func TestRentalBoardEscExits(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "rental-board-esc")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	port := reservePort(t)
	hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: "+hubURL+"\n"+
			"tensorhub_token: rental-idle-test\n"+
			"rentals:\n  max_hourly_spend_usd: 10.00\n  idle_release_s: 3600\n"), 0o600))
	hub := newFakeRentalHub(t, port)
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	// More machines than a 14-row terminal pages, so the viewport has somewhere to go.
	for i := range 12 {
		id, machine := fmt.Sprintf("pr-board-%02d", i), fmt.Sprintf("machine%02d", i)
		hub.add(id, machine)
		fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
			ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU",
			HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
			Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
		}))
	}
	code, tty := ptyRunInput(t, root, 14, boardKeys(), "rental", "list")
	scrolledThenExited(t, "rental", code, tty)
	if !strings.Contains(tty, "q/Esc/Ctrl-C exits") {
		t.Fatalf("the board does not offer Esc in its own guidance line\n%q", tty)
	}
}

// TestRunBoardEscExits is `cozy run list` on a terminal — the second board over the same
// input layer, exercised as its own verb rather than assumed from the first.
func TestRunBoardEscExits(t *testing.T) {
	root := filepath.Join(os.TempDir(), "cozy-product-test", "run-board-esc")
	must(t, os.RemoveAll(root))
	must(t, os.MkdirAll(root, 0o755))
	t.Cleanup(func() { _ = os.RemoveAll(root) })
	// An unreachable hub on purpose: the runs WAIT, which holds the row count still.
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
		"tensorhub_url: http://127.0.0.1:1\n"+
			"tensorhub_token: run-board-esc-test\n"+
			"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	startDaemonProcess(t, root)
	for i := range 12 {
		id := fmt.Sprintf("req-board-%02d", i)
		if _, _, problem := store.Submit(records.Request{
			ID: id, IdemKey: "idem-" + id, BodyDigest: "sha256:" + strings.Repeat("ab", 32),
			Package: "fake/board", Entrypoint: "generate", Payload: []byte("{}"),
			Rental: true,
		}); problem != nil {
			t.Fatal(problem.Message)
		}
	}
	code, tty := ptyRunInput(t, root, 14, boardKeys(), "run", "list")
	scrolledThenExited(t, "run", code, tty)
}
