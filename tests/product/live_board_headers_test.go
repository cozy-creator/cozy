//go:build !windows

package producttest

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
)

// readLiveFrame observes complete writes through the real PTY. A quiet interval
// ends a frame; the deadline diagnoses a board that never answers a resize/key.
func readLiveFrame(t *testing.T, terminal *os.File, accept func(string) bool) string {
	t.Helper()
	var transcript strings.Builder
	buffer := make([]byte, 64*1024)
	poll := []unix.PollFd{{Fd: int32(terminal.Fd()), Events: unix.POLLIN}}
	for deadline := time.Now().Add(5 * time.Second); time.Now().Before(deadline); {
		n, err := unix.Poll(poll, 100)
		if err == unix.EINTR {
			continue
		}
		must(t, err)
		if n > 0 {
			n, err := terminal.Read(buffer)
			if err != nil {
				t.Fatalf("board exited before the expected frame: %v\n%q", err, transcript.String())
			}
			transcript.Write(buffer[:n])
			continue
		}
		const start = "\x1b[H\x1b[J"
		if index := strings.LastIndex(transcript.String(), start); index >= 0 {
			frame := transcript.String()[index+len(start):]
			if accept == nil || accept(frame) {
				return frame
			}
		}
	}
	t.Fatalf("board did not produce the expected frame: %q", transcript.String())
	return ""
}

// These are physical-screen checks, not a search for a heading anywhere in a
// transcript: every line must fit, and no newline may scroll the bottom row.
func assertLiveFrameFits(t *testing.T, frame, heading string, rows, columns int) []string {
	t.Helper()
	lines := strings.Split(frame, "\r\n")
	if len(lines) > rows || strings.HasSuffix(frame, "\n") {
		t.Fatalf("frame scrolls a %d-row terminal (%d lines): %q", rows, len(lines), frame)
	}
	header := -1
	for index, line := range lines {
		if strings.ContainsAny(line, "\r\n\x1b") || len([]rune(line)) >= columns {
			t.Fatalf("frame line wraps or moves the cursor in %d columns: %q", columns, line)
		}
		if strings.HasPrefix(line, heading) {
			header = index
		}
	}
	if header < 0 {
		t.Fatalf("no visible %s column heading: %q", heading, frame)
	}
	if rows <= 2 && (header != 0 || len(lines) != rows) {
		t.Fatalf("tiny terminal lost the heading/navigation priority: %q", frame)
	}
	return lines[header:]
}

func TestLiveBoardHeadersSurviveResizeAndRefresh(t *testing.T) {
	for _, board := range []string{"run", "rental"} {
		t.Run(board, func(t *testing.T) {
			root := t.TempDir()
			port := reservePort(t)
			hubURL := fmt.Sprintf("http://127.0.0.1:%d", port)
			hub := newFakeRentalHub(t, port)
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte(
				"tensorhub_url: "+hubURL+"\ntensorhub_token: rental-idle-test\n"+
					"daemon:\n  idle_shutdown_s: 0\n"), 0o600))
			store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
			fatal(t, problem)
			defer store.Close()
			heading := "MACHINE"
			if board == "run" {
				heading = "NUMBER"
				startDaemonProcess(t, root)
			}
			seed := func(i int) {
				t.Helper()
				id := fmt.Sprintf("board-header-%02d", i)
				if board == "run" {
					_, _, problem := store.Submit(records.Request{
						ID: id, IdemKey: id, BodyDigest: "sha256:" + strings.Repeat("ab", 32),
						Package: "proof/" + strings.Repeat("long-target-", 15), Entrypoint: "generate",
						Payload: []byte("{}"), Rental: true,
					})
					fatal(t, problem)
					return
				}
				machine := fmt.Sprintf("machine%02d", i)
				hub.add(id, machine)
				fatal(t, store.RecordRental(records.Rental{AcceleratorCount: 1,
					ID: id, MachineName: machine, SKU: "cpu", AcceleratorModel: "CPU",
					HourlyRateUSDMicros: 100_000, State: "ready", Hub: hubURL,
					Address: "127.0.0.1:1", CertPath: filepath.Join(root, id+".pem"),
				}))
			}
			for i := range 30 {
				seed(i)
			}
			terminal, cmd := startPTY(t, root, 5, 80, board, "list", "--full")
			frame := readLiveFrame(t, terminal, nil)
			lines := assertLiveFrameFits(t, frame, heading, 5, 80)
			first := strings.Fields(lines[1])[0]
			_, err := terminal.Write([]byte("\x1b[B"))
			must(t, err)
			frame = readLiveFrame(t, terminal, nil)
			lines = assertLiveFrameFits(t, frame, heading, 5, 80)
			anchor := strings.Fields(lines[1])[0]
			if anchor == first {
				t.Fatalf("arrow did not scroll data below the heading: %q", frame)
			}
			// New rows arrive while the table is scrolled. The data anchor and
			// heading must survive the next live snapshot independently.
			seed(30)
			frame = readLiveFrame(t, terminal, func(frame string) bool { return strings.Contains(frame, "/31") })
			lines = assertLiveFrameFits(t, frame, heading, 5, 80)
			if current := strings.Fields(lines[1])[0]; current != anchor {
				t.Fatalf("live refresh moved the anchored row: %q to %q", anchor, current)
			}
			for _, size := range [][2]int{{2, 40}, {1, 24}, {6, 80}, {3, 32}} {
				must(t, unix.IoctlSetWinsize(int(terminal.Fd()), unix.TIOCSWINSZ,
					&unix.Winsize{Row: uint16(size[0]), Col: uint16(size[1])}))
				frame = readLiveFrame(t, terminal, nil)
				assertLiveFrameFits(t, frame, heading, size[0], size[1])
			}
			// Growing again restores useful data; End and Home still operate on
			// the complete row set after visits to a zero-data-row viewport.
			must(t, unix.IoctlSetWinsize(int(terminal.Fd()), unix.TIOCSWINSZ,
				&unix.Winsize{Row: 6, Col: 80}))
			_, err = terminal.Write([]byte("\x1b[F"))
			must(t, err)
			frame = readLiveFrame(t, terminal, nil)
			assertLiveFrameFits(t, frame, heading, 6, 80)
			location := boardLocation.FindStringSubmatch(frame)
			if len(location) != 4 || location[2] != "31" {
				t.Fatalf("End cannot reach the final data row after resize: %q", frame)
			}
			_, err = terminal.Write([]byte("\x1b[H"))
			must(t, err)
			frame = readLiveFrame(t, terminal, nil)
			assertLiveFrameFits(t, frame, heading, 6, 80)
			location = boardLocation.FindStringSubmatch(frame)
			if len(location) != 4 || location[1] != "1" {
				t.Fatalf("Home did not restore the first data row: %q", frame)
			}
			_, err = terminal.Write([]byte("q"))
			must(t, err)
			restored, _ := io.ReadAll(terminal)
			must(t, cmd.Wait())
			if !strings.Contains(string(restored), "\x1b[?1049l") || !strings.Contains(string(restored), "\x1b[?25h") {
				t.Fatalf("board failed to restore terminal: %q", restored)
			}
		})
	}
}
