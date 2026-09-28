//go:build !windows

package producttest

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	localapi "github.com/cozy-creator/cozy/internal/client"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/output"
)

// recordedRun is a real run's event stream (`cozy run watch <n> --json`), trimmed to its
// stage changes and each tenth of a counted stage.
func recordedRun(t *testing.T, name string) []localapi.Event {
	t.Helper()
	file, err := os.Open("testdata/run_watch/" + name + ".jsonl")
	must(t, err)
	defer file.Close()
	var events []localapi.Event
	lines := bufio.NewScanner(file)
	lines.Buffer(make([]byte, 1<<20), 1<<20)
	for lines.Scan() {
		var e localapi.Event
		must(t, json.Unmarshal(lines.Bytes(), &e))
		events = append(events, e)
	}
	must(t, lines.Err())
	return events
}

func recordedAt(t *testing.T, e localapi.Event) time.Time {
	at, err := time.Parse(time.RFC3339Nano, e.At)
	must(t, err)
	return at
}

// Run 1510: a rental wait, an install, three references generated at once while one
// downloads the weights, then long_form's segments.
var liveFrames1510 = []struct{ at, frame string }{
	{"09:14:00", `
Run 1510 · paul/minimax-h3/long_form · darkness · queued · 1m3s
  ▸ waiting for a rental machine · 1m3s`},
	{"09:18:12", `
Run 1510 · paul/minimax-h3/long_form · darkness · running · 5m15s
  ✓ waiting for a rental machine       1m39s
  ✓ installing paul/minimax-h3@1.19.0  27s
  ✓ starting                           8s
  ▸ Creating reference Reina · Downloading model weights · 3m2s
  ▸ Creating reference Hero · 3m2s
  ▸ Creating reference background · 3m2s
  ▸ generate_image · Downloading model weights · 1m48s
    ██████░░░░░░░░░░░░░░  30%  step 9943972608/33118370151`},
	{"09:21:19.2", `
Run 1510 · paul/minimax-h3/long_form · darkness · running · 8m23s
  ✓ waiting for a rental machine                1m39s
  ✓ installing paul/minimax-h3@1.19.0           27s
  ✓ starting                                    8s
  ✓ generate_image · Downloading model weights  4m29s
  ▸ Creating reference Reina · generating image · 6m9s
    ████████████░░░░░░░░  60%  step 24/40 · 0.21s/step avg · ETA ~3s
  ▸ Creating reference Hero · saving image · 6m9s
  ▸ Creating reference background · generating image · 6m9s
    ██████████░░░░░░░░░░  50%  step 20/40 · 0.21s/step avg · ETA ~4s`},
	{"09:26:27.8", `
Run 1510 · paul/minimax-h3/long_form · darkness · running · 13m31s
  … 3 earlier
  ✓ generate_image · Downloading model weights  4m29s
  ✓ Creating reference Reina                    6m13s
  ✓ Creating reference Hero                     6m5s
  ✓ Creating reference background               6m14s
  ▸ Segment 1 of 9 · denoising · 5m4s
    ██████████████████░░  88%  step 7/8 · 8.63s/step avg · ETA ~9s
  overall █░░░░░░░░░░░░░░░░░░░   6%`},
}

// Run 1504 settled: the whole record, once.
const settled1504 = `
Run 1504 · paul/minimax-h3/long_form · lafter · completed · 6m55s
  ✓ starting                                    7s
  ✓ generate_image · Downloading model weights  2m7s
  ✓ Creating reference traveler                 2m42s
  ✓ Segment 1 of 2                              1m59s
  ✓ Segment 2 of 2                              2m4s
  ✓ Assembling video                            1s`

// Run 1514: four references on the machine's four GPUs, a fifth waiting behind its own run.
var liveFrames1514 = []struct{ at, frame string }{
	{"09:54:30", `
Run 1514 · paul/minimax-h3/long_form · darkness · running · 8s
  ✓ starting  0.6s
  ▸ Creating reference White1 · generating image · 7s
    ██████████████░░░░░░  70%  step 28/40 · 0.21s/step avg · ETA ~3s
  ▸ Creating reference White2 · generating image · 7s
    ████████████░░░░░░░░  60%  step 24/40 · 0.21s/step avg · ETA ~3s
  ▸ Creating reference White3 · generating image · 7s
    ██████████░░░░░░░░░░  50%  step 20/40 · 0.21s/step avg · ETA ~4s
  ▸ Creating reference Hero · Checking model inputs · 7s
  ▸ Creating reference Background · generating image · 7s
    ██████████░░░░░░░░░░  50%  step 20/40 · 0.20s/step avg · ETA ~4s
  ▸ waiting for GPU (needs 1, 4 in use by this run's other calls) · 6s`},
}

func TestLiveRunViewGoldenFrames(t *testing.T) {
	goldenFrames(t, "1510", "darkness", liveFrames1510)
	goldenFrames(t, "1514", "darkness", liveFrames1514)

	events := recordedRun(t, "run-1504")
	sink := &renderBuffer{}
	ctx := &cli.Context{Inv: &cli.Invocation{Mode: output.Mode{Human: true, Live: true}}, Out: sink, Err: sink}
	p := cli.NewProgress(ctx, false, recordedAt(t, events[0]))
	p.Describe("1504", "paul/minimax-h3/long_form", "lafter", "running")
	for _, e := range events {
		p.On(e)
	}
	p.Done()
	if got := "\n" + strings.Join(p.Frame(time.Now(), 80), "\n"); got != settled1504 {
		t.Fatalf("settled block:\n%s\nwant:%s", got, settled1504)
	}
	// The terminal settled the region at once: one write of the whole record, never a
	// replay of every event.
	if got := screen(sink.String()); got != settled1504[1:]+"\n" {
		t.Fatalf("the terminal was left with:\n%s", got)
	}
}

// goldenFrames replays a recorded run and compares its frames at moments of the run.
func goldenFrames(t *testing.T, run, machine string, frames []struct{ at, frame string }) {
	t.Helper()
	events := recordedRun(t, "run-"+run)
	began := recordedAt(t, events[0])
	sink := &renderBuffer{}
	ctx := &cli.Context{Inv: &cli.Invocation{Mode: output.Mode{Human: true, Live: true}}, Out: sink, Err: sink}
	p := cli.NewProgress(ctx, false, began)
	defer p.Done()
	p.Describe(run, "paul/minimax-h3/long_form", machine, "queued")
	next := 0
	for _, want := range frames {
		clock, err := time.Parse("15:04:05.999", want.at)
		must(t, err)
		at := time.Date(began.Year(), began.Month(), began.Day(), clock.Hour(), clock.Minute(),
			clock.Second(), clock.Nanosecond(), time.UTC)
		for ; next < len(events) && !recordedAt(t, events[next]).After(at); next++ {
			p.On(events[next])
		}
		if got := "\n" + strings.Join(p.Frame(at, 80), "\n"); got != want.frame {
			t.Fatalf("run %s frame at %s:\n%s\nwant:%s", run, want.at, got, want.frame)
		}
	}
}

// TestRunWatchDrawsInPlace drives the built CLI against a daemon serving a recorded stream:
// on a terminal the region redraws in place and settles to one record above the result;
// redirected, the append-only lines are exactly what they were before the live display.
func TestRunWatchDrawsInPlace(t *testing.T) {
	events := recordedRun(t, "run-1504")
	for i := range events {
		events[i].RequestID = "req-1504"
	}
	life := api.Lifecycle{Number: 1504, RequestID: "req-1504", Kind: "invocation", Status: "completed",
		Package: "paul/minimax-h3", Function: "long_form", Machine: "lafter", CreatedAt: events[0].At}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/":
		case "/v1/requests/1504", "/v1/requests/req-1504":
			_ = json.NewEncoder(w).Encode(life)
		case "/v1/requests/req-1504/events":
			w.Header().Set("Content-Type", "text/event-stream")
			for i, e := range events {
				encoded, err := json.Marshal(e)
				must(t, err)
				_, _ = fmt.Fprintf(w, "id: %d\ndata: %s\n\n", e.EventID, encoded)
				if i%(len(events)/3) == 0 { // two pauses mid-run: the terminal draws, then erases
					w.(http.Flusher).Flush()
					time.Sleep(3 * liveFramePause)
				}
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)

	code, _, piped := runCozyStreams(t, root, "run", "watch", "1504")
	want, err := os.ReadFile("testdata/run_watch/run-1504.piped.txt")
	must(t, err)
	if got := regexp.MustCompile(`elapsed \S+`).ReplaceAllString(piped, "elapsed …"); code != 0 || got != string(want) {
		t.Fatalf("redirected watch changed [exit %d]:\n%s", code, got)
	}

	master, cmd := startPTY(t, root, 24, ptyColumns, "run", "watch", "1504")
	raw, _ := io.ReadAll(master) // EIO once the child exits
	_ = cmd.Wait()
	if cmd.ProcessState.ExitCode() != 0 {
		t.Fatalf("terminal watch failed:\n%q", raw)
	}
	shown := screen(string(raw))
	if !strings.HasPrefix(shown, settled1504[1:]+"\n") || strings.ContainsAny(shown, "▸█░") ||
		!strings.Contains(shown, "status:") {
		t.Fatalf("the terminal was not left with one settled record above the result:\n%s\n%q", shown, raw)
	}
	// Paused twice (and at its start), the stream draws a few frames, each erasing the last:
	// never one per event.
	if frames := strings.Count(string(raw), "\033[J"); frames < 2 || frames > 12 {
		t.Fatalf("%d frames for %d events:\n%q", frames, len(events), raw)
	}
}

// liveFramePause is longer than one live frame (100ms).
const liveFramePause = 100 * time.Millisecond

// screen is what a terminal shows after output: carriage returns, line feeds, cursor-up,
// erase-below and erase-line applied; styling dropped.
func screen(output string) string {
	lines := []string{""}
	row, col := 0, 0
	put := func(r rune) {
		line := []rune(lines[row])
		for len(line) < col {
			line = append(line, ' ')
		}
		if col < len(line) {
			line[col] = r
		} else {
			line = append(line, r)
		}
		lines[row], col = string(line), col+1
	}
	control := regexp.MustCompile(`^\x1b\[([0-9;?]*)([A-Za-z])`)
	for i := 0; i < len(output); {
		if m := control.FindStringSubmatch(output[i:]); m != nil {
			i += len(m[0])
			n, _ := strconv.Atoi(m[1])
			switch m[2] {
			case "A":
				row = max(0, row-max(n, 1))
			case "J":
				lines = append(lines[:row], string([]rune(lines[row])[:min(col, len([]rune(lines[row])))]))
			case "K":
				lines[row] = string([]rune(lines[row])[:min(col, len([]rune(lines[row])))])
			}
			continue
		}
		r, size := []rune(output[i:])[0], len(string([]rune(output[i:])[0]))
		i += size
		switch r {
		case '\r':
			col = 0
		case '\n':
			row++
			if row == len(lines) {
				lines = append(lines, "")
			}
		default:
			put(r)
		}
	}
	return strings.Join(lines, "\n")
}
