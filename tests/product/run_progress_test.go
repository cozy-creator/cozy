//go:build !windows

package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/sys/unix"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/output"
)

const ptyColumns = 80

// ptyRun runs the product binary with a real pseudo-terminal on stdout/stderr, so the
// suite observes exactly the bytes a person's terminal receives — escapes included.
func ptyRun(t *testing.T, root string, args ...string) (int, string) {
	return ptyRunInput(t, root, 24, nil, args...)
}

// ptyRunInput additionally binds stdin to the pseudo-terminal and sends each input after
// a short observation window. This drives terminal interaction itself, not parser helpers.
func ptyRunInput(t *testing.T, root string, rows uint16, input [][]byte, args ...string) (int, string) {
	t.Helper()
	master, cmd := startPTY(t, root, rows, ptyColumns, args...)
	defer master.Close()
	go func() {
		for _, keys := range input {
			time.Sleep(500 * time.Millisecond)
			_, _ = master.Write(keys)
		}
	}()
	var out bytes.Buffer
	_, _ = io.Copy(&out, master)
	_ = cmd.Wait()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, out.String()
}

func startPTY(t *testing.T, root string, rows, columns uint16, args ...string) (*os.File, *exec.Cmd) {
	t.Helper()
	master, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
	must(t, err)
	t.Cleanup(func() { _ = master.Close() })
	must(t, unix.IoctlSetPointerInt(int(master.Fd()), unix.TIOCSPTLCK, 0))
	number, err := unix.IoctlGetInt(int(master.Fd()), unix.TIOCGPTN)
	must(t, err)
	must(t, unix.IoctlSetWinsize(int(master.Fd()), unix.TIOCSWINSZ,
		&unix.Winsize{Row: rows, Col: columns}))
	slave, err := os.OpenFile(fmt.Sprintf("/dev/pts/%d", number), os.O_RDWR, 0)
	must(t, err)
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = slave, slave, slave //cozy:stdin-value test pty navigation
	must(t, cmd.Start())
	must(t, slave.Close()) // the child holds the slave now; EOF/EIO on master ends the read
	// The kill is a stuck-terminal stop for a child that never answers its inputs; the
	// proofs' own bounds are their input cadences, so the stop stays far behind them.
	timedOut := time.AfterFunc(20*time.Second, func() { _ = cmd.Process.Kill() })
	t.Cleanup(func() {
		timedOut.Stop()
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
	})
	return master, cmd
}

// TestRunProgressSurfaces (cl-104) proves the three progress surfaces of one live run
// stream against the real daemon and worker. The weightless tile's delay is a measured
// step loop — the fixture's stand-in for a denoising loop — so `--await` receives real
// per-step frames while the run is in flight.
func TestRunProgressSurfaces(t *testing.T) {
	root, err := os.MkdirTemp(os.TempDir(), "cozy-progress-")
	must(t, err)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down", "--all")
		_ = os.RemoveAll(root)
	})
	if code, out := runCozy(t, root, "package", "install", weightlessProject(t), "--editable"); code != 0 {
		t.Fatalf("weightless install failed [exit %d]\n%s", code, out)
	}

	// PIPED HUMAN: sparse append-only lines — plain bytes, one line per tenth, each line
	// a stable spelling of steps + percentage + elapsed. Never the full tick stream.
	code, _, stderr := runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=3", "delay_ms=2500", "--await")
	if code != 0 {
		t.Fatalf("piped run failed [exit %d]\n%s", code, stderr)
	}
	if strings.ContainsAny(stderr, "\r\033") {
		t.Fatalf("piped progress carries terminal control bytes\n%q", stderr)
	}
	stepLine := regexp.MustCompile(`^  tile_steps (\d+)/100 · (\d+)% stage · (\d+)% overall · elapsed [0-9ms.]+$`)
	previous, matched := -1, 0
	for _, line := range strings.Split(stderr, "\n") {
		if !strings.Contains(line, "tile_steps") {
			continue
		}
		fields := stepLine.FindStringSubmatch(line)
		if fields == nil {
			t.Fatalf("piped step line is not the stable spelling: %q", line)
		}
		step, _ := strconv.Atoi(fields[1])
		if step <= previous {
			t.Fatalf("piped step lines are not monotonic: %d after %d\n%s", step, previous, stderr)
		}
		previous, matched = step, matched+1
	}
	if matched < 3 || matched > 15 {
		t.Fatalf("piped lane is not sparse-but-live: %d step lines for 100 steps\n%s", matched, stderr)
	}
	if strings.Contains(stderr, "value=") {
		t.Fatalf("piped lane leaked a diagnostic spelling\n%s", stderr)
	}

	// TTY: the active stage block refreshes in place; completed stages remain above it.
	// Each row is clamped under the terminal's 80 columns.
	code, tty := ptyRun(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=4", "delay_ms=2500", "--await")
	if code != 0 {
		t.Fatalf("pty run failed [exit %d]\n%s", code, tty)
	}
	rewrites := strings.Split(tty, "\r\033[K")
	if len(rewrites) < 4 {
		t.Fatalf("terminal run did not rewrite in place: %d rewrites\n%q", len(rewrites), tty)
	}
	// The 80-column pty clamps the tail (that IS the resize safety); the bar, steps
	// and percentage must always survive the clamp.
	barred := regexp.MustCompile(`tile_steps \d+/100 \[[=.]{10}\] \d+% stage`)
	seen := 0
	for _, chunk := range rewrites[1:] {
		line := chunk
		if cut := strings.IndexAny(line, "\r\n"); cut >= 0 {
			line = line[:cut]
		}
		if width := len([]rune(line)); width >= ptyColumns {
			t.Fatalf("a rewrite is %d runes wide on a %d-column terminal: %q", width, ptyColumns, line)
		}
		if barred.MatchString(line) {
			seen++
		}
	}
	if seen < 2 {
		t.Fatalf("terminal rewrites never showed the step bar\n%q", tty)
	}
	if !strings.Contains(tty, "elapsed") || !strings.Contains(tty, "s/step avg") || !strings.Contains(tty, "ETA ~") {
		t.Fatalf("terminal run never showed elapsed, measured speed and stage ETA\n%q", tty)
	}

	// --await --json: stdout remains one final result; stderr carries typed JSONL events.
	code, stdout, stderr := runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=5", "delay_ms=1000", "--await")
	if code != 0 || !strings.Contains(stdout, `"status":"completed"`) {
		t.Fatalf("--json run failed [exit %d]\n%s", code, stdout)
	}
	if strings.ContainsAny(stderr, "\r\033") {
		t.Fatalf("--json stderr shows terminal control bytes\n%q", stderr)
	}
	var final map[string]any
	if err := json.Unmarshal([]byte(stdout), &final); err != nil {
		t.Fatalf("stdout is not one final JSON document: %v", err)
	}
	progressEvents := 0
	for _, line := range strings.Split(strings.TrimSpace(stderr), "\n") {
		var event map[string]any
		if err := json.Unmarshal([]byte(line), &event); err != nil || event["type"] == nil {
			t.Fatalf("stderr is not a typed JSONL event: %q (%v)", line, err)
		}
		if event["type"] == "request.progress" {
			progressEvents++
		}
	}
	if progressEvents == 0 {
		t.Fatal("awaited JSON omitted available progress")
	}

	// The list reuses the same lossy Runtime progress lane. Whole-job percentage and ETA
	// come only from overall_fraction; the current stage remains explicitly stage-local.
	// Completed rows preserve the successful 100% coordinate after live telemetry is gone.
	code, output := runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=6", "delay_ms=5000")
	if code != 0 {
		t.Fatalf("detached progress run failed [exit %d]\n%s", code, output)
	}
	type progressRow struct {
		Number          int64    `json:"number"`
		Status          string   `json:"status"`
		ProgressStage   string   `json:"progress_stage"`
		StageFraction   *float64 `json:"stage_fraction"`
		OverallFraction *float64 `json:"overall_fraction"`
		Position        *int64   `json:"position"`
		Total           *int64   `json:"total"`
		RemainingMS     *int64   `json:"remaining_ms"`
		ExecutionMS     int64    `json:"execution_ms"`
	}
	list := func(full bool) progressRow {
		t.Helper()
		args := []string{"--json"}
		if full {
			args = append(args, "--full")
		}
		args = append(args, "run", "list", "--limit", "1")
		code, out := runCozy(t, root, args...)
		var document struct {
			Invocations []progressRow `json:"invocations"`
		}
		if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Invocations) != 1 {
			t.Fatalf("could not read the live progress row [exit %d]\n%s", code, out)
		}
		for _, presentation := range []string{
			`"progress":`, `"queued":"`, `"execution":"`,
			`"stage_fraction":"`, `"overall_fraction":"`, `"position":"`, `"total":"`,
		} {
			if strings.Contains(out, presentation) {
				t.Fatalf("machine run list retained presentation value %s\n%s", presentation, out)
			}
		}
		return document.Invocations[0]
	}
	// THE HUMAN CELL IS ONE NUMBER AND THE STAGE. The machine projection above keeps
	// stage and overall apart because a caller may want either; the LIST CELL is read
	// by a person scanning rows, and it carried four facts at once --
	// `35% overall (~2s) · denoise 90% stage` -- two percentages against different
	// denominators plus an estimate that moves faster than the row holding it.
	if code, human := runCozy(t, root, "run", "list", "--limit", "1"); code != 0 {
		t.Fatalf("human run list [exit %d]:\n%s", code, human)
	} else {
		for _, gone := range []string{"% overall", "% stage"} {
			if strings.Contains(human, gone) {
				t.Fatalf("the progress cell still renders %q:\n%s", gone, human)
			}
		}
		if !humanProgressCell.MatchString(human) {
			t.Fatalf("the progress cell does not read as `<stage> <percent>`:\n%s", human)
		}
	}

	live := list(true)
	if live.Status != "in_progress" || live.ProgressStage == "" ||
		live.StageFraction == nil || *live.StageFraction <= 0 ||
		live.OverallFraction == nil || *live.OverallFraction <= 0 ||
		live.Position == nil || live.Total == nil || *live.Position <= 0 ||
		*live.Total <= 0 || *live.Position > *live.Total ||
		live.RemainingMS == nil || *live.RemainingMS <= 0 || live.ExecutionMS <= 0 {
		t.Fatalf("live run does not distinguish overall and stage progress: %+v", live)
	}
	// Default JSON is the same typed machine projection with fewer diagnostic
	// identity/timing fields; it never falls back to the human progress cell.
	if compact := list(false); compact.StageFraction == nil || compact.OverallFraction == nil ||
		compact.ExecutionMS <= 0 {
		t.Fatalf("default JSON list lost its numeric progress projection: %+v", compact)
	}
	if code, out := runCozy(t, root, "run", "watch", strconv.FormatInt(live.Number, 10), "--json"); code != 0 ||
		!strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("progress proof run did not settle [exit %d]\n%s", code, out)
	}
	if terminal := list(true); terminal.Status != "completed" || terminal.ProgressStage != "" ||
		terminal.StageFraction != nil || terminal.OverallFraction == nil || *terminal.OverallFraction != 1 ||
		terminal.Position != nil || terminal.Total != nil || terminal.RemainingMS != nil {
		t.Fatalf("terminal run did not preserve completed progress: %+v", terminal)
	}

	// The live inventory is a real terminal viewport: a wheel event moves the bounded page,
	// q exits cleanly, and every input/mouse/output mode is restored without echoing keys.
	code, tty = ptyRunInput(t, root, 8,
		[][]byte{[]byte("\x1b[<65;2;3M"), []byte("q")}, "run", "list", "--limit", "10")
	if code != 0 {
		t.Fatalf("q did not exit the live list cleanly [exit %d]\n%q", code, tty)
	}
	for _, control := range []string{
		"\x1b[?1049h", "\x1b[?1049l", "\x1b[?25l", "\x1b[?25h",
		"\x1b[?1000h", "\x1b[?1000l", "\x1b[?1006h", "\x1b[?1006l",
	} {
		if !strings.Contains(tty, control) {
			t.Fatalf("live list did not emit terminal restoration %q\n%q", control, tty)
		}
	}
	if !strings.Contains(tty, "rows 1-3/4") || !strings.Contains(tty, "rows 2-4/4") {
		t.Fatalf("mouse wheel did not move the live viewport\n%q", tty)
	}
	if strings.Contains(tty, "[<65;2;3M") || strings.Contains(tty, "\x1b[<65;2;3Mq") {
		t.Fatalf("terminal input was echoed as output\n%q", tty)
	}
}

// humanProgressCell is the shape a person reads off a row: the stage, then one
// percent for the whole job.
var humanProgressCell = regexp.MustCompile(`[a-z_]+ [0-9]+%`)

// The renderer is also driven on a small real terminal without a model/daemon.
// A resize may reflow old rows; redraw must stay inside the current screen and
// retain the stage/count at the front of each bounded row.
func TestLiveProgressFitsResizedTerminal(t *testing.T) {
	master, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
	must(t, err)
	defer master.Close()
	must(t, unix.IoctlSetPointerInt(int(master.Fd()), unix.TIOCSPTLCK, 0))
	number, err := unix.IoctlGetInt(int(master.Fd()), unix.TIOCGPTN)
	must(t, err)
	resize := func(columns uint16) {
		must(t, unix.IoctlSetWinsize(int(master.Fd()), unix.TIOCSWINSZ, &unix.Winsize{Row: 6, Col: columns}))
	}
	resize(80)
	slave, err := os.OpenFile(fmt.Sprintf("/dev/pts/%d", number), os.O_RDWR, 0)
	must(t, err)
	defer slave.Close()
	buf := &renderBuffer{}
	readDone := make(chan struct{})
	go func() { _, _ = io.Copy(buf, master); close(readDone) }()
	p := cli.NewProgress(&cli.Context{Inv: &cli.Invocation{Mode: output.Mode{Human: true, Color: true}}, Err: slave}, false, time.Now())
	t.Cleanup(p.Done)
	models := make([]any, 12)
	for i := range models {
		models[i] = map[string]any{"model": fmt.Sprintf("paul/model-%d", i), "moved_bytes": 5 << 30, "total_bytes": 50 << 30, "rate_bytes_per_second": 400 << 20}
	}
	p.On(liveEvent("phase", map[string]any{"phase": "downloading", "models": models}))
	waitUntil(t, "bounded model viewport", func() bool { return strings.Contains(buf.String(), "more rows") })
	before := len(buf.String())
	resize(32)
	p.On(liveEvent("phase", map[string]any{"phase": "downloading", "models": models}))
	p.On(liveEvent("progress", map[string]any{"stage": "denoise", "position": 5, "total": 30, "step_ms": 15000}))
	p.Done()
	must(t, slave.Close())
	<-readDone
	narrow := buf.String()[before:]
	for _, control := range regexp.MustCompile(`\x1b\[(\d+)A`).FindAllStringSubmatch(narrow, -1) {
		up, _ := strconv.Atoi(control[1])
		if up > 5 {
			t.Fatalf("redraw moved above the 6-row terminal: %q", narrow)
		}
	}
	for _, chunk := range strings.Split(narrow, "\r\033[K")[1:] {
		line := strings.SplitN(chunk, "\r", 2)[0]
		line = strings.SplitN(line, "\n", 2)[0]
		if len([]rune(line)) >= 32 {
			t.Fatalf("redraw wrapped a 32-column terminal: %q", line)
		}
	}
	if !strings.Contains(narrow, "denoising 5/30") {
		t.Fatalf("resize lost the stage and count: %q", narrow)
	}
}
