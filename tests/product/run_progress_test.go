//go:build !windows

package producttest

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"golang.org/x/sys/unix"
)

const ptyColumns = 80

// ptyRun runs the product binary with a real pseudo-terminal on stdout/stderr, so the
// suite observes exactly the bytes a person's terminal receives — escapes included.
func ptyRun(t *testing.T, root string, args ...string) (int, string) {
	return ptyRunInput(t, root, 24, nil, args...)
}

// ptyRunInput additionally binds stdin to the pseudo-terminal. Each input is written once
// the board has drawn a located frame (`rows N-M/T`) after the previous one, so the
// board's own redraws pace the keys, never a clock.
func ptyRunInput(t *testing.T, root string, rows uint16, input [][]byte, args ...string) (int, string) {
	t.Helper()
	return ptyDrive(t, root, rows, len(input), func(step int, drawn string) []byte {
		if !boardLocation.MatchString(drawn) {
			return nil
		}
		return input[step]
	}, args...)
}

// ptyDrive runs the product on a pseudo-terminal for `steps` inputs. Whenever the board
// draws, next is asked with the output drawn since the previous input; it answers the keys
// to write for this step, or nil to keep watching. A board that never answers is killed by
// startPTY's stuck-terminal stop.
func ptyDrive(t *testing.T, root string, rows uint16, steps int,
	next func(step int, drawn string) []byte, args ...string) (int, string) {
	t.Helper()
	master, cmd := startPTY(t, root, rows, ptyColumns, args...)
	defer master.Close()
	var mu sync.Mutex
	var out bytes.Buffer
	drew, closed := make(chan struct{}, 1), make(chan struct{})
	go func() {
		defer close(closed)
		chunk := make([]byte, 4096)
		for {
			n, err := master.Read(chunk)
			if n > 0 {
				touchPTY(cmd)
			}
			mu.Lock()
			out.Write(chunk[:n])
			mu.Unlock()
			select {
			case drew <- struct{}{}:
			default:
			}
			if err != nil {
				return
			}
		}
	}()
	since := 0
	for step := 0; step < steps; {
		mu.Lock()
		drawn := out.String()[since:]
		mu.Unlock()
		if keys := next(step, drawn); keys != nil {
			since += len(drawn)
			_, _ = master.Write(keys)
			step++
			continue
		}
		select {
		case <-drew:
		case <-closed:
			step = steps
		}
	}
	<-closed
	_ = cmd.Wait()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	mu.Lock()
	defer mu.Unlock()
	return code, out.String()
}

// ptyStall is how long a pseudo-terminal child may draw nothing before it is stopped.
const ptyStall = 20 * time.Second

var ptyStops sync.Map // *exec.Cmd -> *time.Timer

// touchPTY renews a child's stuck-terminal stop after it drew.
func touchPTY(cmd *exec.Cmd) {
	if timer, ok := ptyStops.Load(cmd); ok {
		timer.(*time.Timer).Reset(ptyStall)
	}
}

func startPTY(t *testing.T, root string, rows, columns uint16, args ...string) (*os.File, *exec.Cmd) {
	return startPTYSetup(t, root, rows, columns, nil, args...)
}

func startPTYSetup(t *testing.T, root string, rows, columns uint16, setup func(*os.File), args ...string) (*os.File, *exec.Cmd) {
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
	if setup != nil {
		setup(master)
	}
	must(t, cmd.Start())
	must(t, slave.Close()) // the child holds the slave now; EOF/EIO on master ends the read
	// The kill is a stuck-terminal stop for a child that stops drawing: a reader that sees
	// output renews it (touchPTY), so a long run that keeps drawing is never cut off.
	timedOut := time.AfterFunc(ptyStall, func() { _ = cmd.Process.Kill() })
	ptyStops.Store(cmd, timedOut)
	t.Cleanup(func() {
		ptyStops.Delete(cmd)
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
	// a stable spelling of steps + percentage + measured speed/ETA + elapsed.
	// Never the full tick stream.
	code, _, stderr := runCozyStreams(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=3", "delay_ms=2500", "--await")
	if code != 0 {
		t.Fatalf("piped run failed [exit %d]\n%s\n%s\ndaemon log:\n%s", code, stderr, productWorkerLogs(root), tail(filepath.Join(root, "daemon.log")))
	}
	if strings.ContainsAny(stderr, "\r\033") {
		t.Fatalf("piped progress carries terminal control bytes\n%q", stderr)
	}
	stepLine := regexp.MustCompile(`^  tile_steps (\d+)/100 · (\d+)% stage · ETA ~[0-9hms.]+ · (\d+)% overall · elapsed [0-9ms.]+$`)
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

	// TTY: the run's region refreshes in place and settles to one record. Each row is
	// clamped under the terminal's 80 columns.
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
	barred := regexp.MustCompile(`[█░]{20} +\d+%  \d+/100`)
	seen := 0
	for _, chunk := range rewrites[1:] {
		line := regexp.MustCompile(`\x1b\[[0-9;]*m`).ReplaceAllString(chunk, "")
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
	if plain := regexp.MustCompile(`\x1b\[[0-9;]*m`).ReplaceAllString(tty, ""); !strings.Contains(plain, "▸ tile_steps · ") ||
		!strings.Contains(plain, "ETA ~") {
		t.Fatalf("terminal run never showed elapsed and measured stage ETA\n%q", tty)
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
		// A run on a machine streams the Runtime's own progress samples.
		if event["type"] == "request.progress" || event["type"] == "machine.progress" {
			progressEvents++
		}
	}
	if progressEvents == 0 {
		t.Fatal("awaited JSON omitted available progress")
	}

	// The list reuses the same lossy Runtime progress lane. Whole-job percentage
	// comes from overall_fraction; ETA remains explicitly stage-local.
	// Completed rows preserve the successful 100% coordinate after live telemetry is gone.
	code, output := runCozy(t, root, "run", localWeightlessRef+"/tile",
		"size=32", "seed=6", "delay_ms=5000")
	if code != 0 {
		t.Fatalf("detached progress run failed [exit %d]\n%s", code, output)
	}
	type progressRow struct {
		Number           int64    `json:"number"`
		Status           string   `json:"status"`
		ProgressStage    string   `json:"progress_stage"`
		StageFraction    *float64 `json:"stage_fraction"`
		OverallFraction  *float64 `json:"overall_fraction"`
		Position         *int64   `json:"position"`
		Total            *int64   `json:"total"`
		RemainingMS      *int64   `json:"remaining_ms"`
		StageRemainingMS *int64   `json:"stage_remaining_ms"`
		ExecutionMS      *int64   `json:"execution_ms"`
		ExecutionKnown   bool     `json:"execution_known"`
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
		row := document.Invocations[0]
		// Progress remains observable even when an older Runtime did not measure
		// cooperative execution. Unknown timing must stay explicit in either view.
		if row.ExecutionKnown != (row.ExecutionMS != nil) || row.ExecutionMS != nil && *row.ExecutionMS < 0 {
			t.Fatalf("run list has inconsistent measured timing: %+v", row)
		}
		return row
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
		live.RemainingMS != nil || live.StageRemainingMS == nil || *live.StageRemainingMS <= 0 {
		t.Fatalf("live run does not distinguish overall and stage progress: %+v", live)
	}
	// Default JSON is the same typed machine projection with fewer diagnostic
	// identity/timing fields; it never falls back to the human progress cell.
	if compact := list(false); compact.StageFraction == nil || compact.OverallFraction == nil {
		t.Fatalf("default JSON list lost its numeric progress projection: %+v", compact)
	}
	if code, out := runCozy(t, root, "run", "watch", strconv.FormatInt(live.Number, 10), "--json"); code != 0 ||
		!strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("progress proof run did not settle [exit %d]\n%s", code, out)
	}
	if terminal := list(true); terminal.Status != "completed" || terminal.ProgressStage != "" ||
		terminal.StageFraction != nil || terminal.OverallFraction == nil || *terminal.OverallFraction != 1 ||
		terminal.Position != nil || terminal.Total != nil || terminal.RemainingMS != nil || terminal.StageRemainingMS != nil {
		t.Fatalf("terminal run did not preserve completed progress: %+v", terminal)
	}

	// The live inventory is a real terminal viewport: a wheel event moves the bounded page,
	// q exits cleanly, and every input/mouse/output mode is restored without echoing keys.
	// Five terminal rows hold the heading, three data rows and navigation; four must
	// scroll. Paged run history has no census footer (its pages are not a count of all
	// retained history), so nothing else takes a row.
	code, tty = ptyRunInput(t, root, 5,
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
