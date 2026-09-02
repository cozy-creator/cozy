//go:build !windows

package producttest

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

const ptyColumns = 80

// ptyRun runs the product binary with a real pseudo-terminal on stdout/stderr, so the
// suite observes exactly the bytes a person's terminal receives — escapes included.
func ptyRun(t *testing.T, root string, args ...string) (int, string) {
	t.Helper()
	master, err := os.OpenFile("/dev/ptmx", os.O_RDWR, 0)
	must(t, err)
	defer master.Close()
	must(t, unix.IoctlSetPointerInt(int(master.Fd()), unix.TIOCSPTLCK, 0))
	number, err := unix.IoctlGetInt(int(master.Fd()), unix.TIOCGPTN)
	must(t, err)
	must(t, unix.IoctlSetWinsize(int(master.Fd()), unix.TIOCSWINSZ,
		&unix.Winsize{Row: 24, Col: ptyColumns}))
	slave, err := os.OpenFile(fmt.Sprintf("/dev/pts/%d", number), os.O_RDWR, 0)
	must(t, err)
	cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	cmd.Stdin, cmd.Stdout, cmd.Stderr = slave, slave, slave
	must(t, cmd.Start())
	must(t, slave.Close()) // the child holds the slave now; EOF/EIO on master ends the read
	var out bytes.Buffer
	_, _ = io.Copy(&out, master)
	_ = cmd.Wait()
	code := 0
	if cmd.ProcessState != nil {
		code = cmd.ProcessState.ExitCode()
	}
	return code, out.String()
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
	stepLine := regexp.MustCompile(`^  tile_steps (\d+)/100 · (\d+)% · elapsed [0-9ms.]+$`)
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

	// TTY: ONE in-place line — every step write is a \r + erase rewrite carrying the bar,
	// steps, percentage and elapsed, each clamped under the terminal's 80 columns.
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
	barred := regexp.MustCompile(`tile_steps \[[=.]{18}\] \d+/100 · \d+%`)
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
	if !strings.Contains(tty, "elapsed") {
		t.Fatalf("terminal run never showed elapsed time\n%q", tty)
	}

	// --json: the machine surface is untouched — the renderer contributes nothing.
	code, stdout, stderr := runCozyStreams(t, root, "--json", "run", localWeightlessRef+"/tile",
		"size=32", "seed=5", "delay_ms=1000", "--await")
	if code != 0 || !strings.Contains(stdout, `"status":"completed"`) {
		t.Fatalf("--json run failed [exit %d]\n%s", code, stdout)
	}
	if strings.ContainsAny(stderr, "\r\033") || strings.Contains(stderr, "tile_steps") {
		t.Fatalf("--json stderr shows renderer output\n%q", stderr)
	}
}
