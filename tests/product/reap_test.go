package producttest

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"slices"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// ------------------------------------------------------------------ never leave a daemon
//
// A Cozy daemon detaches into its own session (internal/cli.detachProcess) — the correct
// behaviour for a background service, and the reason every daemon this suite causes to
// exist outlives the test that caused it unless something ends it on purpose. Nothing
// did: 143 unreachable daemons holding 3,403 threads accumulated over three days on one
// development host until it could no longer fork.
//
// Three layers end them, because each covers a failure the one before it cannot:
//
//  1. reapDaemonRoot in a t.Cleanup, registered by childEnv — the single chokepoint every
//     daemon-capable child process passes through. Covers a test that passes, fails,
//     panics or is skipped. Does NOT cover the test binary dying.
//  2. TestMain reaps every registered root after m.Run. Covers a panic that unwinds past
//     the cleanups, and the -timeout panic. Does NOT cover a signal it cannot handle.
//  3. The reaper: a detached copy of this same test binary holding the read end of a pipe
//     nobody else has. ANY death of the test binary — SIGKILL, Ctrl-C on `go test`, the
//     box killing it for memory — closes the write end, and the reaper reaps on EOF.
//     This is the only layer that survives a signal the test binary cannot handle.
//
// Reaping is scoped to roots THIS process registered, never to a pattern like
// "/tmp/cozy-product-test-*/*": several suites run concurrently on a shared box, and a
// pattern sweep would end another run's daemon in the middle of its test.

const reapMode = "--cozy-test-reap="

var daemons struct {
	sync.Mutex
	roots   map[string]bool // every root a daemon-capable child was pointed at
	reaper  *os.File        // the write end; closing it is what wakes the reaper
	reaping *exec.Cmd       // the reaper process, so a clean run can wait for it
	armed   map[string]bool // one t.Cleanup per (root, test), not per runCozy call
}

// trackDaemonRoot records one root and arms the reaping layers for it. childEnv calls it,
// so a test gets this by using the suite's own scaffolding and cannot forget it.
func trackDaemonRoot(t *testing.T, root string) {
	t.Helper()
	root = filepath.Clean(root)
	daemons.Lock()
	if daemons.roots == nil {
		daemons.roots, daemons.armed = map[string]bool{}, map[string]bool{}
	}
	if !daemons.roots[root] && daemons.reaper != nil {
		// The reaper learns a root BEFORE any daemon can exist on it, so a kill arriving
		// one instruction after the spawn still finds one.
		_, _ = io.WriteString(daemons.reaper, root+"\n")
	}
	daemons.roots[root] = true
	// One reap per (root, TOP-LEVEL test), never one per subtest: a parent that starts a
	// daemon and then drives it from a table of t.Run arms must keep it for the whole
	// table, and a cleanup armed on the first arm's own t would end it after that arm.
	key := root + "\x00" + topLevel(t.Name())
	arm := !daemons.armed[key]
	daemons.armed[key] = true
	daemons.Unlock()
	if arm {
		t.Cleanup(func() { reapDaemonRoot(root) })
	}
}

func topLevel(name string) string {
	if i := strings.Index(name, "/"); i >= 0 {
		return name[:i]
	}
	return name
}

func trackedRoots() []string {
	daemons.Lock()
	defer daemons.Unlock()
	roots := make([]string, 0, len(daemons.roots))
	for root := range daemons.roots {
		roots = append(roots, root)
	}
	return roots
}

// reapDaemonRoot ends whatever daemon owns root, politely first. `cozy down --all` is the
// product's own guaranteed teardown: it ends the paid rentals a live arm may be holding,
// which a signal cannot. The kill is what makes the outcome certain when it does not, or
// when there is nothing left healthy enough to answer.
func reapDaemonRoot(root string) {
	defer reapMachineRuntimeRoot(root)
	if len(daemonPidsOn(root)) == 0 {
		return
	}
	if cozyBin != "" {
		// The bound is on a post-mortem courtesy, not on the outcome: the kill below is
		// what guarantees the daemon is gone, whatever `down` did or did not manage.
		ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
		down := exec.CommandContext(ctx, cozyBin, "down", "--all")
		cfg, _ := config.Load() // A configuration refusal cannot prevent the guaranteed reap below.
		down.Env = cfg.Child("COZY_HOME="+root,
			"TENSORFS_HOME="+filepath.Join(root, "tensorfs"))
		down.Stdout, down.Stderr = io.Discard, io.Discard
		_ = down.Run()
		cancel()
	}
	pids := daemonPidsOn(root)
	for _, pid := range pids {
		kill(pid)
	}
	// Wait on the OS, not on a duration: a SIGKILLed process leaves, and if one somehow
	// does not, saying so beats pretending the root is clean.
	for i := 0; i < 200 && anyAlive(pids); i++ {
		time.Sleep(25 * time.Millisecond)
	}
	for _, pid := range pids {
		if alive(pid) {
			fmt.Fprintf(os.Stderr, "cozy-daemon %d on %s survived SIGKILL\n", pid, root)
		}
	}
}

// The machine Host intentionally survives the client's ordinary down, as a pod outlives its
// controller. Test teardown owns both, and must stop its exact Host before deleting the home.
func reapMachineRuntimeRoot(root string) bool {
	entries, err := os.ReadDir("/proc")
	if err != nil {
		return true
	}
	machine := filepath.Join(root, "machine")
	if resolved, err := filepath.EvalSymlinks(machine); err == nil {
		machine = resolved
	}
	// The Host runs as pod-supervisor, a symlink to the cozy it installed or a Host binary itself.
	bin := filepath.Join(machine, "root", "usr", "local", "bin")
	hosts := []string{filepath.Join(bin, "cozy-machine"), filepath.Join(bin, "pod-supervisor"), filepath.Join(bin, "cozy")}
	var owned []int
	for _, entry := range entries {
		pid, err := strconv.Atoi(entry.Name())
		if err != nil {
			continue
		}
		if exe, err := os.Readlink("/proc/" + entry.Name() + "/exe"); err != nil || !slices.Contains(hosts, strings.TrimSuffix(exe, " (deleted)")) {
			continue
		}
		process, _ := os.FindProcess(pid)
		_ = process.Signal(syscall.SIGTERM)
		owned = append(owned, pid)
	}
	for i := 0; i < 400 && anyAlive(owned); i++ {
		time.Sleep(25 * time.Millisecond)
	}
	for _, pid := range owned {
		if alive(pid) {
			fmt.Fprintf(os.Stderr, "owned test machine Host %d did not finish explicit teardown\n", pid)
			return false
		}
	}
	return true
}

func anyAlive(pids []int) bool {
	for _, pid := range pids {
		if alive(pid) {
			return true
		}
	}
	return false
}

// daemonOnRoot answers with one live daemon's pid for root, or 0.
func daemonOnRoot(root string) int {
	pids := daemonPidsOn(root)
	if len(pids) == 0 {
		return 0
	}
	return pids[0]
}

// daemonPidsOn names every live cozy-daemon whose COZY_HOME is root. Two sources, because
// neither alone is complete: the lock record is what the product itself publishes, and
// /proc also finds a daemon whose record was truncated by a shutdown it did not finish.
func daemonPidsOn(root string) []int {
	found := map[int]bool{}
	if data, err := os.ReadFile(filepath.Join(root, "daemon.lock")); err == nil {
		for _, line := range strings.Split(string(data), "\n") {
			if value, ok := strings.CutPrefix(strings.TrimSpace(line), "pid="); ok {
				if pid, err := strconv.Atoi(value); err == nil && daemonHome(pid) == root {
					found[pid] = true
				}
			}
		}
	}
	entries, err := os.ReadDir("/proc")
	if err == nil {
		for _, entry := range entries {
			pid, err := strconv.Atoi(entry.Name())
			if err == nil && daemonHome(pid) == root {
				found[pid] = true
			}
		}
	}
	pids := make([]int, 0, len(found))
	for pid := range found {
		pids = append(pids, pid)
	}
	return pids
}

// daemonHome answers the COZY_HOME of pid if — and only if — pid is a live Cozy daemon.
// Both halves are checked against the kernel's copy: a pid read out of a stale lock file
// may have been reused by an unrelated process, and killing that would be far worse than
// leaving a daemon behind.
func daemonHome(pid int) string {
	argv, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/cmdline")
	if err != nil || filepath.Base(string(splitNul(argv, 2)[0])) != daemonArgv0 {
		return ""
	}
	environ, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/environ")
	if err != nil {
		return ""
	}
	for _, entry := range splitNul(environ, -1) {
		if value, ok := strings.CutPrefix(entry, "COZY_HOME="); ok {
			return filepath.Clean(value)
		}
	}
	return ""
}

const daemonArgv0 = "cozy-daemon"

func splitNul(data []byte, n int) []string {
	return strings.SplitN(strings.TrimRight(string(data), "\x00"), "\x00", n)
}

// ------------------------------------------------------------------------- layer 3
//
// startDaemonReaper re-execs this test binary in reap mode, detached, holding the read end
// of a pipe. The write end is inherited by nothing else (Go opens pipes close-on-exec and
// exec.Cmd passes only the files it is given), so EOF on it means one thing and only one
// thing: this test binary is gone.
func startDaemonReaper() {
	self, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "no daemon reaper: %v\n", err)
		return
	}
	read, write, err := os.Pipe()
	if err != nil {
		fmt.Fprintf(os.Stderr, "no daemon reaper: %v\n", err)
		return
	}
	cmd := exec.Command(self, reapMode+cozyBin)
	// stdin is the control pipe and nothing else; stdout is discarded by the kernel
	// rather than by a copier goroutine in a process that is expected to die first.
	cmd.Stdin, cmd.Stdout, cmd.Stderr = read, nil, os.Stderr //cozy:stdin-value controlled reaper liveness pipe
	detachSession(cmd)
	if err := cmd.Start(); err != nil {
		read.Close()
		write.Close()
		fmt.Fprintf(os.Stderr, "no daemon reaper: %v\n", err)
		return
	}
	read.Close()
	daemons.Lock()
	daemons.reaper, daemons.reaping = write, cmd
	daemons.Unlock()
}

// stopDaemonReaper closes the pipe and waits for the reaper to finish, so a clean run
// leaves neither a daemon nor a reaper behind. The reaper has nothing to do by then —
// TestMain has already reaped — so this returns as fast as one process exit.
func stopDaemonReaper() {
	daemons.Lock()
	write, cmd := daemons.reaper, daemons.reaping
	daemons.reaper, daemons.reaping = nil, nil
	daemons.Unlock()
	if write == nil {
		return
	}
	write.Close()
	// TestMain has already reaped, so the reaper wakes to an empty list and exits at once.
	_ = cmd.Wait()
}

// runDaemonReaper is the reaper process itself: read roots until the pipe closes, then end
// every daemon on them. Reached only through the reapMode argv, never as a test.
func runDaemonReaper(cozyPath string, in io.Reader) {
	// The reaper's stderr is inherited from a process that is already gone by the time it
	// has anything to say. A broken pipe on fd 2 ends a Go program by default, and this
	// one must not be endable by the thing it is cleaning up after.
	signal.Ignore(syscall.SIGPIPE)
	cozyBin = cozyPath
	var roots []string
	seen := map[string]bool{}
	scan := bufio.NewScanner(in)
	for scan.Scan() {
		if root := strings.TrimSpace(scan.Text()); root != "" && !seen[root] {
			seen[root] = true
			roots = append(roots, root)
		}
	}
	// EOF. The test binary is gone, however it went.
	for _, root := range roots {
		reapDaemonRoot(root)
	}
}
