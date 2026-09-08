package producttest

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// Abandoned scratch roots are the disk half of the orphan-daemon leak. A product
// test builds an isolated COZY_HOME under os.TempDir() and relies on deferred
// cleanup; SIGKILL, a test timeout and a fork-starved box all skip it. Nothing
// else reaps them, so they accumulate — 65 GB of dead COZY_HOMEs and 143 orphan
// daemons over three days, until the box refused fork/exec outright.
//
// The suite reaps its predecessors on the way in. A run cleans the corpses of
// the run before it, so the resource that creates the garbage is the one that
// collects it: no scheduler to install, no janitor to remember, and nothing new
// that can itself be left running.

const scratchOwnerFile = ".cozy-scratch-owner"

// scratchPrefixes are the temp-root families this suite creates. A root outside
// them is somebody else's and is never touched.
var scratchPrefixes = []string{
	"cozy-product-test", "cozy-product-", "cozy-native-serving-",
	"cozy-serving-preparation-", "cozy-native-memo-", "cozy-schema-",
	"cozy-progress-", "cozy-child-media-", "cozy-image-preparation-",
	"cozy-composition-", "cozy-assets-", "cozy-byte-result-", "cozy-calls-",
	"cozy-script-", "cozy-typed-", "cozy-job-json-", "cozy-fake-attempt",
}

// claimScratch records this process as the owner of root, so a later run can
// tell an abandoned root from a live sibling's. Best effort: a root that cannot
// be marked simply falls back to the age floor.
func claimScratch(root string) {
	_ = os.WriteFile(filepath.Join(root, scratchOwnerFile),
		[]byte(strconv.Itoa(os.Getpid())), 0o644)
}

// processAlive reports whether pid is a live process. Signal 0 checks for
// existence without delivering anything.
func processAlive(pid int) bool {
	if pid <= 0 {
		return false
	}
	p, err := os.FindProcess(pid)
	if err != nil {
		return false
	}
	return p.Signal(syscall.Signal(0)) == nil
}

// ownerPID returns the pid that owns root and whether one was recorded at all.
// The scratch marker is authoritative; a daemon.lock left by an isolated daemon
// is the fallback, since that daemon is the thing holding the root open.
func ownerPID(root string) (pid int, recorded bool) {
	if b, err := os.ReadFile(filepath.Join(root, scratchOwnerFile)); err == nil {
		if v, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
			return v, true
		}
	}
	b, err := os.ReadFile(filepath.Join(root, "daemon.lock"))
	if err != nil {
		return 0, false
	}
	for _, line := range strings.Split(string(b), "\n") {
		if rest, ok := strings.CutPrefix(strings.TrimSpace(line), "pid="); ok {
			if v, err := strconv.Atoi(rest); err == nil {
				return v, true
			}
		}
	}
	return 0, false
}

// removeAllForce removes a scratch tree. Worker-environment contents are
// content-addressed and deliberately read-only (mode 0555 dirs), so a plain
// RemoveAll fails partway with EACCES — and, when it is called from a deferred
// cleanup that ignores the error, fails silently. That is why every previous
// attempt to clean these roots left them on disk.
func removeAllForce(path string) error {
	if err := os.RemoveAll(path); err == nil {
		return nil
	}
	_ = filepath.WalkDir(path, func(p string, d fs.DirEntry, err error) error {
		if err == nil && d.IsDir() {
			_ = os.Chmod(p, 0o700)
		}
		return nil
	})
	return os.RemoveAll(path)
}

// createdAt is the root's creation time, taken as the oldest top-level entry.
// A partly-deleted tree can only ever look newer, so this cannot be fooled by a
// failed reap into treating an ancient root as fresh.
func createdAt(root string) time.Time {
	entries, err := os.ReadDir(root)
	if err != nil {
		return time.Time{}
	}
	oldest := time.Time{}
	for _, e := range entries {
		info, err := e.Info()
		if err != nil {
			continue
		}
		if oldest.IsZero() || info.ModTime().Before(oldest) {
			oldest = info.ModTime()
		}
	}
	return oldest
}

// reapAbandonedScratch removes every scratch root under dir whose owner is gone.
//
// Liveness beats age. A root whose owner pid is dead is reaped immediately,
// however new it is; a root whose owner is alive is spared, however old. Only a
// root that recorded no owner at all falls back to minAge, which must stay well
// clear of the longest product test so a concurrent lane is never reaped out
// from under itself.
func reapAbandonedScratch(dir string, minAge time.Duration) (reaped int, failed int) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, 0
	}
	self := os.Getpid()
	for _, e := range entries {
		if !e.IsDir() || !hasScratchPrefix(e.Name()) {
			continue
		}
		root := filepath.Join(dir, e.Name())
		pid, recorded := ownerPID(root)
		switch {
		case recorded && pid == self:
			continue // ours, in use
		case recorded && processAlive(pid):
			continue // a live sibling's
		case !recorded:
			created := createdAt(root)
			if created.IsZero() || time.Since(created) < minAge {
				continue
			}
		}
		if err := removeAllForce(root); err != nil {
			failed++
			continue
		}
		reaped++
	}
	return reaped, failed
}

func hasScratchPrefix(name string) bool {
	for _, p := range scratchPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// scratchReapMinAge is the floor for roots that recorded no owner. It must
// exceed the longest product test, so an unmarked root belonging to a lane that
// is still running is never taken.
const scratchReapMinAge = 2 * time.Hour

// forkHeadroom reports whether the box can still fork. The suite starts a real
// daemon per test; on a box out of headroom the fork fails partway through with
// EAGAIN, killing tests that each leak a scratch root — the spiral that made the
// box unusable for hours.
//
// It probes by forking rather than by comparing a counter to a constant. There
// is no threshold worth guessing here: Committed_AS routinely runs at twice
// CommitLimit under heuristic overcommit while fork works perfectly, and the
// thread count sat three orders of magnitude below threads-max when fork
// actually began failing. The only honest question is whether a fork succeeds
// right now, so that is the question asked. The guard lives in code rather than
// an environment variable so a stale shell cannot switch it off.
func forkHeadroom() (ok bool, detail string) {
	// Never probe with /proc/self/exe: that is this test binary, and running it
	// re-enters TestMain.
	if err := exec.Command("/bin/true").Run(); err != nil {
		var se syscall.Errno
		if errors.As(err, &se) && (se == syscall.EAGAIN || se == syscall.ENOMEM) {
			return false, "fork probe failed: " + se.Error()
		}
	}
	used, max := threadCount()
	if max > 0 && used > max*80/100 {
		return false, fmt.Sprintf("%d threads of %d max", used, max)
	}
	return true, ""
}

// threadCount returns the live thread count and the kernel ceiling. loadavg's
// fourth field is running/total threads.
func threadCount() (used, max int64) {
	b, err := os.ReadFile("/proc/sys/kernel/threads-max")
	if err != nil {
		return 0, 0
	}
	max, _ = strconv.ParseInt(strings.TrimSpace(string(b)), 10, 64)
	s, err := os.ReadFile("/proc/loadavg")
	if err != nil {
		return 0, max
	}
	f := strings.Fields(string(s))
	if len(f) < 4 {
		return 0, max
	}
	_, total, found := strings.Cut(f[3], "/")
	if !found {
		return 0, max
	}
	used, _ = strconv.ParseInt(total, 10, 64)
	return used, max
}
