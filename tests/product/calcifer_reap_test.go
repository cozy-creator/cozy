package producttest

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
)

// TestCalciferLeavesARootThatNoLongerCarriesItsClaim is the product rule that makes an
// unreachable Calcifer impossible: the lock file IS the claim, so a Calcifer whose claim is no
// longer at the path it published cannot be found by a client, told to stop, or read back
// from its own records. It leaves.
//
// `calcifer.idle_shutdown_s: 0` is set on purpose — it is what every leaked Calcifer in the
// incident had, and it must not be able to switch this off.
func TestCalciferLeavesARootThatNoLongerCarriesItsClaim(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("calcifer:\n  idle_shutdown_s: 0\n"), 0o600))
	live := startCalciferProcess(t, root)
	logPath := filepath.Join(root, "calcifer.log")
	// The listener can answer before startup finishes writing its claim log.
	deadline := time.Now().Add(5 * time.Second)
	for {
		log, _ := os.ReadFile(logPath)
		if strings.Contains(string(log), "claim: "+filepath.Join(root, "calcifer.lock")) {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("Calcifer did not name the claim it holds\n%s", tail(logPath))
		}
		time.Sleep(10 * time.Millisecond)
	}

	// A Calcifer that still owns its root stays, and says nothing about leaving: the rule
	// is an observed fact about the filesystem, not a countdown.
	select {
	case code := <-live.exited:
		t.Fatalf("Calcifer left (exit %d) while it still owned its root\n%s", code, tail(logPath))
	case <-time.After(4 * time.Second):
	}

	// Remove the claimed pathname atomically. Recursive deletion races the live
	// Calcifer's startup writes and can itself fail before testing the lost claim.
	retired := root + "-retired"
	must(t, os.Rename(root, retired))
	t.Cleanup(func() { _ = os.RemoveAll(retired) })
	if code := awaitCalciferExit(t, live, 30*time.Second); code != 0 {
		t.Fatalf("Calcifer exited %d after its root went", code)
	}
}

// TestTheReaperOutlivesAKilledTestBinary is the layer t.Cleanup cannot be: 58 copies of
// one Calcifer accumulated on a single test root because runs were interrupted, and no
// cleanup runs on a signal the test binary cannot handle.
//
// The proof is the real mechanism, not a stand-in for it. The reaper is this test binary
// re-exec'd exactly as TestMain re-execs it; the write end of its pipe is held by another
// process, and that process is SIGKILLed. The kernel closing that last write end is the
// same event, byte for byte, as the kernel closing it when `go test` is SIGKILLed.
func TestTheReaperOutlivesAKilledTestBinary(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName),
		[]byte("calcifer:\n  idle_shutdown_s: 0\n"), 0o600))
	if code, out := runCozy(t, root, "up", "--json"); code != 0 {
		t.Fatalf("cozy up [exit %d]: %s", code, out)
	}
	pid := calciferOnRoot(root)
	if pid == 0 {
		t.Fatal("cozy up left no Calcifer on the root")
	}

	self, err := os.Executable()
	must(t, err)
	read, write, err := os.Pipe()
	must(t, err)
	reaper := exec.Command(self, reapMode+cozyBin)
	reaper.Stdin, reaper.Stdout, reaper.Stderr = read, os.Stderr, os.Stderr //cozy:stdin-value controlled reaper liveness pipe
	detachSession(reaper)
	must(t, reaper.Start())
	read.Close()
	t.Cleanup(func() { kill(reaper.Process.Pid) })

	_, err = write.WriteString(root + "\n")
	must(t, err)

	// One other process now holds the write end, and nothing else does. It stands in for
	// the test binary: while it lives the reaper waits, and when it dies the reaper acts.
	holder := exec.Command("/bin/sleep", "300")
	holder.ExtraFiles = []*os.File{write}
	must(t, holder.Start())
	write.Close()
	go func() { _ = holder.Wait() }()

	time.Sleep(2 * time.Second)
	if calciferOnRoot(root) != pid {
		t.Fatal("the reaper acted while the process holding its pipe was still alive")
	}

	must(t, holder.Process.Kill()) // SIGKILL: no handler, no cleanup, no warning.
	deadline := time.Now().Add(60 * time.Second)
	for calciferOnRoot(root) != 0 {
		if time.Now().After(deadline) {
			t.Fatalf("Calcifer %d on %s outlived the process that held the reaper's pipe", pid, root)
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err := reaper.Wait(); err != nil {
		t.Fatalf("the reaper did not finish cleanly: %v", err)
	}
}
