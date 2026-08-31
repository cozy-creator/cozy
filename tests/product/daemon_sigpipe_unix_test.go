//go:build !windows

package producttest

import (
	"encoding/json"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
)

func TestDaemonOutlivesClosedStartupDiagnosticPipe(t *testing.T) {
	root := t.TempDir()
	daemonPath := filepath.Join(root, "cozy-daemon")
	must(t, os.Symlink(cozyBin, daemonPath))
	diagnostics, childStderr, err := os.Pipe()
	must(t, err)

	// A real short-lived parent launches the real private daemon and exits. Schedtrace
	// gives the daemon a harmless, delayed stderr write after its startup reader closes.
	testBinary, err := os.Executable()
	must(t, err)
	parent := exec.Command(testBinary, "--cozy-test-daemon-parent="+daemonPath)
	parent.Env = append(childEnv(t, root), "GODEBUG=schedtrace=50")
	parent.Stdout = io.Discard
	parent.Stderr = childStderr
	must(t, parent.Run())
	must(t, childStderr.Close())
	t.Cleanup(func() {
		_ = diagnostics.Close()
		terminateTestDaemon(t, root)
	})

	deadline := time.Now().Add(15 * time.Second)
	state := daemon.State{}
	for time.Now().Before(deadline) {
		state = daemon.Probe(config.Config{Home: root})
		if state.Up {
			break
		}
		time.Sleep(25 * time.Millisecond)
	}
	if !state.Up {
		t.Fatalf("orphaned daemon did not become ready")
	}
	must(t, diagnostics.Close())
	time.Sleep(150 * time.Millisecond) // at least two delayed schedtrace writes

	code, out := runCozy(t, root, "up", "--json", "--full")
	var status struct {
		PID     int  `json:"pid"`
		Changed bool `json:"changed"`
	}
	if err := json.Unmarshal([]byte(out), &status); code != 0 || err != nil ||
		status.Changed || status.PID != state.PID {
		t.Fatalf("daemon died after its parent and diagnostic reader exited [exit %d, parse %v]\n%s", code, err, out)
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("surviving daemon did not shut down cleanly [exit %d]\n%s", code, out)
	}
}
