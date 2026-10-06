package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func crashAndRestartTransactionDaemon(t *testing.T, first *daemonProcess) *daemonProcess {
	t.Helper()
	must(t, first.cmd.Process.Kill())
	select {
	case <-first.exited:
	case <-time.After(10 * time.Second):
		t.Fatal("owned daemon process did not exit after SIGKILL")
	}
	second := startDaemonProcess(t, first.root)
	// A hard crash leaves the old lock. Observe the new process replace it;
	// deleting that lock here would hide the recovery behavior under test.
	waitUntil(t, "restarted daemon publishes its new credential", func() bool {
		data, err := os.ReadFile(filepath.Join(first.root, "daemon.lock"))
		if err != nil {
			return false
		}
		var address, token string
		for _, line := range strings.Split(string(data), "\n") {
			if value, ok := strings.CutPrefix(line, "addr="); ok {
				address = strings.TrimSpace(value)
			}
			if value, ok := strings.CutPrefix(line, "token="); ok {
				token = strings.TrimSpace(value)
			}
		}
		if token == "" || token == first.token || address == "" {
			return false
		}
		second.addr, second.token = address, token
		return true
	})
	return second
}

func waitUntil(t *testing.T, what string, ok func() bool) {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if ok() {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}
