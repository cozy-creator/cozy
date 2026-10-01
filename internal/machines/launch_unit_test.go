package machines

import (
	"context"
	"encoding/base64"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/userunit"
)

// A Host whose user unit has started but whose process is not yet the agent is starting, not
// exited: systemd's executor runs first as the unit's main process (here a shell that waits
// before it execs). Once the unit ends, the Host has exited.
func TestAwaitWaitsOutTheUnitsExecutor(t *testing.T) {
	if !userunit.Available() {
		t.Skip("needs a systemd user manager")
	}
	h := NewHost(t.TempDir(), "", nil)
	sleep, err := exec.LookPath("sleep")
	if err != nil {
		t.Skip("needs sleep")
	}
	raw, err := os.ReadFile(sleep)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Dir(h.binary()), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(h.binary(), raw, 0o755); err != nil {
		t.Fatal(err)
	}
	executor := filepath.Join(h.dir, "executor.sh")
	if err := os.WriteFile(executor, []byte("#!/bin/sh\nsleep 1\nexec "+h.binary()+" 30\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	unit := userunit.Name("cozy-machine-agent", h.dir, false)
	t.Cleanup(func() { _ = userunit.Stop(unit) })
	if _, err := userunit.Start(userunit.Spec{Unit: unit, Argv: []string{executor}}); err != nil {
		t.Fatal(err)
	}
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	closed := listener.Addr().(*net.TCPAddr).Port
	listener.Close()
	record := hostRecord{PID: userunit.MainPID(unit), MediaPort: closed, ReceiptKey: base64.RawURLEncoding.EncodeToString(make([]byte, 32))}
	await := func() string {
		ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
		defer cancel()
		_, problem := h.await(ctx, &record)
		return problem.ErrName()
	}
	if h.alive(record.PID) {
		t.Fatal("the unit's process became the agent before its executor finished")
	}
	if got := await(); got != "machine.host_starting" {
		t.Fatalf("a Host still in its unit's executor answered %s", got)
	}
	for deadline := time.Now().Add(10 * time.Second); !h.alive(record.PID); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatal("the unit's process never became the agent")
		}
	}
	if err := userunit.Stop(unit); err != nil {
		t.Fatal(err)
	}
	if got := await(); got != "machine.host_exited" {
		t.Fatalf("a Host whose unit ended answered %s", got)
	}
}
