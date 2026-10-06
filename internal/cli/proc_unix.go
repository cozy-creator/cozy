//go:build !windows

// Detaching the daemon on platforms that spell it with a session.
package cli

import (
	"os/exec"
	"os/signal"
	"syscall"
)

// The short-lived CLI reads startup diagnostics from the daemon's stderr pipe, then
// closes it once the authenticated API is ready. Later library diagnostics must see
// EPIPE rather than kill the already-detached daemon.
func ignoreCalciferBrokenPipe() {
	signal.Ignore(syscall.SIGPIPE)
}

// detachProcess puts the daemon in its OWN session, so closing the terminal that started it does
// not take it with it.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
