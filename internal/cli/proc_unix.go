//go:build !windows

// Detaching the daemon on platforms that spell it with a session.
package cli

import (
	"os/exec"
	"syscall"
)

// detachProcess puts the daemon in its OWN session, so closing the terminal that started it does
// not take it with it.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
