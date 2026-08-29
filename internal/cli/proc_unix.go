//go:build !windows

// Detaching the controller on platforms that spell it with a session.
package cli

import (
	"os/exec"
	"syscall"
)

// detachProcess puts the service in its OWN session, so closing the terminal that started it does
// not take it with it.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}
