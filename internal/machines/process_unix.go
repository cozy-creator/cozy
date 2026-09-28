//go:build !windows

package machines

import (
	"errors"
	"os/exec"
	"syscall"
)

// detach starts the Host in its own session: it survives this daemon as a pod survives its
// controller.
func detach(command *exec.Cmd) {
	command.SysProcAttr = &syscall.SysProcAttr{Setsid: true}
}

// terminate asks a Host to stop; one already gone is stopped.
func terminate(pid int) error {
	if err := syscall.Kill(pid, syscall.SIGTERM); err != nil && !errors.Is(err, syscall.ESRCH) {
		return err
	}
	return nil
}
