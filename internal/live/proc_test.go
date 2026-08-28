package live

import (
	"os/exec"
	"syscall"
)

// The suite kills process GROUPS: a worker deliberately launches in its own group, so an
// owner-kill leaves it for restart reconciliation to find and fence.
func setProcessGroup(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

func killGroup(pid int) error { return syscall.Kill(-pid, syscall.SIGKILL) }
