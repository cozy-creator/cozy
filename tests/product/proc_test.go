package producttest

import (
	"os/exec"
	"syscall"
)

// The suite kills process GROUPS: a worker deliberately launches in its own group, so an
// owner-kill leaves it for restart reconciliation to find and fence.
func setProcessGroup(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true} }

func killGroup(pid int) error { return syscall.Kill(-pid, syscall.SIGKILL) }

// detachSession puts a child in its OWN session, out of reach of the signal that ends
// `go test`. Only the reaper wants this: it is the process whose whole job is to outlive
// the run that started it, including a run that is interrupted or SIGKILLed.
func detachSession(cmd *exec.Cmd) { cmd.SysProcAttr = &syscall.SysProcAttr{Setsid: true} }

// kill ends one process for certain: its group first, because a daemon detaches into its
// own session and therefore leads a group, then the bare pid for anything that does not.
func kill(pid int) {
	if pid <= 0 {
		return
	}
	_ = syscall.Kill(-pid, syscall.SIGKILL)
	_ = syscall.Kill(pid, syscall.SIGKILL)
}

// alive answers from the OS, not from a record: signal 0 reaches a live process and is
// refused for one that is gone.
func alive(pid int) bool {
	if pid <= 0 {
		return false
	}
	err := syscall.Kill(pid, 0)
	return err == nil || err == syscall.EPERM
}
