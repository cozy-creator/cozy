//go:build !windows

// Detaching and signalling the LocalService — the two process facts this package needs, on
// the platforms that spell them with a session and a signal. `proc_windows.go` says the
// same two things the way Windows has them.
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

// signalProcess reaches ONE process — the service, never a group. `cozy down` asks the service to
// end itself, and the proof of exit is the lock becoming free.
func signalProcess(pid int, sig syscall.Signal) error {
	return syscall.Kill(pid, sig)
}
