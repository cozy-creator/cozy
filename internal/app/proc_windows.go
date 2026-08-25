//go:build windows

// Detaching and signalling the LocalService, on the platform that has neither a session nor
// a signal. Windows detaches with creation flags and ends a process by handle; `cozy down`
// still proves the exit the same way it does everywhere — by the LOCK becoming free, never
// by the pid disappearing.
package app

import (
	"os/exec"
	"syscall"

	"golang.org/x/sys/windows"
)

// detachProcess starts the service in its own process group and with no console attached, which is
// what "survives the terminal that started it" means here.
func detachProcess(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{
		CreationFlags: windows.CREATE_NEW_PROCESS_GROUP | windows.DETACHED_PROCESS,
	}
}

// signalProcess ends ONE process. Windows has no graceful signal to send a service that is not a
// console application, so a TERM and a KILL are the same act — the caller's timeout is what
// distinguishes asking from insisting, and it already reads the lock rather than the pid.
func signalProcess(pid int, sig syscall.Signal) error {
	handle, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(handle)
	return windows.TerminateProcess(handle, 1)
}
