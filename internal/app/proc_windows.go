//go:build windows

// Detaching and signalling the LocalService, on the platform that has neither a session nor
// a signal. Windows detaches with creation flags; `cozy down` asks cooperatively over the
// AUTHENTICATED shutdown route (the service has no console a ctrl event could reach), and
// only the forced tier ends the process by handle. Exit is still proved the same way it is
// everywhere — by the LOCK becoming free, never by the pid disappearing.
package app

import (
	"fmt"
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

// signalProcess refuses to pretend (#449): a TERM here used to TerminateProcess, which is
// a KILL wearing a polite name — it ended the service mid-transaction and called it
// graceful. Now TERM reports that no cooperative signal exists (the shutdown route is the
// cooperative path on this platform) and only an explicit KILL terminates.
func signalProcess(pid int, sig syscall.Signal) error {
	if sig != syscall.SIGKILL {
		return fmt.Errorf("windows has no cooperative process signal; the shutdown route is the ask")
	}
	handle, err := windows.OpenProcess(windows.PROCESS_TERMINATE, false, uint32(pid))
	if err != nil {
		return err
	}
	defer windows.CloseHandle(handle)
	return windows.TerminateProcess(handle, 1)
}
