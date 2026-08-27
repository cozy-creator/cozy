//go:build !windows

// The live driver's process-group instruments. This driver is a LINUX harness — it reads
// `/proc`, waits on an NVIDIA card and kills process groups mid-attempt — so the Windows
// half of this pair exists to keep `go build ./...` honest on that platform, not to make
// the driver run there.
package main

import (
	"os/exec"
	"syscall"
)

func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// killGroup signals exactly the named process group. Workers deliberately launch in
// their own groups, so an owner-kill arm leaves them for restart reconciliation to find
// and fence by pid plus kernel birth identity.
func killGroup(pid int, sig syscall.Signal) error {
	return syscall.Kill(-pid, sig)
}

func killProcess(pid int, sig syscall.Signal) error {
	return syscall.Kill(pid, sig)
}
