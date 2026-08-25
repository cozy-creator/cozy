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

// killGroup signals the whole group: the kill arms are about what survives when NOTHING
// runs a shutdown path, so the grandchildren have to go with the parent.
func killGroup(pid int, sig syscall.Signal) error {
	return syscall.Kill(-pid, sig)
}

func killProcess(pid int, sig syscall.Signal) error {
	return syscall.Kill(pid, sig)
}
