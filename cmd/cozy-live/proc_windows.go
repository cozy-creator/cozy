//go:build windows

// The live driver does not RUN on Windows and this file does not pretend otherwise. It
// exists so `GOOS=windows go build ./...` compiles the whole tree rather than stopping at a
// harness, and every function here refuses by name — a driver that silently did half its
// job on a platform it was never written for would be worse than one that will not start.
package main

import (
	"errors"
	"os/exec"
	"syscall"
)

var errNotOnWindows = errors.New(
	"the cozy-live driver is a Linux harness: it reads /proc, waits on an NVIDIA card and " +
		"kills process groups mid-attempt")

func setProcessGroup(cmd *exec.Cmd) {}

func killGroup(pid int, sig syscall.Signal) error { return errNotOnWindows }

func killProcess(pid int, sig syscall.Signal) error { return errNotOnWindows }
