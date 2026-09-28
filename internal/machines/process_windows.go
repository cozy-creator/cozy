//go:build windows

package machines

import (
	"os"
	"os/exec"
)

// The machine Host is Linux-only; these keep the client building everywhere.
func detach(*exec.Cmd) {}

func terminate(pid int) error {
	process, err := os.FindProcess(pid)
	if err != nil {
		return nil
	}
	return process.Kill()
}
