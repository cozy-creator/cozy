//go:build !windows

package machines

import (
	"errors"
	"io/fs"
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

// diskOf is a file's inode, its link count and the disk its blocks take.
func diskOf(info fs.FileInfo) (uint64, int64, int64) {
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, 1, info.Size()
	}
	return uint64(stat.Ino), int64(stat.Nlink), int64(stat.Blocks) * 512
}
