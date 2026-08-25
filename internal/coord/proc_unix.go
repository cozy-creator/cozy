//go:build !windows

// The PROCESS GROUP, on the platforms that have one. A worker is a supervisor that spawns
// a device executor, which may itself spawn a border subprocess; stopping the worker means
// stopping that whole tree, or a grandchild outlives the terminal that closed its attempt
// and keeps holding a device.
//
// Unix says that with a process group: the child leads its own, and a negative pid signals
// every member. `proc_windows.go` says the same thing with a Job Object, which is the
// analogue that exists there.
package coord

import (
	"os/exec"
	"syscall"
)

// setProcessGroup makes the child the LEADER of a fresh group, before it starts. Every
// process it goes on to spawn inherits the group, which is what makes one signal reach the
// whole tree.
func setProcessGroup(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// adoptProcessGroup has nothing to do here: the group was established at fork. It exists
// because Windows cannot attach its equivalent until the process is running.
func adoptProcessGroup(cmd *exec.Cmd) {}

// killGroup signals the whole group. A NEGATIVE pid is the group, and that is the point:
// signalling the leader alone leaves its children running.
func killGroup(pid int, sig syscall.Signal) error {
	return syscall.Kill(-pid, sig)
}

// alive asks the kernel whether the pid is still there. Signal 0 delivers nothing and
// reports exactly that.
func alive(pid int) bool { return syscall.Kill(pid, 0) == nil }
