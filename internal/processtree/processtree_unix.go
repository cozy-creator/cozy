//go:build !windows

// The PROCESS GROUP, on the platforms that have one. A worker is a supervisor that spawns
// a device executor, which may itself spawn a border subprocess; stopping the worker means
// stopping that whole tree, or a grandchild outlives the terminal that closed its attempt
// and keeps holding a device.
//
// Unix says that with a process group: the child leads its own, and a negative pid signals
// every member. `processtree_windows.go` says the same thing with a Job Object, which is the
// analogue that exists there.
package processtree

import (
	"os/exec"
	"syscall"
)

// Prepare makes the child the LEADER of a fresh group, before it starts. Every
// process it goes on to spawn inherits the group, which is what makes one signal reach the
// whole tree.
func Prepare(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
}

// Adopt has nothing to do here: the group was established at fork. It exists
// because Windows cannot attach its equivalent until the process is running, and it can
// fail there — which is why it returns an error the spawn treats as fatal (#449).
func Adopt(cmd *exec.Cmd) error { return nil }

// Release has nothing to hold here either: a Unix group is a kernel fact that dies
// with its members, not a handle this process keeps.
func Release(pid int) {}

// Kill signals the whole group. A NEGATIVE pid is the group, and that is the point:
// signalling the leader alone leaves its children running.
func Kill(pid int, sig syscall.Signal) error {
	return syscall.Kill(-pid, sig)
}

// alive asks the kernel whether the pid is still there. Signal 0 delivers nothing and
// reports exactly that.
func Alive(pid int) bool { return syscall.Kill(pid, 0) == nil }
