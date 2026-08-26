//go:build darwin

package coord

import (
	"strconv"

	"golang.org/x/sys/unix"
)

// birthOf on Darwin: the kernel's process start time from `kern.proc.pid`, as
// seconds.microseconds. Same contract as the Linux and Windows spellings — a reused pid
// has a different birth.
func birthOf(pid int) string {
	kp, err := unix.SysctlKinfoProc("kern.proc.pid", pid)
	if err != nil {
		return ""
	}
	t := kp.Proc.P_starttime
	return strconv.FormatInt(int64(t.Sec), 10) + "." + strconv.FormatInt(int64(t.Usec), 10)
}
