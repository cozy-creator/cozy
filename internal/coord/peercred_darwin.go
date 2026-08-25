//go:build darwin

package coord

import "golang.org/x/sys/unix"

// Darwin splits the same fact across two socket options: LOCAL_PEERCRED carries the
// uid, LOCAL_PEERPID the pid. Both are the kernel's, not the peer's.
func peerCred(fd uintptr) (peerIdentity, bool) {
	xu, err := unix.GetsockoptXucred(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERCRED)
	if err != nil || xu == nil {
		return peerIdentity{}, false
	}
	pid, err := unix.GetsockoptInt(int(fd), unix.SOL_LOCAL, unix.LOCAL_PEERPID)
	if err != nil {
		return peerIdentity{}, false
	}
	return peerIdentity{PID: int32(pid), UID: xu.Uid}, true
}
