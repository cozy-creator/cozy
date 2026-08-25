//go:build linux

package coord

import "syscall"

// SO_PEERCRED: the kernel's own answer for who opened this socket. It is filled in at
// connect() time and cannot be restated by the peer afterwards.
func peerCred(fd uintptr) (peerIdentity, bool) {
	cred, err := syscall.GetsockoptUcred(int(fd), syscall.SOL_SOCKET, syscall.SO_PEERCRED)
	if err != nil || cred == nil {
		return peerIdentity{}, false
	}
	return peerIdentity{PID: cred.Pid, UID: cred.Uid}, true
}
