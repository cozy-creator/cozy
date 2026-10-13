//go:build linux || darwin

package machinev1

import (
	"net"

	"golang.org/x/sys/unix"
)

// unsentBytes bounds what an upload's socket holds unsent, so a ping is never queued behind more.
const unsentBytes = 128 << 10

func boundUnsent(conn net.Conn) {
	tcp, ok := conn.(*net.TCPConn)
	if !ok {
		return
	}
	if raw, err := tcp.SyscallConn(); err == nil {
		_ = raw.Control(func(fd uintptr) {
			_ = unix.SetsockoptInt(int(fd), unix.IPPROTO_TCP, unix.TCP_NOTSENT_LOWAT, unsentBytes)
		})
	}
}
