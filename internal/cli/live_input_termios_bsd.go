//go:build darwin || dragonfly || freebsd || netbsd || openbsd

package cli

import "golang.org/x/sys/unix"

const liveGetTermios = unix.TIOCGETA
const liveSetTermios = unix.TIOCSETA
