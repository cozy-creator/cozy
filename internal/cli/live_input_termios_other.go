//go:build aix || linux || solaris || zos

package cli

import "golang.org/x/sys/unix"

const liveGetTermios = unix.TCGETS
const liveSetTermios = unix.TCSETS
