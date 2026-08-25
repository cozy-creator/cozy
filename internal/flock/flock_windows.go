//go:build windows

package flock

import (
	"os"

	"golang.org/x/sys/windows"
)

// Windows has no flock(2); LockFileEx is the same law under another name.
// EXCLUSIVE_LOCK|FAIL_IMMEDIATELY is LOCK_EX|LOCK_NB, the range is the whole file, and
// the lock dies with the handle — including on a terminated process, which is the
// property the liveness proof rests on.
const allBytes = ^uint32(0)

func Exclusive(f *os.File) error {
	return windows.LockFileEx(windows.Handle(f.Fd()),
		windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
		0, allBytes, allBytes, new(windows.Overlapped))
}

func Release(f *os.File) error {
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, allBytes, allBytes,
		new(windows.Overlapped))
}
