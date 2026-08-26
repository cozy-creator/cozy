//go:build windows

package flock

import (
	"os"

	"golang.org/x/sys/windows"
)

// Windows has no flock(2); LockFileEx is the same law under another name — with one
// difference that matters: an exclusive range lock is MANDATORY there, and a whole-file
// range blocked every OTHER process's ReadFile of the lock body (Probe read an empty
// document and reported pid 0; found by the windows-proc run). So the lock range is ONE
// BYTE at offset 2^32 — far beyond any data the file will hold, which Windows permits —
// and the published bytes at offset 0 stay readable by everyone. Mutual exclusion holds
// because every locker uses the same range, and the lock still dies with the handle —
// including on a terminated process, which is the property the liveness proof rests on.
func Exclusive(f *os.File) error {
	return windows.LockFileEx(windows.Handle(f.Fd()),
		windows.LOCKFILE_EXCLUSIVE_LOCK|windows.LOCKFILE_FAIL_IMMEDIATELY,
		0, 1, 0, &windows.Overlapped{OffsetHigh: 1})
}

func Release(f *os.File) error {
	return windows.UnlockFileEx(windows.Handle(f.Fd()), 0, 1, 0,
		&windows.Overlapped{OffsetHigh: 1})
}
