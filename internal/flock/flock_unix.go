//go:build unix

// Package flock is the ONE spelling of "an exclusive claim on a file, held by a live
// process". Two laws rest on it — the LocalService's liveness lock and the records
// database's single-writer lock — and both need the same property: the KERNEL releases
// the claim when the holder dies, SIGKILL included. A pidfile cannot promise that, which
// is why neither law is written as one.
//
// The pair is per-OS because the call is. Everything above it is not.
package flock

import (
	"os"
	"syscall"
)

// Exclusive asks for the claim WITHOUT blocking: an error is the answer "somebody else
// holds it", not a failure to ask.
func Exclusive(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_EX|syscall.LOCK_NB)
}

func Release(f *os.File) error {
	return syscall.Flock(int(f.Fd()), syscall.LOCK_UN)
}
