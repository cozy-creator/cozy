//go:build !windows

package cli

import "golang.org/x/sys/unix"

func raiseOpenFileLimit() error {
	var limit unix.Rlimit
	if err := unix.Getrlimit(unix.RLIMIT_NOFILE, &limit); err != nil {
		return err
	}
	if limit.Cur == limit.Max {
		return nil
	}
	limit.Cur = limit.Max
	return unix.Setrlimit(unix.RLIMIT_NOFILE, &limit)
}
