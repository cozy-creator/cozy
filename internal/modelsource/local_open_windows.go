//go:build windows

package modelsource

import "os"

func openLocalNoFollow(path string) (*os.File, error) {
	info, err := os.Lstat(path)
	if err != nil || info.Mode()&os.ModeSymlink != 0 {
		return nil, os.ErrInvalid
	}
	return os.Open(path)
}
