//go:build windows

package install

import (
	"io/fs"

	"golang.org/x/sys/windows"
)

func freeBytes(dir string) (int64, bool) {
	p, err := windows.UTF16PtrFromString(dir)
	if err != nil {
		return 0, false
	}
	var avail, total, free uint64
	if err := windows.GetDiskFreeSpaceEx(p, &avail, &total, &free); err != nil {
		return 0, false
	}
	// The caller's quota, not the volume's — the same thing statfs's Bavail reports.
	return int64(avail), true
}

// inodeKey mirrors the unix identity; unused on Windows, where the Win32 find data
// carries no link count.
type inodeKey struct {
	dev uint64
	ino uint64
}

// Windows' FileInfo carries no link count, so a hardlinked file is indistinguishable
// from an exclusive one here and every byte is charged to this install by logical
// length. It over-reports exclusive bytes; it never invents shared ones, and `cozy
// package list` prints a number that is at worst pessimistic.
func inode(fs.FileInfo) (inodeKey, uint64, int64, bool) { return inodeKey{}, 0, 0, false }

// Windows requires the writable bit for a read-only file to be removed. Directories
// also need traversal permission while the retired tree is walked.
func removalMode(info fs.FileInfo) fs.FileMode {
	if info.IsDir() {
		return info.Mode().Perm() | 0o700
	}
	return info.Mode().Perm() | 0o200
}
