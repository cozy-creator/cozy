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

// Windows' FileInfo carries no link count (the Win32 find data has none), so a
// hardlinked file is indistinguishable from an exclusive one here and every byte is
// charged to this generation. It over-reports exclusive bytes; it never invents shared
// ones, and `cozy endpoint list` prints a number that is at worst pessimistic.
func hardlinked(fs.FileInfo) bool { return false }

// deviceOf is the volume serial number of the volume a path lives on — Windows' answer
// to st_dev, and the same question pickLinkMode asks: are uv's cache and this generation
// on one filesystem, so hardlink dedup can work at all.
func deviceOf(p string) (uint64, bool) {
	name, err := windows.UTF16PtrFromString(p)
	if err != nil {
		return 0, false
	}
	root := make([]uint16, windows.MAX_PATH+1)
	if err := windows.GetVolumePathName(name, &root[0], uint32(len(root))); err != nil {
		return 0, false
	}
	var serial uint32
	if err := windows.GetVolumeInformation(&root[0], nil, 0, &serial, nil, nil, nil, 0); err != nil {
		return 0, false
	}
	return uint64(serial), true
}
