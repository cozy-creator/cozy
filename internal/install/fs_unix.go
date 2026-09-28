//go:build linux || darwin

// The filesystem facts the install pipeline needs that Go's portable API does not
// expose: free space under a directory and whether a file is one of several links to
// the same inode. Each is one syscall here and one Windows API call in the sibling file.
package install

import (
	"io/fs"
	"syscall"
)

func freeBytes(dir string) (int64, bool) {
	var st syscall.Statfs_t
	if err := syscall.Statfs(dir, &st); err != nil {
		return 0, false
	}
	return int64(st.Bavail) * int64(st.Bsize), true
}

// inodeKey identifies one file's storage across every name that reaches it.
type inodeKey struct {
	dev uint64
	ino uint64
}

// inode answers the storage identity, link count and ALLOCATED bytes of one file —
// what makes uv's hardlink dedup measurable as one inode instead of many names, and
// what makes a sparse or tail-packed file report the blocks it really holds.
func inode(info fs.FileInfo) (inodeKey, uint64, int64, bool) {
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return inodeKey{}, 0, 0, false
	}
	return inodeKey{dev: uint64(st.Dev), ino: uint64(st.Ino)}, uint64(st.Nlink), st.Blocks * 512, true
}

// Unix unlinks files through their parent directory. Leave read-only regular files
// alone (they may share an inode) and restore only the directory traversal and write
// bits needed to remove the retired tree.
func removalMode(info fs.FileInfo) fs.FileMode {
	if info.IsDir() {
		return info.Mode().Perm() | 0o700
	}
	return info.Mode().Perm()
}
