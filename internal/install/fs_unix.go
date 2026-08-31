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

// hardlinked answers whether this file's bytes are shared with another name — what
// makes uv's hardlink dedup visible as SHARED rather than exclusive bytes.
func hardlinked(info fs.FileInfo) bool {
	st, ok := info.Sys().(*syscall.Stat_t)
	return ok && st.Nlink > 1
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
