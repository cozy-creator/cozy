//go:build linux || darwin

// The three filesystem facts the install pipeline needs that Go's portable API does not
// expose: free space under a directory, whether a file is one of several links to the
// same inode, and which device a path lives on. Each is one syscall here and one Windows
// API call in the sibling file; nothing above this line knows which.
package install

import (
	"io/fs"
	"os"
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

func deviceOf(p string) (uint64, bool) {
	info, err := os.Stat(p)
	if err != nil {
		return 0, false
	}
	st, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return 0, false
	}
	return uint64(st.Dev), true
}
