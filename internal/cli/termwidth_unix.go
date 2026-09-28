//go:build !windows

package cli

import (
	"io"
	"os"

	"golang.org/x/sys/unix"
)

// terminalWidth is the CURRENT column count of the terminal behind w, or 0 when w is
// not a terminal. Read per render, so a resize takes effect on the next rewrite.
func terminalWidth(w io.Writer) int {
	file, ok := w.(*os.File)
	if !ok {
		return 0
	}
	size, err := unix.IoctlGetWinsize(int(file.Fd()), unix.TIOCGWINSZ)
	if err != nil || size.Col == 0 {
		return 0
	}
	return int(size.Col)
}
