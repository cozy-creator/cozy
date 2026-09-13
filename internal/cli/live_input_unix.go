//go:build !windows

package cli

import "golang.org/x/sys/unix"

// Cbreak input keeps ordinary output processing. Full raw mode would turn the
// progress renderer's newlines into line feeds without returning to column zero.
func setLiveInputMode(fd int) error {
	state, err := unix.IoctlGetTermios(fd, liveGetTermios)
	if err != nil {
		return err
	}
	state.Iflag &^= unix.IXON | unix.IXOFF
	state.Lflag &^= unix.ECHO | unix.ECHONL | unix.ICANON | unix.IEXTEN | unix.ISIG
	state.Cc[unix.VMIN], state.Cc[unix.VTIME] = 1, 0
	return unix.IoctlSetTermios(fd, liveSetTermios, state)
}
