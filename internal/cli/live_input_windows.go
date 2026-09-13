//go:build windows

package cli

import "golang.org/x/term"

// Console raw mode changes input flags only; output newline handling is separate.
func setLiveInputMode(fd int) error {
	_, err := term.MakeRaw(fd)
	return err
}
