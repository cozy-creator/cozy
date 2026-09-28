//go:build windows

package cli

import "io"

// terminalWidth has no winsize ioctl here; 0 means the rewrite lane never clamps.
func terminalWidth(io.Writer) int { return 0 }
