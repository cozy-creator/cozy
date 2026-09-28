// Package units formats small human-readable measurements without depending on
// the CLI output encoder.
package units

import "fmt"

// Bytes renders a binary byte count.
func Bytes(n int64) string {
	const unit = 1024
	if n < unit {
		return fmt.Sprintf("%dB", n)
	}
	divisor, exponent := int64(unit), 0
	for value := n / unit; value >= unit; value /= unit {
		divisor *= unit
		exponent++
	}
	return fmt.Sprintf("%.1f%ciB", float64(n)/float64(divisor), "KMGTPE"[exponent])
}
