// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"regexp"
	"strings"
)

var pattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)

// Valid reports whether id is one portable opaque name, not a path or device.
func Valid(id string) bool {
	if !pattern.MatchString(id) {
		return false
	}
	base := strings.ToUpper(strings.SplitN(id, ".", 2)[0])
	if base == "CON" || base == "PRN" || base == "AUX" || base == "NUL" {
		return false
	}
	return !(len(base) == 4 &&
		(strings.HasPrefix(base, "COM") || strings.HasPrefix(base, "LPT")) &&
		base[3] >= '1' && base[3] <= '9')
}
