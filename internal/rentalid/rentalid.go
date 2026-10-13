// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"regexp"
	"strings"
)

var pattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`)
var machinePattern = regexp.MustCompile(`^[a-z][a-z0-9-]{0,31}$`)

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

// ValidMachineName is the portable booking-alias grammar. Canonical host names
// are supplied by Tensorhub and are never drawn or hashed by this client.
func ValidMachineName(name string) bool {
	return name != "local" && machinePattern.MatchString(name)
}
