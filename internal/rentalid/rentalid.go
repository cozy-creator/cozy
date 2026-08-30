// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"crypto/sha256"
	"encoding/hex"
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

// ValidMachineName reports whether a user-facing machine alias is short, shell-safe,
// and distinct from the reserved name for this computer.
func ValidMachineName(name string) bool {
	return name != "local" && machinePattern.MatchString(name)
}

// MachineName derives the stable fallback shown when the renter did not choose --name.
func MachineName(id string) string {
	sum := sha256.Sum256([]byte(id))
	return "rental-" + hex.EncodeToString(sum[:4])
}
