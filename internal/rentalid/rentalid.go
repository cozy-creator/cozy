// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"regexp"
	"strings"
)

var adjectives = [...]string{"brisk", "bright", "calm", "gentle", "lucid", "quiet", "swift", "vivid"}
var nouns = [...]string{"badger", "falcon", "heron", "otter", "raven", "tiger", "turtle", "wolf"}

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

// NewMachineName returns a private, memorable provider name with 64 bits of
// random collision resistance. It carries no package, model, or workload fact.
func NewMachineName() (string, error) {
	var raw [10]byte
	if _, err := rand.Read(raw[:]); err != nil {
		return "", fmt.Errorf("mint private rental name: %w", err)
	}
	return adjectives[int(raw[0])%len(adjectives)] + "-" +
		nouns[int(raw[1])%len(nouns)] + "-" + hex.EncodeToString(raw[2:]), nil
}
