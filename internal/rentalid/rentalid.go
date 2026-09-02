// Package rentalid owns the portable grammar of a Tensorhub-issued rental id.
// The id becomes both a URL segment and a local credential subject, so accepting
// path syntax or a Windows device name at either boundary would be unsafe.
package rentalid

import (
	"crypto/rand"
	"errors"
	"fmt"
	"math/big"
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

// ErrNoFreeMachineName means every word in the vocabulary names a live rental.
var ErrNoFreeMachineName = errors.New("every machine word is in use by a live rental")

// NewMachineName draws one word this owner is not already using. A machine name
// only has to be unambiguous among the owner's own live rentals — Tensorhub's
// identity for the rental is its `pr-…` id — so a single memorable word is enough,
// and a word comes back into the draw once its rental is released.
func NewMachineName(taken map[string]bool) (string, error) {
	free := make([]string, 0, len(words))
	for _, word := range words {
		if !taken[word] {
			free = append(free, word)
		}
	}
	if len(free) == 0 {
		return "", ErrNoFreeMachineName
	}
	index, err := rand.Int(rand.Reader, big.NewInt(int64(len(free))))
	if err != nil {
		return "", fmt.Errorf("mint private rental name: %w", err)
	}
	return free[index.Int64()], nil
}

// Words is the whole machine vocabulary, sorted.
func Words() []string {
	return append([]string(nil), words...)
}
