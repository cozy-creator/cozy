// Package packageprofile owns Creator's syntax-only handling of Tensorhub's
// four-axis package compatibility profile. Approval remains Tensorhub policy; the
// client accepts only the currently frozen public spellings and never invents a
// default profile.
package packageprofile

import (
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
)

const (
	CU126 = "torch2.13.0-cu126-cp312-linux-x86"
	CU130 = "torch2.13.0-cu130-cp312-linux-x86"
	CPU   = "torch2.13.0-cpu-cp312-linux-x86"
)

var approved = map[string]bool{CPU: true, CU126: true, CU130: true}

var grammar = regexp.MustCompile(`^torch[0-9]+\.[0-9]+\.[0-9]+-[a-z0-9]+-cp[0-9]+-(linux|windows|macos)-(x86|arm64)$`)

// NormalizeSet validates, deduplicates, and sorts one explicit non-empty profile
// set. There is deliberately no default: the exact sorted set becomes release
// meaning and must survive later policy changes byte-for-byte.
func NormalizeSet(raw []string) ([]string, *exit.Error) {
	seen := map[string]bool{}
	for _, value := range raw {
		profile := strings.TrimSpace(value)
		if !grammar.MatchString(profile) {
			return nil, exit.Named(exit.Usage, "package_profile_malformed",
				"%q is not torch<semver>-<accelerator-build>-<python-abi>-<os-cpu>", value).
				WithRemedy("use one exact approved profile: %s, %s, or %s", CPU, CU126, CU130)
		}
		if !approved[profile] {
			return nil, exit.Named(exit.Validation, "package_profile_unapproved",
				"%s is well-formed but is not in this release's approved profile vocabulary", profile).
				WithRemedy("approved profiles are %s, %s, and %s; a policy addition requires a new Creator/Tensorhub contract", CPU, CU126, CU130)
		}
		seen[profile] = true
	}
	if len(seen) == 0 {
		return nil, exit.Usagef("at least one --profile is required; package publication has no mutable default").
			WithRemedy("repeat --profile for the exact frozen set, e.g. --profile %s", CU126)
	}
	out := make([]string, 0, len(seen))
	for profile := range seen {
		out = append(out, profile)
	}
	sort.Strings(out)
	return out, nil
}
