package install

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
)

// Ref is `org/package[@vN]` — the hub's grammar, consumed verbatim. "unset" has no
// spelling in it: Major is present or it is not.
type Ref struct {
	Package  string // org/name
	Major    int
	HasMajor bool
}

func (r Ref) String() string {
	if r.HasMajor {
		return fmt.Sprintf("%s@v%d", r.Package, r.Major)
	}
	return r.Package
}

func ParseRef(s string) (Ref, *exit.Error) {
	bad := func(why string) *exit.Error {
		return exit.Usagef("%q is not a package ref: %s", s, why).
			WithRemedy("the grammar is org/package[@vN], e.g. cozy-creator/demo@v1").
			WithNext("cozy help package install")
	}
	name, ver, hasVer := strings.Cut(s, "@")
	org, ep, ok := strings.Cut(name, "/")
	if !ok || org == "" || ep == "" || strings.Contains(ep, "/") {
		return Ref{}, bad("expected exactly one org/package separator")
	}
	r := Ref{Package: name}
	if !hasVer {
		return r, nil
	}
	if !strings.HasPrefix(ver, "v") {
		return Ref{}, bad("a major is spelled vN, e.g. @v1")
	}
	n, err := strconv.Atoi(ver[1:])
	if err != nil || n < 0 {
		return Ref{}, bad("a major is a non-negative integer after v")
	}
	r.Major, r.HasMajor = n, true
	return r, nil
}

// MajorOf is the semver major of a release version string.
func MajorOf(version string) (int, *exit.Error) {
	head, _, _ := strings.Cut(version, ".")
	n, err := strconv.Atoi(strings.TrimSpace(head))
	if err != nil || n < 0 {
		return 0, exit.New(exit.Validation, "release version %q has no semver major", version).
			WithRemedy("a release version starts with a non-negative integer major")
	}
	return n, nil
}
