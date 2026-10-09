package hub

import "github.com/cozy-creator/cozy/internal/exit"

// ValidPublicationDestination checks a repository a machine is to publish into before a
// machine is selected or bought: a Tensorhub org/name, never a local/ alias.
func ValidPublicationDestination(name string) *exit.Error {
	ref, problem := ParseRef(name)
	if problem != nil {
		return problem
	}
	if ref.Org == "local" {
		return exit.New(exit.Validation, "a publication destination is a public model repository name")
	}
	return nil
}
