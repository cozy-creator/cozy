package hub

import (
	"slices"

	"github.com/cozy-creator/cozy/internal/exit"
)

type PublicationRepository struct {
	Org  string `json:"org"`
	Name string `json:"name"`
}

// PublicationRepositories are names as a publication grant's authorization_details carry them.
func PublicationRepositories(names []string) []PublicationRepository {
	out := make([]PublicationRepository, 0, len(names))
	for _, name := range names {
		ref, _ := ParseRef(name)
		out = append(out, PublicationRepository{Org: ref.Org, Name: ref.Name})
	}
	return out
}

// NormalizePublicationRepositories validates explicit consent before selecting or
// buying a machine. The order and duplicates do not change request identity.
func NormalizePublicationRepositories(repositories []string) ([]string, *exit.Error) {
	names := slices.Clone(repositories)
	slices.Sort(names)
	names = slices.Compact(names)
	if len(names) < 1 || len(names) > 16 {
		return nil, exit.New(exit.Validation, "publication permission needs 1..16 explicit model repositories")
	}
	for _, name := range names {
		ref, problem := ParseRef(name)
		if problem != nil {
			return nil, problem
		}
		if ref.Org == "local" {
			return nil, exit.New(exit.Validation, "publication permission needs public model repository names")
		}
	}
	return names, nil
}
