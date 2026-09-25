package install

import (
	"net/url"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/pelletier/go-toml/v2"
)

// PublishedDependency joins a committed same-org package to selected immutable wheel bytes.
type PublishedDependency struct {
	Package, Version string
	Digests          map[string]bool
}

// PublishedDependencies joins selected requirement hashes to a verified uv.lock.
// A same-named PyPI library is not a Tensorhub package dependency.
func PublishedDependencies(pkg string, lockBytes, locked []byte) ([]PublishedDependency, *exit.Error) {
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		return nil, problem
	}
	var lock struct {
		Packages []struct {
			Name    string `toml:"name"`
			Version string `toml:"version"`
			Source  struct {
				Registry string `toml:"registry"`
			} `toml:"source"`
			Wheels []struct {
				Hash string `toml:"hash"`
			} `toml:"wheels"`
		} `toml:"package"`
	}
	if err := toml.Unmarshal(lockBytes, &lock); err != nil {
		return nil, exit.New(exit.Structural, "published dependency lock is invalid")
	}
	selectedRows := map[string][]string{}
	for _, line := range strings.Split(string(locked), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || strings.HasPrefix(fields[0], "-") || strings.HasPrefix(fields[0], "#") {
			continue
		}
		key := fields[0]
		if _, duplicate := selectedRows[key]; duplicate {
			return nil, exit.New(exit.Conflict, "published dependency requirements are ambiguous")
		}
		selectedRows[key] = fields[1:]
	}
	selected := map[string]PublishedDependency{}
	for _, row := range lock.Packages {
		index, err := url.Parse(row.Source.Registry)
		if err != nil || index.User != nil || index.RawQuery != "" || index.Fragment != "" || (index.Scheme != "https" && index.Scheme != "http") || strings.Trim(index.Path, "/") != "v1/index/"+ref.Org+"/simple" {
			continue
		}
		hashes := selectedRows[row.Name+"=="+row.Version]
		if direct := selectedRows[row.Name]; len(direct) >= 3 && direct[0] == "@" {
			if len(hashes) != 0 {
				return nil, exit.New(exit.Conflict, "published dependency requirements are ambiguous")
			}
			hashes = direct[2:]
		}
		if len(hashes) == 0 {
			continue
		}
		allowed := map[string]bool{}
		for _, wheel := range row.Wheels {
			for _, hash := range hashes {
				if hash == "--hash="+wheel.Hash {
					allowed[wheel.Hash] = true
				}
			}
		}
		if len(allowed) == 0 {
			return nil, exit.New(exit.Conflict, "published callable dependency differs from its locked wheel")
		}
		if _, duplicate := selected[row.Name]; duplicate {
			return nil, exit.New(exit.Conflict, "published callable dependency is ambiguous")
		}
		selected[row.Name] = PublishedDependency{ref.Org + "/" + row.Name, row.Version, allowed}

	}
	out := make([]PublishedDependency, 0, len(selected))
	for _, row := range selected {
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Package < out[j].Package })
	return out, nil
}
