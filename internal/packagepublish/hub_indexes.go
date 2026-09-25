package packagepublish

import (
	"net/url"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

var uvIndexName = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)

// selectedHubIndexes uses uv's named-index override, never edits authored metadata
// or locks. Only an explicitly scoped, canonical Tensorhub index for this package's
// own organization moves with --tensorhub. No credentials enter arguments or logs.
func selectedHubIndexes(project string, locked bool) ([]string, *exit.Error) {
	document, problem := readProjectDocument(filepath.Join(project, "pyproject.toml"))
	if problem != nil {
		return nil, problem
	}
	organization := document.Tool.Cozy.Organization
	base := config.Frozen().HubURL
	if organization == "" || base == "" {
		return nil, nil
	}
	selected := strings.TrimRight(base, "/") + "/v1/index/" + url.PathEscape(organization) + "/simple/"
	var args []string
	for _, index := range document.Tool.UV.Index {
		if !index.Explicit || !uvIndexName.MatchString(index.Name) {
			continue
		}
		parsed, err := url.Parse(index.URL)
		if err != nil || parsed.Scheme != "https" || parsed.Host != "tensorhub.com" ||
			parsed.User != nil || parsed.RawQuery != "" || parsed.Fragment != "" ||
			parsed.EscapedPath() != "/v1/index/"+url.PathEscape(organization)+"/simple/" {
			continue
		}
		args = append(args, "--index", index.Name+"="+selected)
	}
	if len(args) == 0 || !locked {
		return args, nil
	}
	raw, err := os.ReadFile(filepath.Join(project, "uv.lock"))
	if err != nil || int64(len(raw)) > maxLockBytes {
		return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid", "cannot read the bounded uv.lock for selected Tensorhub")
	}
	var lock struct {
		Packages []struct {
			Source struct {
				Registry string `toml:"registry"`
			} `toml:"source"`
		} `toml:"package"`
	}
	if err := toml.Unmarshal(raw, &lock); err != nil {
		return nil, exit.Named(exit.Validation, "registry_dependency_lock_invalid", "uv.lock is invalid TOML")
	}
	for _, pkg := range lock.Packages {
		if orgIndexNamespace(pkg.Source.Registry) == organization && strings.TrimRight(pkg.Source.Registry, "/") != strings.TrimRight(selected, "/") {
			return nil, exit.Named(exit.Validation, "registry_dependency_hub_mismatch", "uv.lock selects a different Tensorhub for this package's organization").WithRemedy("run `uv lock %s`, review uv.lock, then retry; publication never relocks", strings.Join(args, " "))
		}
	}
	return args, nil
}
