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
	hub, err := url.Parse(base)
	if err != nil || hub.Host == "" || (hub.Scheme != "http" && hub.Scheme != "https") || hub.User != nil || hub.RawQuery != "" || hub.Fragment != "" {
		return nil, exit.Named(exit.Validation, "registry_dependency_hub_invalid", "selected Tensorhub must be an HTTP(S) base URL without credentials, query, or fragment")
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
			return nil, exit.Named(exit.Validation, "registry_dependency_hub_mismatch", "uv.lock selects a different Tensorhub for this package's organization").WithRemedy("in an owned project copy, set this organization index URL to %s while retaining explicit=true; run uv lock there, review and copy back uv.lock, then retry; publication never relocks", selected)
		}
	}
	return args, nil
}

// selectCapturedHubIndexes applies the same selection to an already-owned source
// copy before resolving it. CLI --index loses uv's explicit=true property while
// locking, so captures retain the standard index table and change only its URL.
func selectCapturedHubIndexes(project string) *exit.Error {
	args, problem := selectedHubIndexes(project, false)
	if problem != nil || len(args) == 0 {
		return problem
	}
	selected := map[string]string{}
	for i := 1; i < len(args); i += 2 {
		name, target, _ := strings.Cut(args[i], "=")
		selected[name] = target
	}
	path := filepath.Join(project, "pyproject.toml")
	raw, err := os.ReadFile(path)
	if err != nil {
		return exit.Internalf("cannot read owned index metadata: %s", err)
	}
	var document map[string]any
	if err := toml.Unmarshal(raw, &document); err != nil {
		return exit.Internalf("cannot parse owned index metadata: %s", err)
	}
	tool, _ := document["tool"].(map[string]any)
	uv, _ := tool["uv"].(map[string]any)
	indexes, _ := uv["index"].([]any)
	for _, value := range indexes {
		index, _ := value.(map[string]any)
		name, _ := index["name"].(string)
		if target := selected[name]; target != "" {
			index["url"] = target
		}
	}
	raw, err = toml.Marshal(document)
	if err != nil {
		return exit.Internalf("cannot encode owned index metadata: %s", err)
	}
	if err := os.WriteFile(path, raw, 0600); err != nil {
		return exit.Internalf("cannot retain owned index metadata: %s", err)
	}
	return nil
}
