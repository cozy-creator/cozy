package packagepublish

// cl-084: with cozy-runtime and tensorfs on PyPI and packages on the org
// index, a uv.lock row whose source is an editable checkout or a path outside
// the package tree has no excuse — no machine but the author's holds those
// bytes, so the release cannot travel. Publish refuses the row by name.
// In-tree path and wheel rows stay legal: the collector vendors them.

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

type uvLockDocument struct {
	Packages []uvLockPackage `toml:"package"`
}

type uvLockPackage struct {
	Name    string         `toml:"name"`
	Version string         `toml:"version"`
	Source  map[string]any `toml:"source"`
}

// The uv.lock source keys that name a path on the author's machine. Registry,
// git, and URL sources are judged elsewhere; these rows are the untransportable
// class when they escape the tree.
var authorLocalSourceKeys = []string{"editable", "directory", "path", "virtual"}

func refuseAuthorLocalLockRows(root, lockPath string) *exit.Error {
	raw, err := os.ReadFile(lockPath)
	if err != nil || len(raw) == 0 || int64(len(raw)) > maxLockBytes {
		return exit.Named(exit.Validation, "package_publish.lock_invalid",
			"uv.lock must be a readable, non-empty file at or below %d B", maxLockBytes)
	}
	var lock uvLockDocument
	if err := toml.Unmarshal(raw, &lock); err != nil {
		return exit.Named(exit.Validation, "package_publish.lock_invalid",
			"uv.lock is not valid TOML: %v", err)
	}
	for _, row := range lock.Packages {
		for _, key := range authorLocalSourceKeys {
			value, ok := row.Source[key].(string)
			if !ok {
				continue
			}
			if withinPackageTree(root, value) {
				continue
			}
			return exit.Named(exit.Validation, "package_publish.author_local_row",
				"uv.lock row %s %s (%s = %q) resolves outside the package tree and cannot travel",
				row.Name, row.Version, key, value).
				WithRemedy("publish %s to PyPI or your org index and depend on it by name, or move it into the package tree so publish vendors it",
					row.Name)
		}
	}
	return nil
}

func withinPackageTree(root, value string) bool {
	candidate := value
	if !filepath.IsAbs(candidate) {
		candidate = filepath.Join(root, candidate)
	}
	candidate = filepath.Clean(candidate)
	if resolved, err := filepath.EvalSymlinks(candidate); err == nil {
		candidate = resolved
	}
	relative, err := filepath.Rel(root, candidate)
	if err != nil {
		return false
	}
	return relative != ".." && !strings.HasPrefix(filepath.ToSlash(relative), "../")
}
