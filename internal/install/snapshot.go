package install

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/pelletier/go-toml/v2"
)

// snapshotSource copies only the existing bounded, admissible package source set.
// Its identity is checked after copying so an edit cannot mix two revisions.
func snapshotSource(installDir string, local *LocalSource) (string, *exit.Error) {
	pack, problem := packagepublish.PrepareLocalFrom(local.Tree)
	if problem != nil {
		return "", problem
	}
	defer pack.Close()
	root := filepath.Join(installDir, "source")
	dependencies, problem := packagepublish.LocalDependencyPaths(pack.Tree)
	if problem != nil {
		return "", problem
	}
	relocated := map[string]string{pack.Tree: root}
	files := map[string]string{}
	for name, from := range pack.Files {
		files[filepath.Join(root, filepath.FromSlash(name))] = from
	}
	for name, from := range dependencies {
		to := filepath.Join(root, ".cozy-dependencies", name)
		inside := false
		if rel, err := filepath.Rel(pack.Tree, from); err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
			to = filepath.Join(root, rel)
			inside = true
		}
		info, err := os.Stat(from)
		if err != nil {
			return "", exit.Internalf("cannot inspect snapshot dependency: %s", err)
		}
		if !info.IsDir() {
			if !inside {
				to = filepath.Join(to, filepath.Base(from))
			}
			files[to], relocated[from] = from, to
			continue
		}
		relocated[from] = to
		_, members, problem := packagepublish.LibrarySourceTree(from)
		if problem != nil {
			return "", problem
		}
		for member, source := range members {
			files[filepath.Join(to, filepath.FromSlash(member))] = source
		}
	}
	for _, to := range packagepublish.Paths(files) {
		if err := os.MkdirAll(filepath.Dir(to), 0o700); err != nil {
			return "", exit.Internalf("cannot create invocation source snapshot: %s", err)
		}
		if problem := copySnapshotFile(files[to], to); problem != nil {
			return "", problem
		}
	}
	// Recheck every original source before allowing any build/import. Dependency
	// changes matter even when the root project's bytes did not move.
	after, _, _, problem := pack.SourceIdentity()
	if problem != nil {
		return "", problem
	}
	if after != local.SourceDigest {
		return "", exit.Named(exit.Conflict, "local_package_source_changed", "local source or a dependency changed during snapshot")
	}
	for from, to := range relocated {
		info, err := os.Stat(from)
		if err != nil {
			return "", exit.Internalf("cannot inspect source metadata: %s", err)
		}
		if !info.IsDir() {
			continue
		}
		for _, name := range []string{"pyproject.toml", "uv.lock"} {
			if problem := relocateMetadata(filepath.Join(to, name), from, to, relocated); problem != nil {
				return "", problem
			}
		}
	}
	frozen, problem := packagepublish.PrepareLocalFrom(root)
	if problem != nil {
		return "", problem
	}
	defer frozen.Close()
	digest, count, size, problem := frozen.SourceIdentity()
	if problem != nil {
		return "", problem
	}
	if "local/"+frozen.Name != local.Package || frozen.Release != local.Release {
		return "", exit.Named(exit.Conflict, "local_package_source_changed",
			"local package changed while capturing its invocation snapshot").
			WithRemedy("retry after the source has stopped changing")
	}
	local.SourceDigest, local.Tree, local.Files, local.Bytes = digest, root, count, size
	return root, nil
}

// Paths in uv source metadata are relative to the project owning the document.
// Rewrite only local-source fields; requirements/registry hashes remain frozen.
func relocateMetadata(path, oldRoot, newRoot string, relocated map[string]string) *exit.Error {
	raw, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return exit.Internalf("cannot read snapshot metadata: %s", err)
	}
	var document map[string]any
	if err := toml.Unmarshal(raw, &document); err != nil {
		return exit.Named(exit.Validation, "snapshot_metadata_invalid", "%s", err)
	}
	var visit func(any) *exit.Error
	visit = func(value any) *exit.Error {
		switch node := value.(type) {
		case map[string]any:
			for key, child := range node {
				if str, ok := child.(string); ok && (key == "path" || key == "editable" || key == "directory" || key == "virtual") {
					original := str
					if !filepath.IsAbs(original) {
						original = filepath.Join(oldRoot, original)
					}
					original = filepath.Clean(original)
					if target, ok := relocated[original]; ok {
						rel, err := filepath.Rel(newRoot, target)
						if err != nil {
							return exit.Internalf("cannot relocate source dependency: %s", err)
						}
						node[key] = filepath.ToSlash(rel)
					}
				}
				if problem := visit(child); problem != nil {
					return problem
				}
			}
		case []any:
			for _, child := range node {
				if problem := visit(child); problem != nil {
					return problem
				}
			}
		}
		return nil
	}
	if problem := visit(document); problem != nil {
		return problem
	}
	raw, err = toml.Marshal(document)
	if err != nil {
		return exit.Internalf("cannot encode snapshot metadata: %s", err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return exit.Internalf("cannot retain snapshot metadata: %s", err)
	}
	return nil
}

func copySnapshotFile(from, to string) *exit.Error {
	in, err := os.Open(from)
	if err != nil {
		return exit.Internalf("cannot read invocation source: %s", err)
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return exit.Internalf("cannot create invocation source: %s", err)
	}
	n, copyErr := io.Copy(out, io.LimitReader(in, packagepublish.MaxSourceFileBytes+1))
	syncErr := out.Sync()
	closeErr := out.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || n > packagepublish.MaxSourceFileBytes {
		return exit.Named(exit.Conflict, "local_package_source_changed",
			"cannot capture a bounded complete invocation source file")
	}
	return nil
}
