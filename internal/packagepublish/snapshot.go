package packagepublish

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/pelletier/go-toml/v2"
)

// SnapshotSource copies a bounded project and its local dependencies, relocating uv paths.
// The destination belongs to one invocation; no source fingerprint is calculated.
func SnapshotSource(tree, root string) (*Package, *exit.Error) {
	pack, problem := PrepareLocalFrom(tree)
	if problem != nil {
		return nil, problem
	}
	defer pack.Close()
	dependencies, problem := LocalDependencyPaths(pack.Tree)
	if problem != nil {
		return nil, problem
	}
	relocated := map[string]string{pack.Tree: root}
	files := map[string]string{}
	limits := map[string]int64{}
	for name, from := range pack.Files {
		to := filepath.Join(root, filepath.FromSlash(name))
		files[to] = from
		limits[to] = SourceFileLimit(name)
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
			return nil, exit.Internalf("cannot inspect snapshot dependency: %s", err)
		}
		if !info.IsDir() {
			if !inside {
				to = filepath.Join(to, filepath.Base(from))
			}
			files[to], relocated[from] = from, to
			limits[to] = SourceFileLimit(filepath.Base(from))
			continue
		}
		relocated[from] = to
		_, members, problem := LibrarySourceTree(from)
		if problem != nil {
			return nil, problem
		}
		for member, source := range members {
			destination := filepath.Join(to, filepath.FromSlash(member))
			files[destination] = source
			limits[destination] = SourceFileLimit(member)
		}
	}
	for _, to := range Paths(files) {
		if err := os.MkdirAll(filepath.Dir(to), 0o700); err != nil {
			return nil, exit.Internalf("cannot create invocation source snapshot: %s", err)
		}
		if problem := copySnapshotFile(files[to], to, limits[to]); problem != nil {
			return nil, problem
		}
	}
	for from, to := range relocated {
		info, err := os.Stat(from)
		if err != nil {
			return nil, exit.Internalf("cannot inspect source metadata: %s", err)
		}
		if !info.IsDir() {
			continue
		}
		for _, name := range []string{"pyproject.toml", "uv.lock"} {
			if problem := relocateMetadata(filepath.Join(to, name), from, to, relocated); problem != nil {
				return nil, problem
			}
		}
	}
	frozen, problem := PrepareLocalFrom(root)
	if problem != nil {
		return nil, problem
	}
	return frozen, nil
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

func copySnapshotFile(from, to string, limit int64) *exit.Error {
	in, err := os.Open(from)
	if err != nil {
		return exit.Internalf("cannot read invocation source: %s", err)
	}
	defer in.Close()
	out, err := os.OpenFile(to, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return exit.Internalf("cannot create invocation source: %s", err)
	}
	n, copyErr := io.Copy(out, io.LimitReader(in, limit+1))
	syncErr := out.Sync()
	closeErr := out.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || n > limit {
		return exit.Named(exit.Conflict, "local_package_source_changed",
			"cannot capture a bounded complete invocation source file")
	}
	return nil
}
