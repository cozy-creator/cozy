// Package packagepublish stages the current project tree for one package release.
package packagepublish

import (
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy-creator/internal/exit"
	"github.com/cozy-creator/cozy-creator/internal/wheel"
)

const MaxSourceBytes int64 = 512 << 20

// Package is the local input to one begin/upload/finalize operation. Tensorhub
// computes identities and package facts after the bytes arrive.
type Package struct {
	Files map[string]string // source-relative path -> local path
	Wheel string
	Root  string // disposable wheel output
}

func (p *Package) Close() { _ = os.RemoveAll(p.Root) }

type Request struct {
	Tree    string
	Release string
}

// Prepare reads the current working tree and builds its wheel through the
// project's standard PEP 517 backend. Git and commit state are irrelevant.
func Prepare(req Request) (*Package, *exit.Error) {
	release := strings.TrimSpace(req.Release)
	if release == "" || strings.ContainsAny(release, `/\\`) || release == "." || release == ".." {
		return nil, exit.Usagef("--release needs one safe immutable release id")
	}
	tree, files, problem := sourceTree(req.Tree)
	if problem != nil {
		return nil, problem
	}
	root, err := os.MkdirTemp("", "cozy-package-publish-")
	if err != nil {
		return nil, exit.Internalf("cannot create package publication staging: %s", err)
	}
	project, problem := wheel.Build(wheel.Request{Tree: tree, OutDir: root})
	if problem != nil {
		_ = os.RemoveAll(root)
		return nil, problem
	}
	return &Package{Files: files, Wheel: project.Path, Root: root}, nil
}

var ignoredDir = map[string]bool{
	".git": true, ".hg": true, ".svn": true, ".jj": true,
	"__pycache__": true, ".mypy_cache": true, ".ruff_cache": true,
	".pytest_cache": true, ".tox": true,
	".aws": true, ".ssh": true, "credentials": true, "secrets": true,
}

var ignoredRootDir = map[string]bool{
	".venv": true, "venv": true, "node_modules": true,
	".idea": true, ".vscode": true, "build": true, "dist": true,
}

var ignoredFile = map[string]bool{
	".ds_store": true, "thumbs.db": true, ".gitignore": true, ".gitattributes": true,
	"package.descriptor.json": true, "package.release.json": true,
	"package.evaluated-config.json": true,
}

func sourceTree(tree string) (string, map[string]string, *exit.Error) {
	root, err := filepath.Abs(tree)
	if err != nil {
		return "", nil, exit.Usagef("--dir %q is not resolvable: %s", tree, err)
	}
	info, err := os.Stat(root)
	if err != nil || !info.IsDir() {
		return "", nil, exit.Named(exit.NotFound, "package_tree_absent", "%s is not a directory", root)
	}
	files := map[string]string{}
	var total int64
	err = filepath.WalkDir(root, func(file string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if file == root {
			return nil
		}
		name := strings.ToLower(entry.Name())
		if entry.IsDir() {
			rel, err := filepath.Rel(root, file)
			if err != nil {
				return err
			}
			atRoot := !strings.Contains(filepath.ToSlash(rel), "/")
			if ignoredDir[name] || strings.HasSuffix(name, ".egg-info") ||
				(atRoot && ignoredRootDir[name]) {
				return fs.SkipDir
			}
			return nil
		}
		if ignoredFile[name] || credentialFile(name) || strings.HasSuffix(name, ".pyc") ||
			strings.HasSuffix(name, ".pyo") {
			return nil
		}
		rel, err := filepath.Rel(root, file)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if entry.Type()&fs.ModeSymlink != 0 {
			return exit.Named(exit.Validation, "package_source_entry_invalid", "%s is a symlink", rel)
		}
		info, err := entry.Info()
		if err != nil || !info.Mode().IsRegular() {
			return exit.Named(exit.Validation, "package_source_entry_invalid", "%s is not a regular file", rel)
		}
		total += info.Size()
		if total > MaxSourceBytes {
			return exit.Named(exit.Validation, "package_source_too_large",
				"package source exceeds %d B", MaxSourceBytes)
		}
		files[rel] = file
		return nil
	})
	if err != nil {
		if problem := exit.As(err); problem != nil {
			return "", nil, problem
		}
		return "", nil, exit.Named(exit.Structural, "package_tree_unreadable", "%s: %v", root, err)
	}
	for _, required := range []string{"package.toml", "pyproject.toml", "uv.lock"} {
		if files[required] == "" {
			return "", nil, exit.Named(exit.Validation, "package_source_required_file_missing",
				"package source has no %s", required)
		}
	}
	return root, files, nil
}

func credentialFile(name string) bool {
	return name == ".env" || strings.HasPrefix(name, ".env.") || name == ".netrc" ||
		name == ".npmrc" || name == ".pypirc" || strings.HasPrefix(name, "id_rsa") ||
		strings.HasSuffix(name, ".pem") || strings.HasSuffix(name, ".key")
}

func Paths(files map[string]string) []string {
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}
