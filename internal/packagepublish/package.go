// Package packagepublish stages the current project tree for one package release.
package packagepublish

import (
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

const MaxSourceBytes int64 = 512 << 20
const maxProjectMetadataBytes = 1 << 20

var projectNameSeparator = regexp.MustCompile(`[-_.]+`)

// Package is the local input to one begin/upload/finalize operation. Tensorhub
// computes identities and package facts after the bytes arrive.
type Package struct {
	Files            map[string]string // source-relative path -> local path
	Wheel            string
	DependencyWheels []DependencyWheel
	Root             string // disposable wheel output
	Organization     string
	Name             string
	Release          string
}

func (p *Package) Close() { _ = os.RemoveAll(p.Root) }

// Prepare reads the current working tree and builds its wheel through the
// project's standard PEP 517 backend. Git and commit state are irrelevant.
func Prepare() (*Package, *exit.Error) {
	return PrepareFrom(".")
}

// PrepareFrom exists so the product suite can drive publication staging from
// an isolated project directory without changing the process working directory.
func PrepareFrom(projectDir string) (*Package, *exit.Error) {
	tree, files, problem := sourceTree(projectDir)
	if problem != nil {
		return nil, problem
	}
	document, problem := readProjectDocument(files["pyproject.toml"])
	if problem != nil {
		return nil, problem
	}
	metadata, problem := document.publicationMetadata()
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
	fact, problem := wheel.Inspect(project.Path)
	if problem != nil {
		_ = os.RemoveAll(root)
		return nil, problem
	}
	if normalizedProjectName(metadata.Name) != fact.Distribution || metadata.Version != fact.Version {
		_ = os.RemoveAll(root)
		return nil, exit.Named(exit.Validation, "project_metadata_mismatch",
			"pyproject.toml declares %s==%s but the built wheel declares %s==%s",
			metadata.Name, metadata.Version, fact.Distribution, fact.Version).
			WithRemedy("fix the build backend so wheel identity comes from [project] name and version")
	}
	dependencies, problem := collectLocalDependencies(tree, document, root)
	if problem != nil {
		_ = os.RemoveAll(root)
		return nil, problem
	}
	return &Package{
		Files: files, Wheel: project.Path, DependencyWheels: dependencies, Root: root,
		Organization: metadata.Organization, Name: fact.Distribution, Release: fact.Version,
	}, nil
}

type projectMetadata struct {
	Project struct {
		Name         string   `toml:"name"`
		Version      string   `toml:"version"`
		Dependencies []string `toml:"dependencies"`
	} `toml:"project"`
	Tool struct {
		Cozy struct {
			Organization string `toml:"organization"`
		} `toml:"cozy"`
		UV struct {
			Sources   map[string]any `toml:"sources"`
			Workspace struct {
				Members []string `toml:"members"`
				Exclude []string `toml:"exclude"`
			} `toml:"workspace"`
		} `toml:"uv"`
	} `toml:"tool"`
}

func readProjectDocument(path string) (projectMetadata, *exit.Error) {
	var document projectMetadata
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) == 0 || len(raw) > maxProjectMetadataBytes {
		return document, exit.Named(exit.Validation, "project_metadata_unreadable",
			"pyproject.toml must be a non-empty TOML file at or below %d bytes", maxProjectMetadataBytes)
	}
	if err := toml.Unmarshal(raw, &document); err != nil {
		return document, exit.Named(exit.Validation, "project_metadata_invalid",
			"pyproject.toml is not valid TOML: %v", err)
	}
	return document, nil
}

func (document projectMetadata) publicationMetadata() (struct {
	Name, Version, Organization string
}, *exit.Error) {
	var out struct {
		Name, Version, Organization string
	}
	out.Name = strings.TrimSpace(document.Project.Name)
	out.Version = strings.TrimSpace(document.Project.Version)
	out.Organization = strings.TrimSpace(document.Tool.Cozy.Organization)
	if out.Name == "" || out.Name != document.Project.Name || out.Version == "" ||
		out.Version != document.Project.Version {
		return out, exit.Named(exit.Validation, "project_identity_missing",
			"pyproject.toml [project] must declare one untrimmed name and version")
	}
	if out.Organization == "" || out.Organization != document.Tool.Cozy.Organization {
		return out, exit.Named(exit.Validation, "project_organization_missing",
			"pyproject.toml must declare [tool.cozy] organization").
			WithRemedy("add `[tool.cozy]` and `organization = \"your-org\"`")
	}
	return out, nil
}

func normalizedProjectName(value string) string {
	return projectNameSeparator.ReplaceAllString(strings.ToLower(value), "-")
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
		if ignoredFile[name] || refusedSourceFile(name) {
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

func refusedSourceFile(name string) bool {
	if name == ".env" || strings.HasPrefix(name, ".env.") || name == ".netrc" ||
		name == ".npmrc" || name == ".pypirc" || name == "credentials.json" ||
		name == "service-account.json" || name == "id_ed25519" || strings.HasPrefix(name, "id_rsa") {
		return true
	}
	for _, suffix := range []string{".pyc", ".pyo", ".pem", ".key", ".p12", ".pfx", ".jks",
		".pickle", ".pkl", ".pt", ".ckpt", ".safetensors", ".onnx", ".gguf"} {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

func Paths(files map[string]string) []string {
	paths := make([]string, 0, len(files))
	for path := range files {
		paths = append(paths, path)
	}
	sort.Strings(paths)
	return paths
}

func WheelFilenames(wheels []DependencyWheel) []string {
	if len(wheels) == 0 {
		return nil
	}
	names := make([]string, 0, len(wheels))
	for _, wheel := range wheels {
		names = append(names, wheel.Filename)
	}
	sort.Strings(names)
	return names
}
