// Package packagepublish stages the current project tree for one package release.
package packagepublish

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/wheel"
	"github.com/pelletier/go-toml/v2"
)

const (
	MaxSourceBytes          int64 = 512 << 20
	MaxSourceFileBytes      int64 = 64 << 20
	MaxSourceFiles                = 20_000
	maxLockBytes            int64 = 16 << 20
	maxProjectMetadataBytes       = 1 << 20
)

var projectNameSeparator = regexp.MustCompile(`[-_.]+`)

// Package is the local input to one begin/upload/finalize operation. Tensorhub
// computes identities and package facts after the bytes arrive.
type Package struct {
	Files            map[string]string // source-relative path -> local path
	Descriptor       string
	Wheel            string
	DependencyWheels []DependencyWheel
	Registry         []RegistryRow     // locked registry rows; Tensorhub fetches (cl-078)
	Evidence         map[string]string // envelope-relative path -> local path
	Tree             string
	Root             string // disposable wheel output, empty until Build
	Name             string
	Release          string
}

type sourceIdentityFile struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
	Path   string `json:"path"`
}

type sourceIdentityDocument struct {
	Sources []sourceIdentityFile `json:"sources"`
}

func (p *Package) Close() { _ = os.RemoveAll(p.Root) }

// Prepare reads publication identity and source paths without executing the
// project's build backend. The caller can therefore ask Tensorhub whether the
// release is already committed before doing any wheel work.
func Prepare() (*Package, *exit.Error) {
	return PrepareFrom(".")
}

// PrepareFrom exists so the product suite can drive publication staging from
// an isolated project directory without changing the process working directory.
func PrepareFrom(projectDir string) (*Package, *exit.Error) {
	return prepareFrom(projectDir)
}

// PrepareLocalFrom applies the same bounded source rules to a local install.
func PrepareLocalFrom(projectDir string) (*Package, *exit.Error) {
	return prepareFrom(projectDir)
}

func prepareFrom(projectDir string) (*Package, *exit.Error) {
	tree, files, problem := sourceTree(projectDir)
	if problem != nil {
		return nil, problem
	}
	document, problem := readProjectDocument(files["pyproject.toml"])
	if problem != nil {
		return nil, problem
	}
	metadata, problem := document.projectIdentity()
	if problem != nil {
		return nil, problem
	}
	return &Package{
		Files: files, Tree: tree,
		Name: normalizedProjectName(metadata.Name), Release: metadata.Version,
	}, nil
}

// Build runs the standard PEP 517 backend and builds local dependency wheels.
// It is deliberately separate from Prepare so committed replays do no builds.
func (p *Package) Build(ctx context.Context) *exit.Error {
	if p.Root != "" || p.Wheel != "" {
		return exit.Internalf("package publication wheel staging was built more than once")
	}
	document, problem := readProjectDocument(p.Files["pyproject.toml"])
	if problem != nil {
		return problem
	}
	root, err := os.MkdirTemp("", "cozy-package-publish-")
	if err != nil {
		return exit.Internalf("cannot create package publication staging: %s", err)
	}
	p.Root = root
	project, problem := wheel.Build(wheel.Request{Context: ctx, Tree: p.Tree, OutDir: root})
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	fact, problem := wheel.InspectIdentity(project.Path)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	if p.Name != fact.Distribution || p.Release != fact.Version {
		p.Close()
		p.Root = ""
		return exit.Named(exit.Validation, "project_metadata_mismatch",
			"pyproject.toml declares %s==%s but the built wheel declares %s==%s",
			p.Name, p.Release, fact.Distribution, fact.Version).
			WithRemedy("fix the build backend so wheel identity comes from [project] name and version")
	}
	dependencies, needsRegistry, problem := collectLocalDependencies(ctx, p.Tree, document, root)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	registry := []RegistryRow{}
	if needsRegistry {
		registry, problem = collectRegistryRows(ctx, p.Tree, root, dependencies)
		if problem != nil {
			p.Close()
			p.Root = ""
			return problem
		}
	}
	descriptor, problem := describe(ctx, p.Tree, root)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	p.Wheel, p.Descriptor, p.DependencyWheels, p.Registry = project.Path, descriptor, dependencies, registry
	return nil
}

func describe(ctx context.Context, tree, root string) (string, *exit.Error) {
	cmd := exec.CommandContext(ctx, "uv", "run", "--locked", "--no-progress",
		"cozy-runtime", "--json", "--dir", tree, "describe")
	cmd.Dir = tree
	cmd.Env = config.Frozen().Tool()
	var stdout, stderr strings.Builder
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		return "", exit.Named(exit.Validation, "package_descriptor_refused",
			"cozy-runtime could not describe the package").WithRemedy("%s", strings.TrimSpace(stderr.String()))
	}
	raw := []byte(strings.TrimSpace(stdout.String()))
	if len(raw) == 0 || len(raw) > 1<<20 || !json.Valid(raw) {
		return "", exit.Named(exit.Structural, "package_descriptor_invalid",
			"cozy-runtime returned an invalid package descriptor")
	}
	path := filepath.Join(root, "descriptor.json")
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		return "", exit.Internalf("cannot stage package descriptor: %s", err)
	}
	return path, nil
}

// Paths returns the sorted source-relative paths of one prepared tree.
func Paths(files map[string]string) []string {
	out := make([]string, 0, len(files))
	for path := range files {
		out = append(out, path)
	}
	sort.Strings(out)
	return out
}

// SourceIdentity binds an editable install to the exact publishable source tree.
// It neither builds nor claims a wheel: editable execution uses this live tree,
// while published execution remains the separate wheel-backed path.
func (p *Package) SourceIdentity() (string, int, int64, *exit.Error) {
	document := sourceIdentityDocument{}
	var sourceBytes int64
	for _, path := range Paths(p.Files) {
		row, problem := sourceIdentityFileAt(path, p.Files[path])
		if problem != nil {
			return "", 0, 0, problem
		}
		document.Sources = append(document.Sources, row)
		sourceBytes += row.Length
	}
	raw, err := json.Marshal(document)
	if err != nil {
		return "", 0, 0, exit.Internalf("cannot encode local package source identity: %s", err)
	}
	digest := sha256.Sum256(raw)
	return "sha256:" + hex.EncodeToString(digest[:]), len(document.Sources), sourceBytes, nil
}

func sourceIdentityFileAt(name, file string) (sourceIdentityFile, *exit.Error) {
	input, err := os.Open(file)
	if err != nil {
		return sourceIdentityFile{}, exit.Named(exit.Structural, "local_package_source_changed",
			"cannot read %s while fixing the local build identity: %s", name, err).
			WithRemedy("stop changing the project while `cozy package install` is building it")
	}
	hash := sha256.New()
	length, copyErr := io.Copy(hash, input)
	closeErr := input.Close()
	if copyErr != nil || closeErr != nil {
		return sourceIdentityFile{}, exit.Named(exit.Structural, "local_package_source_changed",
			"cannot finish reading %s while fixing the local build identity", name).
			WithRemedy("stop changing the project while `cozy package install` is building it")
	}
	return sourceIdentityFile{Path: name, Length: length,
		Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil))}, nil
}

type projectMetadata struct {
	Project struct {
		Name                 string              `toml:"name"`
		Version              string              `toml:"version"`
		Dependencies         []string            `toml:"dependencies"`
		OptionalDependencies map[string][]string `toml:"optional-dependencies"`
	} `toml:"project"`
	Tool struct {
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

func (document projectMetadata) projectIdentity() (struct {
	Name, Version string
}, *exit.Error) {
	var out struct {
		Name, Version string
	}
	out.Name = strings.TrimSpace(document.Project.Name)
	out.Version = strings.TrimSpace(document.Project.Version)
	if out.Name == "" || out.Name != document.Project.Name || out.Version == "" ||
		out.Version != document.Project.Version {
		return out, exit.Named(exit.Validation, "project_identity_missing",
			"pyproject.toml [project] must declare one untrimmed name and version")
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
}

var refusedDir = map[string]bool{
	".aws": true, ".ssh": true, "credentials": true, "secrets": true,
}

var ignoredRootDir = map[string]bool{
	".venv": true, "venv": true, "node_modules": true,
	".idea": true, ".vscode": true, "build": true, "dist": true,
}

var ignoredFile = map[string]bool{
	".ds_store": true, "thumbs.db": true, ".gitignore": true, ".gitattributes": true,
	"package.descriptor.json":       true,
	"package.evaluated-config.json": true,
}

func sourceTree(tree string) (string, map[string]string, *exit.Error) {
	root, err := filepath.Abs(tree)
	if err != nil {
		return "", nil, exit.Usagef("package directory %q is not resolvable: %s", tree, err)
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
			if refusedDir[name] {
				return exit.Named(exit.Validation, "package_source_private_directory",
					"package source contains private directory %s", filepath.ToSlash(rel)).
					WithRemedy("remove credentials and secrets from the project tree before publication")
			}
			atRoot := !strings.Contains(filepath.ToSlash(rel), "/")
			if ignoredDir[name] || strings.HasSuffix(name, ".egg-info") ||
				(atRoot && ignoredRootDir[name]) {
				return fs.SkipDir
			}
			return nil
		}
		if ignoredFile[name] {
			return nil
		}
		rel, err := filepath.Rel(root, file)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		if refusedSourceFile(name) {
			return exit.Named(exit.Validation, "package_source_file_refused",
				"package source contains non-publishable file %s", rel).
				WithRemedy("remove credentials, generated bytecode, keys, and model weights from the package source tree")
		}
		if entry.Type()&fs.ModeSymlink != 0 {
			return exit.Named(exit.Validation, "package_source_entry_invalid", "%s is a symlink", rel)
		}
		info, err := entry.Info()
		if err != nil || !info.Mode().IsRegular() {
			return exit.Named(exit.Validation, "package_source_entry_invalid", "%s is not a regular file", rel)
		}
		limit := MaxSourceFileBytes
		if rel == "uv.lock" {
			limit = maxLockBytes
		}
		if info.Size() > limit {
			return exit.Named(exit.Validation, "package_source_file_too_large",
				"%s is %d B; package source files may be at most %d B", rel, info.Size(), limit)
		}
		if len(files) >= MaxSourceFiles {
			return exit.Named(exit.Validation, "package_source_file_count_exceeded",
				"package source has more than %d files", MaxSourceFiles)
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
		if info, err := os.Stat(files[required]); err != nil || info.Size() == 0 {
			return "", nil, exit.Named(exit.Validation, "package_source_required_file_empty",
				"package source requires a non-empty %s", required)
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
