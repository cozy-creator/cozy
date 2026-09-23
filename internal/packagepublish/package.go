// Package packagepublish stages the current project tree for one package release.
package packagepublish

import (
	"archive/tar"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
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
	PythonVersion          string
	Files                  map[string]string // source-relative path -> local path
	PackageInterface       string
	Wheel                  string
	SourceArchive          string
	DependencyWheels       []DependencyWheel
	DependencyRequirements []byte
	Vendored               []VendoredDependency // auto-vendored local deps, for the publish nudge (th-113)
	Registry               []RegistryRow        // locked registry rows; Tensorhub fetches (cl-078)
	Tree                   string
	Root                   string // disposable wheel output, empty until Build
	Name                   string
	Release                string
	ScriptModels           map[string]string // plain-main default model refs; ordinary CLI overrides win
	temporarySource        string            // generated single-file project, copied into a retained install
}

type sourceIdentityFile struct {
	Digest string `json:"digest"`
	Length int64  `json:"length"`
	Path   string `json:"path"`
}

type sourceIdentityDocument struct {
	Sources []sourceIdentityFile `json:"sources"`
}

func (p *Package) Close() {
	_ = os.RemoveAll(p.Root)
	_ = os.RemoveAll(p.temporarySource)
}

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
	return p.build(ctx, false)
}

// BuildForPublish also rejects lock rows available only on the author's machine.
// Declared compatibility bounds are preserved in the package metadata.
func (p *Package) BuildForPublish(ctx context.Context) *exit.Error {
	return p.build(ctx, true)
}

func (p *Package) build(ctx context.Context, publish bool) *exit.Error {
	if p.Root != "" || p.Wheel != "" {
		return exit.Internalf("package publication wheel staging was built more than once")
	}
	document, problem := readProjectDocument(p.Files["pyproject.toml"])
	if problem != nil {
		return problem
	}
	if publish {
		canonicalTree, problem := canonicalLocalPath(p.Tree)
		if problem != nil {
			return problem
		}
		if problem := refuseAuthorLocalLockRows(canonicalTree, p.Files["uv.lock"]); problem != nil {
			return problem
		}
	}
	python, problem := hostruntime.ProjectPython(ctx, p.Tree)
	if problem != nil {
		return problem
	}
	p.PythonVersion = python.Version
	root, err := os.MkdirTemp("", "cozy-package-publish-")
	if err != nil {
		return exit.Internalf("cannot create package publication staging: %s", err)
	}
	p.Root = root
	// Declared-metadata refusals and dependency staging run before the project
	// wheel build, so a doomed publication is refused before the expensive work.
	dependencies, needsRegistry, vendored, problem := collectLocalDependencies(ctx, p.Tree, document, root, python.Executable)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	project, problem := projectWheel(ctx, p.Tree, root, p.Name, p.Release)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	registry := []RegistryRow{}
	if needsRegistry {
		organization := strings.TrimSpace(document.Tool.Cozy.Organization)
		registry, problem = collectRegistryRows(ctx, p.Tree, root, organization, dependencies, python)
		if problem != nil {
			p.Close()
			p.Root = ""
			return problem
		}
	}
	packageInterface, problem := describe(ctx, p.Tree, root, publish)
	if problem != nil {
		p.Close()
		p.Root = ""
		return problem
	}
	sourceArchive := ""
	if publish {
		sourceArchive, err = sourceArchiveFile(root, p.Name, p.Release, p.Files)
		if err != nil {
			p.Close()
			p.Root = ""
			return exit.Named(exit.Structural, "package_source_archive_failed", "cannot create deterministic source archive: %v", err)
		}
	}
	p.Wheel, p.SourceArchive, p.PackageInterface, p.DependencyWheels, p.Registry = project, sourceArchive, packageInterface, dependencies, registry
	p.Vendored = vendored
	return nil
}

// sourceArchiveFile writes one deterministic, regular-file-only source
// distribution. It is the public inspection artifact; package metadata remains
// uploaded separately so workers can consume exact bytes without unpacking it.
func sourceArchiveFile(root, name, release string, files map[string]string) (string, error) {
	path := filepath.Join(root, name+"-"+release+".tar.gz")
	out, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return "", err
	}
	gz := gzip.NewWriter(out)
	gz.Header.ModTime = time.Unix(0, 0)
	gz.Header.Name = ""
	tarWriter := tar.NewWriter(gz)
	for _, relative := range Paths(files) {
		input, err := os.Open(files[relative])
		if err != nil {
			_ = tarWriter.Close()
			_ = gz.Close()
			_ = out.Close()
			return "", err
		}
		info, err := input.Stat()
		if err != nil || !info.Mode().IsRegular() {
			_ = input.Close()
			_ = tarWriter.Close()
			_ = gz.Close()
			_ = out.Close()
			if err == nil {
				err = fmt.Errorf("%s is not a regular file", relative)
			}
			return "", err
		}
		header := &tar.Header{Name: filepath.ToSlash(relative), Mode: 0o644, Size: info.Size(), ModTime: time.Unix(0, 0), Typeflag: tar.TypeReg}
		if err := tarWriter.WriteHeader(header); err == nil {
			_, err = io.Copy(tarWriter, input)
		}
		_ = input.Close()
		if err != nil {
			_ = tarWriter.Close()
			_ = gz.Close()
			_ = out.Close()
			return "", err
		}
	}
	if err := tarWriter.Close(); err != nil {
		_ = gz.Close()
		_ = out.Close()
		return "", err
	}
	if err := gz.Close(); err != nil {
		_ = out.Close()
		return "", err
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	return path, nil
}

// projectWheel builds the tree's own wheel and refuses one a worker could not run:
// its identity must come from [project], and it must install at least one Python
// module or package. A backend left to guess a flat layout can emit a wheel holding
// nothing but .dist-info; that wheel would fail on a rented pod, so it fails here.
func projectWheel(ctx context.Context, tree, out, name, release string) (string, *exit.Error) {
	python, problem := hostruntime.ProjectPython(ctx, tree)
	if problem != nil {
		return "", problem
	}
	built, problem := wheel.Build(wheel.Request{Context: ctx, Tree: tree, OutDir: out, Python: python.Executable})
	if problem != nil {
		return "", problem
	}
	fact, problem := wheel.InspectIdentity(built.Path)
	if problem != nil {
		return "", problem
	}
	if name != fact.Distribution || release != fact.Version {
		return "", exit.Named(exit.Validation, "project_metadata_mismatch",
			"pyproject.toml declares %s==%s but the built wheel declares %s==%s",
			name, release, fact.Distribution, fact.Version).
			WithRemedy("fix the build backend so wheel identity comes from [project] name and version")
	}
	if problem := validateProjectWheelDependencies(built.Path); problem != nil {
		return "", problem
	}
	contents, problem := wheel.InspectContents(built.Path)
	if problem != nil {
		return "", problem
	}
	if len(contents.ImportRoots) == 0 {
		return "", exit.Named(exit.Validation, "project_wheel_no_import_roots",
			"the built wheel %s installs no Python module or package: %s",
			fact.Filename, contents.Describe()).
			WithRemedy("declare the project's modules or packages for its build backend in " +
				"pyproject.toml (hatchling: `[tool.hatch.build.targets.wheel] only-include = [...]`; " +
				"setuptools: `[tool.setuptools] py-modules = [...]`), then confirm `uv build --wheel` " +
				"lists them in the wheel's RECORD")
	}
	if problem := applicationEntrypoint(tree, fact.Filename, contents); problem != nil {
		return "", problem
	}
	return built.Path, nil
}

const applicationGroup = "cozy.application"

// applicationEntrypoint holds the two application spellings to one object. An
// editable run discovers the application from package.toml's [application]
// object; a worker discovers it from the installed wheel's one `cozy.application`
// entry point. A wheel that registers none, several, a different object, or an
// object whose module the wheel does not ship runs locally and fails on the pod.
func applicationEntrypoint(tree, filename string, contents wheel.Contents) *exit.Error {
	object, problem := applicationObject(filepath.Join(tree, "package.toml"))
	if problem != nil {
		return problem
	}
	remedy := fmt.Sprintf("declare `[project.entry-points.%q]` in pyproject.toml with one entry, "+
		"`default = %q`, matching package.toml's [application] object", applicationGroup, object)
	entries := contents.Group(applicationGroup)
	switch len(entries) {
	case 0:
		return exit.Named(exit.Validation, "project_wheel_application_entrypoint_missing",
			"the built wheel %s registers no %s entry point; a worker discovers the application "+
				"from the installed wheel, not from package.toml", filename, applicationGroup).
			WithRemedy("%s", remedy)
	case 1:
	default:
		spelled := make([]string, 0, len(entries))
		for _, entry := range entries {
			spelled = append(spelled, entry.Name+" = "+entry.Object)
		}
		return exit.Named(exit.Validation, "project_wheel_application_entrypoint_ambiguous",
			"the built wheel %s registers %d %s entry points (%s); exactly one is required",
			filename, len(entries), applicationGroup, strings.Join(spelled, ", ")).
			WithRemedy("%s", remedy)
	}
	entry := entries[0]
	if entry.Object != object {
		return exit.Named(exit.Validation, "project_wheel_application_entrypoint_mismatch",
			"the built wheel %s registers %s entry point %s = %s, but package.toml's [application] "+
				"object is %s; the two must name one object", filename, applicationGroup,
			entry.Name, entry.Object, object).
			WithRemedy("%s", remedy)
	}
	module, _, _ := strings.Cut(entry.Object, ":")
	root, _, _ := strings.Cut(strings.TrimSpace(module), ".")
	if !slices.Contains(contents.ImportRoots, root) {
		return exit.Named(exit.Validation, "project_wheel_application_module_absent",
			"the built wheel %s registers the application %s but installs no module %s "+
				"(import roots: %s)", filename, entry.Object, root, strings.Join(contents.ImportRoots, ", ")).
			WithRemedy("include the module that defines the application in the wheel: %s",
				"hatchling `[tool.hatch.build.targets.wheel] only-include = [...]`, "+
					"setuptools `[tool.setuptools] py-modules = [...]`")
	}
	return nil
}

// applicationObject is package.toml's [application] object, the spelling the
// editable path runs from. It must be a `module:attribute` reference.
func applicationObject(path string) (string, *exit.Error) {
	raw, err := os.ReadFile(path)
	if err != nil || len(raw) == 0 || len(raw) > maxProjectMetadataBytes {
		return "", exit.Named(exit.Validation, "package_config_unreadable",
			"package.toml must be a non-empty TOML file at or below %d bytes", maxProjectMetadataBytes)
	}
	var document struct {
		Application struct {
			Object string `toml:"object"`
		} `toml:"application"`
	}
	if err := toml.Unmarshal(raw, &document); err != nil {
		return "", exit.Named(exit.Validation, "package_config_invalid",
			"package.toml is not valid TOML: %v", err)
	}
	object := document.Application.Object
	module, attribute, ok := strings.Cut(object, ":")
	if !ok || strings.TrimSpace(object) != object || strings.TrimSpace(module) == "" ||
		strings.TrimSpace(attribute) == "" {
		return "", exit.Named(exit.Validation, "package_application_object_missing",
			"package.toml [application] must name one object as `module:attribute`").
			WithRemedy("set `[application] object = \"<module>:app\"` in package.toml")
	}
	return object, nil
}

// VerifyProjectWheel is the same fence for an editable install: the tree must build
// into a wheel a worker could run BEFORE it is pinned, so `cozy run --rental` never
// discovers on a paid pod what `cozy package install` could have said at once. The
// wheel is disposable; the local revision builds its own from the pinned source.
func VerifyProjectWheel(ctx context.Context, tree, name, release string) *exit.Error {
	out, err := os.MkdirTemp("", "cozy-project-wheel-")
	if err != nil {
		return exit.Internalf("cannot create project wheel staging: %s", err)
	}
	defer os.RemoveAll(out)
	_, problem := projectWheel(ctx, tree, out, name, release)
	return problem
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
func (p *Package) SourceIdentity(extras ...string) (string, int, int64, *exit.Error) {
	document := sourceIdentityDocument{}
	var sourceBytes int64
	files := make(map[string]string, len(p.Files))
	for name, path := range p.Files {
		files[name] = path
	}
	dependencies, problem := LocalDependencySelections(p.Tree, extras...)
	if problem != nil {
		return "", 0, 0, problem
	}
	for name, selection := range dependencies {
		source := selection.Path
		info, err := os.Stat(source)
		if err != nil {
			return "", 0, 0, exit.Internalf("cannot inspect local dependency: %s", err)
		}
		if !info.IsDir() {
			files["dependencies/"+name+"/"+filepath.Base(source)] = source
			continue
		}
		_, members, problem := LibrarySourceTree(source)
		if problem != nil {
			return "", 0, 0, problem
		}
		for member, path := range members {
			files["dependencies/"+name+"/"+member] = path
		}
	}
	for _, path := range Paths(files) {
		row, problem := sourceIdentityFileAt(path, files[path])
		if problem != nil {
			return "", 0, 0, problem
		}
		document.Sources = append(document.Sources, row)
		sourceBytes += row.Length
		if len(document.Sources) > MaxSourceFiles || sourceBytes > MaxSourceBytes {
			return "", 0, 0, exit.Named(exit.Validation, "local_source_closure_too_large", "local project and dependency sources exceed package bounds")
		}
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
		Name                 string                       `toml:"name"`
		Version              string                       `toml:"version"`
		Dependencies         []string                     `toml:"dependencies"`
		OptionalDependencies map[string][]string          `toml:"optional-dependencies"`
		EntryPoints          map[string]map[string]string `toml:"entry-points"`
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
	return document, validateProjectDependencyPolicy(document)
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
	"package-interface.json":        true,
	"package.evaluated-config.json": true,
}

// ignoredSourceFile is the file half of the source rules: the fixed names above plus an
// editor's own scratch (vim swap and backup, emacs lock and autosave), which is never source.
func ignoredSourceFile(name string) bool {
	return ignoredFile[name] || strings.HasSuffix(name, ".swp") || strings.HasSuffix(name, ".swo") ||
		strings.HasSuffix(name, ".swx") || strings.HasSuffix(name, "~") ||
		strings.HasPrefix(name, ".#") || (strings.HasPrefix(name, "#") && strings.HasSuffix(name, "#"))
}

// IgnoredSourcePath answers whether a source-relative path falls outside the tree the
// source rules read: an ignored directory on its way (`.egg-info`, `__pycache__`, a
// root-level `.venv`/`build`) or an ignored file at its end. The editable watcher asks
// this so what the walk would never hash never wakes it.
func IgnoredSourcePath(rel string) bool {
	parts := strings.Split(filepath.ToSlash(rel), "/")
	for index, part := range parts {
		name := strings.ToLower(part)
		if ignoredDir[name] || strings.HasSuffix(name, ".egg-info") || (index == 0 && ignoredRootDir[name]) {
			return true
		}
	}
	return ignoredSourceFile(strings.ToLower(parts[len(parts)-1]))
}

func sourceTree(tree string) (string, map[string]string, *exit.Error) {
	return boundedSourceTree(tree, []string{"package.toml", "pyproject.toml", "uv.lock"})
}

// LibrarySourceTree applies package source bounds to a normal Python library.
// A library needs no Cozy App or package.toml and may use its parent's uv.lock.
func LibrarySourceTree(tree string) (string, map[string]string, *exit.Error) {
	return boundedSourceTree(tree, []string{"pyproject.toml"})
}

func boundedSourceTree(tree string, requiredFiles []string) (string, map[string]string, *exit.Error) {
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
		if ignoredSourceFile(name) {
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
		limit := SourceFileLimit(rel)
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
	for _, required := range requiredFiles {
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

// SourceFileLimit is the admission and snapshot bound for one source member.
// Wheel dependencies retain their existing larger bound during invocation capture.
func SourceFileLimit(name string) int64 {
	if name == "uv.lock" {
		return maxLockBytes
	}
	if strings.HasSuffix(strings.ToLower(name), ".whl") {
		return MaxDependencyWheelBytes
	}
	return MaxSourceFileBytes
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
