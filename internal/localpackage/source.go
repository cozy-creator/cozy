// Package localpackage is unpublished code as its machine takes it: the files of a project where
// its author keeps it (and of the local path dependencies it names), each named by its
// content. Nothing is staged or copied here. This computer's machine reads the directory in
// place; another keeps one tree per package and is sent only what changed.
package localpackage

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// File is one source file, or one wheel its lock selects from a Tensorhub index that this
// home keeps.
type File struct {
	Name       string // its slash path under Root; a wheel's file name
	Digest     string // sha256:<hex>
	Length     int64
	Executable bool
	Path       string
}

// Installation is one unpublished install's code.
type Installation struct {
	ID, Package, Release string
	PythonRequires       string
	// Root is the directory holding the project and its local path dependencies; Project is
	// the project's slash path under it ("" when it is Root).
	Root, Project string
	Files, Wheels []File
	// Callees name the package each Tensorhub dependency is.
	Callees map[string]string
	// Indexes are the uv indexes the project's sources name without declaring, by name.
	Indexes map[string]string
	// Locals are the local path dependencies its lock installs, as slash paths under Root.
	Locals []string
}

// Open reads install's project as it is now. hubWheels is where this home keeps the Tensorhub
// wheels its lock selects; namespace answers the caller's account index for a project that
// names it before it has a lock.
func Open(install records.PackageInstall, hubWheels string, namespace packagepublish.NamespaceSource) (Installation, *exit.Error) {
	project := install.ProjectDir
	if install.SourceKind != "local" || project == "" {
		return Installation{}, exit.New(exit.NotFound, "install %s holds no package source to send", install.ID)
	}
	pack, problem := packagepublish.PrepareLocalFrom(project)
	if problem != nil {
		return Installation{}, problem
	}
	defer pack.Close()
	// Source is portable across compatible interpreters: the Python this computer built with
	// is not a requirement of the machine, the author's bounds are.
	out := Installation{ID: install.ID, Package: install.Package, Release: install.Version, PythonRequires: pack.PythonRequires}
	members := map[string]string{} // absolute path -> itself, project and dependencies alike
	for _, path := range pack.Files {
		members[path] = path
	}
	dependencies, problem := packagepublish.LocalDependencyPaths(pack.Tree)
	if problem != nil {
		return Installation{}, problem
	}
	roots := []string{pack.Tree}
	for _, dependency := range dependencies {
		info, err := os.Stat(dependency)
		if err != nil {
			return Installation{}, exit.New(exit.NotFound, "local dependency %s is unreadable: %s", dependency, err)
		}
		if !info.IsDir() {
			members[dependency], roots = dependency, append(roots, filepath.Dir(dependency))
			continue
		}
		_, files, problem := packagepublish.LibrarySourceTree(dependency)
		if problem != nil {
			return Installation{}, problem
		}
		for _, path := range files {
			members[path] = path
		}
		roots = append(roots, dependency)
	}
	out.Root = common(roots)
	if rel, _ := filepath.Rel(out.Root, pack.Tree); rel != "." {
		out.Project = filepath.ToSlash(rel)
	}
	for _, dependency := range dependencies {
		rel, _ := filepath.Rel(out.Root, dependency)
		out.Locals = append(out.Locals, filepath.ToSlash(rel))
	}
	sort.Strings(out.Locals)
	for path := range members {
		file, problem := read(out.Root, path)
		if problem != nil {
			return Installation{}, problem
		}
		out.Files = append(out.Files, file)
	}
	sort.Slice(out.Files, func(i, j int) bool { return out.Files[i].Name < out.Files[j].Name })
	lock, _ := os.ReadFile(filepath.Join(pack.Tree, "uv.lock"))
	wheels, callees, problem := packagepublish.HubWheels(lock)
	if problem != nil {
		return Installation{}, problem
	}
	out.Callees = callees
	for _, wheel := range wheels {
		path := packagepublish.KeptHubWheel(hubWheels, wheel)
		info, err := os.Stat(path)
		if err != nil {
			return Installation{}, exit.Named(exit.NotFound, "local_package_wheel_absent",
				"this computer keeps no %s, which %s's lock selects from a Tensorhub index", wheel.Filename, install.Package).
				WithRemedy("install the package again: `cozy package install %s`", project)
		}
		out.Wheels = append(out.Wheels, File{Name: wheel.Filename, Digest: "sha256:" + wheel.SHA256, Length: info.Size(), Path: path})
	}
	out.Indexes, problem = packagepublish.AccountIndexes(pack.Tree, namespace)
	return out, problem
}

// common is the deepest directory holding every one of dirs.
func common(dirs []string) string {
	root := filepath.Clean(dirs[0])
	for _, dir := range dirs[1:] {
		for dir = filepath.Clean(dir); root != dir && !strings.HasPrefix(dir, root+string(filepath.Separator)); {
			parent := filepath.Dir(root)
			if parent == root {
				break
			}
			root = parent
		}
	}
	return root
}

// A file's digest is read again only once its size or modification time moved.
var digests sync.Map

type stamp struct {
	size, modified int64
	digest         string
}

func read(root, path string) (File, *exit.Error) {
	info, err := os.Stat(path)
	if err != nil || !info.Mode().IsRegular() {
		return File{}, exit.New(exit.Validation, "%s is not a readable regular file", path)
	}
	name, _ := filepath.Rel(root, path)
	file := File{Name: filepath.ToSlash(name), Length: info.Size(), Executable: info.Mode()&0o111 != 0, Path: path}
	if held, ok := digests.Load(path); ok {
		if held := held.(stamp); held.size == info.Size() && held.modified == info.ModTime().UnixNano() {
			file.Digest = held.digest
			return file, nil
		}
	}
	opened, err := os.Open(path)
	if err != nil {
		return File{}, exit.New(exit.Validation, "%s is unreadable: %s", path, err)
	}
	defer opened.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, opened); err != nil {
		return File{}, exit.New(exit.Validation, "%s is unreadable: %s", path, err)
	}
	file.Digest = "sha256:" + hex.EncodeToString(hash.Sum(nil))
	digests.Store(path, stamp{info.Size(), info.ModTime().UnixNano(), file.Digest})
	return file, nil
}
