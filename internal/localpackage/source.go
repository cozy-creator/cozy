// Package localpackage is unpublished code as its machine takes it: the files of an install's
// own source root, each named by its content. Nothing is staged or copied for it here. The
// machine keeps one tree of them per package and is sent only what changed.
package localpackage

import (
	"cmp"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/pelletier/go-toml/v2"
)

// File is one file of the source root, or one wheel its lock selects from a Tensorhub index,
// which the install keeps beside its source.
type File struct {
	Name       string // its slash path under the root; a wheel's file name
	Digest     string // sha256:<hex>
	Length     int64
	Executable bool
	Path       string
}

// Installation is one unpublished install's code.
type Installation struct {
	ID, Package, Release string
	PythonRequires       string
	Files, Wheels        []File
	// Callees name the package each Tensorhub dependency is.
	Callees map[string]string
}

// An install's source root never changes, so it is read once per process.
var opened sync.Map

// Open reads install's source root. An install without one (a dependency captured as a wheel)
// is no root a machine runs.
func Open(install records.PackageInstall) (Installation, *exit.Error) {
	if held, ok := opened.Load(install.ID); ok {
		return held.(Installation), nil
	}
	root := install.ProjectDir
	raw, err := os.ReadFile(filepath.Join(root, "pyproject.toml"))
	if install.SourceKind != "local" || root == "" || err != nil {
		return Installation{}, exit.New(exit.NotFound, "install %s holds no package source to send", install.ID)
	}
	var project struct {
		Project struct {
			RequiresPython string `toml:"requires-python"`
		} `toml:"project"`
	}
	if err := toml.Unmarshal(raw, &project); err != nil {
		return Installation{}, exit.Named(exit.Validation, "project_metadata_invalid", "pyproject.toml is not valid TOML: %v", err)
	}
	// Source is portable across compatible interpreters: the Python this computer built with
	// is not a requirement of the machine, the author's bounds are.
	out := Installation{ID: install.ID, Package: install.Package, Release: install.Version,
		PythonRequires: project.Project.RequiresPython}
	hubWheels := filepath.Join(root, filepath.FromSlash(packagepublish.HubWheelDir))
	err = filepath.WalkDir(root, func(path string, entry fs.DirEntry, walkErr error) error {
		if walkErr != nil || entry.IsDir() && path == hubWheels {
			return cmp.Or(walkErr, filepath.SkipDir) // its wheels are sent as Wheels
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		if !info.Mode().IsRegular() || len(out.Files) >= packagepublish.MaxSourceFiles {
			return fs.ErrInvalid
		}
		name, err := filepath.Rel(root, path)
		if err != nil {
			return err
		}
		digest, err := digestOf(path)
		out.Files = append(out.Files, File{Name: filepath.ToSlash(name), Digest: digest, Length: info.Size(),
			Executable: info.Mode()&0o111 != 0, Path: path})
		return err
	})
	if err != nil {
		return Installation{}, exit.New(exit.Validation, "install %s's source is not a bounded tree of regular files: %s", install.ID, err)
	}
	sort.Slice(out.Files, func(i, j int) bool { return out.Files[i].Name < out.Files[j].Name })
	lock, _ := os.ReadFile(filepath.Join(root, "uv.lock"))
	wheels, callees, problem := packagepublish.HubWheels(lock)
	if problem != nil {
		return Installation{}, problem
	}
	out.Callees = callees
	for _, wheel := range wheels {
		path := filepath.Join(root, filepath.FromSlash(packagepublish.HubWheelPath(wheel)))
		info, err := os.Stat(path)
		if err != nil {
			return Installation{}, exit.Named(exit.NotFound, "local_package_wheel_absent",
				"install %s does not hold %s, which its lock selects from a Tensorhub index", install.ID, wheel.Filename).
				WithRemedy("install the package again: `cozy package install <its directory>`")
		}
		out.Wheels = append(out.Wheels, File{Name: wheel.Filename, Digest: "sha256:" + wheel.SHA256, Length: info.Size(), Path: path})
	}
	opened.Store(install.ID, out)
	return out, nil
}

func digestOf(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	hash := sha256.New()
	if _, err := io.Copy(hash, file); err != nil {
		return "", err
	}
	return "sha256:" + hex.EncodeToString(hash.Sum(nil)), nil
}
