// Package privatepackage freezes an editable checkout into one exact rented-worker revision.
// The directory is Creator-private staging, not a Tensorhub release or package cache API.
package privatepackage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const maxFiles = packagepublish.MaxDependencyWheels + 1

type File struct {
	Digest, Filename, Kind, Path string
	Length                       int64
}

type Revision struct {
	Package, Release, SourceDigest, Digest string
	Files                                  []File
}

func Stage(ctx context.Context, layout home.Layout, install records.PackageInstall) (Revision, *exit.Error) {
	if install.SourceKind != "local" || install.SourceRef == "" || install.SourceDigest == "" ||
		!strings.HasPrefix(install.Package, "local/") {
		return Revision{}, exit.Named(exit.Validation, "private_package_install_invalid",
			"install %s is not one editable local package", install.ID)
	}
	pack, problem := packagepublish.PrepareLocalFrom(install.SourceRef)
	if problem != nil {
		return Revision{}, problem
	}
	defer pack.Close()
	if "local/"+pack.Name != install.Package || pack.Release != install.Version {
		return Revision{}, exit.Named(exit.Conflict, "private_package_identity_changed",
			"editable source now names local/%s@%s, not installed %s@%s",
			pack.Name, pack.Release, install.Package, install.Version)
	}
	sourceDigest, _, _, problem := pack.SourceIdentity()
	if problem != nil {
		return Revision{}, problem
	}
	if sourceDigest != install.SourceDigest {
		return Revision{}, exit.Named(exit.Conflict, "private_package_source_changed",
			"editable source changed after its install revision was selected").
			WithRemedy("retry the command to select and build the new revision")
	}
	if problem := pack.Build(ctx); problem != nil {
		return Revision{}, problem
	}
	after, _, _, problem := pack.SourceIdentity()
	if problem != nil || after != sourceDigest {
		return Revision{}, exit.Named(exit.Conflict, "private_package_source_changed",
			"editable source changed while its private wheel revision was being built").
			WithRemedy("stop editing briefly and retry")
	}
	paths := []string{pack.Wheel}
	for _, dependency := range pack.DependencyWheels {
		paths = append(paths, dependency.Path)
	}
	stage, err := os.MkdirTemp(layout.PrivatePackages, ".stage-")
	if err != nil {
		return Revision{}, exit.Internalf("cannot create private package staging: %s", err)
	}
	keep := false
	defer func() {
		if !keep {
			_ = os.RemoveAll(stage)
		}
	}()
	wheelDir := filepath.Join(stage, "wheels")
	if err := os.Mkdir(wheelDir, 0o700); err != nil {
		return Revision{}, exit.Internalf("cannot create private wheel staging: %s", err)
	}
	files := make([]File, 0, len(paths))
	for index, source := range paths {
		kind := "dependency"
		if index == 0 {
			kind = "project"
		}
		file, problem := copyWheel(source, wheelDir, kind)
		if problem != nil {
			return Revision{}, problem
		}
		files = append(files, file)
	}
	revision, problem := identity(install.Package, install.Version, sourceDigest, files)
	if problem != nil {
		return Revision{}, problem
	}
	final := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(revision.Digest, "sha256:"))
	if err := os.Rename(stage, final); err != nil {
		if info, statErr := os.Stat(final); statErr != nil || !info.IsDir() {
			return Revision{}, exit.Internalf("cannot publish private package revision: %s", err)
		}
		existing, problem := Open(layout, install, revision.Digest)
		if problem != nil {
			return Revision{}, problem
		}
		return existing, nil
	}
	keep = true
	for index := range revision.Files {
		revision.Files[index].Path = filepath.Join(final, "wheels", revision.Files[index].Filename)
	}
	if err := os.Chmod(filepath.Join(final, "wheels"), 0o500); err != nil ||
		os.Chmod(final, 0o500) != nil {
		return Revision{}, exit.Internalf("cannot seal private package revision")
	}
	return revision, nil
}

func Open(layout home.Layout, install records.PackageInstall, digest string) (Revision, *exit.Error) {
	if len(digest) != 71 || !strings.HasPrefix(digest, "sha256:") {
		return Revision{}, exit.Named(exit.Validation, "private_package_digest_invalid",
			"private package revision digest is malformed")
	}
	root := filepath.Join(layout.PrivatePackages, strings.TrimPrefix(digest, "sha256:"))
	entries, err := os.ReadDir(root)
	if err != nil || len(entries) != 1 || entries[0].Name() != "wheels" || !entries[0].IsDir() {
		return Revision{}, exit.Named(exit.NotFound, "private_package_revision_absent",
			"private package revision %s is absent or incomplete", digest)
	}
	wheelDir := filepath.Join(root, "wheels")
	wheels, err := os.ReadDir(wheelDir)
	if err != nil || len(wheels) == 0 || len(wheels) > maxFiles {
		return Revision{}, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision has an invalid wheel count")
	}
	files := make([]File, 0, len(wheels))
	projects := 0
	for _, entry := range wheels {
		path := filepath.Join(wheelDir, entry.Name())
		if entry.Type()&os.ModeSymlink != 0 || !entry.Type().IsRegular() {
			return Revision{}, exit.Named(exit.Structural, "private_package_revision_invalid",
				"private package carrier %s is not a regular file", entry.Name())
		}
		fact, problem := wheel.InspectIdentity(path)
		if problem != nil {
			return Revision{}, problem
		}
		kind := "dependency"
		if fact.Distribution == strings.TrimPrefix(install.Package, "local/") &&
			fact.Version == install.Version {
			kind, projects = "project", projects+1
		}
		file, problem := measured(path, fact, kind)
		if problem != nil {
			return Revision{}, problem
		}
		files = append(files, file)
	}
	if projects != 1 {
		return Revision{}, exit.Named(exit.Structural, "private_package_project_wheel_count",
			"private package revision has %d project wheels", projects)
	}
	revision, problem := identity(install.Package, install.Version, install.SourceDigest, files)
	if problem != nil {
		return Revision{}, problem
	}
	if revision.Digest != digest {
		return Revision{}, exit.Named(exit.Conflict, "private_package_revision_changed",
			"private package revision bytes no longer match %s", digest)
	}
	return revision, nil
}

func copyWheel(source, destination, kind string) (File, *exit.Error) {
	fact, problem := wheel.InspectIdentity(source)
	if problem != nil {
		return File{}, problem
	}
	input, err := os.Open(source)
	if err != nil {
		return File{}, exit.Named(exit.Structural, "private_package_wheel_unreadable", "%s", err)
	}
	defer input.Close()
	outputPath := filepath.Join(destination, fact.Filename)
	output, err := os.OpenFile(outputPath, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o400)
	if err != nil {
		return File{}, exit.Internalf("cannot stage private package wheel: %s", err)
	}
	hash := sha256.New()
	written, copyErr := io.Copy(io.MultiWriter(output, hash), input)
	syncErr, closeErr := output.Sync(), output.Close()
	if copyErr != nil || syncErr != nil || closeErr != nil || written != fact.Length {
		return File{}, exit.Named(exit.Structural, "private_package_wheel_changed",
			"wheel %s changed while it was being staged", fact.Filename)
	}
	staged, problem := wheel.InspectIdentity(outputPath)
	if problem != nil || staged != fact {
		return File{}, exit.Named(exit.Structural, "private_package_wheel_changed",
			"wheel %s identity changed while it was being staged", fact.Filename)
	}
	return File{Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Filename: fact.Filename,
		Kind: kind, Length: written, Path: outputPath}, nil
}

func measured(path string, fact wheel.Identity, kind string) (File, *exit.Error) {
	input, err := os.Open(path)
	if err != nil {
		return File{}, exit.Named(exit.Structural, "private_package_wheel_unreadable", "%s", err)
	}
	defer input.Close()
	hash := sha256.New()
	length, err := io.Copy(hash, input)
	if err != nil || length != fact.Length {
		return File{}, exit.Named(exit.Structural, "private_package_wheel_changed", "%s", fact.Filename)
	}
	return File{Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil)), Filename: fact.Filename,
		Kind: kind, Length: length, Path: path}, nil
}

func identity(packageName, release, sourceDigest string, files []File) (Revision, *exit.Error) {
	rows := append([]File(nil), files...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Digest < rows[j].Digest })
	values := make([]canonical.Value, 0, len(rows))
	for _, file := range rows {
		values = append(values, map[string]canonical.Value{"digest": file.Digest,
			"filename": file.Filename, "kind": file.Kind, "length": file.Length})
	}
	raw, err := canonical.Write(map[string]canonical.Value{"package": packageName,
		"release": release, "source_digest": sourceDigest, "wheels": values})
	if err != nil {
		return Revision{}, exit.Internalf("cannot encode private package identity: %s", err)
	}
	digest, err := canonical.Spell(canonical.Digest(raw))
	if err != nil {
		return Revision{}, exit.Internalf("cannot digest private package identity: %s", err)
	}
	return Revision{Package: packageName, Release: release, SourceDigest: sourceDigest,
		Digest: digest, Files: rows}, nil
}
