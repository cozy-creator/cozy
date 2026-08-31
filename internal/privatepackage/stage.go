// Package privatepackage freezes an editable checkout into one exact rented-worker revision.
// The directory is Creator-private staging, not a Tensorhub release or package cache API.
package privatepackage

import (
	"bytes"
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
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

const (
	maxFiles              = pb.MaxPrivatePackageFiles
	privateDescriptorFile = "descriptor.json"
	privateRevisionFile   = "revision.json"
)

type File struct {
	Digest, Filename, Kind, Path string
	Length                       int64
}

type Revision struct {
	Package, Release, SourceDigest, Digest, DescriptorDigest string
	DescriptorLength                                         int64
	Files                                                    []File
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
	descriptorBytes, err := os.ReadFile(pack.Descriptor)
	if err != nil || len(descriptorBytes) == 0 || len(descriptorBytes) > canonical.DocMax {
		return Revision{}, exit.Named(exit.Structural, "private_package_descriptor_invalid",
			"private package descriptor is absent or exceeds the canonical document bound")
	}
	normalized, normalizeErr := canonical.NormalizeJCS(descriptorBytes)
	descriptorDigest, err := canonical.Spell(canonical.Digest(descriptorBytes))
	if normalizeErr != nil || err != nil || !bytes.Equal(normalized, descriptorBytes) {
		return Revision{}, exit.Named(exit.Structural, "private_package_descriptor_invalid",
			"private package descriptor bytes are not their canonical identity")
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
	if problem := copyDescriptor(descriptorBytes, filepath.Join(stage, privateDescriptorFile)); problem != nil {
		return Revision{}, problem
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
	revision, revisionBytes, problem := identity(install.Package, install.Version, sourceDigest,
		descriptorDigest, int64(len(descriptorBytes)), files)
	if problem != nil {
		return Revision{}, problem
	}
	if problem := copyDescriptor(revisionBytes,
		filepath.Join(stage, privateRevisionFile)); problem != nil {
		return Revision{}, problem
	}
	if err := syncDirectory(wheelDir); err != nil {
		return Revision{}, exit.Internalf("cannot sync private wheel staging: %s", err)
	}
	if err := syncDirectory(stage); err != nil {
		return Revision{}, exit.Internalf("cannot sync private package staging: %s", err)
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
	if err := syncDirectory(layout.PrivatePackages); err != nil {
		return Revision{}, exit.Internalf("cannot commit private package revision: %s", err)
	}
	keep = true
	for index := range revision.Files {
		revision.Files[index].Path = filepath.Join(final, "wheels", revision.Files[index].Filename)
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
	if err != nil || len(entries) != 3 || entries[0].Name() != privateDescriptorFile ||
		!entries[0].Type().IsRegular() || entries[1].Name() != privateRevisionFile ||
		!entries[1].Type().IsRegular() || entries[2].Name() != "wheels" || !entries[2].IsDir() {
		return Revision{}, exit.Named(exit.NotFound, "private_package_revision_absent",
			"private package revision %s is absent or incomplete", digest)
	}
	descriptorPath := filepath.Join(root, privateDescriptorFile)
	descriptorBytes, err := os.ReadFile(descriptorPath)
	if err != nil || len(descriptorBytes) == 0 || len(descriptorBytes) > canonical.DocMax {
		return Revision{}, exit.Named(exit.Structural, "private_package_descriptor_invalid",
			"private package descriptor is absent or exceeds the canonical document bound")
	}
	normalized, normalizeErr := canonical.NormalizeJCS(descriptorBytes)
	descriptorDigest, err := canonical.Spell(canonical.Digest(descriptorBytes))
	if normalizeErr != nil || err != nil || !bytes.Equal(normalized, descriptorBytes) {
		return Revision{}, exit.Named(exit.Structural, "private_package_descriptor_invalid",
			"private package descriptor bytes changed")
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
	revision, revisionBytes, problem := identity(install.Package, install.Version, install.SourceDigest,
		descriptorDigest, int64(len(descriptorBytes)), files)
	if problem != nil {
		return Revision{}, problem
	}
	if revision.Digest != digest {
		return Revision{}, exit.Named(exit.Conflict, "private_package_revision_changed",
			"private package revision bytes no longer match %s", digest)
	}
	storedRevision, err := os.ReadFile(filepath.Join(root, privateRevisionFile))
	if err != nil || !bytes.Equal(storedRevision, revisionBytes) {
		return Revision{}, exit.Named(exit.Conflict, "private_package_revision_changed",
			"private package revision document no longer matches %s", digest)
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

func copyDescriptor(data []byte, destination string) *exit.Error {
	output, err := os.OpenFile(destination, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o400)
	if err != nil {
		return exit.Internalf("cannot stage private package descriptor: %s", err)
	}
	written, writeErr := output.Write(data)
	syncErr, closeErr := output.Sync(), output.Close()
	if writeErr != nil || syncErr != nil || closeErr != nil || written != len(data) {
		return exit.Named(exit.Structural, "private_package_descriptor_changed",
			"private package descriptor changed while it was being staged")
	}
	return nil
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

func identity(packageName, release, sourceDigest, descriptorDigest string,
	descriptorLength int64, files []File,
) (Revision, []byte, *exit.Error) {
	rows := append([]File(nil), files...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Digest < rows[j].Digest })
	if len(rows) == 0 || len(rows) > maxFiles {
		return Revision{}, nil, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision has an invalid wheel count")
	}
	refs := make([]*pb.PrivatePackageFileRef, 0, len(rows))
	for index, file := range rows {
		digest, err := canonical.Raw(file.Digest)
		if err != nil || file.Length <= 0 || index > 0 && rows[index-1].Digest == file.Digest {
			return Revision{}, nil, exit.Named(exit.Structural, "private_package_file_invalid",
				"private package file %s has an invalid or duplicate identity", file.Filename)
		}
		kind := pb.LocalDownloadKind_LOCAL_DOWNLOAD_KIND_DEPENDENCY_WHEEL
		if file.Kind == "project" {
			kind = pb.LocalDownloadKind_LOCAL_DOWNLOAD_KIND_PROJECT_WHEEL
		} else if file.Kind != "dependency" {
			return Revision{}, nil, exit.Named(exit.Structural, "private_package_file_invalid",
				"private package file %s has an invalid identity", file.Filename)
		}
		refs = append(refs, &pb.PrivatePackageFileRef{Digest: digest, Filename: file.Filename,
			Kind: kind, Length: uint64(file.Length)})
	}
	source, sourceErr := canonical.Raw(sourceDigest)
	descriptor, descriptorErr := canonical.Raw(descriptorDigest)
	if sourceErr != nil || descriptorErr != nil || descriptorLength <= 0 {
		return Revision{}, nil, exit.Named(exit.Structural, "private_package_revision_invalid",
			"private package revision has invalid source or descriptor identity")
	}
	revisionBytes, rawDigest, err := canonical.Identity(&pb.PrivatePackageRevision{Package: packageName,
		Release: release, SourceDigest: source, PackageDescriptor: &pb.Ref{Digest: descriptor,
			Length: uint64(descriptorLength)}, Files: refs})
	if err != nil {
		return Revision{}, nil, exit.Internalf("cannot digest private package identity: %s", err)
	}
	digest, err := canonical.Spell(rawDigest)
	if err != nil {
		return Revision{}, nil, exit.Internalf("cannot spell private package identity: %s", err)
	}
	return Revision{Package: packageName, Release: release, SourceDigest: sourceDigest,
		Digest: digest, DescriptorDigest: descriptorDigest, DescriptorLength: descriptorLength,
		Files: rows}, revisionBytes, nil
}

func syncDirectory(path string) error {
	directory, err := os.Open(path)
	if err != nil {
		return err
	}
	defer directory.Close()
	return directory.Sync()
}
