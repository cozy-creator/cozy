// Package localpackage transports invocation-owned private installations.
package localpackage

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/wheel"
)

const installationFile = "installation.json"

type File struct {
	Digest, Filename, Kind, Path string
	Length                       int64
}

// ID belongs to the ordinary install record. It is never derived from content.
type Installation struct {
	ID, Package, Release          string
	PackageInterface              []byte
	SourceArchive                 string
	Files                         []File
	DependencyRequirements        []byte
	PythonRequires, PythonVersion string
}

func Stage(ctx context.Context, layout home.Layout, install records.PackageInstall) (Installation, *exit.Error) {
	if !validInstallationID(install.ID) {
		return Installation{}, exit.New(exit.Validation, "source sync requires an install ID")
	}
	if _, err := os.Stat(filepath.Join(layout.LocalPackages, install.ID, installationFile)); err == nil {
		return Open(layout, install, install.ID)
	}
	if install.SourceKind != "local" || install.SourceRef == "" {
		return Installation{}, exit.New(exit.NotFound, "private installation has not been staged")
	}
	if ctx.Err() != nil {
		return Installation{}, exit.New(exit.Failed, "source sync canceled")
	}
	stage, problem := newStage(layout)
	if problem != nil {
		return Installation{}, problem
	}
	defer os.RemoveAll(stage)
	pack, problem := packagepublish.SnapshotSource(install.SourceRef, filepath.Join(stage, "source"))
	if problem != nil {
		return Installation{}, problem
	}
	defer pack.Close()
	if "local/"+pack.Name != install.Package || pack.Release != install.Version {
		return Installation{}, exit.New(exit.Conflict, "source declares a different package name or version")
	}
	length, problem := packagepublish.WriteSourceArchive(pack.Tree, filepath.Join(stage, "source.tar"))
	if problem != nil {
		return Installation{}, problem
	}
	surface, err := os.ReadFile(filepath.Join(install.Dir, "documents", "package-interface.json"))
	if err != nil {
		return Installation{}, exit.Internalf("cannot read installed interface: %s", err)
	}
	result := Installation{ID: install.ID, Package: install.Package, Release: install.Version, PackageInterface: surface, SourceArchive: "source.tar", PythonVersion: install.Python,
		Files: []File{{Filename: "source.tar", Kind: "source", Length: length}}}
	return retain(layout, stage, result)
}

// StageWheels retains ordinary published/builtin wheels. Editable projects use Stage.
func StageWheels(layout home.Layout, install records.PackageInstall, surface []byte, paths []string, requirements []byte) (Installation, *exit.Error) {
	if !validInstallationID(install.ID) || len(paths) == 0 || len(paths) > 256 {
		return Installation{}, exit.New(exit.Validation, "wheel installation inputs are incomplete")
	}
	stage, problem := newStage(layout)
	if problem != nil {
		return Installation{}, problem
	}
	defer os.RemoveAll(stage)
	result := Installation{ID: install.ID, Package: install.Package, Release: install.Version, PackageInterface: append([]byte(nil), surface...), DependencyRequirements: append([]byte(nil), requirements...), PythonVersion: install.Python}
	for index, source := range paths {
		fact, problem := wheel.InspectIdentity(source)
		if problem != nil {
			return Installation{}, problem
		}
		input, err := os.Open(source)
		if err != nil {
			return Installation{}, exit.Internalf("cannot read dependency wheel: %s", err)
		}
		output, err := os.OpenFile(filepath.Join(stage, fact.Filename), os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0600)
		if err != nil {
			input.Close()
			return Installation{}, exit.Internalf("cannot stage dependency wheel: %s", err)
		}
		hash := sha256.New()
		length, copyErr := io.Copy(io.MultiWriter(output, hash), io.LimitReader(input, (512<<20)+1))
		input.Close()
		closeErr := output.Close()
		if copyErr != nil || closeErr != nil || length != fact.Length || length > 512<<20 {
			return Installation{}, exit.New(exit.Validation, "dependency wheel exceeds its declared size")
		}
		kind := "dependency"
		if index == 0 {
			kind = "project"
		}
		result.Files = append(result.Files, File{Filename: fact.Filename, Kind: kind, Length: length, Digest: "sha256:" + hex.EncodeToString(hash.Sum(nil))})
	}
	return retain(layout, stage, result)
}

func newStage(layout home.Layout) (string, *exit.Error) {
	if err := os.MkdirAll(layout.LocalPackages, 0700); err != nil {
		return "", exit.Internalf("cannot create source sync staging: %s", err)
	}
	stage, err := os.MkdirTemp(layout.LocalPackages, ".stage-")
	if err != nil {
		return "", exit.Internalf("cannot stage source sync: %s", err)
	}
	return stage, nil
}

func retain(layout home.Layout, stage string, result Installation) (Installation, *exit.Error) {
	sort.Slice(result.Files, func(i, j int) bool { return result.Files[i].Filename < result.Files[j].Filename })
	data, err := json.Marshal(result)
	if err != nil {
		return Installation{}, exit.Internalf("cannot encode source sync record: %s", err)
	}
	if err := os.WriteFile(filepath.Join(stage, installationFile), data, 0600); err != nil {
		return Installation{}, exit.Internalf("cannot retain source sync record: %s", err)
	}
	final := filepath.Join(layout.LocalPackages, result.ID)
	if err := os.Rename(stage, final); err != nil {
		return Installation{}, exit.Internalf("cannot retain private installation: %s", err)
	}
	for i := range result.Files {
		result.Files[i].Path = filepath.Join(final, result.Files[i].Filename)
	}
	return result, nil
}

func Open(layout home.Layout, install records.PackageInstall, id string) (Installation, *exit.Error) {
	if !validInstallationID(id) || id != install.ID {
		return Installation{}, exit.New(exit.Validation, "private installation belongs to a different install record")
	}
	root := filepath.Join(layout.LocalPackages, id)
	raw, err := os.ReadFile(filepath.Join(root, installationFile))
	var result Installation
	if err != nil || len(raw) > 1<<20 || json.Unmarshal(raw, &result) != nil || result.ID != id {
		return Installation{}, exit.New(exit.NotFound, "private installation is unavailable")
	}
	for i := range result.Files {
		if filepath.Base(result.Files[i].Filename) != result.Files[i].Filename {
			return Installation{}, exit.New(exit.Validation, "private installation contains an invalid file path")
		}
		result.Files[i].Path = filepath.Join(root, result.Files[i].Filename)
	}
	return result, nil
}

func validInstallationID(id string) bool {
	return id != "" && id != "." && id != ".." && !strings.ContainsAny(id, "/\\:\x00")
}
