package install

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/pelletier/go-toml/v2"
)

const executionRequirementsFile = "execution-requirements.json"

// InstalledRequirements survives the local environment: remote placement needs
// its selected declarations, not another copy of its Python packages.
func InstalledRequirements(ctx context.Context, inst records.PackageInstall) (packagepublish.RequirementSelection, *exit.Error) {
	path := filepath.Join(inst.Dir, executionRequirementsFile)
	raw, err := os.ReadFile(path)
	if err == nil {
		var selected packagepublish.RequirementSelection
		if len(raw) > 4<<20 || json.Unmarshal(raw, &selected) != nil {
			return selected, exit.New(exit.Structural, "installed execution requirements are invalid")
		}
		return selected, nil
	}
	if !os.IsNotExist(err) {
		return packagepublish.RequirementSelection{}, exit.Internalf("cannot read installed execution requirements: %s", err)
	}
	return ExecutionRequirements(ctx, filepath.Join(inst.Dir, "venv"),
		strings.TrimPrefix(inst.Package, "local/"), strings.Fields(inst.Extra))
}

func retainExecutionRequirements(dir string, selected packagepublish.RequirementSelection) *exit.Error {
	raw, err := json.Marshal(selected)
	if err != nil {
		return exit.Internalf("cannot encode selected execution requirements: %s", err)
	}
	if err := os.WriteFile(filepath.Join(dir, executionRequirementsFile), raw, 0600); err != nil {
		return exit.Internalf("cannot retain selected execution requirements: %s", err)
	}
	return nil
}

// Reuse only static dependency metadata and unchanged dependency source. The
// root program may change: its interface is always read from the fresh snapshot.
// Comparing ordinary files here is a reuse decision, never a content identity.
func reusableSnapshotEnvironment(prior records.PackageInstall, source string) bool {
	if prior.SourceKind != "local" || prior.ProjectDir == "" {
		return false
	}
	for _, name := range []string{"pyproject.toml", "uv.lock", ".python-version"} {
		if !sameSnapshotFile(filepath.Join(prior.ProjectDir, name), filepath.Join(source, name)) {
			return false
		}
	}
	raw, err := os.ReadFile(filepath.Join(source, "pyproject.toml"))
	var project struct {
		Project struct {
			Dynamic []string `toml:"dynamic"`
		} `toml:"project"`
	}
	if err != nil || toml.Unmarshal(raw, &project) != nil || len(project.Project.Dynamic) != 0 {
		return false
	}
	old, problem := packagepublish.LocalDependencyPaths(prior.ProjectDir)
	if problem != nil {
		return false
	}
	current, problem := packagepublish.LocalDependencyPaths(source)
	if problem != nil || len(old) != len(current) {
		return false
	}
	for name, path := range current {
		previous := old[name]
		if previous == "" {
			return false
		}
		info, err := os.Stat(path)
		if err != nil {
			return false
		}
		if !info.IsDir() {
			if !sameSnapshotFile(previous, path) {
				return false
			}
			continue
		}
		_, before, problem := packagepublish.LibrarySourceTree(previous)
		if problem != nil {
			return false
		}
		_, after, problem := packagepublish.LibrarySourceTree(path)
		if problem != nil || len(before) != len(after) {
			return false
		}
		for member, file := range after {
			if before[member] == "" || !sameSnapshotFile(before[member], file) {
				return false
			}
		}
	}
	_, err = os.Stat(filepath.Join(prior.Dir, "venv", "pyvenv.cfg"))
	return err == nil
}

func sameSnapshotFile(left, right string) bool {
	a, ae := os.Open(left)
	b, be := os.Open(right)
	if a != nil {
		defer a.Close()
	}
	if b != nil {
		defer b.Close()
	}
	if ae != nil || be != nil {
		return os.IsNotExist(ae) && os.IsNotExist(be)
	}
	ai, ae := a.Stat()
	bi, be := b.Stat()
	if ae != nil || be != nil || !ai.Mode().IsRegular() || !bi.Mode().IsRegular() || ai.Size() != bi.Size() {
		return false
	}
	x, y := make([]byte, 64<<10), make([]byte, 64<<10)
	for {
		nx, ex := io.ReadFull(a, x)
		ny, ey := io.ReadFull(b, y)
		if nx != ny || ex != ey || !bytes.Equal(x[:nx], y[:ny]) {
			return false
		}
		if ex == io.EOF || ex == io.ErrUnexpectedEOF {
			return true
		}
		if ex != nil {
			return false
		}
	}
}
