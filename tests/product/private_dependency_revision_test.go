package producttest

import (
	"archive/zip"
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestPrivateDependencyRequirementsBindReplayAndMachineCapture(t *testing.T) {
	root := t.TempDir()
	wheelPath := filepath.Join(root, "fixture-1.0-py3-none-any.whl")
	file, err := os.Create(wheelPath)
	must(t, err)
	writer := zip.NewWriter(file)
	metadata, err := writer.Create("fixture-1.0.dist-info/METADATA")
	must(t, err)
	_, err = metadata.Write([]byte("Metadata-Version: 2.3\nName: fixture\nVersion: 1.0\n"))
	must(t, err)
	must(t, writer.Close())
	must(t, file.Close())
	layout := home.Layout{LocalPackages: filepath.Join(root, "revisions")}
	inst := records.PackageInstall{ID: "fixture", Package: "local/fixture", Version: "1.0"}
	requirements := []byte("torch @ https://files.pythonhosted.org/torch-2.13.0-py3-none-any.whl --hash=sha256:" + strings.Repeat("a", 64) + "\n")
	old, problem := localpackage.StageWheels(layout, inst, []byte("{}"), []string{wheelPath}, nil)
	fatal(t, problem)
	inst.ID = "fixture-with-requirements"
	revision, problem := localpackage.StageWheels(layout, inst, []byte("{}"), []string{wheelPath}, requirements)
	fatal(t, problem)
	if revision.ID == old.ID || len(revision.Files) != 1 {
		t.Fatal("distinct accepted installations lost ownership or requirements became a wheel")
	}
	reopened, problem := localpackage.Open(layout, inst, revision.ID)
	fatal(t, problem)
	if !bytes.Equal(reopened.DependencyRequirements, requirements) {
		t.Fatal("retained requirements changed")
	}
	selected, problem := orchestrator.LocalPackageSelection("capture", reopened)
	fatal(t, problem)
	if !bytes.Equal(selected.DependencyRequirements, requirements) || selected.Package.InstallationId != inst.ID {
		t.Fatal("Host selection lost installation or requirements")
	}
	capture, problem := localpackage.CaptureExecution(inst.ID, reopened, func(string) ([]records.ChildBinding, *exit.Error) { return nil, nil }, func(string, string) (localpackage.Installation, *exit.Error) {
		t.Fatal("unexpected child lookup")
		return localpackage.Installation{}, nil
	})
	fatal(t, problem)
	var doc pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(capture.Canonical, &doc))
	if doc.RootInstallationId != inst.ID || len(doc.InstalledPackages) != 1 || doc.InstalledPackages[0].InstallationId != inst.ID || !bytes.Equal(doc.InstalledPackages[0].PackageInterface, []byte("{}")) {
		t.Fatal("machine capture lost accepted installation or interface")
	}
	if len(capture.Installations) != 1 || !bytes.Equal(capture.Installations[0].DependencyRequirements, requirements) {
		t.Fatal("capture staging lost dependency requirements")
	}
	if _, problem := localpackage.Open(layout, inst, old.ID); problem == nil {
		t.Fatal("another install opened this installation")
	}
	first, problem := localpackage.Open(layout, records.PackageInstall{ID: old.ID}, old.ID)
	fatal(t, problem)
	if len(first.DependencyRequirements) != 0 {
		t.Fatal("later installation rewrote the earlier closure")
	}

}
