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
	inst := records.PackageInstall{ID: "fixture", Package: "local/fixture", Version: "1.0", SourceDigest: "sha256:" + strings.Repeat("1", 64)}
	requirements := []byte("torch @ https://files.pythonhosted.org/torch-2.13.0-py3-none-any.whl --hash=sha256:" + strings.Repeat("a", 64) + "\n")
	old, problem := localpackage.StageWheels(layout, inst, []byte("{}"), []string{wheelPath}, nil)
	fatal(t, problem)
	revision, problem := localpackage.StageWheels(layout, inst, []byte("{}"), []string{wheelPath}, requirements)
	fatal(t, problem)
	if revision.Digest == old.Digest || len(revision.Files) != 1 {
		t.Fatal("public requirements were omitted from identity or became a private wheel")
	}
	reopened, problem := localpackage.Open(layout, inst, revision.Digest)
	fatal(t, problem)
	if !bytes.Equal(reopened.DependencyRequirements, requirements) {
		t.Fatal("retained requirements changed")
	}
	selected, problem := orchestrator.LocalPackageSelection("capture", reopened)
	fatal(t, problem)
	if !bytes.Equal(selected.DependencyRequirements, requirements) {
		t.Fatal("Host selection lost requirements")
	}
	capture, problem := localpackage.CaptureExecution(inst.ID, reopened, func(string) ([]records.ChildBinding, *exit.Error) { return nil, nil }, func(string, string) (localpackage.Revision, *exit.Error) {
		t.Fatal("unexpected child lookup")
		return localpackage.Revision{}, nil
	})
	fatal(t, problem)
	var doc pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(capture.Canonical, &doc))
	ref := doc.Revisions[0].DependencyRequirements
	if ref == nil || ref.Length != uint64(len(requirements)) || !bytes.Equal(ref.Digest, canonical.Digest(requirements)) {
		t.Fatal("machine capture lost requirements identity")
	}
	retained := filepath.Join(layout.LocalPackages, strings.TrimPrefix(revision.Digest, "sha256:"), "dependency-requirements.txt")
	must(t, os.Chmod(retained, 0600))
	must(t, os.WriteFile(retained, bytes.Replace(requirements, []byte("2.13.0"), []byte("2.12.0"), 1), 0600))
	if _, problem := localpackage.Open(layout, inst, revision.Digest); problem == nil {
		t.Fatal("changed dependency selection was accepted on replay")
	}
	if _, problem := localpackage.Open(layout, inst, old.Digest); problem != nil {
		t.Fatal("complete all-local closure was refused", problem)
	}
	oldPayload := filepath.Join(layout.LocalPackages, strings.TrimPrefix(old.Digest, "sha256:"), "dependency-requirements.txt")
	must(t, os.Remove(oldPayload))
	if _, problem := localpackage.Open(layout, inst, old.Digest); problem == nil || problem.Name != "local_package_recapture_required" {
		t.Fatalf("old incomplete capture was reused: %v", problem)
	}
	otherPath := filepath.Join(root, "fixture-1.1-py3-none-any.whl")
	other, err := os.Create(otherPath)
	must(t, err)
	otherZip := zip.NewWriter(other)
	otherMetadata, err := otherZip.Create("fixture-1.1.dist-info/METADATA")
	must(t, err)
	_, err = otherMetadata.Write([]byte("Metadata-Version: 2.3\nName: fixture\nVersion: 1.1\n"))
	must(t, err)
	must(t, otherZip.Close())
	must(t, other.Close())
	if _, problem := localpackage.StageWheels(layout, inst, []byte("{}"), []string{wheelPath, otherPath}, requirements); problem == nil {
		t.Fatal("duplicate local distribution was accepted")
	}
}
