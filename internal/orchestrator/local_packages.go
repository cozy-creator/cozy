package orchestrator

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Bounds match the host ledger. Captured wheel bytes travel only over PodHost.
const (
	maxLocalWheelBytes    = int64(512 << 20)
	maxLocalWheelSetBytes = int64(1 << 30)
)

// LocalPackageSelection is the passive code-preparation document for one private
// installation. It neither converges machine state nor creates an execution.
func LocalPackageSelection(operationID string, revision localpackage.Installation) (*pb.DesiredLocalPackageSet, *exit.Error) {
	if operationID == "" || revision.ID == "" || revision.Package == "" || revision.Release == "" || len(revision.Files) == 0 || len(revision.Files) > pb.MaxLocalPackageFiles {
		return nil, exit.Named(exit.Validation, "local_package_revision_invalid", "private installation inputs are incomplete")
	}
	selected := &pb.DesiredLocalPackageSet{PythonRequires: revision.PythonRequires, PythonVersion: hostruntime.PythonMinor(revision.PythonVersion), OperationId: operationID,
		SourceArchive: revision.SourceArchive, DependencyRequirements: revision.DependencyRequirements,
		Package: &pb.DevelopmentPackage{Package: revision.Package, Release: revision.Release, InstallationId: revision.ID}}
	var total int64
	prior := ""
	for _, file := range revision.Files {
		var digest []byte
		if file.Kind != "source" {
			var err error
			digest, err = canonical.Raw(file.Digest)
			if err != nil {
				return nil, exit.New(exit.Validation, "wheel integrity hash is invalid")
			}
		}
		if file.Length <= 0 || file.Length > maxLocalWheelSetBytes-total || file.Filename <= prior {
			return nil, exit.New(exit.Validation, "source transfer exceeds its bound or repeats a filename")
		}
		total += file.Length
		prior = file.Filename
		selected.Files = append(selected.Files, &pb.LocalPackageFileRef{Digest: digest, Filename: file.Filename, Length: uint64(file.Length)})
	}
	return selected, nil
}
