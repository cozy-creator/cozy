package launch

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// PreparationSpec names the install without asserting any serving binding; the machine's
// preparation reports the exact placement later.
func (f *Facts) PreparationSpec() orchestrator.WorkerLaunchSpec {
	return orchestrator.WorkerLaunchSpec{
		Placement: orchestrator.DesiredPlacement{
			Package: f.Install.Package, InstallID: f.Install.ID, InstallationID: f.Install.ID, Release: f.Install.Version,
		},
		Preparation: &orchestrator.LocalServingPreparation{
			PythonVersion: hostruntime.PythonMinor(f.Install.Python), Published: f.Install.SourceKind == "tensorhub", Application: f.PackageInterface.Application,
			ModelSlotPaths:     f.PackageInterface.ModelSlotPaths(),
			PackageInterface:   append([]byte(nil), f.PackageInterface.Raw...),
			LockedRequirements: filepath.Join(f.Install.Dir, "locked-requirements.txt"),
		},
	}
}
