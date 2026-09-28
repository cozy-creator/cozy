package launch

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
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

// JobInstallationID selects an installed resource. It does not identify package
// bytes or participate in operation computation identity.
func (f *Facts) JobInstallationID() (orchestrator.DesiredPlacement, string, *exit.Error) {
	if f.Install.PlacementSetDigest != "" {
		placement, problem := f.Placement()
		return placement, placement.InstallationID, problem
	}
	placement := orchestrator.DesiredPlacement{Package: f.Install.Package, InstallID: f.Install.ID, InstallationID: f.Install.ID, Release: f.Install.Version}
	return placement, f.Install.ID, nil
}
