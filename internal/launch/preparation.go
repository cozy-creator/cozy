package launch

import (
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// PreparationSpec starts the ordinary per-install worker without asserting any
// serving binding. The exact placement is returned by worker preparation later.
func (f *Facts) PreparationSpec(devices []string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	runtime, problem := hostruntime.Path(f.RuntimeCLI.Env)
	if problem != nil {
		return orchestrator.WorkerLaunchSpec{}, problem
	}
	slots := f.PackageInterface.ModelSlotPaths()
	spec := orchestrator.WorkerLaunchSpec{
		Python: runtime, Args: []string{"serve"}, Dir: f.Install.Dir,
		EnvironmentPython: f.environmentPython(),
		Devices:           devices, GraceSec: 3,
		ArtifactCache: filepath.Join(f.Install.Dir, "artifact-cache"),
		InstallRoot:   filepath.Join(f.Install.Dir, "worker-environments"),
		TensorFSRoot:  config.Frozen().TensorFSRoot,
		Placement: orchestrator.DesiredPlacement{
			Package: f.Install.Package, InstallID: f.Install.ID, InstallationID: f.Install.ID, Release: f.Install.Version,
		},
		Preparation: &orchestrator.LocalServingPreparation{
			PythonVersion: hostruntime.PythonMinor(f.Install.Python), Published: f.Install.SourceKind == "tensorhub", Application: f.PackageInterface.Application,
			ModelSlotPaths:     slots,
			PackageInterface:   append([]byte(nil), f.PackageInterface.Raw...),
			LockedRequirements: filepath.Join(f.Install.Dir, "locked-requirements.txt"),
		},
	}
	return spec, nil
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
