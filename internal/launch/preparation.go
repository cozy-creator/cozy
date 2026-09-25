package launch

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"os"
	"path/filepath"
	"sort"

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
	slots := []string{}
	for _, entry := range append(append([]Entrypoint(nil), f.PackageInterface.Entrypoints...), f.PackageInterface.Jobs...) {
		for _, model := range entry.Models {
			slots = append(slots, model.Path)
		}
	}
	sort.Strings(slots)
	spec := orchestrator.WorkerLaunchSpec{
		Python: runtime, Args: []string{"serve"}, Dir: f.Install.Dir,
		EnvironmentPython: f.environmentPython(),
		Devices:           devices, GraceSec: 3,
		ArtifactCache: filepath.Join(f.Install.Dir, "artifact-cache"),
		InstallRoot:   filepath.Join(f.Install.Dir, "worker-environments"),
		TensorFSRoot:  config.Frozen().TensorFSRoot,
		Placement: orchestrator.DesiredPlacement{
			Package: f.Install.Package, InstallID: f.Install.ID, Release: f.Install.Version,
			SourceDigest: f.Install.SourceDigest,
		},
		Preparation: &orchestrator.LocalServingPreparation{
			PythonVersion: f.Install.Python, Published: f.Install.SourceKind == "tensorhub", Application: f.PackageInterface.Application,
			ModelSlotPaths: slots, PackageInterfaceDigest: f.Install.PackageInterface,
			LockedRequirements: filepath.Join(f.Install.Dir, "locked-requirements.txt"),
		},
	}
	if spec.Preparation.Published {
		spec.Placement.SourceDigest = ""
	}
	return spec, nil
}

// JobCodeIdentity is the already-retained code identity. Jobs do not require a
// serving placement, and no component order is inferred here.
func (f *Facts) JobCodeIdentity() (orchestrator.DesiredPlacement, string, *exit.Error) {
	if f.Install.PlacementSetDigest != "" {
		placement, problem := f.Placement()
		if problem != nil {
			return placement, "", problem
		}
		build, problem := orchestrator.JobBuildID(placement.PlacementSetBytes, f.Install.Package)
		return placement, build, problem
	}
	placement := orchestrator.DesiredPlacement{Package: f.Install.Package, InstallID: f.Install.ID, Release: f.Install.Version}
	if f.Install.SourceKind == "local" || f.Install.SourceKind == "wheel" {
		placement.SourceDigest = f.Install.SourceDigest
		return placement, placement.SourceDigest, nil
	}
	locked, err := os.ReadFile(filepath.Join(f.Install.Dir, "locked-requirements.txt"))
	if err != nil {
		return placement, "", exit.Internalf("cannot read retained package code identity: %s", err)
	}
	_, digest, err := canonical.Identity(&pb.Environment{LockedRequirements: &pb.Ref{Digest: canonical.Digest(locked), Length: uint64(len(locked))}})
	if err != nil {
		return placement, "", exit.Internalf("cannot identify retained package environment: %s", err)
	}
	placement.EnvironmentDigest, err = canonical.Spell(digest)
	if err != nil {
		return placement, "", exit.Internalf("cannot spell package environment: %s", err)
	}
	return placement, placement.EnvironmentDigest, nil
}
