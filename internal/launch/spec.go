package launch

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// Facts is everything one package install needs to be served, gathered once.
type Facts struct {
	Install           records.PackageInstall
	Source            string
	PackageDescriptor *PackageDescriptor
	RuntimeCLI        RuntimeCLI
}

// Read gathers a generation's facts: where its source is, the surface it proved at
// install, and the runtime that proved it.
func Read(gen records.PackageInstall, cozyHome string, env []string) (*Facts, *exit.Error) {
	source := SourceDir(gen)
	d, e := ReadDescriptor(DescriptorPath(gen.Dir), gen.PackageDescriptor)
	if e != nil {
		return nil, e
	}
	runtimeBin, descriptor := Binary(gen), ""
	if gen.SourceKind == "tensorhub" {
		runtimeBin, e = HostRuntime()
		if e != nil {
			return nil, e
		}
		descriptor = DescriptorPath(gen.Dir)
	}
	return &Facts{
		Install:           gen,
		Source:            source,
		PackageDescriptor: d,
		RuntimeCLI: RuntimeCLI{
			Bin: runtimeBin, Dir: source, Descriptor: descriptor, Home: cozyHome, Env: env,
		},
	}, nil
}

// SourceDir is where a generation's package tree lives. Editable installs retain
// their explicit absolute author-controlled path.
func SourceDir(gen records.PackageInstall) string {
	if gen.ProjectDir != "" {
		return gen.ProjectDir
	}
	if gen.SourceKind == "local" && gen.SourceRef != "" {
		return gen.SourceRef
	}
	return filepath.Join(gen.Dir, "source")
}

// PackageRevisionDigest is the exact published release or editable source digest
// this generation serves. The same digest pins invocation and worker identity.
func PackageRevisionDigest(gen records.PackageInstall) string {
	return gen.SourceDigest
}

// Placement reads the exact Hub-selected PlacementSet stored at install. Creator never
// rebuilds bindings from package source or asks Runtime for a parallel plan document.
func (f *Facts) Placement() (orchestrator.DesiredPlacement, *exit.Error) {
	if f.Install.PlacementSetDigest == "" {
		return orchestrator.DesiredPlacement{}, exit.Named(exit.Structural,
			"package_selection_missing",
			"%s was installed without an exact PlacementSet",
			f.Install.Package).WithRemedy(
			"reinstall the package; published and editable installs both retain their exact execution selection")
	}
	path := filepath.Join(f.Install.Dir, "artifact-cache",
		strings.TrimPrefix(f.Install.PlacementSetDigest, "sha256:"))
	data, err := os.ReadFile(path)
	if err != nil {
		return orchestrator.DesiredPlacement{}, exit.New(exit.NotFound,
			"cannot read selected PlacementSet %s: %s", f.Install.PlacementSetDigest, err)
	}
	outputs := map[string][]string{}
	for i := range f.PackageDescriptor.Entrypoints {
		ep := &f.PackageDescriptor.Entrypoints[i]
		outputs[ep.Name] = AssetPaths(ep.Result)
	}
	return orchestrator.PlacementFromExact(f.Install.Package, f.Install.ID,
		f.Install.PlacementSetDigest, data, outputs)
}

// Spec adds this host's target-environment materialization to a placement. The trusted
// host Runtime owns worker control; the generation venv is only the selected executor
// overlay. A connected worker never calls this method.
func (f *Facts) Spec(devices []string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	placement, e := f.Placement()
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	cache := filepath.Join(f.Install.Dir, "artifact-cache")
	if f.Install.SourceKind == "local" {
		return orchestrator.WorkerLaunchSpec{
			Placement: placement,
			Python:    Binary(f.Install), Args: []string{"serve",
				"--development-project", f.Source,
				"--development-package", placement.Package,
				"--development-release", placement.Release,
				"--development-source-digest", placement.SourceDigest},
			Dir: f.Source, Devices: devices, GraceSec: 3,
			ArtifactStore: filepath.Join(filepath.Dir(filepath.Dir(f.Install.Dir)), "cas"),
		}, nil
	}
	runtimeBin, e := HostRuntime()
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	return orchestrator.WorkerLaunchSpec{
		Placement: placement,
		Python:    runtimeBin, Args: []string{"serve"},
		Dir:             f.Source,
		Devices:         devices,
		GraceSec:        3,
		ArtifactCache:   cache,
		EnvironmentRoot: filepath.Join(f.Install.Dir, "venv"),
		ArtifactStore:   filepath.Join(filepath.Dir(filepath.Dir(f.Install.Dir)), "cas"),
	}, nil
}
