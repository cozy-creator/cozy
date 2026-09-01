package launch

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
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

// Read gathers an install's facts: where its source is, the surface it proved at
// install, and the runtime that proved it.
func Read(inst records.PackageInstall, cozyHome string, env []string) (*Facts, *exit.Error) {
	source := SourceDir(inst)
	d, e := ReadDescriptor(DescriptorPath(inst.Dir), inst.PackageDescriptor)
	if e != nil {
		return nil, e
	}
	runtimeBin, descriptor := Binary(inst), ""
	if inst.SourceKind == "tensorhub" {
		descriptor = DescriptorPath(inst.Dir)
	}
	return &Facts{
		Install:           inst,
		Source:            source,
		PackageDescriptor: d,
		RuntimeCLI: RuntimeCLI{
			Bin: runtimeBin, Dir: source, Descriptor: descriptor, Home: cozyHome, Env: env,
		},
	}, nil
}

// SourceDir is where an install's package tree lives. Editable installs retain
// their explicit absolute author-controlled path.
func SourceDir(inst records.PackageInstall) string {
	if inst.ProjectDir != "" {
		return inst.ProjectDir
	}
	if inst.SourceKind == "local" && inst.SourceRef != "" {
		return inst.SourceRef
	}
	return filepath.Join(inst.Dir, "source")
}

// PackageRevisionDigest is the exact published release or editable source digest
// this install serves. The same digest pins invocation and worker identity.
func PackageRevisionDigest(inst records.PackageInstall) string {
	return inst.SourceDigest
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
// host Runtime owns worker control; the install venv supplies the selected executor.
// A connected worker never calls this method.
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
		Dir:               f.Source,
		Devices:           devices,
		GraceSec:          3,
		ArtifactCache:     cache,
		EnvironmentPython: home.VenvPython(filepath.Join(f.Install.Dir, "venv")),
		ArtifactStore:     filepath.Join(filepath.Dir(filepath.Dir(f.Install.Dir)), "cas"),
	}, nil
}
