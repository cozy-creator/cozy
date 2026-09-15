package launch

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

// Facts is everything one package install needs to be served, gathered once.
type Facts struct {
	CPUOrchestration bool // exact captured child bindings give model-free composition its CPU role
	// SelfCallable names the install's own exports it captured as its own children.
	// They are the CALLEES of a composition, never its parent, so the CPU role above
	// is not read onto them — one install holds both halves of a self call.
	SelfCallable     map[string]bool
	Install          records.PackageInstall
	Source           string
	PackageInterface *PackageInterface
	RuntimeCLI       RuntimeCLI
}

func (f *Facts) environmentPython() string {
	if f.RuntimeCLI.EnvironmentPython != "" {
		return f.RuntimeCLI.EnvironmentPython
	}
	return home.VenvPython(filepath.Join(f.Install.Dir, "venv"))
}

// EnvironmentPython preserves the ordinary venv layout and resolves only the
// fixed builtin's independently prepared Runtime generation through bound metadata.
func EnvironmentPython(inst records.PackageInstall) (string, *exit.Error) {
	if inst.Package != "local/"+runtimeoperation.Name {
		return home.VenvPython(filepath.Join(inst.Dir, "venv")), nil
	}
	if inst.SourceKind != "wheel" {
		return "", exit.New(exit.Conflict, "Runtime builtin is not a captured metadata carrier")
	}
	return runtimeoperation.ReadEnvironment(inst.Dir, inst.LockDigest, inst.PackageInterface, inst.SourceDigest)
}

// Read gathers an install's facts: where its source is, the surface it proved at install,
// and how this host's Runtime is asked about it — immutable installs use their verified
// interface, while live editable checkouts use source. The tool is resolved when a question is
// asked (Job), so a host that never asks one needs none.
func Read(inst records.PackageInstall, cozyHome string, env []string) (*Facts, *exit.Error) {
	source := SourceDir(inst)
	d, e := ReadPackageInterface(PackageInterfacePath(inst.Dir), inst.PackageInterface)
	if e != nil {
		return nil, e
	}
	packageInterface := ""
	capturedSource := inst.SourceKind == "local" &&
		filepath.Clean(inst.SourceRef) == filepath.Join(inst.Dir, "source") &&
		filepath.Clean(inst.ProjectDir) == filepath.Join(inst.Dir, "source")
	if inst.SourceKind == "tensorhub" || inst.SourceKind == "wheel" || capturedSource {
		packageInterface = PackageInterfacePath(inst.Dir)
	}
	environmentPython, e := EnvironmentPython(inst)
	if e != nil {
		return nil, e
	}
	if inst.Package == "local/"+runtimeoperation.Name {
		if inst.SourceKind != "wheel" || d.Application != runtimeoperation.Application {
			return nil, exit.New(exit.Conflict, "Runtime builtin install changed its fixed application")
		}
	}
	return &Facts{
		Install:          inst,
		Source:           source,
		PackageInterface: d,
		RuntimeCLI: RuntimeCLI{
			Dir: source, PackageInterface: packageInterface, Home: cozyHome, Env: env,
			EnvironmentPython: environmentPython,
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
	for i := range f.PackageInterface.Entrypoints {
		ep := &f.PackageInterface.Entrypoints[i]
		outputs[ep.Name] = AssetPaths(ep.Result)
	}
	return orchestrator.PlacementFromExact(f.Install.Package, f.Install.ID,
		f.Install.PlacementSetDigest, data, outputs)
}

// Spec adds this host's target-environment materialization to a placement. The trusted
// host Runtime owns worker control; the install venv supplies the selected executor.
// A connected worker never calls this method.
func (f *Facts) Spec(devices []string) (orchestrator.WorkerLaunchSpec, *exit.Error) {
	if f.Install.SourceKind == "local" || f.Install.PlacementSetDigest == "" {
		return f.PreparationSpec(devices)
	}
	placement, e := f.Placement()
	if e != nil {
		return orchestrator.WorkerLaunchSpec{}, e
	}
	cache := filepath.Join(f.Install.Dir, "artifact-cache")

	runtimeBin, e := hostruntime.Path(f.RuntimeCLI.Env)
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
		EnvironmentPython: f.environmentPython(),
		TensorFSRoot:      config.Frozen().TensorFSRoot,
	}, nil
}
