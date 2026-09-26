package cli

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// Wheel intake extends the same bindings/revision path as source dependencies.
// The parent is installed once to observe uv's actual selected closure; its
// install remains the final parent unless callable wheels need substitution.
func (i *childIntake) prepareWheelIntake(ctx context.Context, sourceOverlays map[string]string) *exit.Error {
	parent, problem := i.Install()
	if problem != nil {
		return problem
	}
	environmentDir := parent.Install.Dir
	if parent.RemoteSnapshot {
		environmentDir = parent.MetadataEnvironment.Dir
	}
	python := home.VenvPython(filepath.Join(environmentDir, "venv"))
	possible, problem := install.HasInstalledApplications(python, parent.Install.Closure, i.Package.Name, sourceOverlays)
	if problem != nil || !possible {
		return problem
	}
	selection, problem := install.ExecutionRequirements(ctx, filepath.Join(environmentDir, "venv"),
		i.Package.Name, strings.Fields(parent.Install.Extra))
	if problem != nil {
		return problem
	}
	basePython, problem := install.BasePython(filepath.Join(environmentDir, "venv"))
	if problem != nil {
		return problem
	}
	graph, problem := packagepublish.WheelClosures(ctx, parent.Install.SourceRef, basePython, parent.Install.Closure, i.Package.Name, parent.Install.Extra)
	if problem != nil {
		return problem
	}
	if i.staging == "" {
		if err := os.MkdirAll(i.layout.Tmp, 0o700); err != nil {
			return exit.Internalf("cannot create callable wheel staging")
		}
		var err error
		i.staging, err = os.MkdirTemp(i.layout.Tmp, "child-interfaces-")
		if err != nil {
			return exit.Internalf("cannot create callable wheel staging")
		}
	}
	stage, err := os.MkdirTemp(i.staging, "selected-wheels-")
	if err != nil {
		return exit.Internalf("cannot capture selected callable wheels")
	}
	captured, problem := packagepublish.CaptureWheelDependencies(ctx, parent.Install.SourceRef, i.Package.Name, parent.Install.Closure, stage, graph, parent.Install.Python)
	if problem != nil {
		return problem
	}
	candidates := map[string]bool{}
	for name, dependency := range captured {
		if dependency.Application && !packagepublish.ImageOwnedDistribution(name) && sourceOverlays[name] == "" {
			candidates[name] = true
		}
	}
	if len(candidates) == 0 {
		return nil
	}
	if len(candidates) > 32 {
		return exit.New(exit.Validation, "unpublished parent exceeds 32 callable dependencies")
	}
	sourceBindings := map[string][]records.ChildBinding{}
	for _, binding := range i.Bindings {
		child, problem := i.store.Install(binding.ChildInstallID)
		if problem != nil {
			return problem
		}
		if child == nil {
			return exit.New(exit.Conflict, "captured source dependency disappeared")
		}
		name := strings.TrimPrefix(child.Package, "local/")
		sourceBindings[name] = append(sourceBindings[name], binding)
	}
	bindings := map[string][]records.ChildBinding{}
	overlays := map[string]packagepublish.CapturedDependency{}
	visiting := map[string]bool{}
	var prepare func(string, int) *exit.Error
	prepare = func(name string, depth int) *exit.Error {
		if _, ok := overlays[name]; ok {
			return nil
		}
		if visiting[name] || depth > 16 {
			return exit.New(exit.Validation, "unpublished callable wheel graph is cyclic or exceeds 16 levels")
		}
		visiting[name] = true
		defer delete(visiting, name)
		selected := graph[name]
		if selected[name] == "" {
			return exit.New(exit.Conflict, "callable wheel has no exact selected dependency closure")
		}
		// CaptureWheel adds the worker Runtime contract to every managed App.
		// Validate it against the Runtime already selected by this caller, even
		// when the ordinary library metadata did not depend on Runtime itself.
		for dependency, version := range graph["cozy-runtime"] { //cozy:allow distribution metadata, not a Runtime command
			selected[dependency] = version
		}
		dependencyNames := make([]string, 0, len(selected))
		for dependency := range selected {
			dependencyNames = append(dependencyNames, dependency)
		}
		sort.Strings(dependencyNames)
		closure := map[string]packagepublish.CapturedDependency{}
		var childBindings []records.ChildBinding
		for _, dependency := range dependencyNames {
			wheel, ok := captured[dependency]
			if !ok || wheel.Version != selected[dependency] {
				return exit.New(exit.Conflict, "callable wheel closure is missing an exact dependency")
			}
			if dependency != name && candidates[dependency] {
				if problem := prepare(dependency, depth+1); problem != nil {
					return problem
				}
				wheel = overlays[dependency]
				childBindings = append(childBindings, bindings[dependency]...)
			}
			childBindings = append(childBindings, sourceBindings[dependency]...)
			closure[dependency] = wheel
		}
		if len(childBindings) > 32 {
			return exit.New(exit.Validation, "callable wheel exceeds 32 dependency exports")
		}
		surface, problem := install.ReadInstalledInterface(ctx, python, name)
		if problem != nil {
			return problem
		}
		var exports []launch.Entrypoint
		for _, entry := range append(append([]launch.Entrypoint(nil), surface.Entrypoints...), surface.Jobs...) {
			if entry.Invocable != nil && !entry.Internal {
				exports = append(exports, entry)
			}
		}
		if len(exports) == 0 {
			overlays[name] = captured[name]
			return nil
		}
		var result *install.Result
		if parent.RemoteSnapshot {
			selected, err := install.ExecutionRequirements(ctx, filepath.Join(environmentDir, "venv"), name, selection.Extras[name])
			if err != nil {
				return err
			}
			result, problem = install.CaptureRemoteWheel(ctx, i.layout, i.store, name, parent.Install.Python, selection.Extras[name], closure, surface, selected)
		} else {
			result, problem = install.CaptureWheel(ctx, i.layout, i.store, basePython, name, selection.Extras[name], closure, surface)
		}
		if problem != nil {
			return problem
		}
		i.created = append(i.created, result.Install.ID)
		for n := range childBindings {
			childBindings[n].ParentInstallID = result.Install.ID
		}
		if problem := i.store.RecordChildBindings(childBindings); problem != nil {
			return problem
		}
		paths := []string{result.CapturedProjectWheel}
		var requirements []string
		for _, dependency := range dependencyNames {
			if dependency == name {
				continue
			}
			captured := closure[dependency]
			if captured.RegistryRequirement != "" {
				requirements = append(requirements, captured.RegistryRequirement)
			} else {
				paths = append(paths, captured.Path)
			}
		}
		var dependencyRequirements []byte
		if len(requirements) > 0 {
			dependencyRequirements = []byte(strings.Join(requirements, "\n") + "\n")
		}
		_, problem = localpackage.StageWheels(i.layout, result.Install, surface.Raw, paths, dependencyRequirements)
		if problem != nil {
			return problem
		}
		overlays[name] = captured[name]
		for _, entry := range exports {
			bindings[name] = append(bindings[name], records.ChildBinding{Module: entry.Invocable.Module, Export: entry.Invocable.Export,
				ChildInstallID: result.Install.ID, Entrypoint: entry.Name})
		}
		return nil
	}
	names := make([]string, 0, len(candidates))
	for name := range candidates {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		if problem := prepare(name, 0); problem != nil {
			return problem
		}
		if len(bindings[name]) == 0 {
			continue
		}
		i.Bindings = append(i.Bindings, bindings[name]...)
		if len(i.Bindings) > 32 {
			return exit.New(exit.Validation, "unpublished parent exceeds 32 invocable dependency exports")
		}
	}
	return nil
}
