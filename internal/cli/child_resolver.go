package cli

import (
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// composesChildren is Request.ComposesChildren for an installed request, with one exception:
// a client script is a composition root whose captured bindings are the calls it makes, so a
// script that writes weights still reserves capacity for its children.
func (r *Resolver) composesChildren(request records.Request) (bool, *exit.Error) {
	if request.InstallID == "" {
		return false, nil
	}
	if request.ComposesChildren() || request.SizedByOwnModels() {
		return request.ComposesChildren(), nil
	}
	_, surface, problem := r.installPackageInterface(request.InstallID)
	if problem != nil {
		return false, problem
	}
	return surface.Application == packagepublish.ScriptApplication, nil
}

// PrivateRentalNeedsAccelerator sizes the rental for its captured children while
// keeping the parent itself on the separate CPU orchestration slot.
func (r *Resolver) PrivateRentalNeedsAccelerator(request records.Request) (bool, *exit.Error) {
	if request.NeedsAccelerator {
		return true, nil
	}
	if composes, problem := r.composesChildren(request); problem != nil || !composes {
		return false, problem
	}
	queue := []string{request.InstallID}
	seen := map[string]bool{}
	needed := false
	for len(queue) > 0 {
		install := queue[0]
		queue = queue[1:]
		if seen[install] {
			continue
		}
		seen[install] = true
		bindings, problem := r.store.ChildBindings(install)
		if problem != nil {
			return false, problem
		}
		for _, binding := range bindings {
			child, problem := r.store.Install(binding.ChildInstallID)
			if problem != nil {
				return false, problem
			}
			if child == nil {
				return false, exit.Named(exit.Conflict, "child.install_absent", "captured child implementation is unavailable for rental sizing")
			}
			surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(child.Dir))
			if problem != nil {
				return false, problem
			}
			job, problem := surface.Function(binding.Entrypoint)
			if problem != nil {
				return false, problem
			}
			if (job.Kind != "job" && job.Kind != "entrypoint") || job.Invocable == nil || job.Invocable.Module != binding.Module || job.Invocable.Export != binding.Export {
				return false, exit.Named(exit.Conflict, "child.export_changed", "captured child has no exact callable for rental sizing")
			}
			// The callable's own declaration, else the closure JobsInstall reads; no
			// package code needs importing again merely to choose a machine class.
			needed = needed || job.NeedsAccelerator(strings.Split(child.Closure, "\n"))
			queue = append(queue, binding.ChildInstallID)
		}
	}
	return needed, nil
}

// CapturedByteOutputBound uses the captured result schema, with the same finite
// asset bounds as ordinary inputs. A manifest entry cannot invent a result field.
func (r *Resolver) CapturedByteOutputBound(request records.Request, path, mediaType string) (int64, *exit.Error) {
	surface, problem := r.capturedResultInterface(request)
	if problem != nil {
		return 0, problem
	}
	entry, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return 0, problem
	}
	spec, ok := launch.ResultAssetSpec(entry, path)
	if !ok || !spec.AcceptsMediaType(mediaType) {
		return 0, exit.New(exit.Validation, "native byte output is not a declared asset field")
	}
	maximum := spec.MaxBytes
	if maximum <= 0 {
		maximum = inputasset.MaxBytes
	}
	return maximum, nil
}
