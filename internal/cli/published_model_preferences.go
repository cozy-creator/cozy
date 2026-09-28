package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// PublishedChildModels uses the same exact lock and callable metadata as capture.
// A CPU workflow needs capacity for its children; their group counts are combined
// by placement's maximum, since each child reserves its own execution group. The
// request's own callable is never its own child: its slots are already selected.
func (r *Resolver) PublishedChildModels(command *Context, request records.Request) ([]records.ModelRef, *exit.Error) {
	// An unversioned request has no exact closure to walk; RentalConstraints answers it.
	if !request.ComposesChildren() || request.Release == "" {
		return nil, nil
	}
	root := request.Package + "@" + request.Release
	ctx, cancel := hub.Context()
	defer cancel()
	seen, visiting := map[string]bool{}, map[string]bool{}
	var models []records.ModelRef
	var walk func(records.Request, int) *exit.Error
	walk = func(selected records.Request, depth int) *exit.Error {
		key := selected.Package + "@" + selected.Release
		if visiting[key] || depth > 16 || len(seen) > 128 {
			return exit.New(exit.Validation, "published model preference closure is cyclic or exceeds its bound")
		}
		if seen[key] {
			return nil
		}
		seen[key], visiting[key] = true, true
		defer delete(visiting, key)
		plan, iface, locked, problem := r.publishedChildPreparation(ctx, command, selected)
		if problem != nil {
			return problem
		}
		for _, entry := range iface.Entrypoints {
			if entry.Invocable == nil || entry.Internal && key != root || key == root && entry.Name == request.Entrypoint {
				continue
			}
			for _, slot := range entry.Models {
				model, problem := r.childModelLadder(request.Hub, selected.Package, entry.Name, slot)
				if problem != nil {
					continue
				} // Unused or explicitly overridden defaults stay optional.
				model.Slot = slot.Path
				model.Callable = selected.Package + "/" + entry.Name
				models = append(models, model)
			}
		}
		lock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
		if problem != nil {
			return problem
		}
		children, problem := install.PublishedDependencies(selected.Package, lock.Bytes, locked)
		if problem != nil {
			return problem
		}
		for _, child := range children {
			next := selected
			next.Package, next.Release = child.Package, child.Version
			next.Models = nil
			if problem := walk(next, depth+1); problem != nil {
				return problem
			}
		}
		return nil
	}
	if problem := walk(request, 0); problem != nil {
		return nil, problem
	}
	return models, nil
}

func publishedPlanInterface(plan hub.PackageDownloadPlan) (*launch.PackageInterface, *exit.Error) {
	document, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, problem
	}
	return launch.DecodePackageInterface(document.Bytes)
}

// publishedChildPreparation is one published release's plan, interface and locked
// closure, read to size a rental for a workflow's children before any machine is known.
func (r *Resolver) publishedChildPreparation(ctx context.Context, command *Context, request records.Request) (*hub.PackageDownloadPlan, *launch.PackageInterface, []byte, *exit.Error) {
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return nil, nil, nil, problem
	}
	plan, problem := r.catalog(request.Hub).PackageDownloads(ctx, ref, request.Release)
	if problem != nil {
		return nil, nil, nil, problem
	}
	iface, problem := publishedPlanInterface(plan)
	if problem != nil {
		return nil, nil, nil, problem
	}
	config, problem := exactPackageInstallDocument("package.toml", plan.PackageConfig)
	if problem != nil {
		return nil, nil, nil, problem
	}
	ifaceDoc, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, nil, nil, problem
	}
	pyproject, problem := exactPackageInstallDocument("pyproject.toml", plan.Pyproject)
	if problem != nil {
		return nil, nil, nil, problem
	}
	uvLock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return nil, nil, nil, problem
	}
	source, problem := packageInstallPlanFacts(command, ref, request.Release, plan, config, ifaceDoc, pyproject, uvLock)
	if problem != nil {
		return nil, nil, nil, problem
	}
	locked, problem := install.PublishedRequirements(source)
	if problem != nil {
		return nil, nil, nil, problem
	}
	return &plan, iface, locked, nil
}
