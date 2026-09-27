package cli

import (
	"context"
	"sort"
	"strings"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func publishedPlanInterface(plan hub.PackageDownloadPlan) (*launch.PackageInterface, *exit.Error) {
	document, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, problem
	}
	return launch.DecodePackageInterface(document.Bytes)
}

// dependencies joins exact committed callee wheels before the machine accepts
// execution. Subsequent child scheduling needs no client or catalog lookup.
func (m *machineRuns) capturePublishedDependencies(ctx context.Context, request records.Request, connection *machineConnection, capture *pb.MachineExecutionCapture, rootLocked []byte) *exit.Error {
	type node struct {
		installationID string
		plan           hub.PackageDownloadPlan
		iface          *launch.PackageInterface
		locked         []byte
	}
	nodes := map[string]node{}
	visiting := map[string]bool{}
	rootKey := request.Package + "@" + request.Release
	rootRef, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return problem
	}
	rootPlan, problem := m.resolver.catalog(request.Hub).PackageDownloads(ctx, rootRef, request.Release)
	if problem != nil {
		return problem
	}
	rootInterface, problem := launch.DecodePackageInterface(capture.InstalledPackages[0].PackageInterface)
	if problem != nil {
		return problem
	}
	nodes[rootKey] = node{capture.RootInstallationId, rootPlan, rootInterface, rootLocked}
	var walk func(string, int) *exit.Error
	walk = func(key string, depth int) *exit.Error {
		if depth > 16 || len(nodes) > 128 || visiting[key] {
			return exit.New(exit.Validation, "published callable closure is cyclic or exceeds its bound")
		}
		visiting[key] = true
		defer delete(visiting, key)
		parent := nodes[key]
		document, problem := exactPackageInstallDocument("uv.lock", parent.plan.UVLock)
		if problem != nil {
			return problem
		}
		dependencies, problem := install.PublishedDependencies(strings.SplitN(key, "@", 2)[0], document.Bytes, parent.locked)
		if problem != nil {
			return problem
		}
		for _, dependency := range dependencies {
			childKey := dependency.Package + "@" + dependency.Version
			child, known := nodes[childKey]
			if !known {
				ref, problem := hub.ParseRef(dependency.Package)
				if problem != nil {
					return problem
				}
				plan, problem := m.resolver.catalog(request.Hub).PackageDownloads(ctx, ref, dependency.Version)
				if problem != nil {
					return problem
				}
				if plan.Release != dependency.Version {
					return exit.New(exit.Conflict, "published child release changed")
				}
				sub := request
				// The callee's own model selections ride its own preparation.
				sub.Package, sub.Release, sub.InstallID = dependency.Package, dependency.Version, ""
				began := time.Now()
				prepared, problem := connection.preparePublished(ctx, sub)
				if problem != nil {
					return problem
				}
				m.submissionStage(request.ID, "package_preparation", preparedDetail(sub.Package, sub.Release, prepared), began)
				installed := prepared.InstalledPackage
				if installed == nil || installed.InstallationId == "" || installed.Package != dependency.Package || installed.Release != dependency.Version {
					return exit.New(exit.Conflict, "published child preparation returned another installation")
				}
				iface, problem := launch.DecodePackageInterface(installed.PackageInterface)
				if problem != nil {
					return problem
				}
				child = node{installed.InstallationId, plan, iface, prepared.LockedRequirements}
				nodes[childKey] = child
				capture.InstalledPackages = append(capture.InstalledPackages, installed)
				addPublishedBindings(capture, installed.InstallationId, installed.InstallationId, iface, true)
				if problem := walk(childKey, depth+1); problem != nil {
					return problem
				}
			} else {
				if visiting[childKey] {
					return exit.New(exit.Validation, "published callable closure is cyclic")
				}
			}
			addPublishedBindings(capture, parent.installationID, child.installationID, child.iface, false)
		}
		return nil
	}
	if problem := walk(rootKey, 0); problem != nil {
		return problem
	}
	sort.Slice(capture.InstalledPackages, func(i, j int) bool {
		return capture.InstalledPackages[i].InstallationId < capture.InstalledPackages[j].InstallationId
	})
	// A callee's omitted Model is resolved by the machine at this origin when called.
	capture.CatalogOrigin = connection.publicOrigin
	return nil
}

func addPublishedBindings(capture *pb.MachineExecutionCapture, caller, callee string, iface *launch.PackageInterface, self bool) {
	for _, entry := range append(append([]launch.Entrypoint(nil), iface.Jobs...), iface.Entrypoints...) {
		if entry.Invocable == nil || entry.Internal && !self {
			continue
		}
		capture.Bindings = append(capture.Bindings, &pb.MachineCallableBinding{CallerInstallationId: caller, CalleeInstallationId: callee, Module: entry.Invocable.Module, Export: entry.Invocable.Export, Entrypoint: entry.Name})
	}
}

// The worker installs a published child directly. This does not change the
// user's active local package pin or create a client-side inference environment.
func (r *Resolver) publishedChildPreparation(ctx context.Context, command *Context, request records.Request) (*hub.PackageDownloadPlan, *launch.PackageInterface, []byte, string, *exit.Error) {
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	plan, problem := r.catalog(request.Hub).PackageDownloads(ctx, ref, request.Release)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	iface, problem := publishedPlanInterface(plan)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	config, problem := exactPackageInstallDocument("package.toml", plan.PackageConfig)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	ifaceDoc, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	pyproject, problem := exactPackageInstallDocument("pyproject.toml", plan.Pyproject)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	uvLock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	source, problem := packageInstallPlanFacts(command, ref, request.Release, plan, config, ifaceDoc, pyproject, uvLock)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	locked, problem := install.PublishedRequirements(source)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	detail, problem := r.catalog(request.Hub).PackageRelease(ctx, ref, request.Release)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	return &plan, iface, locked, detail.RequiresPython, nil
}
