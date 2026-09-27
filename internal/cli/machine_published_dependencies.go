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
	"github.com/cozy-creator/cozy/internal/orchestrator"
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

// dependencies joins exact committed callee releases before the machine accepts
// execution. A machine that installs callees on first selection receives each code-only
// callee as a deferred row instead; the rest are prepared now. Child scheduling needs no
// client or catalog lookup either way.
func (m *machineRuns) capturePublishedDependencies(ctx context.Context, request records.Request, connection *machineConnection, capture *pb.MachineExecutionCapture, rootLocked []byte) *exit.Error {
	type node struct {
		name   captureName
		plan   hub.PackageDownloadPlan
		iface  *launch.PackageInterface
		locked []byte
	}
	workspace, problem := currentExecutionWorkspace(ctx, connection)
	if problem != nil {
		return problem
	}
	defer sort.Slice(capture.DeferredInstallations, func(i, j int) bool {
		return capture.DeferredInstallations[i].Key < capture.DeferredInstallations[j].Key
	})
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
	nodes[rootKey] = node{captureName{ID: capture.RootInstallationId}, rootPlan, rootInterface, rootLocked}
	var walk func(string) *exit.Error
	walk = func(key string) *exit.Error {
		if len(nodes) > 128 || visiting[key] {
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
				if workspace.DeferredInstallations && len(orchestrator.DownloadModelRefs(sub.PreparedModels())) == 0 {
					preparation, problem := connection.publishedRequest(ctx, sub)
					if problem != nil {
						return problem
					}
					iface, problem := launch.DecodePackageInterface(preparation.PackageInterface)
					if problem != nil {
						return problem
					}
					child = node{captureName{Key: childKey}, plan, iface, preparation.LockedRequirements}
					nodes[childKey] = child
					capture.DeferredInstallations = append(capture.DeferredInstallations, &pb.DeferredInstallation{
						Key: childKey, Package: dependency.Package, Release: dependency.Version, Preparation: preparation})
					addPublishedBindings(capture, child.name, child.name, iface, true)
					if problem := walk(childKey); problem != nil {
						return problem
					}
					addPublishedBindings(capture, parent.name, child.name, child.iface, false)
					continue
				}
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
				child = node{captureName{ID: installed.InstallationId}, plan, iface, prepared.LockedRequirements}
				nodes[childKey] = child
				capture.InstalledPackages = append(capture.InstalledPackages, installed)
				addPublishedBindings(capture, child.name, child.name, iface, true)
				if problem := walk(childKey); problem != nil {
					return problem
				}
			} else {
				if visiting[childKey] {
					return exit.New(exit.Validation, "published callable closure is cyclic")
				}
			}
			addPublishedBindings(capture, parent.name, child.name, child.iface, false)
		}
		return nil
	}
	if problem := walk(rootKey); problem != nil {
		return problem
	}
	sort.Slice(capture.InstalledPackages, func(i, j int) bool {
		return capture.InstalledPackages[i].InstallationId < capture.InstalledPackages[j].InstallationId
	})
	// Every installation's defaults, the root's included, are recorded once the entire
	// binding inventory exists, probing each exact checkpoint once for the capture.
	began, reads := time.Now(), modelDefaultReads{}
	for key, node := range nodes {
		m.resolver.captureDefaultRows(capture, strings.SplitN(key, "@", 2)[0], node.name, node.iface, request, connection.publicOrigin, reads)
	}
	m.submissionStage(request.ID, "model_defaults", modelDefaultsDetail(capturedRungs(capture), len(reads)), began)
	return nil
}

// captureName is how a capture names one installation: its id, or its release key while it
// is deferred to first selection.
type captureName struct{ ID, Key string }

func (n captureName) String() string { return n.ID + n.Key }

func bindingCaller(b *pb.MachineCallableBinding) string {
	return b.CallerInstallationId + b.CallerDeferredKey
}

func addPublishedBindings(capture *pb.MachineExecutionCapture, caller, callee captureName, iface *launch.PackageInterface, self bool) {
	for _, entry := range append(append([]launch.Entrypoint(nil), iface.Jobs...), iface.Entrypoints...) {
		if entry.Invocable == nil || entry.Internal && !self {
			continue
		}
		capture.Bindings = append(capture.Bindings, &pb.MachineCallableBinding{
			CallerInstallationId: caller.ID, CallerDeferredKey: caller.Key, CalleeInstallationId: callee.ID, CalleeDeferredKey: callee.Key,
			Module: entry.Invocable.Module, Export: entry.Invocable.Export, Entrypoint: entry.Name})
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
