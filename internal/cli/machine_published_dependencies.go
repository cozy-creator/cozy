package cli

import (
	"bytes"
	"context"
	"net/url"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"github.com/pelletier/go-toml/v2"
)

type publishedDependency struct {
	name, version string
	digests       map[string]bool
}

// Only wheels actually selected in the parent's immutable plan can authorize
// children. A same-named PyPI library is not a Tensorhub package dependency.
func publishedDependencies(pkg string, plan hub.PackageDownloadPlan, locked []byte) ([]publishedDependency, *exit.Error) {
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		return nil, problem
	}
	document, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return nil, problem
	}
	var lock struct {
		Packages []struct {
			Name    string `toml:"name"`
			Version string `toml:"version"`
			Source  struct {
				Registry string `toml:"registry"`
			} `toml:"source"`
			Wheels []struct {
				Hash string `toml:"hash"`
			} `toml:"wheels"`
		} `toml:"package"`
	}
	if err := toml.Unmarshal(document.Bytes, &lock); err != nil {
		return nil, exit.New(exit.Structural, "published dependency lock is invalid")
	}
	selectedRows := map[string][]string{}
	for _, line := range strings.Split(string(locked), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 || strings.HasPrefix(fields[0], "-") {
			continue
		}
		selectedRows[fields[0]] = fields[1:]
	}
	selected := map[string]publishedDependency{}
	for _, row := range lock.Packages {
		index, err := url.Parse(row.Source.Registry)
		if err != nil || index.User != nil || index.RawQuery != "" || index.Fragment != "" || (index.Scheme != "https" && index.Scheme != "http") || strings.Trim(index.Path, "/") != "v1/index/"+ref.Org+"/simple" {
			continue
		}
		hashes := selectedRows[row.Name+"=="+row.Version]
		if len(hashes) == 0 {
			continue
		}
		allowed := map[string]bool{}
		for _, wheel := range row.Wheels {
			for _, hash := range hashes {
				if hash == "--hash="+wheel.Hash {
					allowed[wheel.Hash] = true
				}
			}
		}
		if len(allowed) == 0 {
			return nil, exit.New(exit.Conflict, "published callable dependency differs from its locked wheel")
		}
		if _, duplicate := selected[row.Name]; duplicate {
			return nil, exit.New(exit.Conflict, "published callable dependency is ambiguous")
		}
		selected[row.Name] = publishedDependency{ref.Org + "/" + row.Name, row.Version, allowed}

	}
	out := make([]publishedDependency, 0, len(selected))
	for _, row := range selected {
		out = append(out, row)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].name < out[j].name })
	return out, nil
}

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
		digest []byte
		plan   hub.PackageDownloadPlan
		iface  *launch.PackageInterface
		locked []byte
	}
	nodes := map[string]node{}
	visiting := map[string]bool{}
	rootKey := request.Package + "@" + request.Release
	rootRef, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return problem
	}
	rootPlan, problem := m.resolver.catalog.PackageDownloads(ctx, rootRef, request.Release)
	if problem != nil {
		return problem
	}
	rootInterface, problem := publishedPlanInterface(rootPlan)
	if problem != nil {
		return problem
	}
	nodes[rootKey] = node{capture.RootRevisionDigest, rootPlan, rootInterface, rootLocked}
	var walk func(string, int) *exit.Error
	walk = func(key string, depth int) *exit.Error {
		if depth > 16 || len(nodes) > 128 || visiting[key] {
			return exit.New(exit.Validation, "published callable closure is cyclic or exceeds its bound")
		}
		visiting[key] = true
		defer delete(visiting, key)
		parent := nodes[key]
		dependencies, problem := publishedDependencies(strings.SplitN(key, "@", 2)[0], parent.plan, parent.locked)
		if problem != nil {
			return problem
		}
		for _, dependency := range dependencies {
			childKey := dependency.name + "@" + dependency.version
			child, known := nodes[childKey]
			if !known {
				ref, problem := hub.ParseRef(dependency.name)
				if problem != nil {
					return problem
				}
				plan, problem := m.resolver.catalog.PackageDownloads(ctx, ref, dependency.version)
				if problem != nil {
					return problem
				}
				if plan.Release != dependency.version {
					return exit.New(exit.Conflict, "published child release changed")
				}
				matched := false
				for _, wheel := range plan.Downloads {
					if wheel.Kind == "project_wheel" && dependency.digests[wheel.Digest] && wheel.Version == dependency.version {
						matched = true
					}
				}
				if !matched {
					return exit.New(exit.Conflict, "published dependency wheel is not the committed callee implementation")
				}
				iface, problem := publishedPlanInterface(plan)
				if problem != nil {
					return problem
				}
				sub := request
				sub.Package, sub.Release, sub.InstallID = dependency.name, dependency.version, ""
				sub.Models = nil
				prepared, problem := connection.preparePublished(ctx, sub)
				if problem != nil {
					return problem
				}
				var set pb.PlacementSet
				if err := canonical.Unmarshal(prepared.PlacementSetCanonicalBytes, &set); err != nil || len(set.Placements) != 1 {
					return exit.New(exit.Conflict, "published child preparation needs one exact placement")
				}
				placement := set.Placements[0]
				if placement.GetPackage().GetPackage() != dependency.name || placement.GetPackage().GetRelease() != dependency.version || placement.Environment == nil || placement.PackageInterface == nil || !bytes.Equal(placement.PackageInterface.Digest, canonical.Digest(iface.Raw)) {
					return exit.New(exit.Conflict, "published child preparation changed its exact release")
				}
				_, digest, err := canonical.Identity(placement.Environment)
				if err != nil || !bytes.Equal(digest, placement.EnvironmentDigest) {
					return exit.New(exit.Conflict, "published child environment changed")
				}
				child = node{digest, plan, iface, prepared.LockedRequirements}
				nodes[childKey] = child
				capture.PublishedRevisions = append(capture.PublishedRevisions, &pb.PublishedPackageRevision{Package: placement.GetPackage(), Environment: placement.Environment, PackageInterface: placement.PackageInterface})
				addPublishedBindings(capture, digest, digest, iface, true)
				if problem := walk(childKey, depth+1); problem != nil {
					return problem
				}
			} else {
				matched := false
				for _, wheel := range child.plan.Downloads {
					if wheel.Kind == "project_wheel" && dependency.digests[wheel.Digest] {
						matched = true
					}
				}
				if !matched {
					return exit.New(exit.Conflict, "published parents disagree on a callee wheel")
				}
				if visiting[childKey] {
					return exit.New(exit.Validation, "published callable closure is cyclic")
				}
			}
			addPublishedBindings(capture, parent.digest, child.digest, child.iface, false)
		}
		return nil
	}
	if problem := walk(rootKey, 0); problem != nil {
		return problem
	}
	sort.Slice(capture.PublishedRevisions, func(i, j int) bool {
		_, a, _ := canonical.Identity(capture.PublishedRevisions[i].Environment)
		_, b, _ := canonical.Identity(capture.PublishedRevisions[j].Environment)
		return bytes.Compare(a, b) < 0
	})
	if connection.wireMinor >= pb.CapturedModelDefaultsWireMinor {
		// Root defaults are recorded by the caller after the entire binding inventory exists.
		for key, node := range nodes {
			if key != rootKey {
				m.resolver.captureDefaultRows(capture, strings.SplitN(key, "@", 2)[0], node.digest, node.iface, request.Rental, connection.publicOrigin)
			}
		}
	}
	return nil
}

func addPublishedBindings(capture *pb.MachineExecutionCapture, caller, callee []byte, iface *launch.PackageInterface, self bool) {
	for _, entry := range append(append([]launch.Entrypoint(nil), iface.Jobs...), iface.Entrypoints...) {
		if entry.Invocable == nil || entry.Internal && !self {
			continue
		}
		capture.Bindings = append(capture.Bindings, &pb.MachineCallableBinding{CallerRevisionDigest: caller, CalleeRevisionDigest: callee, InterfaceDigest: canonical.Digest(iface.Raw), Module: entry.Invocable.Module, Export: entry.Invocable.Export, Entrypoint: entry.Name})
	}
}

// The worker installs a published child directly. This does not change the
// user's active local package pin or create a client-side inference environment.
func (m *machineRuns) publishedChildPreparation(ctx context.Context, request records.Request) (*hub.PackageDownloadPlan, *launch.PackageInterface, []byte, string, *exit.Error) {
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	plan, problem := m.resolver.catalog.PackageDownloads(ctx, ref, request.Release)
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
	source, problem := packageInstallPlanFacts(m.context, ref, request.Release, plan, config, ifaceDoc, pyproject, uvLock)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	locked, problem := install.PublishedRequirements(ctx, source)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	detail, problem := m.resolver.catalog.PackageRelease(ctx, ref, request.Release)
	if problem != nil {
		return nil, nil, nil, "", problem
	}
	if _, problem := detail.Requirements(); problem != nil {
		return nil, nil, nil, "", problem
	}
	return &plan, iface, locked, detail.RequiresPython, nil
}
