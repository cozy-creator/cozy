package cli

import (
	"context"
	"os"
	"path/filepath"
	"sort"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

// childIntake is bounded installation staging under the caller's existing
// install-writer lock. Finish attaches its exact dependency facts to the parent's
// normal install before that snapshot can be submitted for execution.
type childIntake struct {
	Package           *packagepublish.Package
	Bindings          []records.ChildBinding
	layout            home.Layout
	store             *records.Store
	created           []string
	staging           string
	ownedPackage      bool
	prepared          *install.Result
	remoteEnvironment *records.PackageInstall
	remoteCapture     bool
	namespace         packagepublish.NamespaceSource
}

func (i *childIntake) Finish(parentInstall string) *exit.Error {
	bindings := append([]records.ChildBinding(nil), i.Bindings...)
	for n := range bindings {
		bindings[n].ParentInstallID = parentInstall
	}
	if problem := i.store.RecordChildBindings(bindings); problem != nil {
		return problem
	}
	if len(bindings) > 0 && i.prepared != nil && i.prepared.Install.ID == parentInstall {
		if _, problem := localpackage.StagePrepared(context.Background(), i.layout, i.prepared.Install, i.Package); problem != nil {
			return problem
		}
	}
	if i.prepared != nil && i.prepared.Install.ID == parentInstall {
		i.prepared = nil
	}
	return nil
}

func (i *childIntake) Close() {
	if i.prepared != nil {
		_, _ = install.Reclaim(i.layout, i.store, i.prepared.Install.ID)
	}
	if i.ownedPackage {
		i.Package.Close()
	}
	_ = os.RemoveAll(i.staging)
	for _, id := range i.created {
		_, _ = install.Reclaim(i.layout, i.store, id)
	}
}

func (i *childIntake) Install() (*install.Result, *exit.Error) {
	if i.prepared != nil {
		return i.prepared, nil
	}
	files, size, problem := i.Package.SourceInventory()
	if problem != nil {
		return nil, problem
	}
	result, problem := install.Run(i.layout, i.store, install.Request{Ref: install.Ref{Package: "local/" + i.Package.Name}, Snapshot: true,
		RemoteEnvironment: i.remoteEnvironment,
		RemoteCapture:     i.remoteCapture,
		Local:             &install.LocalSource{Bytes: size, Files: files, Package: "local/" + i.Package.Name, Release: i.Package.Release, Tree: i.Package.Tree, Namespace: i.namespace}})
	if problem == nil {
		i.prepared = result
	}
	return result, problem
}

func prepareChildIntake(ctx *Context, pack *packagepublish.Package, layout home.Layout, store *records.Store, remote ...*records.PackageInstall) (*childIntake, *exit.Error) {
	var environment *records.PackageInstall
	if len(remote) > 0 {
		environment = remote[0]
	}
	return prepareChildIntakeGraph(ctx, pack, layout, store, map[string]bool{}, environment)
}

func prepareChildIntakeGraph(ctx *Context, pack *packagepublish.Package, layout home.Layout, store *records.Store, stack map[string]bool, remote *records.PackageInstall) (*childIntake, *exit.Error) {
	if stack[pack.Tree] {
		return nil, exit.New(exit.Validation, "private invocable dependency graph is cyclic at %s", pack.Tree)
	}
	stack[pack.Tree] = true
	defer delete(stack, pack.Tree)
	intake := &childIntake{Package: pack, layout: layout, store: store, remoteEnvironment: remote,
		remoteCapture: remoteRun(ctx),
		namespace:     commandNamespace(ctx)}
	fail := func(problem *exit.Error) (*childIntake, *exit.Error) { intake.Close(); return nil, problem }
	dependencies, problem := packagepublish.LocalDependencySelections(pack.Tree)
	if problem != nil {
		return fail(problem)
	}
	names := make([]string, 0, len(dependencies))
	for name := range dependencies {
		names = append(names, name)
	}
	sort.Strings(names)
	replacements := map[string]string{}
	for _, name := range names {
		path := dependencies[name].Path
		if stack[path] {
			return fail(exit.New(exit.Validation, "private invocable dependency graph is cyclic"))
		}
		if info, err := os.Stat(filepath.Join(path, "package.toml")); err != nil || !info.Mode().IsRegular() {
			continue
		}
		dependency, problem := packagepublish.PrepareUnpublishedFrom(context.Background(), path, commandNamespace(ctx), dependencies[name].Extras...)
		if problem != nil {
			return fail(problem)
		}
		nested, problem := prepareChildIntakeGraph(ctx, dependency, layout, store, stack, nil)
		if problem != nil {
			dependency.Close()
			return fail(problem)
		}
		result, problem := nested.Install()
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		intake.created = append(intake.created, result.Install.ID)
		if problem := nested.Finish(result.Install.ID); problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(result.Install.Dir))
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		var exports []launch.Entrypoint
		for _, job := range append(append([]launch.Entrypoint(nil), surface.Entrypoints...), surface.Jobs...) {
			if job.Invocable != nil && !job.Internal {
				exports = append(exports, job)
			}
		}
		if len(exports) == 0 {
			nested.Close()
			dependency.Close()
			continue
		}
		_, problem = localpackage.StagePrepared(context.Background(), layout, result.Install, dependency)
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		replacements[name] = path
		for _, job := range exports {
			intake.Bindings = append(intake.Bindings, records.ChildBinding{Module: job.Invocable.Module, Export: job.Invocable.Export, ChildInstallID: result.Install.ID, Entrypoint: job.Name})
		}
		nested.Close()
		dependency.Close()
	}
	if problem := intake.prepareWheelIntake(context.Background(), replacements); problem != nil {
		return fail(problem)
	}
	return intake, nil
}
