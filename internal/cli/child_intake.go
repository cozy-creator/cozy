package cli

import (
	"context"
	"os"
	"path/filepath"
	"sort"

	"github.com/cozy-creator/cozy/internal/config"
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
	Package      *packagepublish.Package
	Bindings     []records.ChildBinding
	layout       home.Layout
	store        *records.Store
	created      []string
	staging      string
	ownedPackage bool
	prepared     *install.Result
}

func (i *childIntake) Finish(parentInstall string) *exit.Error {
	bindings := append([]records.ChildBinding(nil), i.Bindings...)
	for n := range bindings {
		bindings[n].ParentInstallID = parentInstall
	}
	if problem := i.store.RecordChildBindings(bindings); problem != nil {
		return problem
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
	source, files, size, problem := i.Package.SourceIdentity()
	if problem != nil {
		return nil, problem
	}
	result, problem := install.Run(i.layout, i.store, install.Request{Ref: install.Ref{Package: "local/" + i.Package.Name}, Snapshot: true,
		Local: &install.LocalSource{SourceDigest: source, Bytes: size, Files: files, Package: "local/" + i.Package.Name, Release: i.Package.Release, Tree: i.Package.Tree}})
	if problem == nil {
		i.prepared = result
	}
	return result, problem
}

func prepareChildIntake(ctx *Context, pack *packagepublish.Package, layout home.Layout, store *records.Store) (*childIntake, *exit.Error) {
	return prepareChildIntakeDepth(ctx, pack, layout, store, map[string]bool{}, 0)
}

func prepareChildIntakeDepth(ctx *Context, pack *packagepublish.Package, layout home.Layout, store *records.Store, stack map[string]bool, depth int) (*childIntake, *exit.Error) {
	if depth > 16 || stack[pack.Tree] {
		return nil, exit.New(exit.Validation, "private invocable dependency graph is cyclic or exceeds 16 levels")
	}
	stack[pack.Tree] = true
	defer delete(stack, pack.Tree)
	intake := &childIntake{Package: pack, layout: layout, store: store}
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
		if len(intake.Bindings) >= 32 {
			return fail(exit.New(exit.Validation, "unpublished parent exceeds 32 invocable dependency exports"))
		}
		dependency, problem := packagepublish.PrepareUnpublishedFrom(context.Background(), path, dependencies[name].Extras...)
		if problem != nil {
			return fail(problem)
		}
		stack[path] = true
		nested, problem := prepareChildIntakeDepth(ctx, dependency, layout, store, stack, depth+1)
		delete(stack, path)
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
		surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(result.Install.Dir), result.Install.PackageInterface)
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		var exports []launch.Entrypoint
		for _, job := range append(append([]launch.Entrypoint(nil), surface.Entrypoints...), surface.Jobs...) {
			if job.Invocable != nil {
				exports = append(exports, job)
			}
		}
		if len(exports) == 0 {
			nested.Close()
			dependency.Close()
			continue
		}
		revision, problem := localpackage.Stage(context.Background(), layout, result.Install)
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		if intake.staging == "" {
			if err := os.MkdirAll(layout.Tmp, 0o700); err != nil {
				nested.Close()
				dependency.Close()
				return fail(exit.Internalf("cannot create interface staging parent: %s", err))
			}
			var err error
			intake.staging, err = os.MkdirTemp(layout.Tmp, "child-interfaces-")
			if err != nil {
				nested.Close()
				dependency.Close()
				return fail(exit.Internalf("cannot stage child interfaces: %s", err))
			}
		}
		var projectWheel localpackage.File
		for _, file := range revision.Files {
			if file.Kind == "project" {
				projectWheel = file
			}
		}
		wheel, problem := launch.GenerateInterfaceWheel(context.Background(), result.Install, layout.Root, config.Frozen().Tool(), revision.Digest, projectWheel.Path, projectWheel.Digest, intake.staging)
		if problem != nil {
			nested.Close()
			dependency.Close()
			return fail(problem)
		}
		replacements[name] = wheel.Path
		for _, job := range exports {
			intake.Bindings = append(intake.Bindings, records.ChildBinding{InterfaceDigest: surface.Digest, Module: job.Invocable.Module, Export: job.Invocable.Export, ChildInstallID: result.Install.ID, LocalRevisionDigest: revision.Digest, Entrypoint: job.Name})
		}
		nested.Close()
		dependency.Close()
	}
	if len(replacements) > 0 {
		overlay, problem := packagepublish.WithChildInterfaces(context.Background(), pack, replacements)
		if problem != nil {
			return fail(problem)
		}
		intake.Package, intake.ownedPackage = overlay, true
	}
	if problem := intake.prepareWheelIntake(context.Background(), replacements); problem != nil {
		return fail(problem)
	}
	return intake, nil
}
