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
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/scratch"
)

func scriptTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	pack, problem := packagepublish.PrepareScript(context.Background(), ctx.Inv.Args[0], commandNamespace(ctx))
	if problem != nil {
		return Target{}, nil, problem
	}
	defer pack.Close()
	target, surface, problem := snapshotTarget(ctx, pack)
	if problem != nil {
		return Target{}, nil, problem
	}
	callables := append(append([]launch.Entrypoint{}, surface.Entrypoints...), surface.Jobs...)
	if len(callables) != 1 {
		reclaimSnapshot(ctx, target)
		return Target{}, nil, exit.Named(exit.Validation, "script_entrypoint_count",
			"script must expose exactly one main function; found %d", len(callables)).
			WithRemedy("define main() or main(ctx); ordinary helper functions are allowed")
	}
	target.Function = callables[0].Name
	// Defaults enter the existing model.<parameter>=<ref> admission path. The
	// user's explicit term wins; neither metadata nor scripts resolve credentials.
	names := make([]string, 0, len(pack.ScriptModels))
	for name := range pack.ScriptModels {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		prefix := "model." + name + "="
		fullPrefix := "model." + target.Function + ".models." + name + "="
		overridden := false
		for _, arg := range ctx.Inv.Args[1:] {
			if strings.HasPrefix(arg, prefix) || strings.HasPrefix(arg, fullPrefix) {
				overridden = true
				break
			}
		}
		if !overridden {
			ctx.Inv.Args = append(ctx.Inv.Args, prefix+pack.ScriptModels[name])
		}
	}
	return target, surface, nil
}

// directoryTarget captures an explicitly addressed authored package. Named org/package
// references never enter this path, even when a matching local directory exists.
func directoryTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	directory := filepath.Clean(ctx.Inv.Args[0])
	function := ""
	if info, err := os.Stat(directory); err != nil || !info.IsDir() {
		parent := filepath.Dir(directory)
		if info, err := os.Stat(filepath.Join(parent, "package.toml")); err == nil && info.Mode().IsRegular() {
			function, directory = filepath.Base(directory), parent
		} else {
			return Target{}, nil, exit.Named(exit.NotFound, "local_package_not_found",
				"local package directory %q does not exist", ctx.Inv.Args[0]).
				WithRemedy("use ./project or ./project/function for an authored package")
		}
	}
	author, problem := packagepublish.AuthorTree(directory)
	if problem != nil {
		return Target{}, nil, problem
	}
	// An unchanged tree runs the snapshot it already has, before any lock or build.
	target, surface, problem := capturedTarget(ctx, author, func() (*packagepublish.Package, *exit.Error) {
		if _, locked := author.Files["uv.lock"]; locked {
			return packagepublish.PrepareLocalFrom(directory)
		}
		// A one-off local run needs no publication lock and does not write one
		// into the author's working tree. Resolve in the existing owned copy.
		return packagepublish.PrepareUnpublishedFrom(context.Background(), directory, commandNamespace(ctx))
	})
	if problem != nil {
		return Target{}, nil, problem
	}
	target.Function = function
	if target.Function == "" {
		if names := surface.PublicNames(); len(names) == 1 {
			target.Function = names[0]
		}
	}
	return target, surface, nil
}

// snapshotTarget uses the ordinary installer while keeping the user's editable
// pin unchanged. Both the program and its environment are owned by the run.
func snapshotTarget(ctx *Context, pack *packagepublish.Package, remote ...*records.PackageInstall) (Target, *launch.PackageInterface, *exit.Error) {
	return capturedTarget(ctx, pack, func() (*packagepublish.Package, *exit.Error) { return pack, nil }, remote...)
}

// capturedTarget is the run's snapshot of the authored tree: the one already captured from
// exactly these files, else prepare's package captured now and recorded against them.
func capturedTarget(ctx *Context, author *packagepublish.Package, prepare func() (*packagepublish.Package, *exit.Error), remote ...*records.PackageInstall) (Target, *launch.PackageInterface, *exit.Error) {
	layout, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	// A capture writes only its own new installs, so captures share the claim; it keeps
	// a sweep off the unreferenced snapshot until the run that owns it is submitted.
	writer, problem := home.WaitWriter(layout, true, ctx.Err)
	if problem != nil {
		return Target{}, nil, problem
	}
	handedOff := false
	defer func() {
		if !handedOff {
			writer.Unlock()
		}
	}()
	// An unchanged tree (its files and its local dependencies) runs the snapshot it already
	// has: the same installation, which its machine already holds.
	live, _, problem := author.SourceStats()
	if problem != nil {
		return Target{}, nil, problem
	}
	if held, surface := heldSnapshot(store, "local/"+author.Name, author.Release, live); held != nil {
		handedOff = true
		return Target{Package: held.Package, InstallID: held.ID, Release: held.Version, Snapshot: true,
			releaseCapture: writer.Unlock}, surface, nil
	}
	stage, problem := scratch.Temp(layout.Tmp, "invocation-source-")
	if problem != nil {
		return Target{}, nil, problem
	}
	defer stage.Release()
	pack, problem := prepare()
	if problem != nil {
		return Target{}, nil, problem
	}
	if pack != author {
		defer pack.Close()
	}
	frozen, problem := packagepublish.SnapshotSource(context.Background(), pack.Tree, filepath.Join(stage.Path, "source"))
	if problem != nil {
		return Target{}, nil, problem
	}
	defer frozen.Close()
	pack = frozen
	var intake *childIntake
	var result *install.Result
	problem = packagePublishStage(ctx, "Capturing local package and dependencies", func() *exit.Error {
		var problem *exit.Error
		intake, problem = prepareChildIntake(ctx, pack, layout, store, remote...)
		if problem != nil {
			return problem
		}
		result, problem = intake.Install()
		return problem
	})
	if intake != nil {
		defer intake.Close()
	}
	if problem != nil {
		return Target{}, nil, problem
	}
	if problem := intake.Finish(result.Install.ID); problem != nil {
		_, _ = install.Reclaim(layout, store, result.Install.ID)
		return Target{}, nil, problem
	}
	raw, err := os.ReadFile(launch.PackageInterfacePath(result.Install.Dir))
	if err != nil {
		_, _ = install.Reclaim(layout, store, result.Install.ID)
		return Target{}, nil, exit.Internalf("cannot read unpublished invocation interface: %s", err)
	}
	surface, problem := launch.DecodePackageInterface(raw)
	if problem != nil {
		_, _ = install.Reclaim(layout, store, result.Install.ID)
		return Target{}, nil, problem
	}
	_ = packagepublish.RecordInvocationSource(result.Install.Dir, live)
	handedOff = true
	return Target{Package: result.Install.Package, InstallID: result.Install.ID,
		Release: result.Install.Version, Snapshot: true, releaseCapture: writer.Unlock}, surface, nil
}

// heldSnapshot is a retained snapshot of pkg@release captured from exactly this live tree, and
// its interface. The capture's writer claim keeps it from collection until the run owns it.
func heldSnapshot(store *records.Store, pkg, release string, live map[string]packagepublish.SourceStamp) (*records.PackageInstall, *launch.PackageInterface) {
	candidates, problem := store.SourceEnvironments(pkg, release)
	if problem != nil {
		return nil, nil
	}
	for i := range candidates {
		if !packagepublish.InvocationSourceUnchanged(candidates[i].Dir, live) {
			continue
		}
		raw, err := os.ReadFile(launch.PackageInterfacePath(candidates[i].Dir))
		if err != nil {
			continue
		}
		if surface, problem := launch.DecodePackageInterface(raw); problem == nil {
			return &candidates[i], surface
		}
	}
	return nil, nil
}

func snapshotLocalJob(ctx *Context, target Target) (Target, *launch.PackageInterface, *exit.Error) {
	_, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	current, problem := exactInvocationInstall(ctx, target)
	if problem != nil {
		return Target{}, nil, problem
	}
	project := current.ProjectDir
	if current.SourceKind == "local" && current.SourceRef != "" {
		project = current.SourceRef
	}
	if project == "" {
		project = current.SourceRef
	}
	pack, problem := packagepublish.PrepareLocalFrom(project)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer pack.Close()
	// An editable install records its own exports. A job also needs the App
	// dependency graph, including raw source dependencies such as Qwen, so use
	// the same intake that a one-off client script uses before accepting it.
	var reusable *records.PackageInstall
	if remoteRun(ctx) {
		reusable = current
	}
	frozen, surface, problem := snapshotTarget(ctx, pack, reusable)
	if problem != nil {
		return Target{}, nil, problem
	}
	frozen.Function = target.Function
	return frozen, surface, nil
}

func reclaimSnapshot(ctx *Context, target Target) {
	releaseSnapshotReader(target)
	if !target.Snapshot || target.InstallID == "" {
		return
	}
	layout, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return
	}
	defer store.Close()
	// Accepted requests keep the install through the existing in-use predicate.
	// A refusal/describe-only command has no owner and can reclaim it immediately.
	_, _ = install.Reclaim(layout, store, target.InstallID)
}

func releaseSnapshotReader(target Target) {
	if target.releaseCapture != nil {
		target.releaseCapture()
	}
	target.lease.Release()
}

// isScriptTarget answers whether a run target names a Python script. A package ref never
// ends in .py except a two-part org/name.py (a function is an identifier), so every other
// .py target is a script, and org/name.py is one when that file exists.
func isScriptTarget(value string) bool {
	if !strings.EqualFold(filepath.Ext(value), ".py") {
		return false
	}
	if explicitPackageDirectory(value) || strings.Count(filepath.ToSlash(value), "/") != 1 {
		return true
	}
	return scriptFileExists(value)
}

// ambiguousScriptTarget is a script target that could also be read as org/name.py.
func ambiguousScriptTarget(value string) bool {
	return !explicitPackageDirectory(value) && strings.Count(filepath.ToSlash(value), "/") == 1 && scriptFileExists(value)
}

func scriptFileExists(value string) bool {
	info, err := os.Stat(value)
	return err == nil && info.Mode().IsRegular()
}
