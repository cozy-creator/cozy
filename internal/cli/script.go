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
)

func scriptTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	layout := home.Paths(ctx.Cfg.Home)
	pack, problem := packagepublish.PrepareScript(context.Background(), ctx.Inv.Args[0], layout.Scripts(), commandNamespace(ctx))
	if problem != nil {
		return Target{}, nil, problem
	}
	target, surface, problem := localTarget(ctx, pack)
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

// directoryTarget runs an explicitly addressed authored package where it is. Named org/package
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
	author, problem := packagepublish.PrepareLocalFrom(directory)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer author.Close()
	target, surface, problem := localTarget(ctx, author)
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

// localTarget is the install a run of an authored tree uses: one that already read exactly
// these files, else one that reads them now. Either is the tree itself, as an editable
// install is: nothing is copied, and only the interface is read here.
func localTarget(ctx *Context, author *packagepublish.Package) (Target, *launch.PackageInterface, *exit.Error) {
	layout, store, _, problem := open(ctx.Cfg, false)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	// Runs share the claim; it keeps a sweep off the install until the run that owns it is
	// submitted.
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
	live, _, problem := author.SourceStats()
	if problem != nil {
		return Target{}, nil, problem
	}
	if held, surface := heldInstall(store, "local/"+author.Name, author.Release, author.Tree, live); held != nil {
		handedOff = true
		return Target{Package: held.Package, InstallID: held.ID, Release: held.Version, Snapshot: true,
			releaseCapture: writer.Unlock}, surface, nil
	}
	files, bytes, problem := author.SourceInventory()
	if problem != nil {
		return Target{}, nil, problem
	}
	var result *install.Result
	problem = packagePublishStage(ctx, "Reading local package", func() *exit.Error {
		result, problem = install.Run(layout, store, install.Request{Ref: install.Ref{Package: "local/" + author.Name}, Snapshot: true,
			Local: &install.LocalSource{Bytes: bytes, Files: files, Package: "local/" + author.Name, Release: author.Release, Tree: author.Tree}})
		return problem
	})
	if problem != nil {
		return Target{}, nil, problem
	}
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(result.Install.Dir))
	if problem != nil {
		_, _ = install.Reclaim(layout, store, result.Install.ID)
		return Target{}, nil, problem
	}
	handedOff = true
	return Target{Package: result.Install.Package, InstallID: result.Install.ID,
		Release: result.Install.Version, Snapshot: true, releaseCapture: writer.Unlock}, surface, nil
}

// heldInstall is an install of pkg@release from tree that read it as it is now, and its
// interface. The run's writer claim keeps it from collection until the run owns it.
func heldInstall(store *records.Store, pkg, release, tree string, live map[string]packagepublish.SourceStamp) (*records.PackageInstall, *launch.PackageInterface) {
	candidates, problem := store.SourceEnvironments(pkg, release)
	if problem != nil {
		return nil, nil
	}
	for i := range candidates {
		if candidates[i].SourceRef != tree || !packagepublish.SourceStatsUnchanged(candidates[i].Dir, live) {
			continue
		}
		if surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(candidates[i].Dir)); problem == nil {
			return &candidates[i], surface
		}
	}
	return nil, nil
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
