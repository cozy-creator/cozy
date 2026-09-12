package cli

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func scriptTarget(ctx *Context) (Target, *launch.PackageInterface, *exit.Error) {
	pack, problem := packagepublish.PrepareScript(context.Background(), ctx.Inv.Args[0])
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

// snapshotTarget uses the ordinary installer while keeping the user's editable
// pin unchanged. Both the program and its environment are owned by the run.
func snapshotTarget(ctx *Context, pack *packagepublish.Package) (Target, *launch.PackageInterface, *exit.Error) {
	layout, store, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	defer writer.Unlock()
	var intake *childIntake
	var result *install.Result
	problem = packagePublishStage(ctx, "Preparing local script environment", func() *exit.Error {
		var problem *exit.Error
		intake, problem = prepareChildIntake(ctx, pack, layout, store)
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
	return Target{Package: result.Install.Package, InstallID: result.Install.ID,
		Release: result.Install.Version, Snapshot: true}, surface, nil
}

func snapshotLocalJob(ctx *Context, target Target) (Target, *launch.PackageInterface, *exit.Error) {
	current, problem := exactInvocationInstall(ctx, target)
	if problem != nil {
		return Target{}, nil, problem
	}
	pack, problem := packagepublish.PrepareLocalFrom(current.SourceRef)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer pack.Close()
	frozen, surface, problem := snapshotTarget(ctx, pack)
	frozen.Function = target.Function
	return frozen, surface, problem
}

func reclaimSnapshot(ctx *Context, target Target) {
	if !target.Snapshot || target.InstallID == "" {
		return
	}
	layout, store, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return
	}
	defer store.Close()
	defer writer.Unlock()
	// Accepted requests keep the install through the existing in-use predicate.
	// A refusal/describe-only command has no owner and can reclaim it immediately.
	_, _ = install.Reclaim(layout, store, target.InstallID)
}

func isScriptTarget(value string) bool {
	return strings.EqualFold(filepath.Ext(value), ".py") &&
		(explicitPackageDirectory(value) || !strings.ContainsAny(value, `/\`))
}
