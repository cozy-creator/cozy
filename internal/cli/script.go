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
	"github.com/cozy-creator/cozy/internal/records"
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
func snapshotTarget(ctx *Context, pack *packagepublish.Package, remote ...*records.PackageInstall) (Target, *launch.PackageInterface, *exit.Error) {
	layout, store, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	handedOff := false
	defer func() {
		if !handedOff {
			writer.Unlock()
		}
	}()
	if err := os.MkdirAll(layout.Tmp, 0700); err != nil {
		return Target{}, nil, exit.Internalf("cannot create invocation staging: %s", err)
	}
	stage, err := os.MkdirTemp(layout.Tmp, "invocation-source-")
	if err != nil {
		return Target{}, nil, exit.Internalf("cannot stage invocation source: %s", err)
	}
	defer os.RemoveAll(stage)
	frozen, problem := packagepublish.SnapshotSource(pack.Tree, filepath.Join(stage, "source"))
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
	handedOff = true
	return Target{Package: result.Install.Package, InstallID: result.Install.ID,
		Release: result.Install.Version, Snapshot: true, releaseCapture: writer.Unlock}, surface, nil
}

func snapshotLocalJob(ctx *Context, target Target) (Target, *launch.PackageInterface, *exit.Error) {
	_, store, writer, problem := open(ctx.Cfg, true)
	if problem != nil {
		return Target{}, nil, problem
	}
	defer store.Close()
	current, problem := exactInvocationInstall(ctx, target)
	writer.Unlock()
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
	if ctx.Inv.Bool("--rental-only") || ctx.Inv.Value("--rental") != "" {
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

func releaseSnapshotReader(target Target) {
	if target.releaseCapture != nil {
		target.releaseCapture()
	}
}

func isScriptTarget(value string) bool {
	return strings.EqualFold(filepath.Ext(value), ".py") &&
		(explicitPackageDirectory(value) || !strings.ContainsAny(value, `/\`))
}
