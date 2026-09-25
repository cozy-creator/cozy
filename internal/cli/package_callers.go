package cli

import (
	"context"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/hostruntime"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// A callable publication has one source-preserving wheel: App retains the serving
// implementation, while normal Python imports expose its generated managed callers.
// Generate before Tensorhub seals the wheel/Environment identity, never at install.
func publishedCallableWheels(ctx context.Context, p *packagepublish.Package) ([]string, *exit.Error) {
	surface, projects := p.PackageInterface, p.ProjectWheels
	iface, problem := launch.ReadPackageInterface(surface, "")
	if problem != nil {
		return nil, problem
	}
	callable := false
	for _, entry := range append(append([]launch.Entrypoint(nil), iface.Entrypoints...), iface.Jobs...) {
		callable = callable || entry.Invocable != nil && !entry.Internal
	}
	if !callable {
		return projects, nil
	}
	if problem := packagepublish.ValidateCallerRuntime(p, "0.18.21"); problem != nil {
		return nil, problem
	}
	env := config.Frozen().Tool()
	bin, problem := hostruntime.Path(env)
	if problem != nil {
		return nil, problem
	}
	return launch.CallerInterfaceWheels(ctx, bin, p.Tree, p.Root, env, surface, p.Name, p.Release, projects)
}
