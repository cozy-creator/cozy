package cli

import (
	"context"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/canonical"
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
	env := config.Frozen().Tool()
	bin, problem := hostruntime.Path(env)
	if problem != nil {
		return nil, problem
	}
	runtime := launch.RuntimeCLI{Bin: bin, Dir: p.Tree, Home: p.Root, Env: env}
	output := make([]string, 0, len(projects))
	for _, source := range projects {
		body, err := os.ReadFile(source)
		if err != nil {
			return nil, exit.Internalf("cannot read callable project wheel: %s", err)
		}
		digest, _ := canonical.Spell(canonical.Digest(body))
		stage, err := os.MkdirTemp(p.Root, "caller-")
		if err != nil {
			return nil, exit.Internalf("cannot stage callable project wheel: %s", err)
		}
		generated, problem := runtime.InterfaceWheel(ctx, surface, p.Name, p.Release, digest, source, digest, stage)
		if problem != nil {
			return nil, problem
		}
		if filepath.Base(source) != generated.Filename {
			return nil, exit.New(exit.Validation, "callable publication changed its wheel platform")
		}
		output = append(output, generated.Path)
	}
	return output, nil
}
