package cli

import (
	"context"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
)

type captureReuse struct {
	caller string
	input  *install.CaptureInput
	pack   *packagepublish.Package
	layout home.Layout
	store  *records.Store
}

func prepareCaptureReuse(ctx *Context, pack *packagepublish.Package, layout home.Layout, store *records.Store) (*captureReuse, *exit.Error) {
	caller := pack.Tree
	if len(ctx.Inv.Args) > 0 && isScriptTarget(ctx.Inv.Args[0]) {
		caller = ctx.Inv.Args[0]
	}
	key, problem := install.CaptureCaller(caller)
	if problem != nil {
		return nil, problem
	}
	input, problem := install.CaptureIdentity(context.Background(), pack)
	if problem != nil {
		return nil, problem
	}
	return &captureReuse{caller: key, input: input, pack: pack, layout: layout, store: store}, nil
}

func (c *captureReuse) lookup() (*records.PackageInstall, *launch.PackageInterface, *exit.Error) {
	if c.input.Key == "" {
		return nil, nil, nil
	}
	pin, problem := c.store.CapturePin(c.caller)
	if problem != nil || pin == nil || pin.InputDigest != c.input.Key {
		return nil, nil, problem
	}
	inst, problem := c.store.Install(pin.InstallID)
	if problem != nil {
		return nil, nil, problem
	}
	if inst == nil || inst.SourceKind != "local" || inst.SourceRef != filepath.Join(inst.Dir, "source") ||
		inst.Python != c.input.Python || inst.Extra != c.input.Extra ||
		inst.Package != "local/"+c.pack.Name || inst.Version != c.pack.Release {
		return nil, nil, nil
	}
	if _, problem := install.BasePython(filepath.Join(inst.Dir, "venv")); problem != nil {
		return nil, nil, nil
	}
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(inst.Dir), inst.PackageInterface)
	if problem != nil {
		return nil, nil, nil
	}
	marker, err := os.ReadFile(filepath.Join(inst.Dir, "private-revision"))
	if err != nil || string(marker) != pin.RevisionDigest {
		return nil, nil, nil
	}
	root, problem := localpackage.Open(c.layout, *inst, pin.RevisionDigest)
	if problem != nil || root.PackageInterfaceDigest != surface.Digest {
		return nil, nil, nil
	}
	if problem := c.validateBindings(*inst, root); problem != nil {
		return nil, nil, nil
	}
	return inst, surface, nil
}

func (c *captureReuse) validateBindings(inst records.PackageInstall, root localpackage.Revision) *exit.Error {
	_, problem := localpackage.CaptureExecution(inst.ID, root, c.store.ChildBindings,
		func(id, digest string) (localpackage.Revision, *exit.Error) {
			child, problem := c.store.Install(id)
			if problem != nil {
				return localpackage.Revision{}, problem
			}
			if child == nil {
				return localpackage.Revision{}, exit.New(exit.Conflict, "captured dependency install disappeared")
			}
			return localpackage.Open(c.layout, *child, digest)
		})
	return problem
}

// complete runs after final interface substitutions and binding publication.
// It publishes no pin unless every revision exists and the original inputs and
// capture tools still match the ones selected before preparation.
func (c *captureReuse) complete(inst records.PackageInstall) *exit.Error {
	if c.input.Key == "" || inst.Python != c.input.Python || inst.Extra != c.input.Extra {
		return nil
	}
	root, problem := localpackage.Stage(context.Background(), c.layout, inst)
	if problem != nil {
		return problem
	}
	if root.PackageInterfaceDigest != inst.PackageInterface {
		return exit.New(exit.Conflict, "captured interface changed during preparation")
	}
	if problem := c.validateBindings(inst, root); problem != nil {
		return problem
	}
	after, problem := install.CaptureIdentity(context.Background(), c.pack)
	if problem != nil {
		return problem
	}
	if after.Source != c.input.Source || after.Key != c.input.Key {
		return exit.New(exit.Conflict, "local source, dependencies, or capture tools changed during preparation")
	}
	if problem := localpackage.RetainRevision(inst, root.Digest); problem != nil {
		return problem
	}
	prior, problem := c.store.ReplaceCapturePin(records.CapturePin{Caller: c.caller,
		InputDigest: c.input.Key, InstallID: inst.ID, RevisionDigest: root.Digest})
	if problem != nil {
		return problem
	}
	if prior != "" && prior != inst.ID {
		c.reclaim(prior, map[string]bool{})
	}
	return nil
}

func (c *captureReuse) reclaim(id string, seen map[string]bool) {
	if seen[id] || len(seen) >= 128 {
		return
	}
	seen[id] = true
	bindings, problem := c.store.ChildBindings(id)
	if problem != nil {
		return
	}
	_, _ = install.Reclaim(c.layout, c.store, id)
	for _, binding := range bindings {
		c.reclaim(binding.ChildInstallID, seen)
	}
}
