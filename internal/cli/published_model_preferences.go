package cli

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/install"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// PublishedChildModels size a rental for a published composition before any machine is known:
// each callee slot's authored default at the lane its machine's rules would serve. A run never
// reads them. Everything they read is fixed by the exact release and the run's bits, so they
// are kept under them and read once; an answer that missed a card or lane is read again next
// time.
func (r *Resolver) PublishedChildModels(command *Context, request records.Request) ([]records.ModelRef, *exit.Error) {
	if !request.ComposesChildren() || request.Release == "" {
		return nil, nil
	}
	kept := ""
	if identity, err := json.Marshal([]any{"child-models/3", command.Cfg.HubURL, request.Package, request.Release,
		request.Entrypoint, request.Bits, client(command).CredentialIdentity()}); err == nil {
		name := sha256.Sum256(identity)
		kept = filepath.Join(home.Paths(command.Cfg.Home).Root, "releases", "children", hex.EncodeToString(name[:16])+".json")
	}
	var models []records.ModelRef
	if raw, err := os.ReadFile(kept); kept != "" && err == nil && json.Unmarshal(raw, &models) == nil {
		return models, nil
	}
	slots, problem := r.publishedChildSlots(command, request)
	if problem != nil {
		return nil, problem
	}
	var refs []string
	for _, slot := range slots {
		if ref := slot.slot.DefaultRef(packageOwner(slot.pkg, []launch.Slot{slot.slot}, r.namespace)); ref != "" {
			refs = append(refs, ref)
		}
	}
	cards, complete := readCards(command, refs), true
	for _, slot := range slots {
		model, problem := r.childSizedDefault(cards, slot.pkg, slot.entry, slot.slot, request.Bits)
		if problem != nil {
			// A slot with no default stays optional; a card or lane the Hub lacked is asked again.
			complete = complete && problem.ErrName() == "model.default_absent"
			continue
		}
		models = append(models, model)
	}
	if raw, err := json.Marshal(models); complete && kept != "" && err == nil &&
		os.MkdirAll(filepath.Dir(kept), 0o700) == nil && os.WriteFile(kept+".tmp", raw, 0o600) == nil {
		_ = os.Rename(kept+".tmp", kept)
	}
	return models, nil
}

// childSlot is one model slot a composition's closure declares.
type childSlot struct {
	pkg, entry string
	slot       launch.Slot
}

// publishedChildSlots are the model slots of every callable the release's locked closure
// publishes, except the request's own: its slots are already selected.
func (r *Resolver) publishedChildSlots(command *Context, request records.Request) ([]childSlot, *exit.Error) {
	root := request.Package + "@" + request.Release
	ctx, cancel := hub.Context()
	defer cancel()
	seen, visiting := map[string]bool{}, map[string]bool{}
	var slots []childSlot
	var walk func(records.Request, int) *exit.Error
	walk = func(selected records.Request, depth int) *exit.Error {
		key := selected.Package + "@" + selected.Release
		if visiting[key] || depth > 16 || len(seen) > 128 {
			return exit.New(exit.Validation, "published model preference closure is cyclic or exceeds its bound")
		}
		if seen[key] {
			return nil
		}
		seen[key], visiting[key] = true, true
		defer delete(visiting, key)
		plan, iface, locked, problem := r.publishedChildPreparation(ctx, command, selected)
		if problem != nil {
			return problem
		}
		for _, entry := range iface.Entrypoints {
			if entry.Invocable == nil || entry.Internal && key != root || key == root && entry.Name == request.Entrypoint {
				continue
			}
			for _, slot := range entry.Models {
				slots = append(slots, childSlot{pkg: selected.Package, entry: entry.Name, slot: slot})
			}
		}
		lock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
		if problem != nil {
			return problem
		}
		children, problem := install.PublishedDependencies(selected.Package, lock.Bytes, locked)
		if problem != nil {
			return problem
		}
		for _, child := range children {
			next := selected
			next.Package, next.Release = child.Package, child.Version
			next.Models = nil
			if problem := walk(next, depth+1); problem != nil {
				return problem
			}
		}
		return nil
	}
	return slots, walk(request, 0)
}

func publishedPlanInterface(plan hub.PackageDownloadPlan) (*launch.PackageInterface, *exit.Error) {
	document, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, problem
	}
	return launch.DecodePackageInterface(document.Bytes)
}

// publishedChildPreparation is one published release's plan, interface and locked
// closure, read to size a rental for a workflow's children before any machine is known.
func (r *Resolver) publishedChildPreparation(ctx context.Context, command *Context, request records.Request) (*hub.PackageDownloadPlan, *launch.PackageInterface, []byte, *exit.Error) {
	ref, problem := hub.ParseRef(request.Package)
	if problem != nil {
		return nil, nil, nil, problem
	}
	plan, problem := r.catalog(request.Hub).PackageDownloads(ctx, ref, request.Release)
	if problem != nil {
		return nil, nil, nil, problem
	}
	iface, problem := publishedPlanInterface(plan)
	if problem != nil {
		return nil, nil, nil, problem
	}
	config, problem := exactPackageInstallDocument("package.toml", plan.PackageConfig)
	if problem != nil {
		return nil, nil, nil, problem
	}
	ifaceDoc, problem := exactPackageInstallDocument("package interface", plan.PackageInterface)
	if problem != nil {
		return nil, nil, nil, problem
	}
	pyproject, problem := exactPackageInstallDocument("pyproject.toml", plan.Pyproject)
	if problem != nil {
		return nil, nil, nil, problem
	}
	uvLock, problem := exactPackageInstallDocument("uv.lock", plan.UVLock)
	if problem != nil {
		return nil, nil, nil, problem
	}
	source, problem := packageInstallPlanFacts(command, ref, request.Release, plan, config, ifaceDoc, pyproject, uvLock)
	if problem != nil {
		return nil, nil, nil, problem
	}
	locked, problem := install.PublishedRequirements(source)
	if problem != nil {
		return nil, nil, nil, problem
	}
	return &plan, iface, locked, nil
}
