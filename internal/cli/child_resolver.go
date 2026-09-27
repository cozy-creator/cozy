package cli

import (
	"encoding/json"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/inputasset"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

// ResolveUnpublishedChild resolves only the immutable interface binding captured by
// the parent's intake. Current pins, mutable source trees and package-supplied
// install identities never participate in this execution authority.
func (r *Resolver) ResolveUnpublishedChild(parent records.Request, module, export string, payload []byte) (orchestrator.Submission, string, *exit.Error) {
	var out orchestrator.Submission
	binding, problem := r.store.ChildBinding(parent.InstallID, module, export)
	if problem != nil {
		return out, "", problem
	}
	if binding == nil {
		return out, "", exit.Named(exit.Conflict, "child.binding_absent", "the parent did not capture this invocable dependency")
	}
	install, problem := r.store.Install(binding.ChildInstallID)
	if problem != nil || install == nil {
		return out, "", exit.Named(exit.Conflict, "child.install_absent", "the captured child implementation is unavailable")
	}
	revision := binding.ChildInstallID
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir))
	if problem != nil {
		return out, "", problem
	}
	job, problem := surface.Function(binding.Entrypoint)
	if problem != nil {
		return out, "", problem
	}
	if job.Internal && (binding.ChildInstallID != parent.InstallID || revision != parent.LocalInstallationID) {
		return out, "", exit.Named(exit.Conflict, "callable_internal", "internal functions require an admitted parent from the same package revision")
	}
	if (job.Kind != "job" && job.Kind != "entrypoint") || job.Invocable == nil || job.Invocable.Module != module || job.Invocable.Export != export {
		return out, "", exit.Named(exit.Conflict, "child.export_changed", "the captured implementation does not expose the exact managed callable")
	}
	if job.Invocable.Memoize {
		for _, capability := range job.Invocable.Capabilities {
			if capability == "egress" || capability == "secrets" {
				return out, "", exit.Named(exit.Conflict, "child.memoized_effect", "a memoized operation cannot carry external egress or secret capabilities")
			}
		}
	}
	parentInstall, problem := r.store.Install(parent.InstallID)
	if problem != nil || parentInstall == nil {
		return out, "", exit.Named(exit.Conflict, "child.parent_install_absent", "the parent snapshot is unavailable")
	}
	parentSurface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(parentInstall.Dir))
	if problem != nil {
		return out, "", problem
	}
	parentJob, problem := parentSurface.Function(parent.Entrypoint)
	if problem != nil {
		return out, "", problem
	}
	if parentJob.Invocable != nil && parentJob.Invocable.Memoize && !job.Invocable.Memoize {
		return out, "", exit.Named(exit.Conflict, "child.impure_dependency", "a memoized parent operation cannot call a dependency that does not opt into memoization")
	}
	var arguments map[string]json.RawMessage
	if job.Kind == "entrypoint" {
		payload, arguments, problem = records.ServingCallArguments(payload)
		if problem != nil {
			return out, "", problem
		}
		if len(arguments) != len(job.Models) {
			return out, "", exit.Named(exit.Conflict, "child.model_unbound", "serving model arguments differ from declared slots")
		}
	} else if json.Unmarshal(payload, &arguments) != nil {
		return out, "", exit.New(exit.Validation, "child input is not an object")
	}
	if problem := launch.ValidatePayload(install.Package, job, payload); problem != nil {
		return out, "", problem
	}
	if revision != "" {
		if _, problem := r.LocalInstallation(install.ID, revision); problem != nil {
			return out, "", problem
		}
	}
	out = orchestrator.Submission{Kind: "serving", Package: install.Package, Entrypoint: binding.Entrypoint, Release: install.Version, InstallID: install.ID,
		Payload: append([]byte(nil), payload...), Outputs: launch.AssetPaths(job.Result),
		NeedsAccelerator: launch.AcceleratorRequired(strings.Split(install.Closure, "\n")), Org: parent.Org}
	if job.Kind == "job" {
		jobs, problem := r.JobsInstall(install.ID)
		if problem != nil {
			return out, "", problem
		}
		var facts *launch.JobFacts
		for i := range jobs {
			if jobs[i].Name == binding.Entrypoint {
				facts = &jobs[i]
				break
			}
		}
		if facts == nil {
			return out, "", exit.Named(exit.Conflict, "child.export_changed", "captured child has no matching job facts")
		}
		out.Kind, out.RetainWork, out.PlanID = "job", true, facts.DescriptorID
		out.Outputs, out.WeightsOutputs, out.NeedsAccelerator = facts.Outputs, facts.WeightsOutputs, facts.NeedsAccelerator
	}
	models := make([]orchestrator.ModelRef, 0, len(job.Models))
	var defaulted []int
	for _, slot := range job.Models {
		if _, present := arguments[slot.Param]; job.Kind == "entrypoint" && !present {
			return out, "", exit.Named(exit.Conflict, "child.model_unbound", "serving model arguments differ from declared slots")
		}
		artifact, problem := records.DecodeModelArtifact(arguments[slot.Param])
		if problem != nil {
			return out, "", problem
		}
		if artifact == nil {
			// No retained artifact: the granting host selects the slot itself, from the
			// owner's binding or the callee's authored default (cl-210). The caller named
			// its own callable, never a model.
			selected, problem := r.childModelLadder(parent.Hub, install.Package, binding.Entrypoint, slot)
			if problem != nil {
				return out, "", problem
			}
			defaulted = append(defaulted, len(models))
			models = append(models, selected)
			continue
		}
		source, problem := r.store.ResolveArtifactSource(*artifact)
		if problem != nil {
			return out, "", problem
		}
		held, problem := r.store.ArtifactSourceAllowed(parent, *source)
		if problem != nil {
			return out, "", problem
		}
		if !held {
			return out, "", exit.Named(exit.Conflict, "child.artifact_scope", "model artifact was not received with retained custody in this private store")
		}

		models = append(models, orchestrator.ModelRef{Package: install.Package, Slot: slot.Param, BindingPath: slot.Path, Model: artifact.ProducerRequestID + "/" + artifact.OutputSlot, Manifest: artifact.Manifest.Digest, ManifestLength: artifact.Manifest.Length})
	}
	if len(defaulted) > 0 {
		accelerator, count, problem := r.childAccelerator(parent)
		if problem != nil {
			return out, "", problem
		}
		pinned, _, ok := rental.Pin(models, accelerator, count)
		if !ok {
			return out, "", exit.Named(exit.Conflict, "child.model_rung_absent",
				"%s %s has no authored group for %d× %s (ladders: %s)", install.Package, binding.Entrypoint,
				count, orNone(accelerator), strings.Join(rental.Ladder(models), "; ")).
				WithRemedy("declare a rung for this accelerator, or run the composition on a machine its ladder names")
		}
		for _, i := range defaulted {
			if pinned[i], problem = r.childManifest(parent.Hub, pinned[i]); problem != nil {
				return out, "", problem
			}
		}
		models = pinned
	}
	out.Models = models
	received, problem := r.store.ReceivedByteAssets(parent.ID)
	if problem != nil {
		return out, "", problem
	}
	parentAssets := append(append([]records.AssetBinding(nil), parent.Assets...), received...)
	out.Assets, problem = launch.InheritChildAssets(job, payload, parentAssets)
	if problem != nil {
		return out, "", problem
	}
	out.ChildReusable = job.Invocable.Memoize
	out.ChildArtifacts = len(launch.ModelArtifactPaths(job.Result)) > 0 || len(out.Outputs) > len(out.WeightsOutputs)
	out.LocalInstallationID = revision
	if parent.Worker != "" {
		out.Worker, out.Rental, out.RentalRequired = parent.Worker, true, true
		out.RequestedRental = parent.RequestedRental
	}
	identity, _ := json.Marshal(map[string]any{"installation_id": revision, "entrypoint": binding.Entrypoint, "module": module, "export": export})
	identity, err := canonical.NormalizeJCS(identity)
	if err != nil {
		return out, "", exit.Internalf("cannot encode frozen child identity: %s", err)
	}
	target, _ := canonical.Spell(canonical.Digest(identity))
	return out, target, nil
}

func (r *Resolver) CapturedArtifactPaths(request records.Request) ([][]string, *exit.Error) {
	install, problem := r.store.Install(request.InstallID)
	if problem != nil || install == nil {
		return nil, exit.Named(exit.Conflict, "child.install_absent", "captured artifact result schema is unavailable")
	}
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir))
	if problem != nil {
		return nil, problem
	}
	job, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	return launch.ModelArtifactPaths(job.Result), nil
}

// PrivateRentalNeedsAccelerator sizes the rental for its captured children while
// keeping the parent itself on the separate CPU orchestration slot.
func (r *Resolver) PrivateRentalNeedsAccelerator(request records.Request) (bool, *exit.Error) {
	if request.NeedsAccelerator || request.InstallID == "" || !request.ComposesChildren() {
		return request.NeedsAccelerator, nil
	}
	queue := []string{request.InstallID}
	seen := map[string]bool{}
	needed := false
	for len(queue) > 0 {
		install := queue[0]
		queue = queue[1:]
		if seen[install] {
			continue
		}
		seen[install] = true
		bindings, problem := r.store.ChildBindings(install)
		if problem != nil {
			return false, problem
		}
		for _, binding := range bindings {
			child, problem := r.store.Install(binding.ChildInstallID)
			if problem != nil {
				return false, problem
			}
			if child == nil {
				return false, exit.Named(exit.Conflict, "child.install_absent", "captured child implementation is unavailable for rental sizing")
			}
			surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(child.Dir))
			if problem != nil {
				return false, problem
			}
			job, problem := surface.Function(binding.Entrypoint)
			if problem != nil {
				return false, problem
			}
			if (job.Kind != "job" && job.Kind != "entrypoint") || job.Invocable == nil || job.Invocable.Module != binding.Module || job.Invocable.Export != binding.Export {
				return false, exit.Named(exit.Conflict, "child.export_changed", "captured child has no exact callable for rental sizing")
			}
			// The callable's own declaration, else the closure JobsInstall reads; no
			// package code needs importing again merely to choose a machine class.
			needed = needed || job.NeedsAccelerator(strings.Split(child.Closure, "\n"))
			queue = append(queue, binding.ChildInstallID)
		}
	}
	return needed, nil
}

// CapturedByteOutputBound uses the captured result schema, with the same finite
// asset bounds as ordinary inputs. A manifest entry cannot invent a result field.
func (r *Resolver) CapturedByteOutputBound(request records.Request, path, mediaType string) (int64, *exit.Error) {
	surface, problem := r.capturedResultInterface(request)
	if problem != nil {
		return 0, problem
	}
	entry, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return 0, problem
	}
	spec, ok := launch.ResultAssetSpec(entry, path)
	if !ok || !spec.AcceptsMediaType(mediaType) {
		return 0, exit.New(exit.Validation, "native byte output is not a declared asset field")
	}
	maximum := spec.MaxBytes
	if maximum <= 0 {
		maximum = inputasset.MaxBytes
	}
	return maximum, nil
}

func (r *Resolver) CapturedRetainedResultFields(request records.Request) (map[string]bool, *exit.Error) {
	surface, problem := r.capturedResultInterface(request)
	if problem != nil {
		return nil, problem
	}
	entry, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	return launch.RetainedAssetPaths(entry), nil
}
