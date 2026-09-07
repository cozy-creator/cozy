package cli

import (
	"bytes"
	"encoding/json"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
)

// ResolvePrivateChild resolves only the immutable interface binding captured by
// the parent's intake. Current pins, mutable source trees and package-supplied
// install identities never participate in this execution authority.
func (r *Resolver) ResolvePrivateChild(parent records.Request, iface, module, export string, payload []byte) (orchestrator.Submission, string, *exit.Error) {
	var out orchestrator.Submission
	binding, problem := r.store.ChildBinding(parent.InstallID, iface, module, export)
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
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir), iface)
	if problem != nil {
		return out, "", problem
	}
	job, problem := surface.Function(binding.Entrypoint)
	if problem != nil {
		return out, "", problem
	}
	if job.Kind != "job" || job.Invocable == nil || job.Invocable.Module != module || job.Invocable.Export != export {
		return out, "", exit.Named(exit.Conflict, "child.export_changed", "the captured implementation does not expose the exact invocable job")
	}
	if job.Invocable.Reusable {
		for _, capability := range job.Invocable.Capabilities {
			if capability == "egress" || capability == "secrets" {
				return out, "", exit.Named(exit.Conflict, "child.reusable_effect", "a reusable operation cannot carry external egress or secret capabilities")
			}
		}
	}
	parentInstall, problem := r.store.Install(parent.InstallID)
	if problem != nil || parentInstall == nil {
		return out, "", exit.Named(exit.Conflict, "child.parent_install_absent", "the parent snapshot is unavailable")
	}
	parentSurface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(parentInstall.Dir), parentInstall.PackageInterface)
	if problem != nil {
		return out, "", problem
	}
	parentJob, problem := parentSurface.Function(parent.Entrypoint)
	if problem != nil {
		return out, "", problem
	}
	if parentJob.Invocable != nil && parentJob.Invocable.Reusable && !job.Invocable.Reusable {
		return out, "", exit.Named(exit.Conflict, "child.impure_dependency", "a reusable parent operation cannot call a non-reusable dependency")
	}
	if problem := launch.ValidatePayload(install.Package, job, payload); problem != nil {
		return out, "", problem
	}
	if _, problem := r.LocalRevision(install.ID, binding.LocalRevisionDigest); problem != nil {
		return out, "", problem
	}
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
		return out, "", exit.Named(exit.Conflict, "child.export_changed", "the captured child has no matching job facts")
	}
	if len(facts.Outputs) > len(facts.WeightsOutputs) {
		return out, "", exit.Named(exit.Unavailable, "child.artifact_binding_required", "artifact and model child calls require explicit native reference adoption")
	}
	var arguments map[string]json.RawMessage
	if json.Unmarshal(payload, &arguments) != nil {
		return out, "", exit.New(exit.Validation, "child input is not an object")
	}
	models := make([]orchestrator.ModelRef, 0, len(job.Models))
	for _, slot := range job.Models {
		artifact, problem := records.DecodeModelArtifact(arguments[slot.Param])
		if problem != nil {
			return out, "", problem
		}
		if artifact == nil {
			return out, "", exit.Named(exit.Conflict, "child.model_unbound", "child model %s needs an exact retained ModelArtifact", slot.Param)
		}
		weights, problem := r.store.ArtifactOutput(*artifact)
		if problem != nil {
			return out, "", problem
		}
		producer, problem := r.store.RequestRow(artifact.ProducerRequestID)
		if problem != nil || producer == nil || producer.ReuseScope != parent.ReuseScope || producer.Worker != parent.Worker {
			return out, "", exit.Named(exit.Conflict, "child.artifact_scope", "model artifact does not belong to the parent's retained scope and store")
		}
		held, problem := r.store.ArtifactHasCustody(producer.ID, weights.Attempt, weights.OutputSlot, parent.ReuseScope)
		if problem != nil {
			return out, "", problem
		}
		if !held {
			return out, "", exit.Named(exit.Conflict, "child.artifact_released", "model artifact no longer has retained native custody")
		}
		models = append(models, orchestrator.ModelRef{Package: install.Package, Slot: slot.Param, BindingPath: slot.Path, Model: producer.ID + "/" + artifact.OutputSlot, Manifest: artifact.Manifest.Digest, ManifestLength: artifact.Manifest.Length})
	}
	out = orchestrator.Submission{Kind: "job", RetainWork: true, Package: install.Package, Entrypoint: binding.Entrypoint, Release: install.Version, InstallID: install.ID,
		PlanID: facts.DescriptorID, Payload: append([]byte(nil), payload...), Outputs: facts.Outputs, WeightsOutputs: facts.WeightsOutputs, NeedsAccelerator: facts.NeedsAccelerator, Org: parent.Org}
	out.Models = models
	out.ChildReusable = job.Invocable.Reusable
	resultSchema, _ := json.Marshal(job.Result)
	out.ChildArtifacts = bytes.Contains(resultSchema, []byte(`"input":"model"`))
	if parent.Worker != "" {
		out.Worker, out.Rental, out.RentalRequired = parent.Worker, true, true
		out.LocalPackageDigest = binding.LocalRevisionDigest
	}
	identity, _ := json.Marshal(map[string]any{"local_revision_digest": binding.LocalRevisionDigest, "interface_digest": iface, "entrypoint": binding.Entrypoint, "module": module, "export": export, "models": models})
	identity, err := canonical.NormalizeJCS(identity)
	if err != nil {
		return out, "", exit.Internalf("cannot encode frozen child identity: %s", err)
	}
	target, _ := canonical.Spell(canonical.Digest(identity))
	return out, target, nil
}
