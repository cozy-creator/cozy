package cli

import (
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
	if len(facts.ModelParams) > 0 || len(facts.WeightsOutputs) > 0 || len(facts.Outputs) > 0 {
		return out, "", exit.Named(exit.Unavailable, "child.artifact_binding_required", "artifact and model child calls require explicit native reference adoption")
	}
	out = orchestrator.Submission{Kind: "job", RetainWork: true, Package: install.Package, Entrypoint: binding.Entrypoint, Release: install.Version, InstallID: install.ID,
		PlanID: facts.DescriptorID, Payload: append([]byte(nil), payload...), Outputs: facts.Outputs, WeightsOutputs: facts.WeightsOutputs, NeedsAccelerator: facts.NeedsAccelerator, Org: parent.Org}
	out.ChildReusable = job.Invocable.Reusable
	if parent.Worker != "" {
		out.Worker, out.Rental, out.RentalRequired = parent.Worker, true, true
		out.LocalPackageDigest = binding.LocalRevisionDigest
	}
	identity, _ := json.Marshal(map[string]any{"local_revision_digest": binding.LocalRevisionDigest, "interface_digest": iface, "entrypoint": binding.Entrypoint, "module": module, "export": export})
	identity, err := canonical.NormalizeJCS(identity)
	if err != nil {
		return out, "", exit.Internalf("cannot encode frozen child identity: %s", err)
	}
	target, _ := canonical.Spell(canonical.Digest(identity))
	return out, target, nil
}
