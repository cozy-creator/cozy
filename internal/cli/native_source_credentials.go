package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
)

// NativeSourceCredential is called only after a private parent source call was accepted.
func (r *Resolver) NativeSourceCredential(operation string) string {
	switch operation {
	case "download_huggingface":
		return secret.NativeSourceCredential(r.cfg.HuggingFaceToken)
	case "download_civitai":
		return secret.NativeSourceCredential(r.cfg.CivitaiToken)
	default:
		return ""
	}
}

// NativeSourceEligible reads the captured descriptor, never source code or a mutable package pin.
func (r *Resolver) NativeSourceEligible(parent records.Request, operation string) *exit.Error {
	if operation == "convert_cozytensors" || operation == "source_files" || operation == "commit_file" {
		return nil
	}
	install, problem := r.store.Install(parent.InstallID)
	if problem != nil || install == nil {
		return exit.Named(exit.Conflict, "native.parent_install_absent", "source caller has no captured install")
	}
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir), install.PackageInterface)
	if problem != nil {
		return problem
	}
	job, problem := surface.Function(parent.Entrypoint)
	if problem != nil {
		return problem
	}
	if job.Invocable != nil && job.Invocable.Memoize {
		return exit.Named(exit.Conflict, "native.resolve_in_memoized_parent", "provider resolution must run before memo lookup; a memoized parent cannot download a provider source")
	}
	return nil
}
