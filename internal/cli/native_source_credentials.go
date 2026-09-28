package cli

import (
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// SourceCredentials are the configured provider credentials, keyed by provider, that a
// native source call presents. They travel only in a sent command or submission, never a record.
func (r *Resolver) SourceCredentials() []*pb.SourceCredential {
	var credentials []*pb.SourceCredential
	for _, row := range []struct {
		provider pb.NativeSourceOperation
		token    secret.Value
	}{
		{pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_HUGGINGFACE, r.cfg.HuggingFaceToken},
		{pb.NativeSourceOperation_NATIVE_SOURCE_OPERATION_CIVITAI, r.cfg.CivitaiToken},
	} {
		if row.token.Present() {
			credentials = append(credentials, &pb.SourceCredential{Provider: row.provider, Credential: secret.NativeSourceCredential(row.token)})
		}
	}
	return credentials
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
	surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(install.Dir))
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
