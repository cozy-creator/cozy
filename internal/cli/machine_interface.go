package cli

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/protobuf/proto"
)

func (r *Resolver) machineInterfacePath(request records.Request) (string, *exit.Error) {
	layout, problem := home.Open(r.cfg.Home)
	if problem != nil {
		return "", problem
	}
	return filepath.Join(layout.PublicationRoot(request.Org, request.ID), "package-interface.json"), nil
}

// Keep the already-resolved interface with the request's existing client
// custody files, so collection after reconnect needs no local package install.
func (r *Resolver) captureMachineInterface(request records.Request, surface *launch.PackageInterface) *exit.Error {
	path, problem := r.machineInterfacePath(request)
	if problem != nil {
		return problem
	}
	if raw, err := os.ReadFile(path); err == nil {
		if !bytes.Equal(raw, surface.Raw) {
			return exit.New(exit.Conflict, "captured public interface changed")
		}
		return nil
	} else if !os.IsNotExist(err) {
		return exit.Internalf("cannot read captured public interface: %s", err)
	}
	directory := filepath.Dir(path)
	if err := os.MkdirAll(directory, 0700); err != nil {
		return exit.Internalf("cannot retain public interface: %s", err)
	}
	file, err := os.CreateTemp(directory, ".interface-")
	if err != nil {
		return exit.Internalf("cannot stage public interface: %s", err)
	}
	defer os.Remove(file.Name())
	if _, err = file.Write(surface.Raw); err == nil {
		err = file.Sync()
	}
	if closeError := file.Close(); err == nil {
		err = closeError
	}
	if err == nil {
		err = os.Rename(file.Name(), path)
	}
	if err == nil {
		parent, openError := os.Open(directory)
		err = openError
		if err == nil {
			err = parent.Sync()
			parent.Close()
		}
	}
	if err != nil {
		return exit.Internalf("cannot commit captured public interface: %s", err)
	}
	return nil
}

func (r *Resolver) capturedResultInterface(request records.Request) (*launch.PackageInterface, *exit.Error) {
	link, problem := r.store.MachineExecution(request.ID)
	if problem != nil {
		return nil, problem
	}
	if (link == nil || len(link.Submission) == 0) && request.InstallID != "" {
		_, surface, problem := r.installPackageInterface(request.InstallID)
		return surface, problem
	}
	var submission pb.MachineExecutionSubmit
	var capture pb.MachineExecutionCapture
	if link == nil || proto.Unmarshal(link.Submission, &submission) != nil || canonical.Unmarshal(submission.CaptureCanonicalBytes, &capture) != nil {
		return nil, exit.New(exit.Conflict, "published result has no frozen interface identity")
	}
	var expected *pb.Ref
	local := request.LocalPackageDigest != ""
	if local {
		root, err := canonical.Spell(capture.RootRevisionDigest)
		if err != nil || root != request.LocalPackageDigest {
			return nil, exit.New(exit.Conflict, "local result differs from its captured root")
		}
		for _, revision := range capture.Revisions {
			_, digest, err := canonical.Identity(revision)
			if err == nil && bytes.Equal(digest, capture.RootRevisionDigest) &&
				revision.Package == request.Package && revision.Release == request.Release {
				expected = revision.PackageInterface
			}
		}
	}
	for _, revision := range capture.PublishedRevisions {
		if revision.GetPackage().GetPackage() != request.Package || revision.GetPackage().GetRelease() != request.Release {
			continue
		}
		_, digest, err := canonical.Identity(revision.Environment)
		if err == nil && bytes.Equal(digest, capture.RootRevisionDigest) {
			expected = revision.PackageInterface
		}
	}
	if expected == nil || expected.Length == 0 || expected.Length > 1<<20 {
		return nil, exit.New(exit.Conflict, "result omitted its captured interface")
	}
	digest, err := canonical.Spell(expected.Digest)
	if err != nil {
		return nil, exit.New(exit.Conflict, "published result changed its interface digest")
	}
	path, problem := r.machineInterfacePath(request)
	if problem != nil {
		return nil, problem
	}
	if _, err := os.Stat(path); os.IsNotExist(err) {
		var surface *launch.PackageInterface
		if local {
			layout, problem := home.Open(r.cfg.Home)
			if problem != nil {
				return nil, problem
			}
			// Recover only the exact retained revision, never a newly edited install
			// or a public repository for private source.
			source := filepath.Join(layout.LocalPackages, strings.TrimPrefix(request.LocalPackageDigest, "sha256:"), launch.PackageInterfaceFile)
			surface, problem = launch.ReadPackageInterface(source, digest)
			if problem != nil {
				// Before request-owned interface retention, an edited install could
				// reclaim this revision. Another verified copy of the exact digest
				// is sufficient; its current package version is not authority.
				installed, read := r.store.Installed()
				if read != nil {
					return nil, read
				}
				for _, candidate := range installed {
					if candidate.Package != request.Package || candidate.PackageInterface != digest {
						continue
					}
					if exact, read := launch.ReadPackageInterface(launch.PackageInterfacePath(candidate.Dir), digest); read == nil {
						surface = exact
						break
					}
				}
				if surface == nil {
					return nil, problem
				}
			}
		} else {
			// Older accepted requests can recover only byte-identical public metadata.
			ref, problem := hub.ParseRef(request.Package)
			if problem != nil {
				return nil, problem
			}
			ctx, cancel := hub.Context()
			defer cancel()
			detail, problem := r.catalog.PackageRelease(ctx, ref, request.Release)
			if problem != nil {
				return nil, problem
			}
			surface, problem = launch.DecodePackageInterface(detail.PackageInterface)
			if problem != nil {
				return nil, problem
			}
		}
		if surface.Digest != digest || uint64(len(surface.Raw)) != expected.Length {
			return nil, exit.New(exit.Conflict, "public result interface differs from the accepted capture")
		}
		if problem := r.captureMachineInterface(request, surface); problem != nil {
			return nil, problem
		}
	}
	surface, problem := launch.ReadPackageInterface(path, digest)
	if problem != nil {
		return nil, problem
	}
	if uint64(len(surface.Raw)) != expected.Length {
		return nil, exit.New(exit.Conflict, "captured public interface length changed")
	}
	entry, problem := surface.Function(request.Entrypoint)
	if problem != nil {
		return nil, problem
	}
	if entry.Kind != "job" || entry.DescriptorID != request.PlanID {
		return nil, exit.New(exit.Conflict, "published result changed its captured job declaration")
	}
	return surface, nil
}
