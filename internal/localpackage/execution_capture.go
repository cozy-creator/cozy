package localpackage

import (
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// Capture fixes the accepted invocation's installation graph, not package content.
type ExecutionCapture struct {
	Canonical     []byte
	Digest        []byte // ordinary request-document replay identity
	Installations []Installation
}

func CaptureExecution(rootInstall string, root Installation,
	bindings func(string) ([]records.ChildBinding, *exit.Error),
	resolve func(string, string) (Installation, *exit.Error),
) (ExecutionCapture, *exit.Error) {
	if rootInstall == "" || root.ID != rootInstall {
		return ExecutionCapture{}, exit.New(exit.Validation, "execution requires its accepted root installation")
	}
	document := &pb.MachineExecutionCapture{RootInstallationId: rootInstall}
	queue := []Installation{root}
	seen := map[string]bool{}
	result := ExecutionCapture{}
	for len(queue) > 0 {
		current := queue[0]
		queue = queue[1:]
		if seen[current.ID] {
			continue
		}
		seen[current.ID] = true
		if len(seen) > 128 {
			return ExecutionCapture{}, exit.New(exit.Validation, "execution exceeds 128 installations")
		}
		document.InstalledPackages = append(document.InstalledPackages, &pb.InstalledPackage{InstallationId: current.ID, Package: current.Package, Release: current.Release, PackageInterface: current.PackageInterface})
		result.Installations = append(result.Installations, current)
		rows, problem := bindings(current.ID)
		if problem != nil {
			return ExecutionCapture{}, problem
		}
		for _, row := range rows {
			if row.ParentInstallID != current.ID {
				return ExecutionCapture{}, exit.New(exit.Validation, "callable binding belongs to another installation")
			}
			child := current
			if row.ChildInstallID != current.ID {
				child, problem = resolve(row.ChildInstallID, row.ChildInstallID)
				if problem != nil {
					return ExecutionCapture{}, problem
				}
			}
			document.Bindings = append(document.Bindings, &pb.MachineCallableBinding{CallerInstallationId: current.ID, CalleeInstallationId: child.ID, Module: row.Module, Export: row.Export, Entrypoint: row.Entrypoint})
			queue = append(queue, child)
		}
	}
	sort.Slice(document.InstalledPackages, func(i, j int) bool {
		return document.InstalledPackages[i].InstallationId < document.InstalledPackages[j].InstallationId
	})
	sort.Slice(document.Bindings, func(i, j int) bool {
		a, b := document.Bindings[i], document.Bindings[j]
		return a.CallerInstallationId+"\x00"+a.Module+"\x00"+a.Export < b.CallerInstallationId+"\x00"+b.Module+"\x00"+b.Export
	})
	var err error
	result.Canonical, result.Digest, err = canonical.Identity(document)
	if err != nil {
		return ExecutionCapture{}, exit.Internalf("cannot encode accepted installation graph: %s", err)
	}
	if len(result.Canonical) > 1<<20 {
		return ExecutionCapture{}, exit.New(exit.Validation, "execution graph exceeds 1 MiB")
	}
	return result, nil
}
