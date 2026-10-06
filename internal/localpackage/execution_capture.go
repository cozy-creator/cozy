package localpackage

import (
	"encoding/json"
	"sort"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
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
	document := &CaptureDocument{Format: "cozy.capture/1", RootInstallationID: rootInstall, InstalledPackages: []*CapturedPackage{}, Bindings: []*CapturedBinding{}}
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
		document.InstalledPackages = append(document.InstalledPackages, &CapturedPackage{InstallationID: current.ID, Package: current.Package, Release: current.Release, PackageInterface: current.PackageInterface})
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
			document.Bindings = append(document.Bindings, &CapturedBinding{CallerInstallationID: current.ID, CalleeInstallationID: child.ID, Module: row.Module, Export: row.Export, Entrypoint: row.Entrypoint})
			queue = append(queue, child)
		}
	}
	sort.Slice(document.InstalledPackages, func(i, j int) bool {
		return document.InstalledPackages[i].InstallationID < document.InstalledPackages[j].InstallationID
	})
	sort.Slice(document.Bindings, func(i, j int) bool {
		a, b := document.Bindings[i], document.Bindings[j]
		return a.CallerInstallationID+"\x00"+a.Module+"\x00"+a.Export < b.CallerInstallationID+"\x00"+b.Module+"\x00"+b.Export
	})
	var err error
	result.Canonical, result.Digest, err = captureIdentity(document)
	if err != nil {
		return ExecutionCapture{}, exit.Internalf("cannot encode accepted installation graph: %s", err)
	}
	if len(result.Canonical) > 1<<20 {
		return ExecutionCapture{}, exit.New(exit.Validation, "execution graph exceeds 1 MiB")
	}
	return result, nil
}

// CaptureDocument is this controller's immutable installation graph, not a machine RPC.
type CaptureDocument struct {
	Format             string             `json:"format"`
	RootInstallationID string             `json:"root_installation_id"`
	InstalledPackages  []*CapturedPackage `json:"installed_packages"`
	Bindings           []*CapturedBinding `json:"bindings"`
	ModelChoices       []*v1.ModelChoice  `json:"model_choices,omitempty"`
}
type CapturedPackage struct {
	InstallationID   string `json:"installation_id"`
	Package          string `json:"package"`
	Release          string `json:"release,omitempty"`
	PackageInterface []byte `json:"package_interface,omitempty"`
}
type CapturedBinding struct {
	CallerInstallationID string `json:"caller_installation_id"`
	CalleeInstallationID string `json:"callee_installation_id"`
	Module               string `json:"module,omitempty"`
	Export               string `json:"export,omitempty"`
	Entrypoint           string `json:"entrypoint,omitempty"`
}

func captureIdentity(document *CaptureDocument) ([]byte, []byte, error) {
	raw, err := json.Marshal(document)
	if err != nil {
		return nil, nil, err
	}
	raw, err = canonical.NormalizeJCS(raw)
	return raw, canonical.Digest(raw), err
}
func (capture *ExecutionCapture) SetModelChoices(choices []*v1.ModelChoice) error {
	var document CaptureDocument
	if err := json.Unmarshal(capture.Canonical, &document); err != nil {
		return err
	}
	document.ModelChoices = choices
	var err error
	capture.Canonical, capture.Digest, err = captureIdentity(&document)
	return err
}
