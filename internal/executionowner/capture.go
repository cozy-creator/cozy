package executionowner

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/localpackage"
	"github.com/cozy-creator/cozy/internal/records"
)

type Exported struct {
	Raw      []byte
	Packages map[string]localpackage.Revision
}

// Export reads already-frozen source captures before the root has been offered.
// The caller uploads these exact files before asking Host to accept ownership.
func Export(layout home.Layout, store *records.Store, request records.Request) (*Exported, *exit.Error) {
	if request.ParentRequestID != "" || request.Ordinal != 0 || request.InstallID == "" || request.LocalPackageDigest == "" {
		return nil, invalid("execution export requires a fresh captured root")
	}
	capsule := Capsule{Format: CapsuleFormat, Root: Root{Revision: request.LocalPackageDigest,
		Entrypoint: request.Entrypoint, Input: append(json.RawMessage(nil), request.Payload...), IdempotencyKey: request.IdemKey}, Bindings: []Binding{}}
	out := &Exported{Packages: map[string]localpackage.Revision{}}
	type selected struct{ install, revision string }
	queue := []selected{{request.InstallID, request.LocalPackageDigest}}
	seen := map[string]bool{}
	for len(queue) > 0 {
		next := queue[0]
		queue = queue[1:]
		if seen[next.install] {
			continue
		}
		seen[next.install] = true
		if len(seen) > 1024 {
			return nil, invalid("captured callable inventory exceeds its bound")
		}
		installed, problem := store.Install(next.install)
		if problem != nil || installed == nil {
			return nil, invalid("captured source install is unavailable")
		}
		revision, problem := localpackage.Open(layout, *installed, next.revision)
		if problem != nil {
			return nil, problem
		}
		directory := filepath.Join(layout.LocalPackages, strings.TrimPrefix(revision.Digest, "sha256:"))
		raw, err := os.ReadFile(filepath.Join(directory, "revision.json"))
		if err != nil {
			return nil, invalid("captured revision document is unavailable")
		}
		surface, problem := launch.ReadPackageInterface(launch.PackageInterfacePath(installed.Dir), revision.PackageInterfaceDigest)
		if problem != nil {
			return nil, problem
		}
		if _, exists := out.Packages[revision.Digest]; !exists {
			capsule.Packages = append(capsule.Packages, Package{Revision: raw, Interface: surface.Raw,
				Capture: Capture{Python: installed.Python, Platform: installed.Platform, Closure: installed.Closure, Extra: installed.Extra}})
			out.Packages[revision.Digest] = revision
		}
		bindings, problem := store.ChildBindings(installed.ID)
		if problem != nil {
			return nil, problem
		}
		for _, binding := range bindings {
			childRevision := binding.LocalRevisionDigest
			if binding.ChildInstallID == installed.ID && childRevision == "" {
				childRevision = revision.Digest
			}
			capsule.Bindings = append(capsule.Bindings, Binding{ParentRevision: revision.Digest,
				ChildRevision: childRevision, InterfaceDigest: binding.InterfaceDigest,
				Module: binding.Module, Export: binding.Export, Entrypoint: binding.Entrypoint})
			queue = append(queue, selected{binding.ChildInstallID, childRevision})
		}
	}
	sort.Slice(capsule.Packages, func(i, j int) bool {
		return string(capsule.Packages[i].Revision) < string(capsule.Packages[j].Revision)
	})
	sort.Slice(capsule.Bindings, func(i, j int) bool {
		a, _ := json.Marshal(capsule.Bindings[i])
		b, _ := json.Marshal(capsule.Bindings[j])
		return string(a) < string(b)
	})
	var problem *exit.Error
	out.Raw, problem = Encode(capsule)
	return out, problem
}
