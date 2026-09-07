package producttest

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
)

func TestPrivateRentalIncludesChildGPUWithoutGrantingItToParent(t *testing.T) {
	for _, gpu := range []bool{false, true} {
		t.Run(map[bool]string{false: "cpu_child", true: "gpu_child"}[gpu], func(t *testing.T) {
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			parent := cleanupTestInstall(layout, "1111111111111111", "1.0.0")
			child := cleanupTestInstall(layout, "2222222222222222", "1.0.0")
			child.Closure = "cozy-runtime==0.3.0"
			if gpu {
				child.Closure += "\ntorch==2.13.0"
			}
			raw, err := canonical.NormalizeJCS([]byte(`{"format":"cozy.package.interface/1","application":"private_ops:app","entrypoints":[],"jobs":[{"name":"compute","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false,"weights_outputs":[],"invocable":{"context":"ctx","module":"private_ops","export":"compute","parameters":[],"defaults":{},"type_names":{},"enum_members":{},"memoize":true,"capabilities":[]}}]}`))
			must(t, err)
			child.PackageInterface, _ = canonical.Spell(canonical.Digest(raw))
			must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(child.Dir)), 0o700))
			must(t, os.WriteFile(launch.PackageInterfacePath(child.Dir), raw, 0o444))
			fatal(t, store.RecordInstall(parent))
			fatal(t, store.RecordInstall(child))
			fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: parent.ID, ChildInstallID: child.ID, InterfaceDigest: child.PackageInterface, LocalRevisionDigest: childDigest("b"), Module: "private_ops", Export: "compute", Entrypoint: "compute"}}))
			request := records.Request{InstallID: parent.ID, Kind: "job", NeedsAccelerator: false, RetainWork: true, Rental: true}
			resolver := cli.NewResolver(store, config.Config{Home: layout.Root}, nil)
			requiresGPU, problem := resolver.PrivateRentalNeedsAccelerator(request)
			fatal(t, problem)
			if requiresGPU != gpu || request.NeedsAccelerator {
				t.Fatalf("rental class=%t parent GPU grant=%t", requiresGPU, request.NeedsAccelerator)
			}
			sku, _, ok := rental.Choose(offeredSKUs(), requiresGPU, rental.Constraints{})
			wanted := "cpu"
			if gpu {
				wanted = "rtx-4090"
			}
			if !ok || sku.Name != wanted {
				t.Fatalf("frozen child requires rental %s, got %+v", wanted, sku)
			}
		})
	}
}
