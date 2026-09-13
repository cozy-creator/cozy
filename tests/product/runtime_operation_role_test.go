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
	"github.com/cozy-creator/cozy/internal/packagepublish"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/runtimeoperation"
)

func TestBuiltinAvailabilityDoesNotStripAnOrdinaryJobsGPU(t *testing.T) {
	for _, application := range []string{"ordinary_gpu:app", packagepublish.ScriptApplication} {
		t.Run(application, func(t *testing.T) {
			layout, problem := home.Open(t.TempDir())
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			parent := cleanupTestInstall(layout, "1111111111111111", "1.0.0")
			child := cleanupTestInstall(layout, "2222222222222222", "1.0.0")
			parent.Closure = "cozy-runtime==0.16.8\ntorch==2.13.0"
			raw, err := canonical.NormalizeJCS([]byte(`{"format":"cozy.package.interface/1","application":"` + application + `","entrypoints":[],"jobs":[{"name":"run","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false}]}`))
			must(t, err)
			parent.PackageInterface, _ = canonical.Spell(canonical.Digest(raw))
			must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(parent.Dir)), 0700))
			must(t, os.WriteFile(launch.PackageInterfacePath(parent.Dir), raw, 0400))
			fatal(t, store.RecordInstall(parent))
			fatal(t, store.RecordInstall(child))
			fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: parent.ID, ChildInstallID: child.ID, InterfaceDigest: childDigest("a"), LocalRevisionDigest: childDigest("b"), Module: runtimeoperation.Module, Export: "quantize", Entrypoint: "quantize"}}))
			resolver := cli.NewResolver(store, config.Config{Home: layout.Root}, nil)
			jobs, problem := resolver.JobsInstall(parent.ID)
			fatal(t, problem)
			if len(jobs) != 1 || jobs[0].NeedsAccelerator != (application != packagepublish.ScriptApplication) {
				t.Fatal("builtin availability changed the wrong execution role")
			}
		})
	}
}
