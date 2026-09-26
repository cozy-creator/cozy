package producttest

import (
	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestNativeDownloadsCannotHideResolutionInsideMemoizedParents(t *testing.T) {
	for _, memo := range []bool{false, true} {
		layout, problem := home.Open(t.TempDir())
		fatal(t, problem)
		store, problem := records.Open(layout.DB)
		fatal(t, problem)
		install := cleanupTestInstall(layout, "1111111111111111", "1.0.0")
		text := `{"format":"cozy.package.interface/1","application":"private_ops:app","entrypoints":[],"jobs":[{"name":"compute","models":[],"request":{"fields":[]},"result":{"fields":[]},"publishes":false,"weights_outputs":[],"invocable":{"context":"ctx","module":"private_ops","export":"compute","parameters":[],"defaults":{},"type_names":{},"enum_members":{},"memoize":false,"capabilities":[]}}]}`
		if memo {
			text = strings.Replace(text, `"memoize":false`, `"memoize":true`, 1)
		}
		raw, err := canonical.NormalizeJCS([]byte(text))
		must(t, err)

		must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(install.Dir)), 0o700))
		must(t, os.WriteFile(launch.PackageInterfacePath(install.Dir), raw, 0o444))
		fatal(t, store.RecordInstall(install))
		resolver := cli.NewResolver(store, config.Config{Home: layout.Root}, nil)
		parent := records.Request{InstallID: install.ID, Entrypoint: "compute"}
		for _, operation := range []string{"download_huggingface", "download_civitai"} {
			problem = resolver.NativeSourceEligible(parent, operation)
			if memo {
				if problem == nil || problem.ErrName() != "native.resolve_in_memoized_parent" {
					t.Fatal("memoized parent accepted hidden provider resolution")
				}
			} else {
				fatal(t, problem)
			}
		}
		fatal(t, resolver.NativeSourceEligible(parent, "convert_cozytensors"))
		store.Close()
	}
}
