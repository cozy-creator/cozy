package cli

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// The same package name, even the same version, is a separate identity at each Hub.
// Exercise the daemon's catalog resolver, installed metadata and the CLI's warmed cache
// together, including authored model defaults that must not leak into the other Hub.
func TestSelectedHubScopesInstalledAndCachedPackageMetadata(t *testing.T) {
	for _, otherRelease := range []string{"1.0.0", "2.0.0"} {
		t.Run(otherRelease, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			if problem != nil {
				t.Fatal(problem)
			}
			store, problem := records.Open(layout.DB)
			if problem != nil {
				t.Fatal(problem)
			}
			defer store.Close()
			const pkg = "proof/shared"
			surface := func(name, model string) []byte {
				return []byte(`{"format":"cozy.package.interface/1","application":"shared:app","entrypoints":[{"name":"` + name +
					`","models":[{"class":"Model","path":"` + name + `.models.model","component_use":{},"default_ladder":[{"gpu":"*","lane":"proof/` + model +
					`@1.0.0/bf16"}]}],"request":{"fields":[]},"result":{"fields":[]}}],"jobs":[]}`)
			}
			a, b := surface("from_a", "model-a"), surface("from_b", "model-b")
			served := 0
			source := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/packages/"+pkg+"/releases/"+otherRelease {
					t.Errorf("selected source received unexpected path %s", r.URL.Path)
					http.NotFound(w, r)
					return
				}
				served++
				_ = json.NewEncoder(w).Encode(map[string]any{"release": map[string]string{"release": otherRelease},
					"package_interface": json.RawMessage(b), "execution_requirements": []string{"cozy-runtime>=0.19.0"}})
			}))
			defer source.Close()
			const first = "https://first.example"
			installed := records.PackageInstall{ID: "first-source", Package: pkg, Major: 1, Version: "1.0.0", Hub: first,
				SourceKind: "tensorhub", SourceRef: pkg + "@1.0.0", Verified: true, Dir: layout.InstallDir("first-source")}
			if err := os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0o700); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(launch.PackageInterfacePath(installed.Dir), a, 0o444); err != nil {
				t.Fatal(err)
			}
			if _, problem := store.Activate(installed); problem != nil {
				t.Fatal(problem)
			}
			cfg := config.Config{Home: root, HubURL: source.URL}
			resolver := NewResolver(store, cfg)
			ref, _ := hub.ParseRef(pkg)
			_, selected, problem := resolver.releaseInterface(source.URL, ref, otherRelease)
			if problem != nil {
				t.Fatal(problem)
			}
			entry, problem := selected.Function("from_b")
			if problem != nil || entry.Models[0].Default("proof").Model != "proof/model-b" || served != 1 {
				t.Fatalf("selected Hub reused foreign installed metadata: entry=%+v error=%v reads=%d", entry, problem, served)
			}
			keepReleaseInterface(root, first, pkg, "1.0.0", a, nil)
			keepNewestRelease(root, first, pkg, "1.0.0")
			keepReleaseInterface(root, source.URL, pkg, otherRelease, b, nil)
			keepNewestRelease(root, source.URL, pkg, otherRelease)
			for _, scope := range []struct{ origin, release, function, model string }{
				{source.URL, otherRelease, "from_b", "proof/model-b"},
				{first, "1.0.0", "from_a", "proof/model-a"},
				{source.URL, otherRelease, "from_b", "proof/model-b"},
			} {
				ctx := &Context{Cfg: cfg.ForHub(scope.origin), Out: io.Discard, Err: io.Discard,
					Inv: &Invocation{Args: []string{pkg + "/" + scope.function}}}
				target, metadata, problem := invocationTarget(ctx)
				if problem != nil {
					t.Fatal(problem)
				}
				if target.lease != nil {
					target.lease.Release()
				}
				entry, problem := metadata.Function(scope.function)
				if problem != nil || target.Release != scope.release || entry.Models[0].Default("proof").Model != scope.model || ctx.Cfg.HubURL != scope.origin {
					t.Fatalf("source %s mixed release, interface or model defaults: target=%+v entry=%+v error=%v", scope.origin, target, entry, problem)
				}
			}
		})
	}
}
