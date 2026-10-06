package cli

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Two Hubs may publish the same package and release with different interfaces. A local
// install at one Hub must not replace the other Hub's retained description or closure.
func TestReleaseInterfacesStayWithTheirHubAfterRestart(t *testing.T) {
	var reads atomic.Int32
	server := func() *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			reads.Add(1)
			http.Error(w, "the retained release should need no Hub read", http.StatusServiceUnavailable)
		}))
	}
	a, b := server(), server()
	defer a.Close()
	defer b.Close()
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
	const pkg, release = "proof/shared", "1.0.0"
	ref, problem := hub.ParseRef(pkg)
	if problem != nil {
		t.Fatal(problem)
	}
	interfaceFor := func(application string) []byte {
		return []byte(fmt.Sprintf(`{"format":"cozy.package.interface/1","application":%q,"entrypoints":[],"jobs":[]}`, application))
	}
	for _, held := range []struct{ origin, application, dependency string }{
		{a.URL, "from_a", "dependency-a==1"}, {b.URL, "from_b", "dependency-b==2"},
	} {
		keepReleaseInterface(root, held.origin, pkg, release, interfaceFor(held.application), []string{held.dependency})
		// The machine's description has no dependency closure: refreshing it must retain
		// the closure belonging to this Hub alone.
		keepReleaseInterface(root, held.origin, pkg, release, interfaceFor(held.application), nil)
		keepNewestRelease(root, held.origin, pkg, release)
	}
	installed := records.PackageInstall{ID: "installed_a", Package: pkg, Version: release,
		SourceKind: "tensorhub", Hub: a.URL, Dir: filepath.Join(layout.Installs, "installed_a"), Closure: "installed-dependency==3"}
	path := launch.PackageInterfacePath(installed.Dir)
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, interfaceFor("installed_a"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, problem := store.Activate(installed); problem != nil {
		t.Fatal(problem)
	}
	// A new resolver has no process cache; all answers come from durable retained inputs.
	resolver := NewResolver(store, config.Config{Home: root, HubURL: a.URL})
	for _, want := range []struct{ origin, application, dependency string }{
		{b.URL, "from_b", "dependency-b==2"}, {"", "installed_a", "installed-dependency==3"},
	} {
		requirements, surface, problem := resolver.releaseInterface(want.origin, ref, release)
		if problem != nil || surface == nil || surface.Application != want.application || !slices.Equal(requirements, []string{want.dependency}) {
			t.Fatalf("release at %q answered %v, %v, %v; want %s with %s", want.origin, requirements, surface, problem, want.application, want.dependency)
		}
	}
	for _, want := range []struct{ origin, application string }{{a.URL, "from_a"}, {b.URL, "from_b"}} {
		gotRelease, surface := keptNewestRelease(root, want.origin, pkg)
		if gotRelease != release || surface == nil || surface.Application != want.application {
			t.Fatalf("newest at %s answered %s, %v; want %s from %s", want.origin, gotRelease, surface, release, want.application)
		}
	}
	if got := reads.Load(); got != 0 {
		t.Fatalf("retained resolution read the Hub %d times", got)
	}
}
