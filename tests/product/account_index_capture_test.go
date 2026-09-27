package producttest

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

// A capture writes the account index into its owned copy with explicit scope: only the
// dependencies the source names from `tensorhub` are looked up there, and none falls through
// to the public index. The authored project never changes.
func TestAccountIndexCaptureKeepsExplicitScope(t *testing.T) {
	for _, missing := range []bool{false, true} {
		t.Run(fmt.Sprint("missing-owned=", missing), func(t *testing.T) {
			var mu sync.Mutex
			var requests []string
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				mu.Lock()
				requests = append(requests, r.URL.Path)
				mu.Unlock()
				http.NotFound(w, r)
			}))
			defer server.Close()
			dependencies := []string{"cozy-index-fixture>=1,<2"}
			if !missing {
				dependencies = append(dependencies, "unmapped-fixture>=1,<2")
			}
			metadata := fixturePyproject(dependencies...) + fmt.Sprintf(`
[tool.uv.sources]
cozy-index-fixture = { index = "tensorhub" }
[[tool.uv.index]]
name = "fixture-pypi"
url = %q
default = true
`, server.URL+"/pypi/")
			tree := fixtureTree(t, metadata, minimalFixtureLock)
			namespace := func() (packagepublish.Namespace, *exit.Error) {
				return packagepublish.Namespace{Hub: server.URL, Account: "paul"}, nil
			}
			_, problem := packagepublish.PrepareUnpublishedFrom(t.Context(), tree, namespace)
			if problem == nil || problem.Name != "child.interface_lock_refused" {
				t.Fatalf("capture answered %v", problem)
			}
			mu.Lock()
			defer mu.Unlock()
			ownSeen := false
			for _, path := range requests {
				if path == "/v1/index/paul/simple/cozy-index-fixture/" {
					ownSeen = true
				}
				if path == "/v1/index/paul/simple/unmapped-fixture/" {
					t.Fatal("the account index was queried for a dependency the source does not name")
				}
				if path == "/pypi/cozy-index-fixture/" {
					t.Fatal("a missing account dependency fell through to the public index")
				}
			}
			if !ownSeen {
				t.Fatalf("the account index was never queried: %v", requests)
			}
			authored, err := os.ReadFile(filepath.Join(tree, "pyproject.toml"))
			must(t, err)
			if string(authored) != metadata {
				t.Fatal("capture changed the authored project")
			}
		})
	}
}

// The account index is Creator's to write; authored source declaring it is refused, and a
// source that names it cannot resolve without a namespace.
func TestAuthoredAccountIndexIsRefused(t *testing.T) {
	sources := `
[tool.uv.sources]
cozy-index-fixture = { index = "tensorhub" }
`
	declared := fixtureTree(t, fixturePyproject("cozy-index-fixture>=1,<2")+sources+`
[[tool.uv.index]]
name = "tensorhub"
url = "https://tensorhub.com/v1/index/paul/simple/"
explicit = true
`, minimalFixtureLock)
	if _, problem := packagepublish.UsesAccountIndex(declared); problem == nil || problem.Name != "account_index_declared" {
		t.Fatalf("a declared account index answered %v", problem)
	}
	named := fixtureTree(t, fixturePyproject("cozy-index-fixture>=1,<2")+sources, minimalFixtureLock)
	uses, problem := packagepublish.UsesAccountIndex(named)
	fatal(t, problem)
	if !uses {
		t.Fatal("a named account index was not detected")
	}
	if _, problem := packagepublish.PrepareUnpublishedFrom(t.Context(), named, nil); problem == nil ||
		problem.Name != "account_index_namespace_missing" {
		t.Fatalf("an unresolved account index answered %v", problem)
	}
	if _, problem := packagepublish.Lock(t.Context(), named, nil, nil); problem == nil ||
		problem.Name != "account_index_namespace_missing" || !strings.Contains(problem.Message, "tensorhub") {
		t.Fatalf("lock without a namespace answered %v", problem)
	}
}
