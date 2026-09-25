package producttest

import (
	"bytes"
	"crypto/sha256"
	"flag"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/packagepublish"
)

var indexProofProject = flag.String("index-proof-project", "", "owned publication fixture")
var indexProofCapture = flag.Bool("index-proof-capture", false, "capture owned unpublished project")
var indexProofHub = flag.String("index-proof-hub", "", "selected fixture Hub")

// A fresh process freezes the selected Hub exactly as the publishing CLI does.
func TestSelectedHubIndexBuildProcess(t *testing.T) {
	project := *indexProofProject
	if project == "" {
		t.Skip("subprocess publication fixture")
	}
	selected := *indexProofHub
	_, problem := config.LoadForTensorhub(&selected)
	fatal(t, problem)
	if *indexProofCapture {
		pack, problem := packagepublish.PrepareUnpublishedFrom(t.Context(), project)
		if problem != nil {
			fmt.Println("INDEX_RESULT=" + problem.Name)
			return
		}
		defer pack.Close()
		raw, err := os.ReadFile(pack.Files["uv.lock"])
		must(t, err)
		if !strings.Contains(string(raw), selected+"/v1/index/paul/simple/") {
			t.Fatal("capture lost selected Hub")
		}
		fmt.Println("INDEX_RESULT=capture_selected")
		return
	}
	pack, problem := packagepublish.PrepareFrom(project)
	fatal(t, problem)
	defer pack.Close()
	problem = pack.BuildForPublish(t.Context())
	if problem == nil {
		t.Fatal("fixture unexpectedly described an application")
	}
	fmt.Println("INDEX_RESULT=" + problem.Name)
}

func TestSelectedHubIndexPublication(t *testing.T) {
	wheel := prebuiltProjectWheel(t, t.TempDir(), "cozy-index-fixture", "py3-none-any", "weightless:app")
	data, err := os.ReadFile(wheel)
	must(t, err)
	digest := fmt.Sprintf("%x", sha256.Sum256(data))
	filename := filepath.Base(wheel)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.Contains(r.URL.Path, "/files/") {
			w.Write(data)
			return
		}
		fmt.Fprintf(w, `<a href="/v1/index/paul/files/%s/%s#sha256=%s">%s</a>`, digest, filename, digest, filename)
	}))
	defer server.Close()
	for _, tc := range []struct{ name, index, hub, dependency, want string }{
		{"selected", "https://tensorhub.com/v1/index/paul/simple/", server.URL, "cozy-index-fixture>=1.0,<2", "package_interface_refused"},
		{"capture", "https://tensorhub.com/v1/index/paul/simple/", server.URL, "cozy-index-fixture>=1.0,<2", "capture_selected"},
		{"wrong-hub", "https://tensorhub.com/v1/index/paul/simple/", "http://127.0.0.1:1", "cozy-index-fixture>=1.0,<2", "registry_dependency_hub_mismatch"},
		{"stale", "https://tensorhub.com/v1/index/paul/simple/", server.URL, "cozy-index-fixture>=2.0,<3", "registry_dependency_export_failed"},
		{"third-party", server.URL + "/simple/", server.URL, "cozy-index-fixture>=1.0,<2", "registry_dependency_index_refused"},
		{"foreign-org", server.URL + "/v1/index/other/simple/", server.URL, "cozy-index-fixture>=1.0,<2", "registry_dependency_index_refused"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			metadata := fixturePyproject("cozy-index-fixture>=1.0,<2") + fmt.Sprintf(`
[tool.cozy]
organization = "paul"
[tool.uv.sources]
cozy-index-fixture = { index = "tensorhub-paul" }
[[tool.uv.index]]
name = "tensorhub-paul"
url = %q
explicit = true
`, tc.index)
			tree := fixtureTree(t, metadata, minimalFixtureLock)
			index := tc.index
			if strings.HasPrefix(index, "https://tensorhub.com/") {
				index = server.URL + "/v1/index/paul/simple/"
			}
			lock := exec.Command("uv", "lock", "--index", "tensorhub-paul="+index, "--no-python-downloads")
			lock.Dir = tree
			if out, err := lock.CombinedOutput(); err != nil {
				t.Fatalf("lock: %v\n%s", err, out)
			}
			// Real uv proves the named override validates without contacting any server.
			export := exec.Command("uv", "export", "--locked", "--offline", "--index", "tensorhub-paul="+index, "--format", "pylock.toml", "--no-python-downloads")
			export.Dir = tree
			if out, err := export.CombinedOutput(); err != nil {
				t.Fatalf("offline export: %v\n%s", err, out)
			}
			metadata = strings.Replace(metadata, "cozy-index-fixture>=1.0,<2", tc.dependency, 1)
			must(t, os.WriteFile(filepath.Join(tree, "pyproject.toml"), []byte(metadata), 0600))
			before, err := os.ReadFile(filepath.Join(tree, "uv.lock"))
			must(t, err)
			command := exec.Command(os.Args[0], "-test.run=^TestSelectedHubIndexBuildProcess$", "-test.v", "-index-proof-project="+tree, "-index-proof-hub="+tc.hub)
			if tc.name == "capture" {
				command.Args = append(command.Args, "-index-proof-capture")
			}
			command.Env = childEnv(t, t.TempDir())
			out, err := command.CombinedOutput()
			if err != nil || !strings.Contains(string(out), "INDEX_RESULT="+tc.want) {
				t.Fatalf("publication: %v, want %s\n%s", err, tc.want, out)
			}
			after, err := os.ReadFile(filepath.Join(tree, "uv.lock"))
			must(t, err)
			authored, err := os.ReadFile(filepath.Join(tree, "pyproject.toml"))
			must(t, err)
			if !bytes.Equal(before, after) || string(authored) != metadata {
				t.Fatal("publication changed authored metadata or lock")
			}
		})
	}
}

func TestSelectedHubCaptureKeepsExplicitIndexScope(t *testing.T) {
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
[tool.cozy]
organization = "paul"
[tool.uv.sources]
cozy-index-fixture = { index = "tensorhub-paul" }
[[tool.uv.index]]
name = "tensorhub-paul"
url = "https://tensorhub.com/v1/index/paul/simple/"
explicit = true
[[tool.uv.index]]
name = "fixture-pypi"
url = %q
default = true
`, server.URL+"/pypi/")
			tree := fixtureTree(t, metadata, minimalFixtureLock)
			command := exec.Command(os.Args[0], "-test.run=^TestSelectedHubIndexBuildProcess$", "-test.v", "-index-proof-capture", "-index-proof-project="+tree, "-index-proof-hub="+server.URL)
			command.Env = childEnv(t, t.TempDir())
			out, err := command.CombinedOutput()
			if err != nil || !strings.Contains(string(out), "INDEX_RESULT=child.interface_lock_refused") {
				t.Fatalf("capture: %v\n%s", err, out)
			}
			mu.Lock()
			defer mu.Unlock()
			ownSeen := false
			for _, path := range requests {
				if path == "/v1/index/paul/simple/cozy-index-fixture/" {
					ownSeen = true
				}
				if path == "/v1/index/paul/simple/unmapped-fixture/" {
					t.Fatal("explicit index was queried for an unmapped dependency")
				}
				if path == "/pypi/cozy-index-fixture/" {
					t.Fatal("missing owned dependency fell through to public index")
				}
			}
			if !ownSeen {
				t.Fatalf("selected own index never queried: %v\n%s", requests, out)
			}
			authored, err := os.ReadFile(filepath.Join(tree, "pyproject.toml"))
			must(t, err)
			if string(authored) != metadata {
				t.Fatal("capture changed authored project")
			}
		})
	}
}
