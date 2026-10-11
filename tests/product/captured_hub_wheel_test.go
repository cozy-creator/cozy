package producttest

import (
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestCapturedHubWheelRetainsBytesWithoutRemoteLoopbackRequirement(t *testing.T) {
	source := prebuiltProjectWheel(t, t.TempDir(), "hub-fixture", "py3-none-any", "weightless:app")
	data, err := os.ReadFile(source)
	must(t, err)
	digest := fmt.Sprintf("%x", sha256.Sum256(data))
	var changed atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if changed.Load() {
			w.Write([]byte("changed"))
			return
		}
		w.Write(data)
	}))
	defer server.Close()
	index := server.URL + "/v1/index/paul/simple/"
	object := server.URL + "/v1/index/paul/hub-fixture/1.0.0/" + filepath.Base(source)
	lock := fmt.Sprintf("version=1\n[[package]]\nname='capture-root'\nversion='1.0.0'\nsource={editable='.'}\n[[package]]\nname='hub-fixture'\nversion='1.0.0'\nsource={registry=%q}\nwheels=[{url=%q,hash=%q}]\n", index, object, "sha256:"+digest)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte("[project]\nname='capture-root'\nversion='1.0.0'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600))
	closure := "capture-root==1.0.0\nhub-fixture==1.0.0"
	store := t.TempDir()
	fatal(t, packagepublish.KeepHubWheels(t.Context(), root, store))
	if kept, err := os.ReadFile(filepath.Join(store, digest, filepath.Base(source))); err != nil || string(kept) != string(data) {
		t.Fatalf("the Hub wheel was not kept by its hash: %v", err)
	}
	packRoot := t.TempDir()
	pack := packagepublish.Package{Name: "capture-root", Release: "1.0.0", Root: packRoot, Wheel: prebuiltProjectWheel(t, packRoot, "capture-root", "py3-none-any", "weightless:app"), Files: map[string]string{"uv.lock": filepath.Join(root, "uv.lock")}}
	fatal(t, pack.CaptureUnpublishedClosure(t.Context(), closure, nil, "3.12.12"))
	if len(pack.DependencyWheels) != 1 || len(pack.DependencyRequirements) != 0 || pack.DependencyPackages["hub-fixture"] != "paul/hub-fixture" {
		t.Fatalf("Hub source leaked into remote requirements: wheels=%v requirements=%s", pack.DependencyWheels, pack.DependencyRequirements)
	}
	for _, candidate := range []string{
		strings.Replace(lock, object, strings.Replace(object, server.URL, "https://other.invalid", 1), 1),
		strings.Replace(lock, object, strings.Replace(object, "/paul/hub-fixture/", "/other/hub-fixture/", 1), 1),
		strings.Replace(lock, object, object+"?redirect=1", 1),
		strings.Replace(lock, object, strings.Replace(object, "http://", "http://user:secret@", 1), 1),
		strings.Replace(lock, object, object+"#fragment", 1),
	} {
		if _, _, problem := packagepublish.CapturedRegistryRows([]byte(candidate), closure, "capture-root", "1.0.0", nil); problem == nil {
			t.Fatal("changed Hub object origin/namespace/credentials admitted")
		}
	}
	// Release URLs need not contain the digest: the lock's SHA256 still binds
	// their bytes, and a changed hash must fail even with valid wheel metadata.
	wrongHash := strings.Replace(lock, "sha256:"+digest, "sha256:"+strings.Repeat("1", 64), 1)
	must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(wrongHash), 0600))
	if problem := packagepublish.KeepHubWheels(t.Context(), root, t.TempDir()); problem == nil || problem.Name != "private_dependency_download_changed" {
		t.Fatalf("changed locked hash admitted: %v", problem)
	}
	must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600))
	changed.Store(true)
	if problem := packagepublish.KeepHubWheels(t.Context(), root, t.TempDir()); problem == nil {
		t.Fatal("changed Hub bytes admitted")
	}
}
