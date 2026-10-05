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
	object := server.URL + "/v1/index/paul/files/" + digest + "/" + filepath.Base(source)
	lock := fmt.Sprintf("version=1\n[[package]]\nname='capture-root'\nversion='1.0.0'\nsource={editable='.'}\n[[package]]\nname='hub-fixture'\nversion='1.0.0'\nsource={registry=%q}\nwheels=[{url=%q,hash=%q}]\n", index, object, "sha256:"+digest)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte("[project]\nname='capture-root'\nversion='1.0.0'\n"), 0600))
	must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600))
	closure := "capture-root==1.0.0\nhub-fixture==1.0.0"
	captured, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), map[string]map[string]string{"library": {"hub-fixture": "1.0.0"}})
	fatal(t, problem)
	got := captured["hub-fixture"]
	if got.Path == "" || got.Digest != "sha256:"+digest || got.RegistryRequirement != "" || !got.Application || got.Package != "paul/hub-fixture" {
		t.Fatalf("Hub wheel was not retained privately: %+v", got)
	}
	packRoot := t.TempDir()
	pack := packagepublish.Package{Name: "capture-root", Release: "1.0.0", Root: packRoot, Wheel: prebuiltProjectWheel(t, packRoot, "capture-root", "py3-none-any", "weightless:app"), Files: map[string]string{"uv.lock": filepath.Join(root, "uv.lock")}}
	fatal(t, pack.CaptureUnpublishedClosure(t.Context(), closure, nil, "3.12.12"))
	if len(pack.DependencyWheels) != 1 || len(pack.DependencyRequirements) != 0 || pack.DependencyPackages["hub-fixture"] != "paul/hub-fixture" {
		t.Fatalf("Hub source leaked into remote requirements: wheels=%v requirements=%s", pack.DependencyWheels, pack.DependencyRequirements)
	}
	for _, candidate := range []string{
		strings.Replace(lock, object, strings.Replace(object, server.URL, "https://other.invalid", 1), 1),
		strings.Replace(lock, object, strings.Replace(object, "/paul/files/", "/other/files/", 1), 1),
		strings.Replace(lock, object, object+"?redirect=1", 1),
		strings.Replace(lock, object, strings.Replace(object, digest, strings.Repeat("1", 64), 1), 1),
	} {
		if _, _, problem := packagepublish.CapturedRegistryRows([]byte(candidate), closure, "capture-root", "1.0.0", nil); problem == nil {
			t.Fatal("changed Hub object origin/namespace/hash admitted")
		}
	}
	changed.Store(true)
	if _, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), map[string]map[string]string{"library": {"hub-fixture": "1.0.0"}}); problem == nil {
		t.Fatal("changed Hub bytes admitted")
	}
}
