package producttest

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestCapturedPyTorchRequirementPreservesOfficialObjectInPrivateEnvironment(t *testing.T) {
	const version = "2.13.0+cpu"
	const index = "https://download.pytorch.org/whl/cpu"
	const object = "https://download-r2.pytorch.org/whl/cpu/torch-2.13.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl"
	hash := "sha256:" + strings.Repeat("4", 64)
	capture := func(t *testing.T, name, selectedIndex, selectedURL, selectedHash string) (packagepublish.CapturedDependency, bool) {
		t.Helper()
		root := t.TempDir()
		must(t, os.WriteFile(filepath.Join(root, "pyproject.toml"), []byte("[project]\nname='capture-root'\nversion='1.0.0'\n"), 0600))
		lock := fmt.Sprintf("version=1\n[[package]]\nname='capture-root'\nversion='1.0.0'\nsource={editable='.'}\n[[package]]\nname=%s\nversion=%s\nsource={registry=%s}\nwheels=[{url=%s,hash=%s}]\n", strconv.Quote(name), strconv.Quote(version), strconv.Quote(selectedIndex), strconv.Quote(selectedURL), strconv.Quote(selectedHash))
		must(t, os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600))
		closure := "capture-root==1.0.0\n" + name + "==" + version
		rows, _, problem := packagepublish.CapturedRegistryRows([]byte(lock), closure, "capture-root", "1.0.0", nil)
		if packagepublish.ImageOwnedDistribution(name) && problem == nil && len(rows) != 1 {
			t.Fatal("framework direct reference was omitted")
		}
		captured, problem := packagepublish.CaptureWheelDependencies(t.Context(), root, "capture-root", closure, t.TempDir(), t.TempDir(), map[string]map[string]string{"library": {name: version}})
		return captured[name], problem == nil
	}
	good, ok := capture(t, "torch", index, object, hash)
	if !ok || good.Requirement != "torch @ "+strings.Replace(object, "download-r2.pytorch.org", "download.pytorch.org", 1)+" --hash="+hash || good.Path != "" || good.Digest != "" || good.Application {
		t.Fatalf("official framework capture lost exact local identity or image ownership: %+v", good)
	}
	canonical, ok := capture(t, "torch", index, strings.Replace(object, "download-r2.pytorch.org", "download.pytorch.org", 1), hash)
	if !ok || canonical.Requirement != good.Requirement {
		t.Fatal("official mirrors changed the selected artifact identity")
	}
	for _, arm := range []struct{ name, index, url, hash string }{
		{"foreign-index", "https://example.org/whl/cpu", object, hash},
		{"foreign-object", index, strings.Replace(object, "download-r2.pytorch.org", "example.org", 1), hash},
		{"lookalike-host", index, strings.Replace(object, "download-r2.pytorch.org", "download-r2.pytorch.org.example.org", 1), hash},
		{"wrong-backend", index, strings.Replace(object, "/whl/cpu/", "/whl/cu130/", 1), hash},
		{"traversal", index, strings.Replace(object, "/whl/cpu/", "/whl/cpu/../cpu/", 1), hash},
		{"query", index, object + "?redirect=elsewhere", hash},
		{"wrong-distribution", index, strings.Replace(object, "/torch-", "/torchaudio-", 1), hash},
		{"wrong-version", index, strings.Replace(object, "2.13.0", "2.12.0", 1), hash},
		{"invalid-hash", index, object, "sha256:" + strings.Repeat("z", 64)},
		{"missing-algorithm", index, object, strings.Repeat("4", 64)},
	} {
		t.Run(arm.name, func(t *testing.T) {
			if _, ok := capture(t, "torch", arm.index, arm.url, arm.hash); ok {
				t.Fatal("changed origin or identity was admitted")
			}
		})
	}
	t.Run("other-image-package", func(t *testing.T) {
		if _, ok := capture(t, "numpy", index, strings.Replace(object, "/torch-", "/numpy-", 1), hash); ok {
			t.Fatal("PyTorch exception spread to another image distribution")
		}
	})
	t.Run("ordinary-package", func(t *testing.T) {
		if _, ok := capture(t, "private-library", index, strings.Replace(object, "/torch-", "/private_library-", 1), hash); ok {
			t.Fatal("PyTorch exception spread to private library publication")
		}
	})
}
