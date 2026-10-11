package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/packagepublish"
)

func TestCapturedHubStorageRedirectRetainsExactWheel(t *testing.T) {
	var payload bytes.Buffer
	archive := zip.NewWriter(&payload)
	member, err := archive.Create("fixture-1.0.dist-info/METADATA")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = member.Write([]byte("Metadata-Version: 2.3\nName: fixture\nVersion: 1.0\n")); err != nil {
		t.Fatal(err)
	}
	if err = archive.Close(); err != nil {
		t.Fatal(err)
	}
	storage := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.Write(payload.Bytes()) }))
	defer storage.Close()
	priorTransport := http.DefaultTransport
	http.DefaultTransport = storage.Client().Transport
	t.Cleanup(func() { http.DefaultTransport = priorTransport })
	for _, arm := range []struct {
		name, target, hash, code string
		status                   int
	}{
		{"presigned", storage.URL + "/object?private-signature=secret", fmt.Sprintf("%x", sha256.Sum256(payload.Bytes())), "", http.StatusFound},
		{"changed-bytes", storage.URL + "/object?private-signature=secret", strings.Repeat("1", 64), "private_dependency_download_changed", http.StatusFound},
		{"insecure-redirect", "http://127.0.0.1/object?private-signature=secret", strings.Repeat("1", 64), "private_dependency_download_failed", http.StatusFound},
		{"missing", storage.URL, strings.Repeat("1", 64), "private_dependency_download_failed", http.StatusNotFound},
	} {
		t.Run(arm.name, func(t *testing.T) {
			hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { http.Redirect(w, r, arm.target, arm.status) }))
			defer hub.Close()
			root := t.TempDir()
			metadata := []byte("[project]\nname='root'\nversion='1.0'\n")
			if err := os.WriteFile(filepath.Join(root, "pyproject.toml"), metadata, 0600); err != nil {
				t.Fatal(err)
			}
			object := hub.URL + "/v1/index/paul/files/" + arm.hash + "/fixture-1.0-py3-none-any.whl"
			lock := fmt.Sprintf("version=1\n[[package]]\nname='root'\nversion='1.0'\nsource={editable='.'}\n[[package]]\nname='fixture'\nversion='1.0'\nsource={registry=%q}\nwheels=[{url=%q,hash=%q}]\n", hub.URL+"/v1/index/paul/simple/", object, "sha256:"+arm.hash)
			if err := os.WriteFile(filepath.Join(root, "uv.lock"), []byte(lock), 0600); err != nil {
				t.Fatal(err)
			}
			store := t.TempDir()
			problem := packagepublish.KeepHubWheels(t.Context(), root, store)
			if arm.code == "" {
				if problem != nil {
					t.Fatal(problem)
				}
				data, err := os.ReadFile(filepath.Join(store, arm.hash, "fixture-1.0-py3-none-any.whl"))
				if err != nil || !bytes.Equal(data, payload.Bytes()) {
					t.Fatal("the redirect lost exact wheel custody", err)
				}
				return
			}
			if problem == nil || problem.Name != arm.code {
				t.Fatalf("expected %s, got %v", arm.code, problem)
			}
			if strings.Contains(problem.Error(), "secret") || strings.Contains(problem.Error(), "private-signature") {
				t.Fatal("storage capability leaked into error")
			}
		})
	}
}
