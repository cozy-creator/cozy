package packagepublish

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"
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
			row := RegistryRow{Name: "fixture", Version: "1.0", URL: hub.URL + "/v1/index/paul/files/" + arm.hash + "/fixture-1.0-py3-none-any.whl", SHA256: arm.hash, captureLocally: true}
			problem := fetchCapturedWheel(t.Context(), storage.Client(), row, filepath.Join(t.TempDir(), "fixture-1.0-py3-none-any.whl"))
			if arm.code == "" {
				if problem != nil {
					t.Fatal(problem)
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
