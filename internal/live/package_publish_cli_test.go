package live

import (
	"archive/zip"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy-creator/internal/packageprofile"
	"github.com/cozy-creator/cozy-creator/internal/transfer"
)

func TestEmptyPresignedUpload(t *testing.T) {
	empty := filepath.Join(t.TempDir(), "empty.txt")
	mustWrite(t, empty, "")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut || r.ContentLength != 0 || len(r.TransferEncoding) != 0 {
			w.WriteHeader(http.StatusLengthRequired)
			return
		}
		body, err := io.ReadAll(r.Body)
		if err != nil || len(body) != 0 {
			t.Errorf("empty PUT body = %q, %v", body, err)
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	defer server.Close()

	uploaded, moved, problem := transfer.UploadPresigned(context.Background(), "empty.txt", empty,
		server.URL, nil)
	if problem != nil || !uploaded || moved != 0 {
		t.Fatalf("empty PUT uploaded=%t bytes=%d: %v", uploaded, moved, problem)
	}
}

func TestPackagePublishCLI(t *testing.T) {
	source, testBin := packageFixture(t)
	var server *httptest.Server
	var lock sync.Mutex
	state := "pending"
	put := map[string][]byte{}
	sourcePaths := map[string]string{}
	beginCount, uploadsCount, finalizeCount := 0, 0, 0
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost && r.Header.Get("Authorization") != "Bearer admin" {
			writeHubError(w, http.StatusUnauthorized, "auth.token_invalid", "wrong token")
			return
		}
		if r.Method == http.MethodPost && r.Header.Get("X-Tensorhub-Reason") !=
			"cozy package publish cozy/marco@1.0.0" {
			t.Errorf("package mutation audit reason = %q", r.Header.Get("X-Tensorhub-Reason"))
		}
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1/packages":
			_, _ = w.Write([]byte(`{"package":{"org":"cozy","name":"marco","created_at":"2026-08-28T00:00:00Z"}}`))
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/begin"):
			var body map[string]any
			if json.NewDecoder(r.Body).Decode(&body) != nil || len(body) != 0 {
				t.Errorf("begin body is not empty JSON: %#v", body)
			}
			lock.Lock()
			beginCount++
			current := state
			lock.Unlock()
			wheel := map[string]any{"already_uploaded": current == "committed"}
			if current == "pending" {
				wheel["url"] = server.URL + "/put/wheel"
				wheel["required_headers"] = map[string]string{"X-Test-Path": "project_wheel"}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"release": "1.0.0", "state": current, "created": beginCount == 1,
				"project_wheel_upload": wheel,
			})
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/uploads"):
			var request struct {
				Paths []string `json:"paths"`
			}
			if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
				t.Errorf("source upload request: %s", err)
			}
			rows := make([]map[string]any, 0, len(request.Paths))
			lock.Lock()
			uploadsCount++
			for index, path := range request.Paths {
				id := fmt.Sprintf("source-%d", index)
				sourcePaths[id] = path
				rows = append(rows, map[string]any{"path": path, "url": server.URL + "/put/" + id,
					"required_headers": map[string]string{"X-Test-Path": path}, "already_uploaded": false})
			}
			lock.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"uploads": rows})
		case r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/put/"):
			id := strings.TrimPrefix(r.URL.Path, "/put/")
			lock.Lock()
			path := sourcePaths[id]
			if id == "wheel" {
				path = "project_wheel"
			}
			lock.Unlock()
			if r.Header.Get("X-Test-Path") != path {
				t.Errorf("PUT %s omitted signed header for %s", id, path)
			}
			body, _ := io.ReadAll(r.Body)
			lock.Lock()
			put[path] = body
			lock.Unlock()
			w.WriteHeader(http.StatusNoContent)
		case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/finalize"):
			var body map[string]any
			if json.NewDecoder(r.Body).Decode(&body) != nil || len(body) != 0 {
				t.Errorf("finalize body is not empty JSON: %#v", body)
			}
			lock.Lock()
			finalizeCount++
			created := state != "committed"
			state = "committed"
			lock.Unlock()
			_ = json.NewEncoder(w).Encode(map[string]any{"created": created,
				"release": "1.0.0",
				"profiles": []map[string]any{{"profile": packageprofile.CPU, "state": "qualified",
					"candidate_id": "candidate-cpu", "base_realization_kind": "oci"}},
				"package_executions": []map[string]string{{"profile": packageprofile.CPU,
					"function": "marco", "state": "qualified"}},
			})
		default:
			writeHubError(w, http.StatusNotFound, "route.not_found", r.Method+" "+r.URL.Path)
		}
	})
	server = httptest.NewServer(handler)
	defer server.Close()

	home := t.TempDir()
	run := func(args ...string) (int, string) {
		cmd := exec.Command("/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
		cmd.Env = childEnv(t, home, "TENSORHUB_URL="+server.URL, "TENSORHUB_TOKEN=admin",
			"PATH="+testBin+":/usr/local/bin:/usr/bin:/bin")
		body, _ := cmd.CombinedOutput()
		return cmd.ProcessState.ExitCode(), string(body)
	}
	args := []string{"package", "publish", "cozy/marco", "--release", "1.0.0", "--dir", source}
	if code, out := run(args...); code != 0 || !strings.Contains(out, "status:") ||
		!strings.Contains(out, "published") ||
		!strings.Contains(out, "qualified") || strings.Contains(out, "candidate-cpu") {
		t.Fatalf("package publish [exit %d]\n%s", code, out)
	}
	if code, out := run(append(args, "--full")...); code != 0 ||
		!strings.Contains(out, "candidate-cpu") || !strings.Contains(out, "qualified") ||
		!strings.Contains(out, "uploaded:") || !strings.Contains(out, "0B") {
		t.Fatalf("package publish replay [exit %d]\n%s", code, out)
	}

	lock.Lock()
	defer lock.Unlock()
	if beginCount != 2 || uploadsCount != 1 || finalizeCount != 2 {
		t.Fatalf("calls begin=%d uploads=%d finalize=%d", beginCount, uploadsCount, finalizeCount)
	}
	for _, required := range []string{"package.toml", "marco_polo/__init__.py",
		"marco_polo/untracked.py", "notes.txt", "pyproject.toml", "uv.lock"} {
		if _, ok := put[required]; !ok {
			t.Errorf("current working-tree file %s was not uploaded", required)
		}
	}
	for _, forbidden := range []string{".env.local", ".ssh/id_rsa", ".venv/marker", "dist/stale.whl"} {
		if _, ok := put[forbidden]; ok {
			t.Errorf("local-only file %s was uploaded", forbidden)
		}
	}
	wheelNames := wheelBytesNames(t, put["project_wheel"])
	for _, required := range []string{"marco_polo/__init__.py", "marco_polo/untracked.py"} {
		if !slices.Contains(wheelNames, required) {
			t.Errorf("current working-tree file %s is absent from wheel %v", required, wheelNames)
		}
	}
}

func packageFixture(t *testing.T) (string, string) {
	t.Helper()
	repo := t.TempDir()
	mustWrite(t, filepath.Join(repo, "pyproject.toml"), `[build-system]
requires = ["uv_build>=0.9.18,<0.10"]
build-backend = "uv_build"

[project]
name = "marco-polo"
version = "1.0.0"
dependencies = []

[tool.uv.build-backend]
module-root = ""
`)
	mustWrite(t, filepath.Join(repo, "package.toml"), "[application]\nobject = \"marco_polo:app\"\n")
	mustWrite(t, filepath.Join(repo, "marco_polo", "__init__.py"), "app = object()\n")
	mustWrite(t, filepath.Join(repo, "marco_polo", "untracked.py"), "value = 'current tree'\n")
	mustWrite(t, filepath.Join(repo, "notes.txt"), "ordinary untracked provenance\n")
	mustWrite(t, filepath.Join(repo, ".env.local"), "TOKEN=secret\n")
	mustWrite(t, filepath.Join(repo, ".ssh", "id_rsa"), "secret\n")
	mustWrite(t, filepath.Join(repo, ".venv", "marker"), "local environment\n")
	mustWrite(t, filepath.Join(repo, "dist", "stale.whl"), "stale output\n")
	mustWrite(t, filepath.Join(repo, "uv.lock"), "version = 1\n")
	testBin := t.TempDir()
	realUV, err := exec.LookPath("uv")
	must(t, err)
	fakeUV := filepath.Join(testBin, "uv")
	mustWrite(t, fakeUV, fmt.Sprintf("#!/bin/sh\nexec %q \"$@\"\n", realUV))
	must(t, os.Chmod(fakeUV, 0o755))
	return repo, testBin
}

func wheelBytesNames(t *testing.T, body []byte) []string {
	t.Helper()
	zr, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
	must(t, err)
	names := make([]string, 0, len(zr.File))
	for _, member := range zr.File {
		if !member.FileInfo().IsDir() {
			names = append(names, member.Name)
		}
	}
	sort.Strings(names)
	return names
}

func writeHubError(w http.ResponseWriter, status int, code, message string) {
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]any{"error": map[string]string{
		"code": code, "message": message, "remedy": "fixture"}})
}
