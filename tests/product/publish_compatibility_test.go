package producttest

import (
	"archive/zip"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// The normal CLI builds, uploads and commits a real wheel to an independent local
// publication peer. The peer verifies the declared hashes and retained wheel metadata.
func TestPublishCLIPreservesMajorMinorCompatibility(t *testing.T) {
	project := weightlessProject(t)
	path := filepath.Join(project, "pyproject.toml")
	raw, err := os.ReadFile(path)
	must(t, err)
	runtimeRange := regexp.MustCompile(`"cozy-runtime\[media\]>=([0-9][0-9a-zA-Z.]*),<1"`)
	matched := runtimeRange.FindStringSubmatch(string(raw))
	if matched == nil {
		t.Fatal("fixture lost its Runtime dependency")
	}
	compatibleRuntime := "cozy-runtime[media]>=" + matched[1] + ",<1"
	narrow := "tensorfs>=0.3.35,<0.4"
	authored := strings.Replace(string(raw), matched[0], fmt.Sprintf("%q, %q", compatibleRuntime, narrow), 1)
	must(t, os.WriteFile(path, []byte(authored), 0644))
	lock := exec.Command("uv", "lock")
	lock.Dir = project
	if output, err := lock.CombinedOutput(); err != nil {
		t.Fatalf("lock: %v\n%s", err, output)
	}

	var mu sync.Mutex
	declared := map[string]hub.PackageDeclaredFile{}
	uploaded := map[string][]byte{}
	committed := false
	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	defer server.Close()
	mux.HandleFunc("GET /v1/accounts/current", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.Account{Name: "proof"})
	})
	mux.HandleFunc("GET /v1/packages/proof/cozy-weightless-package/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, `{"error":{"code":"not_found","message":"not published"}}`, http.StatusNotFound)
	})
	mux.HandleFunc("POST /v1/packages/proof/cozy-weightless-package/publish/1.0.0", func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Files []hub.PackageDeclaredFile `json:"files"`
		}
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		answer := hub.PackageReleaseDraft{PublicationID: "proof-publication"}
		mu.Lock()
		for _, file := range body.Files {
			declared[file.Path] = file
			answer.Files = append(answer.Files, hub.PackageFileGrant{PackageDeclaredFile: file, Upload: &hub.PackagePresignedUpload{URL: server.URL + "/upload?path=" + url.QueryEscape(file.Path)}})
		}
		mu.Unlock()
		_ = json.NewEncoder(w).Encode(answer)
	})
	mux.HandleFunc("PUT /upload", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
			w.WriteHeader(400)
			return
		}
		name := r.URL.Query().Get("path")
		digest := sha256.Sum256(raw)
		mu.Lock()
		defer mu.Unlock()
		file := declared[name]
		if int64(len(raw)) != file.Length || "sha256:"+hex.EncodeToString(digest[:]) != file.Digest {
			t.Error("uploaded bytes differ from declaration")
			w.WriteHeader(400)
			return
		}
		uploaded[name] = raw
		w.WriteHeader(http.StatusCreated)
	})
	mux.HandleFunc("POST /v1/packages/proof/cozy-weightless-package/publish/1.0.0/finalize", func(w http.ResponseWriter, _ *http.Request) {
		mu.Lock()
		defer mu.Unlock()
		if len(uploaded) == 0 || len(uploaded) != len(declared) {
			t.Error("commit before all bytes arrived")
			w.WriteHeader(400)
			return
		}
		committed = true
		_ = json.NewEncoder(w).Encode(hub.PackageReleaseCommit{PublicationID: "proof-publication", State: "committed"})
	})
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: publication-proof\n"), 0600))
	command := exec.Command(cozyBin, "package", "publish", "--json")
	command.Dir = project
	command.Env = childEnv(t, root)
	if output, err := command.CombinedOutput(); err != nil {
		t.Fatalf("normal CLI publication: %v\n%s", err, output)
	}
	mu.Lock()
	defer mu.Unlock()
	if !committed {
		t.Fatal("CLI did not commit")
	}
	if _, ok := uploaded["artifacts/source/cozy-weightless-package-1.0.0.tar.gz"]; !ok {
		t.Fatal("publication did not upload the standard source distribution")
	}
	for name := range uploaded {
		if strings.HasPrefix(name, "src/") || strings.HasSuffix(name, ".py") {
			t.Fatalf("publication uploaded loose source file %q", name)
		}
	}
	var metadata string
	for name, body := range uploaded {
		if !strings.HasPrefix(name, "artifacts/project/") {
			continue
		}
		wheel, err := zip.NewReader(bytes.NewReader(body), int64(len(body)))
		must(t, err)
		for _, file := range wheel.File {
			if !strings.HasSuffix(file.Name, ".dist-info/METADATA") {
				continue
			}
			stream, err := file.Open()
			must(t, err)
			raw, err := io.ReadAll(stream)
			must(t, err)
			stream.Close()
			metadata = string(raw)
		}
	}
	compact := strings.ReplaceAll(metadata, " ", "")
	if !strings.Contains(compact, "Requires-Dist:cozy-runtime[media]<1,>="+matched[1]) || !strings.Contains(compact, "Requires-Dist:tensorfs<0.4,>=0.3.35") {
		t.Fatalf("published wheel changed the declared compatibility bounds:\n%s", metadata)
	}
}
