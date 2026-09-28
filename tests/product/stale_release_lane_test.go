package producttest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/tfs"
	"github.com/cozy-creator/cozy/internal/transfer"
)

// A local release lane is a cache of Tensorhub's mutable lane. When the hub re-points the
// lane, the next acquisition fetches the new manifest instead of refusing the stale copy,
// and a resolution whose header fact disagrees with the manifest defers to the manifest.
func TestStaleLocalReleaseLaneFetchesTheHubSelection(t *testing.T) {
	binary, err := exec.LookPath("tfs")
	if err != nil {
		t.Skip("requires the native tfs CLI")
	}
	read := func(name string) []byte {
		body, err := os.ReadFile(filepath.Join("testdata", "unpublished_preparation", name))
		must(t, err)
		return body
	}
	first, header, vocab := read("manifest.json"), read("model.cozytensors"), read("vocab.txt")
	second := bytes.Replace(first, []byte("model.cozytensors"), []byte("other.cozytensors"), 1)
	if bytes.Equal(first, second) {
		t.Fatal("fixture manifest has no native entry to rename")
	}
	objects := map[string][]byte{assessmentDigest(header): header, assessmentDigest(vocab): vocab}
	var current atomic.Pointer[[]byte]
	current.Store(&first)
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, _ *http.Request) {
		manifest := *current.Load()
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/base", Release: "1.0.0", Lane: "bf16",
			ManifestID: assessmentDigest(manifest), HeaderID: "sha256:" + strings.Repeat("0", 64),
			ManifestLength: int64(len(manifest)), Objects: 2, Bytes: int64(len(header) + len(vocab))})
	})
	mux.HandleFunc("GET /v1/models/proof/base/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(*current.Load())
	})
	mux.HandleFunc("POST /v1/models/proof/base/releases/1.0.0/lanes/bf16/reads", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			IDs []string `json:"object_ids"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&ask))
		out := []hub.Read{}
		for _, id := range ask.IDs {
			out = append(out, hub.Read{ObjectID: id, Length: int64(len(objects[id])), URL: server.URL + "/objects/" + id})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"reads": out})
	})
	mux.HandleFunc("GET /objects/{id}", func(w http.ResponseWriter, r *http.Request) { _, _ = w.Write(objects[r.PathValue("id")]) })
	server = httptest.NewServer(mux)
	defer server.Close()

	area := t.TempDir()
	cfg := config.Config{Tfs: binary, TensorFSRoot: filepath.Join(area, "store"), HubURL: server.URL}
	tool, problem := tfs.Open(cfg)
	fatal(t, problem)
	client := hub.New(cfg, "stale-release-lane")
	acquire := func(turn string) transfer.Fetched {
		fetch := &transfer.Fetch{Tool: tool, Hub: client, Spec: "proof/base@1.0.0", Lane: "bf16",
			Scratch: filepath.Join(area, "work-"+turn), Locks: filepath.Join(area, "locks")}
		row, problem := fetch.Resolve(context.Background())
		fatal(t, problem)
		fetched, problem := fetch.Acquire(context.Background(), row)
		fatal(t, problem)
		return fetched
	}
	if got := acquire("first"); got.ManifestID != assessmentDigest(first) || got.HeaderID != assessmentDigest(header) {
		t.Fatalf("first acquisition = %s header %s", got.ManifestID, got.HeaderID)
	}
	current.Store(&second)
	if got := acquire("second"); got.ManifestID != assessmentDigest(second) {
		t.Fatalf("re-pointed lane acquired %s, want %s", got.ManifestID, assessmentDigest(second))
	}
	rows, problem := tool.Releases(filepath.Join(area, "releases.jsonl"))
	fatal(t, problem)
	if len(rows) != 1 || rows[0].Lane != "bf16" || "sha256:"+rows[0].ManifestSHA256 != assessmentDigest(second) {
		t.Fatalf("local lane was not re-pointed: %+v", rows)
	}
}
