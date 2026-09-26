package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/secret"
	"github.com/cozy-creator/cozy/internal/tfs"
)

// Actual native files from Runtime's code/default preparation proof. The catalog
// only serves those bytes; TensorFS owns verification and warm reuse.
func TestCapturedLocalDefaultDownloadsColdModel(t *testing.T) {
	binary, err := exec.LookPath("tfs")
	if err != nil {
		t.Skip("requires the native tfs CLI")
	}
	read := func(name string) []byte {
		body, err := os.ReadFile(filepath.Join("testdata", "unpublished_preparation", name))
		must(t, err)
		return body
	}
	manifest, header, vocab := read("manifest.json"), read("model.cozytensors"), read("vocab.txt")
	manifestID, headerID := assessmentDigest(manifest), assessmentDigest(header)
	objects := map[string][]byte{headerID: header, assessmentDigest(vocab): vocab}
	var reads atomic.Int64
	var changed atomic.Bool
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models/proof/base", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "proof", Name: "base"}, Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: manifestID, Bytes: 4096}}}}})
	})
	mux.HandleFunc("GET /v1/models/proof/base/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(manifest) })
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("ref") != "proof/base@1.0.0@"+manifestID || r.URL.Query().Get("lane") != "bf16" {
			t.Error("cold child lost exact release/lane/manifest constraint")
		}
		id := manifestID
		if changed.Load() {
			id = childDigest("a")
		}
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/base", Release: "1.0.0", Lane: "bf16", ManifestID: id, HeaderID: headerID, ManifestLength: int64(len(manifest)), Objects: 2, Bytes: int64(len(header) + len(vocab))})
	})
	mux.HandleFunc("POST /v1/models/proof/base/releases/1.0.0/lanes/bf16/reads", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			IDs []string `json:"object_ids"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&ask))
		out := []hub.Read{}
		for _, id := range ask.IDs {
			body, ok := objects[id]
			if !ok {
				t.Error("asked for undeclared fixture object")
				continue
			}
			out = append(out, hub.Read{ObjectID: id, Length: int64(len(body)), URL: server.URL + "/objects/" + id})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"reads": out})
	})
	mux.HandleFunc("GET /objects/{id}", func(w http.ResponseWriter, r *http.Request) { reads.Add(1); _, _ = w.Write(objects[r.PathValue("id")]) })
	server = httptest.NewServer(mux)
	defer server.Close()
	layout, problem := home.Open(t.TempDir())
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	installed := cleanupTestInstall(layout, "4444444444444444", "1.0.0")
	installed.Package = "local/captured"
	raw := assessmentJSON(t, map[string]any{
		"format": "cozy.package.interface/1", "application": "captured:app",
		"entrypoints": []any{map[string]any{
			"name": "judge", "models": []any{map[string]any{"class": "Judge", "path": "judge.models.model", "component_use": map[string]any{}, "default_ladder": []any{map[string]any{"gpu": "*", "lane": "proof/base@1.0.0/bf16"}}}},
			"request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}},
			"invocable": map[string]any{"context": "ctx", "module": "captured", "export": "judge", "parameters": []any{}, "defaults": map[string]any{}, "type_names": map[string]any{}, "enum_members": map[string]any{}, "memoize": false, "capabilities": []any{}},
		}},
		"jobs": []any{map[string]any{"name": "parent", "publishes": false, "models": []any{}, "request": map[string]any{"fields": []any{}}, "result": map[string]any{"fields": []any{}}, "weights_outputs": []any{}}},
	})

	must(t, os.MkdirAll(filepath.Dir(launch.PackageInterfacePath(installed.Dir)), 0700))
	must(t, os.WriteFile(launch.PackageInterfacePath(installed.Dir), raw, 0444))
	fatal(t, store.RecordInstall(installed))
	fatal(t, store.RecordChildBindings([]records.ChildBinding{{ParentInstallID: installed.ID, ChildInstallID: installed.ID, Module: "captured", Export: "judge", Entrypoint: "judge"}}))
	cfg := config.Config{Home: layout.Root, HubURL: server.URL, HubToken: secret.New("fixture"), Tfs: binary, TensorFSRoot: filepath.Join(layout.Root, "native-store")}
	resolver := cli.NewResolver(store, cfg, nil)
	parent := records.Request{ID: "parent", InstallID: installed.ID, Entrypoint: "parent", Kind: "job"}
	args := assessmentJSON(t, map[string]any{"payload": map[string]any{}, "models": map[string]any{"model": nil}})
	for turn := range 2 {
		selected, _, problem := resolver.ResolveUnpublishedChild(parent, "captured", "judge", args)
		fatal(t, problem)
		if len(selected.Models) != 1 || selected.Models[0].Manifest != manifestID || selected.Models[0].BindingPath != "judge.models.model" {
			t.Fatal("default input identity changed")
		}
		if turn == 0 && reads.Load() != 0 {
			t.Fatal("child resolution blocked on byte transfer before admission")
		}
		fatal(t, resolver.EnsureLocalModels(selected.Models))
		tool, problem := tfs.Open(cfg)
		fatal(t, problem)
		fatal(t, tool.VerifyManifest(manifestID))
		if reads.Load() != 2 {
			t.Fatalf("cold/warm native fetch %d read %d objects, want 2 total", turn, reads.Load())
		}
	}
	changed.Store(true)
	selected, _, problem := resolver.ResolveUnpublishedChild(parent, "captured", "judge", args)
	fatal(t, problem)
	problem = resolver.EnsureLocalModels(selected.Models)
	if problem == nil || problem.ErrName() != "child.model_manifest_changed" {
		t.Fatal("changed download resolution was accepted", problem)
	}
}
