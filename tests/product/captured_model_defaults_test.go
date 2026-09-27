package producttest

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/hub"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func requireCapturedModelDefaults(t *testing.T, python string) {
	t.Helper()
	out, err := exec.Command(python, "-c", "from cozy.worker.v1.wire_version import WIRE_MINOR; print(WIRE_MINOR)").CombinedOutput()
	if err != nil {
		t.Fatalf("actual Runtime wire: %v %s", err, out)
	}
	minor, err := strconv.Atoi(strings.TrimSpace(string(out)))
	must(t, err)
	if minor < int(pb.CapturedModelDefaultsWireMinor) {
		t.Skip("captured Model defaults require Runtime wire56")
	}
}

// A tiny native checkpoint backs both the metadata and actual public byte routes.
// No placement, execution, receipt, or Model construction is returned by this fixture.
func capturedDefaultCatalog(t *testing.T, seed servingSeed, python string) (*httptest.Server, *atomic.Int32) {
	t.Helper()
	code := `import base64,json,sys,tensorfs
s=tensorfs.Store.open(sys.argv[1]); manifest=sys.argv[2]
rows=s.walk(manifest)
objects={r["id"].removeprefix("sha256:"):base64.b64encode(s.document(r["id"],r["length"])).decode() for r in rows}
objects[manifest[7:]]=base64.b64encode(bytes(s.manifest(manifest)["manifest"])).decode()
print(json.dumps(objects))`
	out, err := exec.Command(python, "-c", code, seed.SourceRoot, seed.Manifest).CombinedOutput()
	if err != nil {
		t.Fatalf("read native catalog fixture: %v %s", err, out)
	}
	var encoded map[string]string
	must(t, json.Unmarshal(out, &encoded))
	objects := map[string][]byte{}
	for id, body := range encoded {
		objects[id], err = base64.StdEncoding.DecodeString(body)
		must(t, err)
	}
	var downloads atomic.Int32
	var server *httptest.Server
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models/proof/ordered", func(w http.ResponseWriter, r *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "proof", Name: "ordered"}, Releases: []hub.ModelReleaseSummary{{
			ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: seed.Manifest,
				Components: seed.HeaderComponents, ComponentBytes: seed.ComponentBytes, Bytes: seed.Bytes}},
		}}})
	})
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("ref") != "proof/ordered@"+seed.Manifest || r.URL.Query().Get("lane") != "" {
			http.Error(w, "unfrozen alias", http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/ordered", ManifestID: seed.Manifest, ManifestLength: seed.Length,
			HeaderID: seed.HeaderDigest, Components: seed.HeaderComponents, ComponentBytes: seed.ComponentBytes, Bytes: seed.Bytes})
	})
	mux.HandleFunc("POST /v1/models/proof/ordered/checkpoints/{manifest}/reads", func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "" || r.PathValue("manifest") != seed.Manifest {
			http.Error(w, "not this anonymous checkpoint", http.StatusForbidden)
			return
		}
		var ask struct {
			IDs []string `json:"object_ids"`
		}
		if json.NewDecoder(r.Body).Decode(&ask) != nil {
			http.Error(w, "malformed", 400)
			return
		}
		reads := []hub.Read{}
		for _, id := range ask.IDs {
			data, ok := objects[strings.TrimPrefix(id, "sha256:")]
			if !ok {
				http.Error(w, "outside checkpoint", 404)
				return
			}
			reads = append(reads, hub.Read{ObjectID: id, Length: int64(len(data)), URL: server.URL + "/objects/" + strings.TrimPrefix(id, "sha256:")})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"reads": reads})
	})
	mux.HandleFunc("POST /v1/tensorfs/closure", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			Ref string `json:"ref"`
		}
		if r.Header.Get("Authorization") != "" || json.NewDecoder(r.Body).Decode(&ask) != nil || ask.Ref != "proof/ordered@"+seed.Manifest {
			http.Error(w, "unfrozen or credentialed read", 403)
			return
		}
		ids := []string{}
		for id := range objects {
			if id != strings.TrimPrefix(seed.Manifest, "sha256:") {
				ids = append(ids, id)
			}
		}
		sort.Strings(ids)
		rows := []map[string]any{}
		for _, id := range ids {
			rows = append(rows, map[string]any{"sha256": id, "length": len(objects[id])})
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"complete": true, "scope": "runtime", "model": "proof/ordered", "release": "", "lane": "",
			"manifest": map[string]any{"sha256": strings.TrimPrefix(seed.Manifest, "sha256:"), "length": seed.Length}, "objects": rows, "presign_max_digests": 128, "server_time_unix": time.Now().Unix()})
	})
	mux.HandleFunc("POST /v1/tensorfs/presign", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			Digests []string `json:"digests"`
		}
		if r.Header.Get("Authorization") != "" || json.NewDecoder(r.Body).Decode(&ask) != nil {
			http.Error(w, "credentialed or malformed", 403)
			return
		}
		urls := map[string]string{}
		for _, id := range ask.Digests {
			if _, ok := objects[id]; !ok {
				http.Error(w, "outside checkpoint", 404)
				return
			}
			urls[id] = server.URL + "/objects/" + id
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"urls": urls, "server_time_unix": time.Now().Unix(), "expires_at_unix": time.Now().Add(time.Hour).Unix()})
	})
	mux.HandleFunc("GET /objects/{id}", func(w http.ResponseWriter, r *http.Request) {
		data, ok := objects[r.PathValue("id")]
		if !ok {
			http.NotFound(w, r)
			return
		}
		downloads.Add(1)
		w.Header().Set("Content-Length", strconv.Itoa(len(data)))
		_, _ = w.Write(data)
	})
	server = httptest.NewServer(mux)
	return server, &downloads
}

func TestCapturedModelDefaultsDoNotAcquireUnusedOrInaccessibleModels(t *testing.T) {
	integration(t)
	if *privateChildRuntimeWheel == "" {
		t.Skip("requires exact Runtime execution peer")
	}
	wheel, err := filepath.Abs(*privateChildRuntimeWheel)
	must(t, err)
	version := runtimeFixtureVersion(t, wheel)
	control := filepath.Join(t.TempDir(), "control")
	python := filepath.Join(control, "bin", "python")
	for _, args := range [][]string{{"venv", control, "--python", "3.12"}, {"pip", "install", "--python", python, wheel}} {
		out, err := exec.Command("uv", args...).CombinedOutput()
		if err != nil {
			t.Fatalf("SDK: %v %s", err, out)
		}
	}
	requireCapturedModelDefaults(t, python)
	root, err := os.MkdirTemp("", "cozy-defaults-")
	must(t, err)
	defer tracePrivateChildWait(t, root)()
	path := filepath.Join(control, "bin")
	for _, item := range childEnv(t, root) {
		if strings.HasPrefix(item, "PATH=") {
			path += string(os.PathListSeparator) + strings.TrimPrefix(item, "PATH=")
		}
	}
	t.Cleanup(func() {
		if _, err := os.Stat(filepath.Join(root, "creator.sqlite")); err == nil {
			compositionDown(t, root, path)
		} else if !os.IsNotExist(err) {
			t.Error(err)
		}
		if t.Failed() {
			t.Log("default capture evidence retained", root)
		} else {
			must(t, removeAllForce(root))
		}
	})
	project := t.TempDir()
	library := filepath.Join(project, "library")
	must(t, os.MkdirAll(library, 0700))
	seedBytes, err := exec.Command(python, filepath.Join("testdata", "local_serving_preparation", "seed.py"), filepath.Join(project, "catalog")).CombinedOutput()
	if err != nil {
		t.Fatalf("native seed: %v %s", err, seedBytes)
	}
	var seed servingSeed
	must(t, json.Unmarshal(seedBytes, &seed))
	catalog, downloads := capturedDefaultCatalog(t, seed, python)
	defer catalog.Close()
	must(t, os.WriteFile(filepath.Join(root, "config.yaml"), []byte("tensorhub_url: "+catalog.URL+"\ntensorhub_token: must-not-reach-byte-probe\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	metadata := fmt.Sprintf(`[project]
name="default-tools"
version="1.0.0"
requires-python=">=3.12,<3.13"
dependencies=["cozy-runtime>=%s"]
[project.entry-points."cozy.application"]
default="default_tools:app"
[tool.uv.sources]
cozy-runtime={path=%s}
[build-system]
requires=["hatchling"]
build-backend="hatchling.build"
[tool.hatch.build.targets.wheel]
only-include=["default_tools.py"]
`, version, strconv.Quote(wheel))
	must(t, os.WriteFile(filepath.Join(library, "pyproject.toml"), []byte(metadata), 0600))
	must(t, os.WriteFile(filepath.Join(library, "package.toml"), []byte("[application]\nobject=\"default_tools:app\"\n"), 0600))
	module := `from typing import Any
import msgspec
from cozy_runtime.author import App, Context, Model
class Request(msgspec.Struct):
    value:int=1
class Result(msgspec.Struct):
    value:int
class Weights(Model[object]):
    def load(self,loader:Any)->None:
        raise AssertionError("unused model initialized")
app=App()
@app.entrypoint(defaults={"model":[{"gpu":"*","lane":"proof/ordered@1.0.0/bf16"}]})
def available(payload:Request,model:Weights)->Result:
    raise AssertionError("unused inference executed")
@app.entrypoint(defaults={"model":[{"gpu":"*","lane":"proof/inaccessible@1.0.0/bf16"}]})
def unavailable(payload:Request,model:Weights)->Result:
    raise AssertionError("inaccessible inference executed")
`
	must(t, os.WriteFile(filepath.Join(library, "default_tools.py"), []byte(module), 0600))
	body := fmt.Sprintf(`# /// script
# requires-python=">=3.12,<3.13"
# dependencies=["cozy-runtime>=%s","default-tools>=1.0.0"]
# [tool.uv.sources]
# cozy-runtime={path=%s}
# default-tools={path="./library",editable=true}
# ///
from default_tools import available,unavailable
async def main()->int:
    return 7
`, version, strconv.Quote(wheel))
	script := filepath.Join(project, "use.py")
	must(t, os.WriteFile(script, []byte(body), 0600))
	status, out := runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status != 0 || !strings.Contains(out, `"value":7`) {
		t.Fatalf("unused defaults [%d]: %s", status, out)
	}
	if downloads.Load() != 0 {
		t.Fatal("importing a library downloaded unused model payloads")
	}
	db, err := sql.Open("sqlite", "file:"+filepath.Join(root, "tensorfs", ".cozy-workspace", "journal.sqlite3")+"?mode=ro")
	must(t, err)
	defer db.Close()
	var capture []byte
	must(t, db.QueryRow("SELECT capture_document FROM executions WHERE json_extract(CAST(capture_document AS TEXT),'$.model_defaults') IS NOT NULL LIMIT 1").Scan(&capture))
	// Decode through generic fields to keep this assertion on the actual retained JSON.
	var document map[string]any
	must(t, json.Unmarshal(capture, &document))
	rows := document["model_defaults"].([]any)
	if len(rows) != 2 {
		t.Fatalf("incomplete default inventory: %s", capture)
	}
	if rows[0].(map[string]any)["entrypoint"] != "available" || rows[1].(map[string]any)["unavailable_code"] != "model_default_unavailable" {
		t.Fatalf("default availability changed: %s", capture)
	}
	if !strings.Contains(string(capture), seed.Manifest) || strings.Contains(string(capture), "must-not-reach") {
		t.Fatal("capture lost immutable checkpoint or retained a credential")
	}
	body = strings.Replace(body, "return 7", "return (await unavailable()).value", 1)
	must(t, os.WriteFile(script, []byte(body), 0600))
	status, out = runCozyPath(t, root, path, "run", script, "--await", "--json")
	if status == 0 || !strings.Contains(out, "captured Model default is unavailable") {
		t.Fatalf("used unavailable default [%d]: %s", status, out)
	}
	var count int
	must(t, db.QueryRow("SELECT count(*) FROM execution_checkpoint_inputs").Scan(&count))
	if count != 0 || downloads.Load() != 0 {
		t.Fatal("unavailable default acquired model data or a pin")
	}
}
