package producttest

import (
	"crypto/rand"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
)

// A real CozyTensors model: the fixture header with its asset list replaced by eight
// distinct 512 KiB blobs. The native tfs CLI encodes it, verifies it and walks its
// closure; the stand-in Hub only serves bytes.
type cancelFixture struct {
	manifest, header       []byte
	manifestID, headerID   string
	assets                 []string
	objects                map[string][]byte
	assetBytes, totalBytes int64
}

func newCancelFixture(t *testing.T, tfs string) cancelFixture {
	t.Helper()
	dir := t.TempDir()
	decoded, err := exec.Command(tfs, "cbor", "decode",
		filepath.Join("testdata", "unpublished_preparation", "model.cozytensors")).Output()
	must(t, err)
	var header map[string]any
	must(t, json.Unmarshal(decoded, &header))
	f := cancelFixture{objects: map[string][]byte{}}
	var assets []any
	for i := range 8 {
		body := make([]byte, 512<<10)
		_, err := rand.Read(body)
		must(t, err)
		id := assessmentDigest(body)
		f.objects[id], f.assets = body, append(f.assets, id)
		f.assetBytes += int64(len(body))
		assets = append(assets, []any{fmt.Sprintf("weights/part-%d.bin", i), id, len(body),
			"application/octet-stream", []any{[]any{id, len(body)}}})
	}
	header["assets"] = assets
	raw, err := json.Marshal(header)
	must(t, err)
	must(t, os.WriteFile(filepath.Join(dir, "header.json"), raw, 0o600))
	f.header, err = exec.Command(tfs, "cbor", "encode", filepath.Join(dir, "header.json")).Output()
	must(t, err)
	f.headerID = assessmentDigest(f.header)
	f.objects[f.headerID] = f.header
	f.manifest = fmt.Appendf(nil, `{"entries":[{"blob":{"length":%d,"sha256":"%s"},"kind":"cozytensors","path":"model.cozytensors"}]}`,
		len(f.header), strings.TrimPrefix(f.headerID, "sha256:"))
	f.manifestID = assessmentDigest(f.manifest)
	f.totalBytes = f.assetBytes + int64(len(f.header))
	return f
}

// cancelHub serves the fixture. While holding, the first `servedBeforeHold` asset bodies
// arrive whole; the next asset body stops halfway and stays open until its client leaves.
type cancelHub struct {
	fixture cancelFixture
	mu      sync.Mutex
	holding bool
	served  int
	heldID  string
	gets    map[string]int
	grants  map[string]int
	held    chan struct{} // an asset body is open and half sent
	left    chan struct{} // its client closed the connection
}

const servedBeforeHold = 3

func (h *cancelHub) serve(t *testing.T) *httptest.Server {
	var server *httptest.Server
	f := h.fixture
	resolution := hub.ModelResolution{Model: "proof/big", Release: "1.0.0", Lane: "bf16",
		ManifestID: f.manifestID, HeaderID: f.headerID, ManifestLength: int64(len(f.manifest)),
		Objects: len(f.objects), Bytes: f.totalBytes}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		if !strings.HasPrefix(r.URL.Query().Get("ref"), "proof/big@") {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"model.not_found","message":"no such model"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(resolution)
	})
	mux.HandleFunc("GET /v1/models/proof/big/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(f.manifest)
	})
	mux.HandleFunc("POST /v1/models/proof/big/releases/1.0.0/lanes/bf16/reads", func(w http.ResponseWriter, r *http.Request) {
		var ask struct {
			IDs []string `json:"object_ids"`
		}
		must(t, json.NewDecoder(r.Body).Decode(&ask))
		reads := []hub.Read{}
		h.mu.Lock()
		for _, id := range ask.IDs {
			h.grants[id]++
			reads = append(reads, hub.Read{ObjectID: id, Length: int64(len(f.objects[id])),
				URL: server.URL + "/objects/" + id})
		}
		h.mu.Unlock()
		_ = json.NewEncoder(w).Encode(map[string]any{"reads": reads})
	})
	mux.HandleFunc("GET /objects/{id}", func(w http.ResponseWriter, r *http.Request) {
		id := r.PathValue("id")
		body := f.objects[id]
		h.mu.Lock()
		h.gets[id]++
		hold := h.holding && id != f.headerID && h.served == servedBeforeHold
		if id != f.headerID {
			h.served++
		}
		if hold {
			h.heldID = id
		}
		h.mu.Unlock()
		if !hold {
			_, _ = w.Write(body)
			return
		}
		w.Header().Set("Content-Length", fmt.Sprint(len(body)))
		_, _ = w.Write(body[:len(body)/2])
		w.(http.Flusher).Flush()
		close(h.held)
		<-r.Context().Done()
		close(h.left)
	})
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		t.Logf("stand-in Hub has no route for %s %s", r.Method, r.URL.Path)
		http.NotFound(w, r)
	})
	server = httptest.NewServer(mux)
	t.Cleanup(server.Close)
	return server
}

func (h *cancelHub) assetGets() (total int, per map[string]int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	per = map[string]int{}
	for _, id := range h.fixture.assets {
		per[id] = h.gets[id]
		total += h.gets[id]
	}
	return total, per
}

// Finding #5: `cozy run cancel` of a local `cozy model download` must stop the Hub
// transfer at once, report canceled, and keep every object that fully arrived so the
// next download resumes after them.
func TestCanceledModelDownloadStopsAndResumes(t *testing.T) {
	tfs, err := exec.LookPath("tfs")
	if err != nil {
		t.Skip("requires the native tfs CLI")
	}
	stand := &cancelHub{fixture: newCancelFixture(t, tfs), holding: true, gets: map[string]int{},
		grants: map[string]int{}, held: make(chan struct{}), left: make(chan struct{})}
	server := stand.serve(t)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+
		"\ntensorhub_token: cancel-download-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)

	code, out, errOut := runCozyStreams(t, root, "model", "download", "proof/big@1.0.0", "local/big",
		"--lane", "bf16", "--json")
	if code != 0 {
		t.Fatalf("download submission failed [exit %d]\n%s\n%s", code, out, errOut)
	}
	run := submittedRunReference(t, code, out)
	select {
	case <-stand.held:
	case <-time.After(2 * time.Minute):
		t.Fatalf("the download never reached its fourth object\n%s", daemonLogTail(root))
	}

	code, out = runCozy(t, root, "run", "cancel", run, "--json")
	if code != 0 || !strings.Contains(out, `"status":"cancelled by user"`) {
		t.Fatalf("cancel did not settle the download [exit %d]\n%s", code, out)
	}
	// The open body is abandoned, not drained: its client hangs up.
	select {
	case <-stand.left:
	case <-time.After(10 * time.Second):
		t.Fatalf("the canceled download still holds its object open\n%s", daemonLogTail(root))
	}
	moved, _ := stand.assetGets()
	if moved != servedBeforeHold+1 {
		t.Fatalf("canceled download asked for %d asset bodies, want %d", moved, servedBeforeHold+1)
	}
	code, out = runCozy(t, root, "run", "watch", run, "--json")
	if !strings.Contains(out, "cancelled by user") {
		t.Fatalf("canceled run does not report canceled [exit %d]\n%s", code, out)
	}
	if after, _ := stand.assetGets(); after != moved {
		t.Fatalf("canceled download kept moving bytes: %d asset requests, then %d", moved, after)
	}

	// The re-run resumes: objects that fully arrived are neither granted nor fetched
	// again; the half-sent one and those never started move exactly once.
	stand.mu.Lock()
	stand.holding = false
	firstGrants := map[string]int{}
	for id, n := range stand.grants {
		firstGrants[id] = n
	}
	stand.mu.Unlock()
	_, before := stand.assetGets()
	code, out, errOut = runCozyStreams(t, root, "model", "download", "proof/big@1.0.0", "local/big",
		"--lane", "bf16", "--json", "--await")
	if code != 0 || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("resumed download did not complete [exit %d]\n%s\n%s\n%s", code, out, errOut,
			daemonLogTail(root))
	}
	_, after := stand.assetGets()
	stand.mu.Lock()
	defer stand.mu.Unlock()
	whole := 0
	for _, id := range stand.fixture.assets {
		fetched, granted := after[id]-before[id], stand.grants[id]-firstGrants[id]
		if before[id] == 1 && id != stand.heldID {
			whole++
			if fetched != 0 || granted != 0 {
				t.Fatalf("object %s arrived whole before the cancel, yet the re-run was granted it %d "+
					"times and fetched it %d times", id, granted, fetched)
			}
			continue
		}
		if fetched != 1 {
			t.Fatalf("object %s moved %d times in the resumed download, want once", id, fetched)
		}
	}
	if whole != servedBeforeHold {
		t.Fatalf("%d objects arrived whole before the cancel, want %d", whole, servedBeforeHold)
	}
}

func daemonLogTail(root string) string {
	raw, _ := os.ReadFile(filepath.Join(root, "daemon.log"))
	if len(raw) > 4000 {
		raw = raw[len(raw)-4000:]
	}
	return string(raw)
}
