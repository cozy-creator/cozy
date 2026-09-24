package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
)

func TestTensorhubFlagOverridesFileAndEnvironment(t *testing.T) {
	var configured, selected atomic.Int32
	serve := func(count *atomic.Int32) *httptest.Server {
		return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			count.Add(1)
			if r.URL.Path != "/v1/models" {
				t.Errorf("unexpected route %s", r.URL.Path)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"models": []any{}, "search": map[string]any{"total": 0, "capped": false}})
		}))
	}
	first, second := serve(&configured), serve(&selected)
	defer first.Close()
	defer second.Close()
	root := t.TempDir()
	file := filepath.Join(root, config.FileName)
	original := []byte("tensorhub_url: " + first.URL + "\n")
	must(t, os.WriteFile(file, original, 0600))
	for _, args := range [][]string{
		{"--tensorhub=" + second.URL + "/", "model", "search", "--json"},
		{"model", "search", "--tensorhub", second.URL, "--json"},
	} {
		cmd := exec.Command(cozyBin, args...)
		cmd.Env = childEnv(t, root, "TENSORHUB_URL="+first.URL)
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("override failed: %v %s", err, out)
		}
	}
	if selected.Load() != 2 || configured.Load() != 0 {
		t.Fatalf("selected=%d configured=%d", selected.Load(), configured.Load())
	}
	raw, err := os.ReadFile(file)
	must(t, err)
	if string(raw) != string(original) {
		t.Fatal("one-command override rewrote configuration")
	}
	code, out := runCozy(t, root, "model", "search", "--json")
	if code != 0 || configured.Load() != 1 {
		t.Fatalf("config fallback failed: %d %s", code, out)
	}
	for _, target := range []string{"", "relative", "file:///tmp/hub", "https://user:secret@hub.invalid", "https://hub.invalid?q=x"} {
		code, out := runCozy(t, root, "--tensorhub="+target, "model", "search", "--json")
		if code == 0 || !strings.Contains(out, "config.tensorhub_url_invalid") {
			t.Fatalf("invalid URL accepted: %d %s", code, out)
		}
		if strings.Contains(out, "secret") {
			t.Fatal("URL credentials leaked in error")
		}
	}
}

func TestTensorhubFlagRefusesDifferentDaemonOrigin(t *testing.T) {
	var submitted atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			submitted.Add(1)
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	fatal(t, held.PublishTensorhub("https://local-hub.invalid"))
	_, problem = api.Mint(layout)
	fatal(t, problem)
	code, out := runCozy(t, root, "--tensorhub=https://other-hub.invalid", "run", "list", "--json")
	if code == 0 || !strings.Contains(out, "daemon.tensorhub_mismatch") {
		t.Fatalf("cross-Hub submission not refused: %d %s", code, out)
	}
	if submitted.Load() != 0 {
		t.Fatalf("cross-Hub command submitted %d API calls", submitted.Load())
	}
}
