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

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
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

// `up` with a one-command hub starts the one daemon on the configured default, and that
// daemon then serves commands for any hub.
func TestTensorhubFlagDoesNotBindTheDaemon(t *testing.T) {
	server := httptest.NewServer(http.NotFoundHandler())
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: https://file-hub.invalid\nport: 0\n"), 0600))
	cmd := exec.Command(cozyBin, "--tensorhub="+server.URL, "up", "--json")
	// Missing local Runtime is allowed for a rental-only host. Keep this proof
	// independent of whichever Python tools the developer has installed.
	cmd.Env = childEnv(t, root, "PATH="+t.TempDir())
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("daemon startup failed: %v %s", err, out)
	}
	state := daemon.Probe(config.Config{Home: root})
	if !state.Up {
		t.Fatalf("daemon did not start as a multi-hub daemon: %+v", state)
	}
	for _, args := range [][]string{{"run", "list", "--json"}, {"--tensorhub=" + server.URL, "run", "list", "--json"}} {
		if code, outText := runCozy(t, root, args...); code != 0 {
			t.Fatalf("%v was refused by the multi-hub daemon: %d %s", args, code, outText)
		}
	}
	if after := daemon.Probe(config.Config{Home: root}); after.PID != state.PID {
		t.Fatalf("a hub selection restarted the daemon: %d -> %d", state.PID, after.PID)
	}
}

// TENSORHUB_URL outranks the saved current hub in its shell; `hub use` and `hub list` say so
// instead of claiming a switch that shell will not see.
func TestHubUseNamesAnEnvironmentOverride(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("hubs:\n  local: http://127.0.0.1:9\n"), 0o600))
	for _, args := range [][]string{{"hub", "use", "local"}, {"hub", "list"}} {
		cmd := exec.Command(cozyBin, args...)
		cmd.Env = childEnv(t, root, "TENSORHUB_URL="+testDefaultHub)
		out, err := cmd.CombinedOutput()
		if err != nil || !strings.Contains(string(out), "TENSORHUB_URL in this environment selects the hub") {
			t.Fatalf("%v hid the environment override: %v\n%s", args, err, out)
		}
	}
}
