package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
)

// A CLI update can precede its daemon update. Success is already authoritative
// in the old response; failure without a saved percentage remains unknown.
func TestRunListCompletedProgressWithOldDaemon(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/" {
			w.WriteHeader(http.StatusOK)
			return
		}
		if r.URL.Path != "/v1/requests" {
			t.Errorf("unexpected daemon route: %s", r.URL.Path)
			w.WriteHeader(http.StatusNotFound)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"requests": []api.Lifecycle{
			{Number: 1, Status: "completed", Package: "proof/old-completed", Function: "run"},
			{Number: 2, Status: "failed", Package: "proof/old-failed", Function: "run"},
		}})
	}))
	defer server.Close()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	held, problem := daemon.Hold(layout, strings.TrimPrefix(server.URL, "http://"), "")
	fatal(t, problem)
	defer held.Release()
	_, problem = api.Mint(layout)
	fatal(t, problem)
	code, output := runCozy(t, root, "run", "list", "--full")
	if code != 0 {
		t.Fatalf("old daemon list [exit %d]: %s", code, output)
	}
	for _, line := range strings.Split(output, "\n") {
		if strings.Contains(line, "proof/old-completed/run") && !strings.Contains(line, "100%") {
			t.Fatalf("old daemon success did not render 100%%: %s", output)
		}
		if strings.Contains(line, "proof/old-failed/run") && strings.Contains(line, "%") {
			t.Fatalf("old daemon failure invented progress: %s", output)
		}
	}
	if !strings.Contains(output, "proof/old-completed/run") || !strings.Contains(output, "proof/old-failed/run") {
		t.Fatalf("old daemon rows absent: %s", output)
	}
}
