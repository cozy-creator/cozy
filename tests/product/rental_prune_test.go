package producttest

import (
	"net/http"
	"runtime"
	"strings"
	"testing"

	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

func TestRentalPruneRequiresAuthenticatedKnownWorkspace(t *testing.T) {
	root := t.TempDir()
	daemon := startDaemonProcess(t, root)
	path := "/v1/local/rentals/rental-prune-absent/prune"
	unauthorized := daemon.call(t, http.MethodPost, path, map[string]any{}, "Authorization", "")
	if unauthorized.Status != http.StatusUnauthorized {
		t.Fatalf("unauthenticated cache prune: %s", unauthorized.brief())
	}
	foreign := daemon.call(t, http.MethodPost, path, map[string]any{"path": "/other-store"})
	if foreign.Status != http.StatusBadRequest {
		t.Fatalf("cache prune admitted a caller-selected path: %s", foreign.brief())
	}
	missing := daemon.call(t, http.MethodPost, path, map[string]any{})
	if missing.Status != http.StatusNotFound {
		t.Fatalf("cache prune did not resolve owned workspace: %s", missing.brief())
	}
	code, out := runCozy(t, root, "rental", "prune", "rental-prune-absent", "--json")
	if code == 0 || !strings.Contains(out, `"code":"not_found"`) {
		t.Fatalf("prune CLI bypassed rental lookup [%d]: %s", code, out)
	}
	if rows := listInvocations(t, root); len(rows) != 0 {
		t.Fatal("pruning created an execution request")
	}
}

func TestLocalCachePruneRequiresItsWorkspaceWithoutCreatingAJob(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("the startup-refusing Runtime fixture is a POSIX shell script")
	}
	// This tool passes version admission but exits before publishing a worker
	// address. A service failure must not become an all-zero successful prune.
	root, path := hostRuntimeRoot(t, "cache-prune-refused", stubRuntime(t, "0.4.0", pb.WireMinor))
	code, out := runCozyPath(t, root, path, "cache", "prune", "--json")
	if code == 0 || !strings.Contains(out, `"code":"workspace.control_unavailable"`) || strings.Contains(out, `"removed_entries"`) {
		t.Fatalf("workspace startup failure became cache-prune success [%d]: %s", code, out)
	}
	if rows := listInvocations(t, root); len(rows) != 0 {
		t.Fatal("local cache pruning created an execution request")
	}
}
