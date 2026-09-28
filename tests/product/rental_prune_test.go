package producttest

import (
	"net/http"
	"os/exec"
	"strings"
	"testing"
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

// This computer's cache is its machine's, pruned through the same Host call as a rental's.
// Without a machine there is nothing to prune, and a refusal is never an all-zero success.
func TestLocalCachePruneRequiresItsMachineWithoutCreatingAJob(t *testing.T) {
	root := t.TempDir()
	cmd := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "cache", "prune", "--json")
	cmd.Env = childEnv(t, root)
	raw, _ := cmd.CombinedOutput()
	out := string(raw)
	if *machineHostBinary == "" && (cmd.ProcessState.ExitCode() == 0 || !strings.Contains(out, `"code":"machine.not_installed"`) || strings.Contains(out, `"removed_entries"`)) {
		t.Fatalf("a missing machine became cache-prune success [%d]: %s", cmd.ProcessState.ExitCode(), out)
	}
	if rows := listInvocations(t, root); len(rows) != 0 {
		t.Fatal("local cache pruning created an execution request")
	}
}
