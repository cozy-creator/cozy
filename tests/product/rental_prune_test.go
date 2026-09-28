package producttest

import (
	"encoding/base64"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
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

// A CPU pod booted on a CUDA-torch image reads back no accelerator, but still the toolkit its
// torch was built for (as Runtime's host facts report it): its machine attaches, and its
// operation cache prunes.
func TestCPURentalOnACUDATorchImageAttaches(t *testing.T) {
	root, err := os.MkdirTemp(scratchBase, "cpu-rental-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	identity, problem := rental.PendingCreatorIdentity(layout, "cpu-rental")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod := &fakePod{controlKey: public, resources: &pb.WorkerResources{BackendVersion: "12.8"},
		prune: func(*pb.PruneOperationCacheCall) (*pb.PruneOperationCacheResult, error) {
			return &pb.PruneOperationCacheResult{RemovedEntries: 3}, nil
		}}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	stand := newFakeRentalHub(t, 0)
	stand.publishListing()
	stand.add(podRental, "attached")
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "attached", State: "ready", SKU: "cpu",
		AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, Hub: stand.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
		ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+stand.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	startDaemonProcess(t, root)
	if code, out := runCozy(t, root, "rental", "prune", "attached", "--json"); code != 0 || !strings.Contains(out, `"removed_entries":3`) {
		t.Fatalf("the CPU rental's machine did not attach [exit %d]: %s", code, out)
	}
}
