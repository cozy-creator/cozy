package producttest

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// attachedRental gives root a ready rental named "attached" whose machine is pod, its Hub
// stand, and a running daemon: what `cozy rental new` leaves once the machine is up.
func attachedRental(t *testing.T, pod *fakePod) (string, *fakeRentalHub) {
	t.Helper()
	root, err := os.MkdirTemp(scratchBase, "rental-gaps-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	identity, problem := rental.PendingCreatorIdentity(layout, "attached")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod.controlKey, pod.resources = public, &pb.WorkerResources{BackendVersion: "12.8"}
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
	return root, stand
}

// `cozy rental show` names the agent, Runtime and TensorFS the machine says it runs. A Host
// that predates DescribeMachine still shows its rental, with a note instead of versions.
func TestRentalShowNamesTheMachinesSoftware(t *testing.T) {
	var describes atomic.Int32
	pod := &fakePod{describe: func(*pb.DescribeMachineQuery) (*pb.MachineDescription, error) {
		if describes.Add(1) > 2 {
			return nil, status.Error(codes.Unimplemented, "an older Host")
		}
		return &pb.MachineDescription{Host: &pb.MachineHost{Version: "0.1.9"},
			Runtime: &pb.MachineRuntime{Version: "0.18.99", TensorfsVersion: "0.3.81"}}, nil
	}}
	root, _ := attachedRental(t, pod)
	code, out := runCozy(t, root, "rental", "show", "attached", "--json")
	var shown map[string]any
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &shown) != nil || shown["runtime_version"] != "0.18.99" ||
		shown["tensorfs_version"] != "0.3.81" || shown["agent_version"] != "0.1.9" {
		t.Fatalf("rental show did not name the machine's software [exit %d]:\n%s", code, out)
	}
	if code, out = runCozy(t, root, "rental", "show", "attached"); code != 0 || !strings.Contains(out, "0.3.81") || !strings.Contains(out, "0.18.99") {
		t.Fatalf("the human rental show hides the versions [exit %d]:\n%s", code, out)
	}
	if code, out = runCozy(t, root, "rental", "show", "attached", "--json"); code != 0 || strings.Contains(out, "runtime_version") ||
		!strings.Contains(out, "software versions unavailable") {
		t.Fatalf("an older Host must leave a note, not fail the show [exit %d]:\n%s", code, out)
	}
}

// `cozy model download --rental --await` waits for the machine's verified receipt, saying
// what it reports on the way; a refusal ends the wait with the machine's reason.
func TestRentalModelDownloadAwaitsWithProgress(t *testing.T) {
	const gib = uint64(1 << 30)
	var downloads atomic.Int32
	pod := &fakePod{prepareModels: func(_ *pb.PreparePackageSetCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if downloads.Add(1) > 1 {
			return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_REFUSED,
				SafeCode: "store_full", SafeDetail: "the store has no room for 3 GiB"})
		}
		setBytes, setDigest, err := canonical.Identity(&pb.PlacementSet{})
		if err != nil {
			return err
		}
		for _, event := range []*pb.PrepareEvent{
			{Stage: pb.PrepareStage_PREPARE_STAGE_RESOLVED, TotalBytes: 3 * gib},
			{Stage: pb.PrepareStage_PREPARE_STAGE_DOWNLOADING, TotalBytes: 3 * gib, TransferredBytes: gib},
			{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, TotalBytes: 3 * gib, TransferredBytes: 3 * gib,
				PlacementSet: &pb.DesiredPlacementSet{PlacementSetDigest: setDigest, PlacementSetCanonicalBytes: setBytes}},
		} {
			if err := stream.Send(event); err != nil {
				return err
			}
			time.Sleep(2500 * time.Millisecond)
		}
		return nil
	}}
	root, stand := attachedRental(t, pod)
	stand.mux.HandleFunc("GET /v1/models/paul/minimax-h3", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "paul", Name: "minimax-h3"},
			Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "2.0", CutAt: "2026-02-01T00:00:00Z"},
				Lanes: []hub.ModelLaneSummary{{Lane: "fp8", ManifestID: "sha256:" + strings.Repeat("b", 64), Bytes: int64(3 * gib),
					ComponentBytes: map[string]int64{"transformer": int64(3 * gib)}}}}}})
	})
	code, out, progress := runCozyStreams(t, root, "model", "download", "paul/minimax-h3#fp8", "--rental=attached", "--await", "--json")
	var settled map[string]any
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &settled) != nil || settled["status"] != "completed" || settled["kind"] != "download" || settled["elapsed"] == nil {
		t.Fatalf("the awaited download did not report its verified end [exit %d]:\n%s\n%s", code, out, progress)
	}
	if !strings.Contains(progress, "on attached · running · downloading models · 1.0GiB / 3.0GiB") {
		t.Fatalf("the wait printed no download progress:\n%s", progress)
	}
	code, out, progress = runCozyStreams(t, root, "model", "download", "paul/minimax-h3#fp8", "--rental=attached", "--await", "--json")
	if code == 0 || !strings.Contains(out+progress, "store_full") {
		t.Fatalf("a refused download must end the wait with the machine's reason [exit %d]:\n%s\n%s", code, out, progress)
	}
}

// `cozy model gc --rental` reclaims the rental's store through its machine, and says when
// the machine deferred collection because bytes were still moving in.
func TestModelGCReclaimsARentalsStore(t *testing.T) {
	var prunes atomic.Int32
	pod := &fakePod{prune: func(*pb.PruneOperationCacheCall) (*pb.PruneOperationCacheResult, error) {
		if prunes.Add(1) > 1 {
			return &pb.PruneOperationCacheResult{StoreBusy: true}, nil
		}
		return &pb.PruneOperationCacheResult{RemovedEntries: 2, ReclaimedBytes: 5 << 30}, nil
	}}
	root, _ := attachedRental(t, pod)
	if code, out := runCozy(t, root, "model", "gc", "--rental=attached"); code != 0 || !strings.Contains(out, "5.0GiB") {
		t.Fatalf("model gc did not reclaim the rental's store [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "model", "gc", "--rental=attached", "--json"); code != 0 ||
		!strings.Contains(out, `"store_busy":true`) || !strings.Contains(out, "collection deferred") {
		t.Fatalf("a busy store must be said, not reported as an empty success [exit %d]:\n%s", code, out)
	}
	if code, out := runCozy(t, root, "model", "gc", "--rental=nobody"); code == 0 || !strings.Contains(out, "no rental") {
		t.Fatalf("an unknown rental must be refused [exit %d]:\n%s", code, out)
	}
}

func lastJSONLine(out string) string {
	lines := strings.Split(strings.TrimSpace(out), "\n")
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.HasPrefix(strings.TrimSpace(lines[i]), "{") {
			return lines[i]
		}
	}
	return ""
}
