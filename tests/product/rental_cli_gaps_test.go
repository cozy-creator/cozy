package producttest

import (
	"context"
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
	if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &settled) != nil || settled["status"] != "succeeded" || settled["elapsed"] == nil {
		t.Fatalf("the awaited download did not report its verified end [exit %d]:\n%s\n%s", code, out, progress)
	}
	if !strings.Contains(progress, "attached: downloading 1.0GiB of 3.0GiB (33%)") {
		t.Fatalf("the wait printed no download progress:\n%s", progress)
	}
	code, out, progress = runCozyStreams(t, root, "model", "download", "paul/minimax-h3#fp8", "--rental=attached", "--await", "--json")
	if code == 0 || !strings.Contains(out+progress, "store_full") {
		t.Fatalf("a refused download must end the wait with the machine's reason [exit %d]:\n%s\n%s", code, out, progress)
	}
}

// A cozy and a machine a release apart share no protocol for some verb. Each way round the
// user reads one message naming the side to upgrade, never the transport's own error.
func TestProtocolSkewNamesTheSideToUpgrade(t *testing.T) {
	// An older machine serves cozy.worker.v1 only. `rental show` reads cozy.machine.v1 Status
	// and still shows what the Hub knows; `rental keepalive` has nothing else to do.
	root, _ := attachedRental(t, &fakePod{})
	code, out := runCozy(t, root, "rental", "show", "attached")
	if code != 0 || !strings.Contains(out, "the machine is older than this cozy") || strings.Contains(out, "upgrade cozy") {
		t.Fatalf("an older machine was not named as the side to upgrade [exit %d]:\n%s", code, out)
	}
	code, out = runCozy(t, root, "rental", "keepalive", "attached")
	if code != 1 || !strings.Contains(out, "the machine is older than this cozy") || !strings.Contains(out, "update the machine") || strings.Contains(out, "upgrade cozy") {
		t.Fatalf("an older machine was not named as the side to upgrade [exit %d]:\n%s", code, out)
	}
	// A newer machine serves cozy.machine.v1 and not every worker.v1 call this cozy still makes
	// (`rental logs --tensorfs` reads ReadMachineLog); one past worker.v1 keeps only
	// ProtocolInfo, to say so. Either way this cozy is the side to upgrade.
	for name, pod := range map[string]*fakePod{"unported call": {servesV1: true},
		"past worker.v1": {protocolInfo: func(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
			return nil, status.Error(codes.FailedPrecondition, "this machine serves cozy.machine.v1 only; upgrade cozy")
		}}} {
		root, _ = attachedRental(t, pod)
		code, out = runCozy(t, root, "rental", "logs", "attached", "--tensorfs")
		if code != 1 || !strings.Contains(out, "is newer than this cozy") || !strings.Contains(out, "upgrade cozy") || strings.Contains(out, "update the machine") {
			t.Fatalf("%s: this cozy was not named as the side to upgrade [exit %d]:\n%s", name, code, out)
		}
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
