package producttest

import (
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// Ordinary CLI -> actual daemon/records -> signed TLS Host. Only explicit calls
// reach the keepalive method; status, duplicate IDs, restart and failures cannot renew.
func TestRentalKeepaliveCLIResetsOnlyAfterAcknowledgment(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "keepalive-cli")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod := &fakePod{controlKey: public}
	var mu sync.Mutex
	receipts := map[string]*pb.KeepRentalAliveResult{}
	clock := time.Now().UTC().Truncate(time.Millisecond)
	calls := 0
	failed := false
	pod.keepalive = func(request *pb.KeepRentalAliveRequest) (*pb.KeepRentalAliveResult, error) {
		mu.Lock()
		defer mu.Unlock()
		calls++
		if failed {
			return nil, status.Error(codes.Unavailable, "unconfirmed reset")
		}
		if prior := receipts[request.RequestId]; prior != nil {
			return proto.Clone(prior).(*pb.KeepRentalAliveResult), nil
		}
		clock = clock.Add(time.Second)
		result := &pb.KeepRentalAliveResult{RequestId: request.RequestId, WorkerId: podWorkerID, WorkerBootId: podBootID, AcknowledgedAtUnixMs: clock.UnixMilli(), IdleDeadlineUnixMs: clock.Add(900 * time.Second).UnixMilli()}
		receipts[request.RequestId] = result
		return proto.Clone(result).(*pb.KeepRentalAliveResult), nil
	}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "keepalive")
	hub.mu.Lock()
	hub.rentals[podRental]["requested_accelerator_model"] = "fake-4090"
	hub.mu.Unlock()
	row := records.Rental{ID: podRental, MachineName: "keepalive", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	daemon := startDaemonProcess(t, root)
	command := func() string {
		t.Helper()
		code, out := runCozy(t, root, "rental", "keepalive", "keepalive", "--json", "--full")
		if code != 0 {
			t.Fatalf("keepalive CLI: %d %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
		}
		var result struct {
			RequestID  string `json:"request_id"`
			ReleaseDue string `json:"release_due"`
		}
		must(t, json.Unmarshal([]byte(out), &result))
		if result.RequestID == "" || result.ReleaseDue == "" {
			t.Fatalf("missing acknowledged receipt: %s", out)
		}
		return result.RequestID
	}
	baseline := func() time.Time {
		t.Helper()
		current, p := store.RentalRow(podRental)
		fatal(t, p)
		idle, p := rental.ObserveIdle(store, *current)
		fatal(t, p)
		return idle.Since
	}
	first := command()
	initial := baseline()
	second := command()
	renewed := baseline()
	if first == second || !renewed.After(initial) {
		t.Fatal("explicit commands did not issue fresh resets")
	}
	reply := daemon.call(t, http.MethodPost, "/v1/local/rentals/"+podRental+"/keepalive", map[string]any{"request_id": first})
	if reply.Status != http.StatusOK || !baseline().Equal(renewed) {
		t.Fatalf("older idempotent receipt changed latest clock: %s", reply.brief())
	}
	mu.Lock()
	before := calls
	mu.Unlock()
	if code, out := runCozy(t, root, "rental", "list", "--json", "--no-watch"); code != 0 {
		t.Fatalf("status: %s", out)
	}
	daemon = crashAndRestartTransactionDaemon(t, daemon)
	reply = daemon.call(t, http.MethodPost, "/v1/local/rentals/"+podRental+"/claim", map[string]any{})
	if reply.Status != http.StatusOK && reply.Status != http.StatusAccepted {
		t.Fatalf("reconnect: %s", reply.brief())
	}
	mu.Lock()
	after := calls
	failed = true
	mu.Unlock()
	if after != before || !baseline().Equal(renewed) {
		t.Fatal("status or restart renewed the rental")
	}
	if code, out := runCozy(t, root, "rental", "keepalive", "keepalive", "--json"); code == 0 || strings.Contains(out, `"release_due"`) {
		t.Fatalf("failed acknowledgment claimed success: %d %s", code, out)
	}
	if !baseline().Equal(renewed) {
		t.Fatal("failed acknowledgment reset local clock")
	}
	reply = daemon.call(t, http.MethodPost, "/v1/local/rentals/"+podRental+"/keepalive", map[string]any{"request_id": "invalid-duration", "duration": 0})
	if reply.Status == http.StatusOK {
		t.Fatal("duration override admitted")
	}
	if code, _ := runCozy(t, root, "rental", "keepalive", "keepalive", "--duration", "0"); code == 0 {
		t.Fatal("CLI duration override admitted")
	}
	current, problem := store.RentalRow(podRental)
	fatal(t, problem)
	current.State = "release_requested"
	fatal(t, store.RecordRental(*current))
	mu.Lock()
	before = calls
	mu.Unlock()
	if code, out := runCozy(t, root, "rental", "keepalive", "keepalive", "--json"); code == 0 {
		t.Fatalf("ending rental kept alive: %s", out)
	}
	mu.Lock()
	after = calls
	mu.Unlock()
	if after != before {
		t.Fatal("ending rental reached Host keepalive")
	}
}
