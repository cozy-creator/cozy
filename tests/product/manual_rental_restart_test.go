package producttest

import (
	"bytes"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/workertls"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

// The independent TLS peer verifies real per-rental signatures and supplies an
// empty canonical snapshot. The actual daemon must claim it again after process
// replacement without any request, package, or explicit second claim call.
func TestIdleManualRentalReclaimsAfterDaemonRestart(t *testing.T) {
	for _, mode := range []string{"healthy", "attached", "weather", "released", "closed", "permanent", "released_health", "closing_health", "concurrent", "retained"} {
		t.Run(mode, func(t *testing.T) { proveIdleManualRentalRestart(t, mode) })
	}
}

func proveIdleManualRentalRestart(t *testing.T, mode string) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "manual-restart")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod := &fakePod{controlKey: public}
	var mu sync.Mutex
	var claims []uint64
	var acknowledged []uint64
	var expected *pb.WorkerSnapshot
	var sequence uint64
	pod.onFrame = func(frame *pb.RecordOwnerFrame, send func(*pb.WorkerFrame) error) (bool, error) {
		mu.Lock()
		defer mu.Unlock()
		if claim := frame.GetClaim(); claim != nil {
			if err := pod.verifyClaim(claim, true); err != nil {
				return true, err
			}
			claims = append(claims, claim.RecordOwnerEpoch)
			sequence++ // this independent peer owns its stream fence
			body, digest, err := canonical.Identity(&pb.WorkerSnapshotBody{
				WorkerPhase:    pb.WorkerPhase_WORKER_PHASE_ONLINE,
				AdmissionEpoch: 1, AdmissionState: pb.AdmissionState_ADMISSION_STATE_CLOSED})
			if err != nil {
				return true, err
			}
			hostBody, hostDigest, err := canonical.Identity(&pb.HostSnapshotBody{})
			if err != nil {
				return true, err
			}
			expected = &pb.WorkerSnapshot{RecordOwnerEpoch: claim.RecordOwnerEpoch,
				ControlStreamEpoch: sequence, WorkerBootId: podBootID,
				SnapshotId: fmt.Sprintf("empty-%d", sequence), SnapshotDigest: digest,
				SnapshotCanonicalBytes: body, HostSnapshotDigest: hostDigest, HostSnapshotCanonicalBytes: hostBody}
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
				RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: sequence, WorkerBootId: podBootID,
				Accepted: true, WireMinor: pb.WireMinor, WorkerId: podWorkerID, WorkerInstanceId: "manual-empty-pod",
				Resources: &pb.WorkerResources{Backend: "none"}}}}); err != nil {
				return true, err
			}
			return true, send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_Snapshot{Snapshot: expected}})
		}
		if ack := frame.GetSnapshotAck(); ack != nil {
			if expected == nil || ack.RecordOwnerEpoch != expected.RecordOwnerEpoch ||
				ack.ControlStreamEpoch != expected.ControlStreamEpoch || ack.WorkerBootId != expected.WorkerBootId ||
				ack.SnapshotId != expected.SnapshotId || !bytes.Equal(ack.SnapshotDigest, expected.SnapshotDigest) ||
				!bytes.Equal(ack.HostSnapshotDigest, expected.HostSnapshotDigest) {
				return true, fmt.Errorf("snapshot acknowledgement identity mismatch")
			}
			acknowledged = append(acknowledged, ack.RecordOwnerEpoch)
		}
		return false, nil
	}
	certPath := standInCertificate(t, root)
	pair, err := tls.LoadX509KeyPair(certPath, certPath+".key")
	must(t, err)
	pin, err := workertls.LoadPin(certPath)
	must(t, err)
	pod.leafDigest = pin.Digest()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow independent TLS worker peer
	must(t, err)
	control := grpc.NewServer(grpc.Creds(credentials.NewServerTLSFromCert(&pair)))
	pb.RegisterWorkerControlServer(control, pod)
	go func() { _ = control.Serve(listener) }()
	defer control.Stop()
	var mediaUnavailable atomic.Bool
	var healthCalls atomic.Int64
	var blockedHealth atomic.Bool
	healthReply := make(chan struct{})
	var replyOnce sync.Once
	releaseHealth := func() { replyOnce.Do(func() { close(healthReply) }) }
	defer releaseHealth()
	media := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/health" {
			t.Errorf("unexpected media request %s", r.URL.Path)
			w.WriteHeader(404)
			return
		}
		healthCalls.Add(1)
		if blockedHealth.Load() {
			<-healthReply
		}
		if mediaUnavailable.Load() {
			if mode == "permanent" {
				w.WriteHeader(http.StatusForbidden)
				_, _ = w.Write([]byte(`{"error":{"code":"proof.owner_refused","message":"fixed credential refusal"}}`))
			} else {
				w.WriteHeader(http.StatusServiceUnavailable)
			}
			return
		}
		revision := mediawire.ContractRev
		_ = json.NewEncoder(w).Encode(mediawire.Health{Service: mediawire.Service, ContractRev: &revision})
	}))
	media.TLS = &tls.Config{Certificates: []tls.Certificate{pair}, MinVersion: tls.VersionTLS12}
	media.StartTLS()
	defer media.Close()
	var mutations atomic.Int64
	// Control reattachment needs the retained identity, not a successful cloud poll.
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			mutations.Add(1)
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.hub_unavailable","message":"controlled cloud outage"}}`))
	}))
	defer hub.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\ntensorhub_token: manual-restart-proof\nrentals:\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	cert, err := os.ReadFile(certPath)
	must(t, err)
	token, problem := rental.PendingMediaToken(layout, "manual-restart")
	fatal(t, problem)
	row := records.Rental{ID: podRental, State: "ready", Hub: hub.URL, MachineName: "manual-empty",
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: listener.Addr().String(), MediaAddress: strings.TrimPrefix(media.URL, "https://"),
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	if mode == "attached" {
		row.State = "attached"
	}
	fatal(t, rental.Attach(layout, store, row, string(cert), token, identity))
	before, problem := store.RentalRow(podRental)
	fatal(t, problem)
	keyBefore, err := os.ReadFile(layout.RentalCreatorIdentity(podRental))
	must(t, err)
	blockedHealth.Store(mode == "concurrent")
	first := startDaemonProcess(t, root)
	claim := func() reply { return first.call(t, "POST", "/v1/local/rentals/"+podRental+"/claim", map[string]any{}) }
	if mode == "concurrent" {
		waitUntil(t, "startup connection waits on media", func() bool { return healthCalls.Load() == 1 })
		replies := make(chan reply, 4)
		for i := 0; i < 4; i++ {
			go func() { replies <- claim() }()
		}
		// Hold the real health request while the explicit claim calls enter the API.
		time.Sleep(100 * time.Millisecond)
		releaseHealth()
		for i := 0; i < 4; i++ {
			reply := <-replies
			if reply.Status != http.StatusOK && reply.Status != http.StatusAccepted {
				t.Fatalf("concurrent rental claim: %s", reply.brief())
			}
		}
	} else if reply := claim(); reply.Status != http.StatusOK && reply.Status != http.StatusAccepted {
		t.Fatalf("initial explicit rental claim: %s", reply.brief())
	}
	waitUntil(t, "first empty rental snapshot acknowledged", func() bool { mu.Lock(); defer mu.Unlock(); return len(acknowledged) == 1 })
	if mode == "retained" {
		request := recordPrivateTransaction(t, store, "control-restart", podRental)
		response := first.call(t, http.MethodPost, "/v1/local/jobs/"+request.ID+"/pause",
			map[string]any{"actor": "product proof"})
		if response.Status != http.StatusOK || !bytes.Contains(response.Body, []byte(`"status":"paused"`)) {
			t.Fatalf("pause retained rental owner: %s", response.brief())
		}
		// The machine now belongs to a retained transaction. Recovery must not
		// rely on the manual-rental exception or an open Python attempt.
		row.ManagedRequestID = request.ID
		fatal(t, store.RecordRental(row))
		before, problem = store.RentalRow(podRental)
		fatal(t, problem)
	}
	stop := func(d *daemonProcess) {
		t.Helper()
		must(t, d.cmd.Process.Signal(syscall.SIGTERM))
		select {
		case code := <-d.exited:
			if code != 0 {
				t.Fatalf("daemon exit=%d", code)
			}
		case <-time.After(10 * time.Second):
			t.Fatal("process-only shutdown did not finish")
		}
	}
	// Other row states are not idle manual capacity. Each has valid retained
	// credentials and a live peer, so an accidental reattachment is observable.
	for _, state := range []string{"managed", "acquiring", "release_requested"} {
		excluded := row
		excluded.ID = "pr-excluded-" + state
		excluded.MachineName = map[string]string{"managed": "cedar", "acquiring": "birch", "release_requested": "oak"}[state]
		if state == "managed" {
			excluded.ManagedRequestID = "finished-managed-request"
		} else {
			excluded.State = state
		}
		fatal(t, rental.Attach(layout, store, excluded, string(cert), token, identity))
	}
	stop(first)
	priorHealth := healthCalls.Load()
	mediaUnavailable.Store(mode == "weather" || mode == "released" || mode == "closed" || mode == "permanent")
	blockedHealth.Store(mode == "released_health" || mode == "closing_health")
	var second *daemonProcess
	var owner *orchestrator.Orchestrator
	if mode == "closing_health" {
		owner, problem = orchestrator.Open(orchestrator.Options{
			Cfg: config.Config{Home: root}, Layout: layout, Store: store,
			Rentals: rental.Resolver(layout, store), ObserveRental: rental.ObserveWorker(store),
			RentalClaimProof: rental.ClaimProof(layout)})
		fatal(t, problem)
		defer owner.Close(0)
		_, _, problem = owner.Reconcile()
		fatal(t, problem)
	} else {
		second = startDaemonProcess(t, root)
	}
	waitUntil(t, "startup attempts the retained media endpoint", func() bool { return healthCalls.Load() > priorHealth })
	expectedClaims := 2
	if mode == "permanent" {
		expectedClaims = 1
	}
	if mode == "released" || mode == "released_health" {
		released, problem := store.RentalRow(podRental)
		fatal(t, problem)
		released.State = "release_requested"
		fatal(t, store.RecordRental(*released))
		expectedClaims = 1
	}
	if mode == "closed" {
		stop(second)
		expectedClaims = 1
	}
	if mode == "closing_health" {
		owner.Close(0)
		expectedClaims = 1
	}
	mediaUnavailable.Store(false)
	releaseHealth()
	if expectedClaims == 2 {
		waitUntil(t, "restarted daemon claims idle manual rental without a request", func() bool { mu.Lock(); defer mu.Unlock(); return len(acknowledged) == 2 })
	} else {
		// Cross the actual startup retry cadence after its authority was removed.
		time.Sleep(orchestrator.ReportCadence + 200*time.Millisecond)
	}
	if mode == "closing_health" || mode == "released_health" {
		rows, problem := store.LiveWorkers()
		fatal(t, problem)
		if len(rows) != 0 {
			t.Fatal("health completion registered a worker after owner close")
		}
	}
	if expectedClaims == 1 && healthCalls.Load() != priorHealth+1 {
		t.Fatal("startup retried after its ownership, process, or credential authority ended")
	}
	mu.Lock()
	if len(claims) != expectedClaims || len(acknowledged) != expectedClaims || sequence != uint64(expectedClaims) {
		t.Fatalf("owner/stream restart identities: claims=%v acknowledgements=%v", claims, acknowledged)
	}
	for i := range claims {
		if claims[i] != claims[0] || acknowledged[i] != claims[i] {
			t.Fatal("the retained owner identity changed")
		}
	}
	mu.Unlock()
	pod.mu.Lock()
	if len(pod.offers) != 0 || len(pod.desired) != 0 || len(pod.prepares) != 0 {
		t.Fatal("empty rental recovery invented application work")
	}
	pod.mu.Unlock()
	after, problem := store.RentalRow(podRental)
	fatal(t, problem)
	if after == nil || after.ID != before.ID || after.ExpectedWorkerID != before.ExpectedWorkerID ||
		after.ExpectedWorkerBootID != before.ExpectedWorkerBootID || after.RentedAt != before.RentedAt || after.ReadyAt != before.ReadyAt || after.ManagedRequestID != before.ManagedRequestID {
		t.Fatalf("reattachment changed rental ownership/lifecycle: before=%+v after=%+v", before, after)
	}
	keyAfter, err := os.ReadFile(layout.RentalCreatorIdentity(podRental))
	must(t, err)
	if !bytes.Equal(keyBefore, keyAfter) || mutations.Load() != 0 {
		t.Fatal("restart changed the key or submitted a paid mutation")
	}
	if mode != "closed" && second != nil {
		stop(second)
	}
}
