package producttest

import (
	"bytes"
	"crypto/sha256"
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
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/home"
	hubapi "github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/mediawire"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
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

// The manual CLI disappeared after its accepted ID was durable. Recovery performs
// only Hub reads and completes the same authenticated local attachment.
func TestInterruptedManualAcquisitionCompletesOnDaemonRestart(t *testing.T) {
	for _, mode := range []string{"interrupted", "interrupted-released", "interrupted-wrong-key"} {
		t.Run(mode, func(t *testing.T) { proveIdleManualRentalRestart(t, mode) })
	}
}

// CPU work has no accelerator requirement; an already-paid idle GPU is usable.
// Accelerator work still cannot use a CPU worker. Both arms cross real daemon
// acquisition, retained rental records, signed Claim and independent TLS peer.
func TestIdleRentalAcceleratorRequirement(t *testing.T) {
	for _, mode := range []string{"cpu_job_on_gpu", "gpu_job_on_cpu"} {
		t.Run(mode, func(t *testing.T) { proveIdleManualRentalRestart(t, mode) })
	}
}

func proveIdleManualRentalRestart(t *testing.T, mode string) {
	root := t.TempDir()
	classProof := mode == "cpu_job_on_gpu" || mode == "gpu_job_on_cpu"
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
			resources := &pb.WorkerResources{Backend: "none"}
			if mode == "cpu_job_on_gpu" {
				resources = &pb.WorkerResources{Backend: "cuda", DeviceName: "RTX 4090", DeviceCount: 1, DeviceMemoryTotalBytes: 24 << 30}
			}
			if err := send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ClaimAck{ClaimAck: &pb.ClaimAck{
				RecordOwnerEpoch: claim.RecordOwnerEpoch, ControlStreamEpoch: sequence, WorkerBootId: podBootID,
				Accepted: true, WireMinor: pb.WireMinor, WorkerId: podWorkerID, WorkerInstanceId: "manual-empty-pod",
				Resources: resources}}}); err != nil {
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
	pb.RegisterPodHostServer(control, pod)
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
	cert, err := os.ReadFile(certPath)
	must(t, err)
	token, problem := rental.PendingMediaToken(layout, "manual-restart")
	fatal(t, problem)
	machineName := "manual-empty"
	var mutations atomic.Int64
	rentalView := map[string]any{"rental_id": podRental, "name": "manual-empty", "state": "ready", "accelerator_count": 1, "hourly_rate_usd_micros": 1}
	// Control reattachment needs the retained identity, not a successful cloud poll.
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			mutations.Add(1)
		}
		if strings.HasPrefix(mode, "interrupted") && r.Method == http.MethodGet {
			if r.URL.Path == "/v1/rentals/"+podRental {
				publicKey := identity.PublicKey()
				if mode == "interrupted-wrong-key" {
					publicKey = base64.RawURLEncoding.EncodeToString(bytes.Repeat([]byte{1}, 32))
				}
				_ = json.NewEncoder(w).Encode(map[string]any{"rental_id": podRental, "name": machineName,
					"state": "ready", "requested_accelerator_model": "CPU", "accelerator_count": 1,
					"hourly_rate_usd_micros": 1, "worker_address": listener.Addr().String(),
					"media_address": strings.TrimPrefix(media.URL, "https://"), "cert_pem": string(cert),
					"worker_id": podWorkerID, "worker_boot_id": podBootID,
					"creator_public_key": publicKey, "media_token_sha256": []string{secret.HashHex(token)}})
				return
			}
			if r.URL.Path == "/v1/rentals" {
				_ = json.NewEncoder(w).Encode([]any{})
				return
			}
		}
		if classProof && r.Method == http.MethodGet {
			switch r.URL.Path {
			case "/v1/packages/proof/idle-job/releases/1":
				_ = json.NewEncoder(w).Encode(rentalReleaseFacts())
				return
			case "/v1/rentals/" + podRental + "/image-inventory":
				_ = json.NewEncoder(w).Encode(map[string]any{"image_inventory": map[string]any{
					"format": "tensorhub.image_inventory/1", "profile": "python3.12-cpu-linux-x86", "python": "3.12.12",
					"interpreters": []map[string]string{{"version": "3.12.12", "abi": "cp312"}}, "distributions": []any{},
				}})
				return
			case "/v1/rentals":
				_ = json.NewEncoder(w).Encode(map[string]any{"rentals": []map[string]any{rentalView}})
				return
			case "/v1/rental-skus":
				skus := offeredSKUs()
				for i := range skus {
					if skus[i].AcceleratorModel != "CPU" {
						skus[i].ComputeCapability, skus[i].VRAMGB, skus[i].MinimumRAMPerGPUGB = "8.9", 24, 64
					}
				}
				_ = json.NewEncoder(w).Encode(skus)
				return
			case "/v1/rentals/" + podRental:
				_ = json.NewEncoder(w).Encode(rentalView)
				return
			}
		}
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.hub_unavailable","message":"controlled cloud outage"}}`))
	}))
	defer hub.Close()
	if strings.HasPrefix(mode, "interrupted") {
		op, _, problem := store.BeginRentalOperation(records.RentalOperation{
			Key: "manual-restart", Hub: hub.URL, Reason: "cozy rental new cpu", HourlyRateUSDMicros: 1,
		}, 2_000_000, 0, func(name string) ([]byte, string, *exit.Error) {
			body, problem := hubapi.RentalRequestBytes(name, "cpu", secret.HashHex(token), identity.PublicKey(), hubapi.DeclaredWorkload{}, nil)
			return body, fmt.Sprintf("sha256:%x", sha256.Sum256(body)), problem
		}, nil)
		fatal(t, problem)
		request, problem := hubapi.ParseRentalRequestBytes(op.RequestBody)
		fatal(t, problem)
		machineName = request.Name
	}
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\ntensorhub_token: manual-restart-proof\nrentals:\n  max_hourly_spend_usd: 2\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))
	row := records.Rental{AcceleratorCount: 1, ID: podRental, State: "ready", Hub: hub.URL, MachineName: machineName,
		SKU: "cpu", AcceleratorModel: "CPU", HourlyRateUSDMicros: 1, Address: listener.Addr().String(), MediaAddress: strings.TrimPrefix(media.URL, "https://"),
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	if mode == "attached" {
		row.State = "attached"
	}
	if mode == "retained" {
		row.ManagedRequestID = "req-private-control-restart"
	}
	if mode == "cpu_job_on_gpu" {
		row.SKU, row.AcceleratorModel = "rtx-4090", "RTX 4090"
	}
	if strings.HasPrefix(mode, "interrupted") {
		pending := row
		pending.State, pending.Address, pending.MediaAddress = "acquiring", "", ""
		pending.ExpectedWorkerID, pending.ExpectedWorkerBootID = "", ""
		fatal(t, store.RecordRental(pending))
		fatal(t, store.AdvanceRentalOperation("manual-restart", podRental, "acquiring"))
		if mode == "interrupted-released" {
			fatal(t, store.AdvanceRentalOperation("manual-restart", podRental, "released"))
			pending.State = "released"
			fatal(t, store.RecordRental(pending))
		}
		startDaemonProcess(t, root)
		if mode == "interrupted-released" || mode == "interrupted-wrong-key" {
			waitUntil(t, "recovery refuses changed ownership", func() bool {
				if mode == "interrupted-released" {
					row, problem := store.RentalRow(podRental)
					fatal(t, problem)
					return row == nil
				}
				return strings.Contains(tail(filepath.Join(root, "daemon.log")), "did not retain the Creator key")
			})
			mu.Lock()
			defer mu.Unlock()
			if len(claims) != 0 || mutations.Load() != 0 {
				t.Fatal("refused acquisition claimed a worker or made a paid mutation")
			}
			if _, err := os.Stat(layout.RentalCert(podRental)); !os.IsNotExist(err) {
				t.Fatal("refused acquisition installed worker credentials")
			}
			return
		}
		waitUntil(t, "interrupted acquisition attaches and acknowledges its snapshot", func() bool {
			mu.Lock()
			defer mu.Unlock()
			return len(acknowledged) == 1
		})
		attached, problem := store.RentalRow(podRental)
		fatal(t, problem)
		op, problem := store.RentalOperation("manual-restart")
		fatal(t, problem)
		if mutations.Load() != 0 || op.State != "attached" || attached.Address != row.Address || attached.ExpectedWorkerBootID != podBootID {
			t.Fatalf("manual acquisition not recovered from its original operation: state=%s mutations=%d row=%+v", op.State, mutations.Load(), attached)
		}
		if _, err := os.Stat(layout.PendingRentalMediaToken("manual-restart")); !os.IsNotExist(err) {
			t.Fatal("pending credential was not retired after authenticated attachment")
		}
		return
	}
	fatal(t, rental.Attach(layout, store, row, string(cert), token, identity))
	before, problem := store.RentalRow(podRental)
	fatal(t, problem)
	keyBefore, err := os.ReadFile(layout.RentalCreatorIdentity(podRental))
	must(t, err)
	blockedHealth.Store(mode == "concurrent")
	if classProof {
		const request = "job-idle-accelerator-requirement"
		_, _, problem := store.Submit(records.Request{ID: request, IdemKey: request,
			BodyDigest: "sha256:" + strings.Repeat("a", 64), Package: "proof/idle-job", Release: "1", Entrypoint: "produce",
			Kind: "job", NeedsAccelerator: mode == "gpu_job_on_cpu", Rental: true, RentalRequired: true,
			Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]"})
		fatal(t, problem)
	}
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
		before, problem = store.RentalRow(podRental)
		fatal(t, problem)
	}
	if classProof {
		const request = "job-idle-accelerator-requirement"
		waitUntil(t, "existing-rental selection or paid ask", func() bool {
			request, problem := store.RequestRow(request)
			fatal(t, problem)
			return mutations.Load() > 0 || request.Worker != "" || request.State == "failed"
		})
		selected, problem := store.RequestRow(request)
		fatal(t, problem)
		if mode == "cpu_job_on_gpu" {
			if selected.Worker != podRental || mutations.Load() != 0 {
				t.Fatalf("CPU job did not reuse the ready GPU: worker=%q paid asks=%d; %s", selected.Worker, mutations.Load(), tail(filepath.Join(root, "daemon.log")))
			}
		} else if selected.Worker == podRental || mutations.Load() == 0 {
			t.Fatalf("GPU job reused a CPU instead of asking for accelerator capacity: worker=%q paid asks=%d", selected.Worker, mutations.Load())
		}
		return
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
