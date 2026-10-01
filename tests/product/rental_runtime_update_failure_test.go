package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Native maintenance uses an actual agent/Runtime pair. The only transport fault
// injector is HTTPS in front of that process; there is no SSH installer stand-in.
type nativeMaintenanceFixture struct {
	root                        string
	store                       *records.Store
	direct                      *machines.Maintenance
	dropUpload                  atomic.Bool
	dropAdmission               atomic.Bool
	hideObservation             atomic.Bool
	posts                       atomic.Int32
	dropped                     atomic.Int32
	runtimeWheel, tensorfsWheel string
}

// nativeMaintenance attaches an actual agent/Runtime pair as rental tessa; serve adjusts
// the stand-in Hub before the machine starts.
func nativeMaintenance(t *testing.T, serve ...func(*machineHub)) *nativeMaintenanceFixture {
	t.Helper()
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		if *requireMachineHost {
			t.Fatal("native maintenance qualification requires agent and paired Runtime/TensorFS wheel fixtures")
		}
		t.Skip("requires actual -machine-host, -machine-runtime-wheel and -machine-tensorfs-wheel fixtures")
	}
	uv, err := exec.LookPath("uv")
	if err != nil {
		if *requireMachineHost {
			t.Fatal("uv is required for native maintenance qualification")
		}
		t.Skip("uv is required for the native machine fixture")
	}
	h := newMachineHub(t)
	for _, adjust := range serve {
		adjust(h)
	}
	root, err := os.MkdirTemp("", "cz-native-update-")
	must(t, err)
	claimScratch(root)
	t.Cleanup(func() {
		_, _ = runCozy(t, root, "down")
		if !t.Failed() {
			_ = removeAllForce(root)
		} else {
			t.Logf("native maintenance evidence: %s", root)
		}
	})
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	st, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(st.Close)
	var authorized atomic.Value
	authorized.Store("")
	prior := h.worker.Config.Handler
	h.worker.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/worker/rental/authorized-keys" {
			prior.ServeHTTP(w, r)
			return
		}
		key := authorized.Load().(string)
		if key == "" || r.Header.Get("X-Cozy-Worker-ID") != parityWorker || len(r.Header.Get("X-Cozy-Worker-Token")) != 43 {
			http.Error(w, "worker not ready", 503)
			return
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"worker_id": parityWorker, "authorized_keys": []string{key}, "lease_seconds": 30})
	})
	source := machines.Source{Host: *machineHostBinary, RuntimeWheel: *machineRuntimeWheel, TensorFSWheel: *machineTensorFSWheel, Pinned: true}
	launch, identity, mediaToken, provider := providerHost(t, h, layout, source, uv)
	authorized.Store(identity.PublicKey())
	f := &nativeMaintenanceFixture{root: root, store: st, runtimeWheel: *machineRuntimeWheel, tensorfsWheel: *machineTensorFSWheel}
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: launch.Leaf})
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(certPEM) {
		t.Fatal("agent leaf missing")
	}
	transport := &http.Transport{TLSClientConfig: &tls.Config{RootCAs: roots, ServerName: "cozy-worker", MinVersion: tls.VersionTLS12}, ForceAttemptHTTP2: true}
	t.Cleanup(transport.CloseIdleConnections)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	f.direct = &machines.Maintenance{Base: "https://" + launch.Addr, Machine: launch.WorkerID, Public: ed25519.PublicKey(public), Sign: identity.Sign, Client: &http.Client{Transport: transport}}
	target, err := url.Parse(f.direct.Base)
	must(t, err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	proxy.Transport = transport
	proxy.ModifyResponse = func(response *http.Response) error {
		if response.Request.Method == http.MethodPost && response.Request.URL.Path == "/v1/machine/runtime/update" && response.StatusCode == http.StatusAccepted && f.dropAdmission.CompareAndSwap(true, false) {
			f.dropped.Add(1)
			response.Body.Close()
			return fmt.Errorf("fixture lost the already-accepted response")
		}
		return nil
	}
	front := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPut && strings.HasPrefix(r.URL.Path, "/v1/machine/runtime/wheels/") && f.dropUpload.CompareAndSwap(true, false) {
			http.Error(w, "fixture refused transfer before forwarding", http.StatusBadGateway)
			return
		}
		if r.Method == http.MethodPost && r.URL.Path == "/v1/machine/runtime/update" {
			f.posts.Add(1)
		}
		if r.Method == http.MethodGet && r.URL.Path == "/v1/machine/runtime" && f.posts.Load() > 0 && f.hideObservation.Load() {
			http.Error(w, "fixture hides operation observation", http.StatusServiceUnavailable)
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	leaf, err := tls.LoadX509KeyPair(filepath.Join(provider, "run/cozy/bootstrap/tls.crt"), filepath.Join(provider, "run/cozy/bootstrap/tls.key"))
	must(t, err)
	front.TLS = &tls.Config{Certificates: []tls.Certificate{leaf}, MinVersion: tls.VersionTLS12}
	front.EnableHTTP2 = true
	front.StartTLS()
	t.Cleanup(front.Close)
	address := strings.TrimPrefix(front.URL, "https://")
	h.mu.Lock()
	h.rentals[parityRental] = map[string]any{"rental_id": parityRental, "name": "tessa", "state": "ready", "requested_accelerator_model": "Virtual Accelerator", "accelerator_count": 4, "hourly_rate_usd_micros": 1, "worker_address": address, "media_address": address, "worker_id": launch.WorkerID, "worker_boot_id": launch.BootID, "cert_pem": string(certPEM), "creator_public_key": identity.PublicKey(), "media_token_sha256": []string{secret.HashHex(secret.New(mediaToken))}}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, st, records.Rental{ID: parityRental, MachineName: "tessa", SKU: "virtual-4", State: "ready", AcceleratorModel: "Virtual Accelerator", AcceleratorCount: 4, HourlyRateUSDMicros: 1, Hub: h.server.URL, Address: address, MediaAddress: address, ExpectedWorkerID: launch.WorkerID, ExpectedWorkerBootID: launch.BootID}, string(certPEM), secret.New(mediaToken), identity))
	startDaemonProcess(t, root)
	ctx, cancel := context.WithTimeout(t.Context(), time.Minute)
	defer cancel()
	_, problem = f.direct.AwaitUpdateAdmission(ctx)
	fatal(t, problem)
	return f
}

func (f *nativeMaintenanceFixture) update(t *testing.T, wheel string) (int, string) {
	t.Helper()
	return cozyWithin(t, f.root, 2*time.Minute, "rental", "update", "tessa", "--runtime-wheel", wheel, "--tensorfs-wheel", f.tensorfsWheel, "--json")
}

func (f *nativeMaintenanceFixture) state(t *testing.T) *machines.RuntimeState {
	t.Helper()
	state, problem := f.direct.State(t.Context())
	fatal(t, problem)
	return state
}

// transportLogs is every update's retained transport diagnostics under root.
func transportLogs(root string) string {
	paths, _ := filepath.Glob(filepath.Join(root, "tmp", "runtime-updates", "*", "transport.log"))
	text := ""
	for _, path := range paths {
		data, _ := os.ReadFile(path)
		text += path + ":\n" + string(data) + "\n"
	}
	return text
}

// eventually waits for ok; the bound only catches a hang on a loaded shared box. A
// shown value is the last observation, reported when it never came.
func eventually(t *testing.T, root, what string, ok func() bool, shown ...*string) {
	t.Helper()
	for deadline := time.Now().Add(5 * time.Minute); !ok(); time.Sleep(50 * time.Millisecond) {
		if time.Now().After(deadline) {
			observed := ""
			for _, value := range shown {
				observed += *value + "\n"
			}
			t.Fatalf("%s did not happen: %s%s\n%s", what, observed, tail(filepath.Join(root, "daemon.log")), transportLogs(root))
		}
	}
}

// cozyWithin runs one CLI command, failing the test if it outlives `within`.
func cozyWithin(t *testing.T, root string, within time.Duration, args ...string) (int, string) {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), within)
	defer cancel()
	cmd := exec.CommandContext(ctx, "/usr/bin/nice", append([]string{"-n", "19", cozyBin}, args...)...)
	cmd.Env = childEnv(t, root)
	data, _ := cmd.CombinedOutput()
	if ctx.Err() != nil {
		t.Fatalf("cozy %s hung past %s: %s\n%s", strings.Join(args, " "), within, data, tail(filepath.Join(root, "daemon.log")))
	}
	return cmd.ProcessState.ExitCode(), string(data)
}

func TestNativeWheelTransferFailureLeavesAgentUnchangedAndRetryUpdates(t *testing.T) {
	f := nativeMaintenance(t)
	before := f.state(t)
	candidate := localBuild(t, f.runtimeWheel, "nativetry")
	f.dropUpload.Store(true)
	if code, out := f.update(t, candidate); code == 0 {
		t.Fatalf("failed transfer succeeded: %s", out)
	}
	after := f.state(t)
	row, problem := f.store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	if f.posts.Load() != 0 || after.Runtime != before.Runtime || after.Update != nil || row == nil || row.State != "failed" {
		t.Fatalf("unsent update mutated or remained pending: agent=%+v row=%+v posts=%d", after, row, f.posts.Load())
	}
	if code, out := f.update(t, candidate); code != 0 {
		t.Fatalf("native retry failed: %s", out)
	}
	after = f.state(t)
	row, problem = f.store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	if after.Update == nil || after.Update.State != "succeeded" || row.State != "succeeded" || after.Update.Operation != row.ID || f.posts.Load() != 1 {
		t.Fatalf("native retry did not settle one operation: %+v %+v", after, row)
	}
}

func TestLostNativeUpdateAdmissionReplyReconcilesActualAgent(t *testing.T) {
	f := nativeMaintenance(t)
	f.dropAdmission.Store(true)
	if code, out := f.update(t, localBuild(t, f.runtimeWheel, "lostreply")); code != 0 {
		t.Fatalf("accepted update was treated as unsent after reply loss: %s", out)
	}
	row, problem := f.store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	for range 2 {
		state := f.state(t)
		if row == nil || row.State != "succeeded" || state.Update == nil || state.Update.State != "succeeded" || state.Update.Operation != row.ID || f.dropped.Load() != 1 || f.posts.Load() != 1 {
			t.Fatalf("lost reply duplicated/lost authoritative update: row=%+v agent=%+v posts=%d drops=%d", row, state, f.posts.Load(), f.dropped.Load())
		}
	}
}

func TestNativeUpdateSurvivesControllerObserverRestart(t *testing.T) {
	f := nativeMaintenance(t)
	f.hideObservation.Store(true)
	candidate := localBuild(t, f.runtimeWheel, "observerrestart")
	done := make(chan struct{})
	var output []byte
	command := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "rental", "update", "tessa", "--runtime-wheel", candidate, "--tensorfs-wheel", f.tensorfsWheel, "--json")
	command.Env = childEnv(t, f.root)
	go func() { output, _ = command.CombinedOutput(); close(done) }()
	t.Cleanup(func() {
		if command.Process != nil {
			_ = command.Process.Kill()
		}
	})
	eventually(t, f.root, "agent accepted native update", func() bool { state := f.state(t); return state.Update != nil })
	original, problem := f.store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	if original == nil {
		t.Fatal("native intent was not durable")
	}
	if code, out := runCozy(t, f.root, "down"); code != 0 {
		t.Fatalf("observer down: %s", out)
	}
	select {
	case <-done:
	case <-time.After(time.Minute):
		t.Fatal("update observer did not detach")
	}
	eventually(t, f.root, "agent completes without controller observer", func() bool { state := f.state(t); return state.Update != nil && state.Update.State == "succeeded" })
	held, problem := f.store.RuntimeUpdate(parityRental)
	fatal(t, problem)
	if held == nil || held.ID != original.ID || held.State == "failed" {
		t.Fatalf("observer loss erased accepted update: %+v %s", held, output)
	}
	f.hideObservation.Store(false)
	startDaemonProcess(t, f.root)
	eventually(t, f.root, "restarted observer reconciles terminal operation", func() bool {
		row, problem := f.store.RuntimeUpdate(parityRental)
		fatal(t, problem)
		return row != nil && row.ID == original.ID && row.State == "succeeded"
	})
	if f.posts.Load() != 1 {
		t.Fatalf("observer restart reissued update %d times", f.posts.Load())
	}
}
