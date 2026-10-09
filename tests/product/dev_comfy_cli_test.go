package producttest

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type comfyKeepalivePeer struct {
	v1.UnimplementedMachineServer
	calls atomic.Int32
}

func (p *comfyKeepalivePeer) Status(request *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	if request.Keepalive {
		p.calls.Add(1)
	}
	return stream.Send(&v1.StatusFrame{WorkerId: "comfy-worker", BootId: "comfy-boot", Phase: "ready", IdleDeadlineUnixMs: time.Now().Add(15 * time.Minute).UnixMilli()})
}

// The real CLI, Hub lookup, pinned-TLS keepalive and SQLite acknowledgement must
// reach transport and emit one result. A nil *exit.Error must not stop dispatch.
func TestDevComfyCLIAcknowledgedKeepaliveReachesTransportAndReportsOutcome(t *testing.T) {
	h := newFakeRentalHub(t, 0)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+h.server.URL+"\ntensorhub_token: rental-idle-test\n"), 0600))
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	must(t, os.MkdirAll(layout.Rentals, 0700))
	owner, problem := rental.OwnerIdentityAt(layout.RentalCreatorIdentity("pr-comfy"))
	fatal(t, problem)
	certServer := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certServer.TLS.Certificates[0]
	certServer.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow pinned loopback machine fixture for the real CLI keepalive
	must(t, err)
	peer := &comfyKeepalivePeer{}
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, peer)
	go server.Serve(listener)
	defer server.Stop()
	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]})
	certPath := layout.RentalCert("pr-comfy")
	must(t, os.WriteFile(certPath, certPEM, 0600))
	fatal(t, store.RecordRental(records.Rental{ID: "pr-comfy", MachineName: "fixture", SKU: "cpu", State: "ready", Hub: h.server.URL, Address: listener.Addr().String(), CertPath: certPath, ExpectedWorkerID: "comfy-worker", ExpectedWorkerBootID: "comfy-boot", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1}))
	h.mu.Lock()
	h.rentals["pr-comfy"] = map[string]any{"rental_id": "pr-comfy", "name": "fixture", "state": "ready", "development": true, "ssh_address": "127.0.0.1:2222", "worker_address": listener.Addr().String(), "cert_pem": string(certPEM), "worker_id": "comfy-worker", "worker_boot_id": "comfy-boot", "creator_public_key": owner.PublicKey(), "hourly_rate_usd_micros": int64(1), "accelerator_count": 1, "requested_accelerator_model": "CPU"}
	h.mu.Unlock()
	bin := filepath.Join(root, "bin")
	must(t, os.Mkdir(bin, 0700))
	calls := filepath.Join(root, "transport.jsonl")
	outcome := filepath.Join(root, "outcome")
	fixture := "#!/usr/bin/python3\nimport sys,json\nr=json.loads(sys.stdin.readline());sys.stdin.read()\nwith open(" + pythonLiteral(calls) + ",'a') as f:f.write(json.dumps(r['action'])+'\\n')\nstatus='prepared' if r['action']=='prepare' else open(" + pythonLiteral(outcome) + ").read().strip()\nprint(json.dumps({'status':status,'prompt_id':'12345678-1234-4234-8234-123456789abc','files':[]}))\n"
	must(t, os.WriteFile(filepath.Join(bin, "ssh"), []byte(fixture), 0700))
	input := filepath.Join(root, "input.json")
	must(t, os.WriteFile(input, []byte(`{"graph_json":"{\"1\":{}}","output_root":"/outputs"}`), 0600))
	key := filepath.Join(root, "ssh-key")
	hosts := filepath.Join(root, "known-hosts")
	must(t, os.WriteFile(key, []byte("fixture"), 0600))
	must(t, os.WriteFile(hosts, []byte("fixture"), 0600))
	for _, status := range []string{"success", "failed"} {
		must(t, os.WriteFile(outcome, []byte(status), 0600))
		command := exec.Command(cozyBin, "dev", "comfy", "--rental=fixture", "--input="+input, "--idempotency-key="+status, "--ssh-key="+key, "--ssh-known-hosts="+hosts, "--out="+filepath.Join(root, "out-"+status), "--json", "--full")
		command.Env = childEnv(t, root)
		for i, entry := range command.Env {
			if strings.HasPrefix(entry, "PATH=") {
				command.Env[i] = "PATH=" + bin + string(os.PathListSeparator) + strings.TrimPrefix(entry, "PATH=")
			}
		}
		var stdout, stderr bytes.Buffer
		command.Stdout = &stdout
		command.Stderr = &stderr
		err := command.Run()
		if status == "success" && err != nil {
			t.Fatalf("%v %s %s", err, stdout.String(), stderr.String())
		}
		if status == "failed" && err == nil {
			t.Fatal("failed Comfy result exited zero")
		}
		var result map[string]any
		if json.Unmarshal(stdout.Bytes(), &result) != nil || result["status"] != status {
			t.Fatalf("missing terminal JSON after acknowledged keepalive: %q %q", stdout.String(), stderr.String())
		}
	}
	data, err := os.ReadFile(calls)
	must(t, err)
	if bytes.Count(data, []byte("start")) != 2 || peer.calls.Load() < 2 {
		t.Fatalf("dispatch skipped: %q / keepalives%d", data, peer.calls.Load())
	}
}

func pythonLiteral(value string) string { raw, _ := json.Marshal(value); return string(raw) }
