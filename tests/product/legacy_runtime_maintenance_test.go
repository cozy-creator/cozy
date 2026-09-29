package producttest

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
)

type legacyMaintenancePeer struct {
	pb.UnimplementedWorkerControlServer
	mu                             sync.Mutex
	arm, version, operation, state string
	posts                          int
}

type legacyProtocolPeer struct{ pb.UnimplementedPodHostServer }

func (p *legacyMaintenancePeer) postCount() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.posts
}

func (*legacyProtocolPeer) ProtocolInfo(context.Context, *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
	return &pb.ProtocolInfoResult{WireMinor: 68, MinimumWireMinor: 64}, nil
}

func (p *legacyMaintenancePeer) DescribeMachine(_ context.Context, q *pb.DescribeMachineQuery) (*pb.MachineDescription, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	worker, agent := "worker-legacy", "0.1.0"
	if p.arm == "wrong_machine" {
		worker = "another-worker"
	}
	if p.arm == "agent_changed" && p.posts > 0 {
		agent = "another-agent"
	}
	return &pb.MachineDescription{WorkerId: worker, WorkerBootId: "boot-legacy", Host: &pb.MachineHost{Version: agent, Phase: "ready", StartedAtUnixMs: 100}, Runtime: &pb.MachineRuntime{Version: p.version, TensorfsVersion: "0.3.78"}}, nil
}

func TestLegacyRuntimeMaintenancePreservesAuthorityAndOperation(t *testing.T) {
	binary := filepath.Join(t.TempDir(), "legacy-runtime-update")
	build := exec.Command("go", "build", "-o", binary, "./scripts/maintenance/legacy-runtime-update")
	build.Dir = filepath.Join("..", "..")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build maintenance command: %v\n%s", err, out)
	}
	for _, arm := range []string{"success", "rolled_back", "agent_changed", "wrong_machine", "foreign_pending"} {
		t.Run(arm, func(t *testing.T) {
			root := t.TempDir()
			layout := home.Paths(root)
			must(t, os.MkdirAll(layout.Rentals, 0700))
			public, private, err := ed25519.GenerateKey(rand.Reader)
			must(t, err)
			raw, err := x509.MarshalPKCS8PrivateKey(private)
			must(t, err)
			must(t, os.WriteFile(layout.RentalCreatorIdentity("pr-legacy-proof"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: raw}), 0600))
			peer := &legacyMaintenancePeer{arm: arm, version: "0.18.85"}
			if arm == "foreign_pending" {
				peer.operation, peer.state = "someone-elses-update", "waiting"
			}
			g := grpc.NewServer()
			pb.RegisterPodHostServer(g, &legacyProtocolPeer{})
			pb.RegisterWorkerControlServer(g, peer)
			defer g.Stop()
			server := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if strings.HasPrefix(r.Header.Get("Content-Type"), "application/grpc") {
					g.ServeHTTP(w, r)
					return
				}
				grant, err := capability.Verify(strings.TrimPrefix(r.Header.Get("Authorization"), "Cozy-Cap "), "worker-legacy", []ed25519.PublicKey{public}, time.Now(), "")
				if err != nil || !grant.Permits(capability.Maintenance) {
					t.Errorf("maintenance lost existing owner authority: %v", err)
					http.Error(w, "unauthorized", 403)
					return
				}
				peer.mu.Lock()
				defer peer.mu.Unlock()
				if r.Method == http.MethodPost && r.URL.Path == "/v1/machine/runtime/update" {
					var body struct {
						Operation         string
						Runtime, TensorFS struct{ Version string }
						Agent             *string
						Pin               *bool
					}
					if json.NewDecoder(r.Body).Decode(&body) != nil || body.Operation == "" || body.Runtime.Version != "0.18.89" || body.TensorFS.Version != "0.3.78" || body.Agent != nil || body.Pin != nil {
						t.Error("legacy update changed requested pair or pretended to select its agent")
					}
					peer.posts++
					peer.operation, peer.state, peer.version = body.Operation, "succeeded", "0.18.89"
					if arm == "rolled_back" {
						peer.state, peer.version = "rolled_back", "0.18.85"
					}
					w.WriteHeader(http.StatusAccepted)
					return
				}
				if r.Method != http.MethodGet || r.URL.Path != "/v1/machine/runtime" {
					t.Errorf("unexpected mutation %s %s", r.Method, r.URL.Path)
					http.Error(w, "unexpected", 400)
					return
				}
				state := map[string]any{"capabilities": []string{"runtime-update/1"}, "runtime": peer.version, "tensorfs": "0.3.78"}
				if peer.operation != "" {
					state["update"] = map[string]any{"operation": peer.operation, "state": peer.state, "error": "fixture outcome", "to": map[string]string{"runtime": "0.18.89", "tensorfs": "0.3.78"}}
				}
				_ = json.NewEncoder(w).Encode(state)
			}))
			server.EnableHTTP2 = true
			server.StartTLS()
			defer server.Close()
			cert := layout.RentalCert("pr-legacy-proof")
			must(t, os.WriteFile(cert, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: server.Certificate().Raw}), 0600))
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			fatal(t, store.RecordRental(records.Rental{ID: "pr-legacy-proof", MachineName: "motonari", State: "ready", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: "https://hub.example", Address: strings.TrimPrefix(server.URL, "https://"), CertPath: cert, ExpectedWorkerID: "worker-legacy", ExpectedWorkerBootID: "boot-legacy"}))
			store.Close()
			invoke := func(apply bool) (string, error) {
				args := []string{"--machine", "motonari", "--runtime-version", "0.18.89", "--tensorfs-version", "0.3.78"}
				if apply {
					args = append(args, "--apply", "--expect-agent-version", "0.1.0", "--expect-agent-started-unix-ms", "100")
				}
				ctx, cancel := context.WithTimeout(t.Context(), 15*time.Second)
				defer cancel()
				cmd := exec.CommandContext(ctx, binary, args...)
				cmd.Env = childEnv(t, root)
				out, err := cmd.CombinedOutput()
				return string(out), err
			}
			out, err := invoke(false)
			if arm == "wrong_machine" {
				if err == nil || !strings.Contains(out, "identity differs") {
					t.Fatalf("wrong machine was not refused: %v %s", err, out)
				}
				return
			}
			if err != nil || !strings.Contains(out, `"status":"plan_only"`) || peer.postCount() != 0 {
				t.Fatalf("default mode mutated: %v %s posts=%d", err, out, peer.postCount())
			}
			out, err = invoke(true)
			if arm == "foreign_pending" {
				if err == nil || peer.postCount() != 0 {
					t.Fatalf("competing update was admitted: %v %s posts=%d", err, out, peer.postCount())
				}
				return
			}
			if arm == "success" && (err != nil || !strings.Contains(out, `"status":"verified"`)) {
				t.Fatalf("ready pair not verified: %v %s", err, out)
			}
			if arm != "success" && err == nil {
				t.Fatalf("%s falsely succeeded: %s", arm, out)
			}
			if peer.postCount() != 1 {
				t.Fatalf("update submitted %d times", peer.postCount())
			}
			_, _ = invoke(true)
			if peer.postCount() != 1 {
				t.Fatal("stable operation retry submitted a second update")
			}
		})
	}
}
