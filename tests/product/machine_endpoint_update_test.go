package producttest

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/tls"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"encoding/pem"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/capability"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// A real authenticated TLS/HTTP2 peer exercises the ordinary CLI's upload,
// restart reconnect and observer-only resume. It is not SDK activation proof.
type endpointUpdatePeer struct {
	v1.UnimplementedMachineServer
	mu         sync.Mutex
	public     ed25519.PublicKey
	objects    map[string][]byte
	specs      map[string]*v1.RunSpec
	accepted   chan struct{}
	finish     chan struct{}
	finishOnce sync.Once
	current    string
	controls   atomic.Int32
	writes     atomic.Int32
	reconnects atomic.Int32
}

func (p *endpointUpdatePeer) authorize(ctx context.Context) error {
	md, _ := metadata.FromIncomingContext(ctx)
	values := md.Get("authorization")
	if len(values) != 1 {
		return status.Error(codes.Unauthenticated, "missing owner")
	}
	token := strings.TrimPrefix(values[0], "Cozy-Cap ")
	grant, err := capability.Verify(token, "endpoint-update-worker", []ed25519.PublicKey{p.public}, time.Now(), "")
	if err != nil || grant.Action != "machine" {
		return status.Error(codes.Unauthenticated, "wrong owner")
	}
	return nil
}

func (p *endpointUpdatePeer) Status(_ *v1.StatusRequest, stream grpc.ServerStreamingServer[v1.StatusFrame]) error {
	if err := p.authorize(stream.Context()); err != nil {
		return err
	}
	p.mu.Lock()
	current := p.current
	p.mu.Unlock()
	return stream.Send(&v1.StatusFrame{WorkerId: "endpoint-update-worker", BootId: "endpoint-update-boot", Phase: "ready", Runtime: current, Tensorfs: "0.4.0"})
}

func (p *endpointUpdatePeer) Write(stream grpc.ClientStreamingServer[v1.WriteFrame, v1.WriteResult]) error {
	if err := p.authorize(stream.Context()); err != nil {
		return err
	}
	first, err := stream.Recv()
	if err != nil {
		return err
	}
	p.writes.Add(1)
	p.mu.Lock()
	held := append([]byte(nil), p.objects[first.Digest]...)
	p.mu.Unlock()
	for {
		frame, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}
		held = append(held, frame.Data...)
	}
	if len(held) > 0 {
		hash := sha256.Sum256(held)
		if uint64(len(held)) != first.Length || "sha256:"+hex.EncodeToString(hash[:]) != first.Digest {
			return status.Error(codes.DataLoss, "wrong wheel bytes")
		}
		p.mu.Lock()
		p.objects[first.Digest] = held
		p.mu.Unlock()
	}
	return stream.SendAndClose(&v1.WriteResult{Held: uint64(len(held))})
}

func (p *endpointUpdatePeer) Control(context.Context, *v1.ControlRequest) (*v1.RunState, error) {
	p.controls.Add(1)
	return nil, status.Error(codes.Unimplemented, "an observer must never control the update")
}

func (p *endpointUpdatePeer) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	if err := p.authorize(stream.Context()); err != nil {
		return err
	}
	p.mu.Lock()
	stored := p.specs[request.Id]
	first := stored == nil && request.Spec != nil
	if stored != nil && request.Spec != nil && !proto.Equal(stored, request.Spec) {
		p.mu.Unlock()
		return status.Error(codes.AlreadyExists, "update id already owns another selection")
	}
	if first {
		stored = proto.Clone(request.Spec).(*v1.RunSpec)
		p.specs[request.Id] = stored
	}
	p.mu.Unlock()
	if stored == nil {
		return status.Error(codes.NotFound, "no accepted update")
	}
	if stored.Kind != v1.RunKind_RUN_KIND_UPDATE {
		return status.Error(codes.InvalidArgument, "not an update")
	}
	if request.After < 1 {
		if err := stream.Send(&v1.RunEvent{Sequence: 1, Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.Id, State: "running"}}}); err != nil {
			return err
		}
	}
	if first {
		close(p.accepted)
		return status.Error(codes.Unavailable, "service restart")
	}
	if request.Spec == nil && request.After == 1 {
		p.reconnects.Add(1)
	}
	select {
	case <-p.finish:
	case <-stream.Context().Done():
		return stream.Context().Err()
	}
	p.mu.Lock()
	p.current = "0.20.0"
	p.mu.Unlock()
	return stream.Send(&v1.RunEvent{Sequence: 2, Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "succeeded", Result: []byte(`{"from":{"runtime":"0.19.0","tensorfs":"0.4.0"},"to":{"runtime":"0.20.0","tensorfs":"0.4.0"}}`)}}})
}

func endpointUpdateFixture(t *testing.T) (string, string, *endpointUpdatePeer, *atomic.Int32) {
	t.Helper()
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	must(t, os.MkdirAll(layout.Machine, 0700))
	owner, problem := machines.NewHost(layout.Machine, "", nil).Owner()
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(owner.PublicKey())
	must(t, err)
	hubCalls := new(atomic.Int32)
	hub := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hubCalls.Add(1)
		http.Error(w, "endpoint maintenance must not ask a Hub", 500)
	}))
	t.Cleanup(hub.Close)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.URL+"\n"), 0600))
	certSource := httptest.NewTLSServer(http.NotFoundHandler())
	cert := certSource.TLS.Certificates[0]
	certSource.Close()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	must(t, err)
	peer := &endpointUpdatePeer{public: public, objects: map[string][]byte{}, specs: map[string]*v1.RunSpec{}, accepted: make(chan struct{}), finish: make(chan struct{}), current: "0.19.0"}
	server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}})))
	v1.RegisterMachineServer(server, peer)
	go server.Serve(listener)
	t.Cleanup(func() { peer.finishOnce.Do(func() { close(peer.finish) }); server.Stop() })
	ep := machineendpoint.Endpoint{Format: machineendpoint.Format, Address: listener.Addr().String(), WorkerID: "endpoint-update-worker", WorkerBootID: "endpoint-update-boot", WorkspaceID: "endpoint-workspace", CertificatePEM: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]}))}
	raw, err := json.Marshal(ep)
	must(t, err)
	path := filepath.Join(root, "endpoint.json")
	must(t, os.WriteFile(path, raw, 0600))
	return root, path, peer, hubCalls
}

func TestEndpointSoftwareUpdateUsesOwnerPreservesTensorFSAndReattaches(t *testing.T) {
	root, endpoint, peer, hubCalls := endpointUpdateFixture(t)
	before, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	wheel := filepath.Join(root, "cozy_runtime-0.20.0-py3-none-any.whl")
	body := bytes.Repeat([]byte("owned-wheel"), 100)
	must(t, os.WriteFile(wheel, body, 0600))
	peer.finishOnce.Do(func() { close(peer.finish) })
	code, out := runCozy(t, root, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "endpoint-update", "--runtime-wheel", wheel, "--keep-agent", "--json")
	if code != 0 || !strings.Contains(out, `"tensorfs":"0.4.0"`) || !strings.Contains(out, `"runtime":"0.20.0"`) {
		t.Fatalf("endpoint update: %d %s", code, out)
	}
	peer.mu.Lock()
	spec := proto.Clone(peer.specs["endpoint-update"]).(*v1.RunSpec)
	count := len(peer.specs)
	peer.mu.Unlock()
	var selection map[string]string
	must(t, json.Unmarshal(spec.Payload, &selection))
	if count != 1 || peer.reconnects.Load() != 1 || selection["agent"] != "explicit" || selection["tensorfs"] != "" || len(spec.Inputs) != 1 || spec.Inputs[0].Field != "runtime" {
		t.Fatalf("selection or reconnect changed: %v %v", selection, spec.Inputs)
	}
	must(t, os.Remove(wheel))
	writes := peer.writes.Load()
	code, out = runCozy(t, root, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "endpoint-update", "--observe", "--json")
	if code != 0 || peer.writes.Load() != writes || !strings.Contains(out, `"previous_runtime":"0.19.0"`) {
		t.Fatalf("resume reopened inputs or lost original result: %d %s", code, out)
	}
	if code, out = runCozy(t, root, "machine", "show", "--machine-endpoint-file", endpoint, "--json"); code != 0 || !strings.Contains(out, "0.20.0") {
		t.Fatalf("endpoint show: %d %s", code, out)
	}
	after, err := os.ReadFile(filepath.Join(root, config.FileName))
	must(t, err)
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	defer store.Close()
	rentals, problem := store.Rentals()
	fatal(t, problem)
	if !bytes.Equal(before, after) || len(rentals) != 0 || hubCalls.Load() != 0 || daemon.Probe(config.Config{Home: root}).Up || peer.controls.Load() != 0 {
		t.Fatal("endpoint maintenance changed config, enrolled a rental, asked a Hub, started the daemon, or sent Control")
	}
}

func TestEndpointUpdateObserverInterruptionLeavesTheUpdateAndWrongOwnerCannotWrite(t *testing.T) {
	root, endpoint, peer, _ := endpointUpdateFixture(t)
	wheel := filepath.Join(root, "cozy_runtime-0.20.0-py3-none-any.whl")
	must(t, os.WriteFile(wheel, []byte("wheel"), 0600))
	command := exec.Command(cozyBin, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "interrupted-update", "--runtime-wheel", wheel, "--json")
	command.Env = childEnv(t, root)
	var output bytes.Buffer
	command.Stdout = &output
	command.Stderr = &output
	must(t, command.Start())
	t.Cleanup(func() {
		if command.ProcessState == nil {
			_ = command.Process.Kill()
			_ = command.Wait()
		}
	})
	select {
	case <-peer.accepted:
	case <-time.After(10 * time.Second):
		t.Fatal("update was not accepted")
	}
	must(t, command.Process.Signal(os.Interrupt))
	if err := command.Wait(); err == nil {
		t.Fatal("detached observer claimed completion")
	}
	if peer.controls.Load() != 0 || !strings.Contains(output.String(), "machine.update_observation_lost") {
		t.Fatalf("observer interrupted work: %s", output.String())
	}
	peer.finishOnce.Do(func() { close(peer.finish) })
	code, out := runCozy(t, root, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "interrupted-update", "--observe", "--json")
	if code != 0 {
		t.Fatalf("reattach after interrupt: %d %s", code, out)
	}
	wrong := t.TempDir()
	layout, problem := home.Open(wrong)
	fatal(t, problem)
	must(t, os.MkdirAll(layout.Machine, 0700))
	_, problem = machines.NewHost(layout.Machine, "", nil).Owner()
	fatal(t, problem)
	writes := peer.writes.Load()
	code, out = runCozy(t, wrong, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "unauthorized", "--runtime-wheel", wheel, "--json")
	if code == 0 || peer.writes.Load() != writes {
		t.Fatalf("wrong owner could write: %d %s", code, out)
	}
}

func TestEndpointUpdateActivatesRealSoftwareAndKeepsThePinnedBoot(t *testing.T) {
	if *machineHostBinary == "" || *machineRuntimeWheel == "" || *machineTensorFSWheel == "" {
		t.Skip("requires a real native machine and its Runtime/TensorFS wheels")
	}
	root := t.TempDir()
	provisionMachine(t, root)
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\n"), 0600))
	t.Cleanup(func() { _, _ = runCozy(t, root, "machine", "stop") })
	if code, out := runCozy(t, root, "machine", "start", "--json"); code != 0 {
		t.Fatalf("start fixture machine: %d %s", code, out)
	}
	host := machines.NewHost(home.Paths(root).Machine, "", nil)
	before, problem := host.ReadStatus(context.Background())
	fatal(t, problem)
	var agent struct {
		WorkerPort int `json:"worker_port"`
	}
	raw, err := os.ReadFile(filepath.Join(root, "machine", "agent.json"))
	must(t, err)
	must(t, json.Unmarshal(raw, &agent))
	leaf, err := os.ReadFile(filepath.Join(root, "machine", "leaf.pem"))
	must(t, err)
	ep := machineendpoint.Endpoint{Format: machineendpoint.Format, Address: net.JoinHostPort("127.0.0.1", strconv.Itoa(agent.WorkerPort)),
		WorkerID: before.WorkerId, WorkerBootID: before.BootId, CertificatePEM: string(leaf), WorkspaceID: "native-update-proof"}
	raw, err = json.Marshal(ep)
	must(t, err)
	endpoint := filepath.Join(root, "endpoint.json")
	must(t, os.WriteFile(endpoint, raw, 0600))
	candidate := localBuild(t, *machineRuntimeWheel, "endpointupdate")
	expected := strings.SplitN(filepath.Base(candidate), "-", 3)[1]
	code, out := runCozy(t, root, "machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "native-endpoint-update",
		"--runtime-wheel", candidate, "--keep-agent", "--json")
	if code != 0 {
		t.Fatalf("real software update: %d %s\n%s", code, out, tail(filepath.Join(root, "machine", "host.log")))
	}
	after, problem := host.ReadStatus(context.Background())
	fatal(t, problem)
	if after.Runtime != expected || after.Tensorfs != before.Tensorfs || after.BootId != before.BootId {
		t.Fatalf("software or boot changed incorrectly: before %v after %v", before, after)
	}
	for _, args := range [][]string{{"machine", "show", "--machine-endpoint-file", endpoint, "--json"},
		{"machine", "update", "--machine-endpoint-file", endpoint, "--idempotency-key", "native-endpoint-update", "--observe", "--json"}} {
		if code, out := runCozy(t, root, args...); code != 0 || !strings.Contains(out, expected) || !strings.Contains(out, before.Tensorfs) {
			t.Fatalf("same pinned endpoint cannot observe activated pair: %d %s", code, out)
		}
	}
}
