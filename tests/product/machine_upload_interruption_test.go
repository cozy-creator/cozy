package producttest

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/localpackage"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// The relay never terminates TLS. Certificate pinning, owner authentication,
// upload parsing, fsync, preparation and execution all remain in the real peers.
type interruptedHostControl struct {
	listener net.Listener
	upstream string
	mu       sync.Mutex
	peers    map[net.Conn]bool
	before   int64
	after    int64
	cut      bool
	resumed  bool
	dropped  chan struct{}
	resume   chan struct{}
	stop     chan struct{}
	wait     sync.WaitGroup
}

func newInterruptedHostControl(t *testing.T, upstream string) *interruptedHostControl {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow test-only loopback relay preserves end-to-end TLS while interrupting an upload
	must(t, err)
	p := &interruptedHostControl{listener: listener, upstream: upstream, peers: map[net.Conn]bool{}, dropped: make(chan struct{}), resume: make(chan struct{}), stop: make(chan struct{})}
	p.wait.Add(1)
	go func() {
		defer p.wait.Done()
		for {
			client, err := listener.Accept()
			if err != nil {
				return
			}
			p.wait.Add(1)
			go p.relay(client)
		}
	}()
	t.Cleanup(func() {
		close(p.stop)
		listener.Close()
		p.closePeers()
		p.wait.Wait()
	})
	return p
}

func (p *interruptedHostControl) closePeers() {
	p.mu.Lock()
	defer p.mu.Unlock()
	for peer := range p.peers {
		peer.Close()
	}
}

func (p *interruptedHostControl) relay(client net.Conn) {
	defer p.wait.Done()
	defer client.Close()
	p.mu.Lock()
	paused := p.cut && !p.resumed
	p.peers[client] = true
	p.mu.Unlock()
	defer func() { p.mu.Lock(); delete(p.peers, client); p.mu.Unlock() }()
	if paused {
		select {
		case <-p.resume:
		case <-p.stop:
			return
		}
	}
	raw, err := os.ReadFile(p.upstream)
	var host actualChildHost
	if err != nil || json.Unmarshal(raw, &host) != nil {
		return
	}
	server, err := net.DialTimeout("tcp", host.Control, 5*time.Second)
	if err != nil {
		return
	}
	defer server.Close()
	p.mu.Lock()
	if p.cut && !p.resumed {
		p.mu.Unlock()
		return
	}
	p.peers[server] = true
	p.mu.Unlock()
	defer func() { p.mu.Lock(); delete(p.peers, server); p.mu.Unlock() }()
	reverse := make(chan struct{})
	go func() {
		defer close(reverse)
		_, _ = io.Copy(client, server)
		client.Close()
	}()
	defer func() { server.Close(); client.Close(); <-reverse }()
	buffer := make([]byte, 32<<10)
	for {
		n, err := client.Read(buffer)
		if n > 0 {
			if _, writeErr := server.Write(buffer[:n]); writeErr != nil {
				return
			}
			p.mu.Lock()
			trip := false
			if p.resumed {
				p.after += int64(n)
			} else if !p.cut {
				p.before += int64(n)
				if p.before >= 4<<20 {
					p.cut, trip = true, true
					close(p.dropped)
				}
			}
			p.mu.Unlock()
			if trip {
				p.closePeers()
				return
			}
			// Make durable prefix acknowledgements observable before the next burst;
			// this is a fault test, never a throughput benchmark.
			time.Sleep(4 * time.Millisecond)
		}
		if err != nil {
			return
		}
	}
}

func (p *interruptedHostControl) reconnect() {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.resumed {
		p.resumed = true
		close(p.resume)
	}
}

type hostUploadFile struct {
	Operation string `json:"operation"`
	Source    string `json:"source"`
	Digest    string `json:"digest"`
	Filename  string `json:"filename"`
	Length    int64  `json:"length"`
	Received  int64  `json:"received"`
	State     string `json:"state"`
	Preparing string `json:"operation_state"`
}

func actualHostUploadFiles(t *testing.T, host actualChildHost) []hostUploadFile {
	t.Helper()
	raw, err := exec.Command("docker", "exec", host.Container, "python3", "-c", `import json,sqlite3
c=sqlite3.connect('file:/var/lib/cozy/pod-supervisor.sqlite?mode=ro',uri=True)
c.row_factory=sqlite3.Row
print(json.dumps([dict(row) for row in c.execute('''SELECT f.operation_id AS operation,
lower(hex(o.source_digest)) AS source, lower(hex(f.digest)) AS digest, f.filename,
f.length, f.received_bytes AS received, f.state, o.state AS operation_state
FROM local_package_files f JOIN local_package_operations o USING(operation_id)
ORDER BY f.operation_id,f.digest''')]))`).CombinedOutput()
	if err != nil {
		t.Fatalf("read actual Host upload journal: %v %s", err, raw)
	}
	var files []hostUploadFile
	must(t, json.Unmarshal(raw, &files))
	return files
}

func TestMachineExecutionActualHostInterruptedUpload(t *testing.T) {
	if *machineExecutionScript == "" || *childHostLauncher == "" || *childHostHome == "" || *childHostRuntimeBin == "" {
		t.Skip("requires a new actual Host home and a captured script with a wheel larger than 4 MiB")
	}
	if _, err := os.Stat(filepath.Join(*childHostHome, "host-container.json")); !os.IsNotExist(err) {
		t.Fatal("interruption proof requires a new task-owned Host home")
	}
	upstream := filepath.Join(*childHostHome, "upload-upstream-host.json")
	proxy := newInterruptedHostControl(t, upstream)
	launcher := *childHostLauncher
	wrapper := filepath.Join(t.TempDir(), "launch-through-control-proxy.py")
	quoted := func(value string) string { raw, _ := json.Marshal(value); return string(raw) }
	program := fmt.Sprintf(`import json,subprocess,sys
from pathlib import Path
host=json.loads(subprocess.check_output(['python3',%s,sys.argv[1]]))
Path(%s).write_text(json.dumps(host))
host['control_address']=%s
print(json.dumps(host))
`, quoted(launcher), quoted(upstream), quoted(proxy.listener.Addr().String()))
	must(t, os.WriteFile(wrapper, []byte(program), 0600))
	*childHostLauncher = wrapper
	t.Cleanup(func() { *childHostLauncher = launcher })
	layout, store, host, path, _ := startActualChildHost(t)
	t.Cleanup(func() {
		proxy.reconnect()
		if t.Failed() {
			t.Logf("retained failed proof home %s and owned container %s", layout.Root, host.Container)
			return
		}
		compositionDown(t, layout.Root, path)
		label, err := exec.Command("docker", "inspect", "--format", `{{index .Config.Labels "com.cozy.task"}}`, host.Container).Output()
		if err != nil || strings.TrimSpace(string(label)) != "proto056-installation-reuse" {
			t.Error("refused removal of container without this proof's ownership label")
			return
		}
		if out, err := exec.Command("docker", "rm", "-f", host.Container).CombinedOutput(); err != nil {
			t.Errorf("remove owned proof container: %v %s", err, out)
		}
	})
	key := "initial-private-upload-interruption"
	args := []string{"run", *machineExecutionScript, "--rental", "child-host", "--await", "--json", "--full", "--idempotency-key", key}
	type answer struct {
		code   int
		output string
	}
	finished := make(chan answer, 1)
	go func() { code, out := runCozyPath(t, layout.Root, path, args...); finished <- answer{code, out} }()
	select {
	case <-proxy.dropped:
	case result := <-finished:
		t.Fatalf("CLI ended before the real network interruption [%d]: %s", result.code, result.output)
	case <-time.After(5 * time.Minute):
		t.Fatal("fixture sent no 4 MiB upload to interrupt")
	}
	paused := actualHostUploadFiles(t, host)
	var partial *hostUploadFile
	for index := range paused {
		row := &paused[index]
		if row.State == "receiving" && row.Received >= 1<<20 && row.Received < row.Length {
			partial = row
			break
		}
	}
	if partial == nil || partial.Preparing != "files" {
		t.Fatalf("no durable partial file before preparation: %+v", paused)
	}
	request, problem := store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request == nil {
		t.Fatal("upload began before the ordinary CLI recorded its request")
	}
	events, problem := store.EventsAfter(request.ID, 0, 1000)
	fatal(t, problem)
	started := false
	for _, event := range events {
		if event.Type == "machine.package_upload_started" && event.Payload["operation_id"] == partial.Operation {
			started = true
		}
		if event.Type == "machine.package_uploaded" && event.Payload["operation_id"] == partial.Operation {
			t.Fatal("client recorded an incomplete upload as complete")
		}
	}
	if !started {
		t.Fatal("initial upload has no durable started operation for reconnect")
	}
	proxy.reconnect()
	result := <-finished
	if result.code != 0 {
		// A command may report the transient disconnect before its next invocation
		// resumes the same request. An installer refusal is never a retry signal.
		if result.code != int(exit.Unavailable) {
			t.Fatalf("interrupted upload became a terminal refusal [%d]: %s", result.code, result.output)
		}
		result.code, result.output = runCozyPath(t, layout.Root, path, args...)
	}
	if result.code != 0 {
		t.Fatalf("same-request upload reconnect [%d]: %s", result.code, result.output)
	}
	request, problem = store.RequestByIdempotencyKey(key)
	fatal(t, problem)
	if request == nil || request.State != "succeeded" {
		t.Fatalf("resumed CLI request did not execute: %+v", request)
	}
	completed := actualHostUploadFiles(t, host)
	remaining := int64(0)
	matched := false
	for _, row := range completed {
		remaining += row.Length
		if row.Operation == partial.Operation && row.Digest == partial.Digest {
			matched = row.State == "verified" && row.Received == row.Length && row.Preparing == "prepared" && row.Source == partial.Source
		}
	}
	for _, row := range paused {
		remaining -= row.Received
	}
	proxy.mu.Lock()
	after := proxy.after
	proxy.mu.Unlock()
	if !matched || after > remaining+(1<<20) {
		t.Fatalf("resume lost its exact operation or resent retained bytes: matched=%v forwarded=%d remaining=%d files=%+v", matched, after, remaining, completed)
	}
	proof, err := json.MarshalIndent(map[string]any{"request": request.ID, "paused": paused, "completed": completed, "forwarded_after_reconnect": after, "remaining_wheel_bytes": remaining, "tls_passthrough": true}, "", "  ")
	must(t, err)
	must(t, os.WriteFile(filepath.Join(layout.Root, "upload-interruption-proof.json"), proof, 0600))
	t.Logf("actual Host upload interruption: %s", proof)
}

type loseVerifiedUploadACK struct {
	pb.UnimplementedPodHostServer
	receiver *uploadReceiver
	lost     atomic.Bool
}

type verifiedACKStream struct {
	grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]
	lost *atomic.Bool
}

func (s *verifiedACKStream) Send(row *pb.LocalPackageFileStatus) error {
	if row.State == pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED && s.lost.CompareAndSwap(false, true) {
		return status.Error(codes.Unavailable, "lost final acknowledgement after durable verification")
	}
	return s.BidiStreamingServer.Send(row)
}

func (r *loseVerifiedUploadACK) LocalPackageUpload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	return r.receiver.LocalPackageUpload(&verifiedACKStream{stream, &r.lost})
}

// Supplementary protocol proof: the real CLI/Host test above cannot selectively
// suppress an encrypted gRPC acknowledgement without terminating TLS.
func TestUploadLostFinalACKReplaysVerifiedHeaderWithoutCarrier(t *testing.T) {
	_, carrier, header := uploadFixture(t, 5<<20+31)
	receiver := &uploadReceiver{path: filepath.Join(t.TempDir(), "received")}
	peer := &loseVerifiedUploadACK{receiver: receiver}
	client, _ := uploadClient(t, peer)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if problem := localpackage.UploadFile(ctx, client, header, carrier, nil); problem == nil || problem.Code != exit.Unavailable || !peer.lost.Load() {
		t.Fatalf("final ACK loss did not remain resumable: %v", problem)
	}
	must(t, os.Remove(carrier))
	if problem := localpackage.UploadFile(ctx, client, header, carrier, nil); problem != nil {
		t.Fatalf("verified header replay reopened a missing carrier: %v", problem)
	}
	receiver.mu.Lock()
	defer receiver.mu.Unlock()
	if len(receiver.streams) != 2 || len(receiver.streams[1]) != 0 {
		t.Fatalf("verified replay resent chunk bytes: %+v", receiver.streams)
	}
}
