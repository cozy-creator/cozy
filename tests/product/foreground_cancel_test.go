package producttest

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/pem"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/machines"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"
)

// The remote acknowledgement is deliberately withheld. Persisting local intent
// must not let a short-lived foreground controller close before its Control arrives.
type foregroundCancelMachine struct {
	v1.UnimplementedMachineServer
	entered     chan struct{}
	ack         chan struct{}
	finish      chan struct{}
	reads       chan struct{}
	once        sync.Once
	calls       atomic.Int32
	applied     atomic.Bool
	unavailable atomic.Bool
}

func (m *foregroundCancelMachine) Control(ctx context.Context, request *v1.ControlRequest) (*v1.RunState, error) {
	m.calls.Add(1)
	if m.unavailable.Load() {
		return nil, status.Error(codes.Unavailable, "fixture transport unavailable")
	}
	if strings.HasSuffix(request.Id, "-queued") {
		return nil, status.Error(codes.NotFound, "this queued run was never sent")
	}
	m.once.Do(func() { close(m.entered) })
	select {
	case <-m.ack:
		m.applied.Store(true)
		return &v1.RunState{Id: request.Id, Number: 1, State: "canceling", Sequence: 1}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (m *foregroundCancelMachine) Run(request *v1.RunRequest, stream grpc.ServerStreamingServer[v1.RunEvent]) error {
	state := "running"
	if m.applied.Load() {
		state = "canceling"
	}
	if err := stream.Send(&v1.RunEvent{Sequence: 1, AtMs: time.Now().UnixMilli(), Event: &v1.RunEvent_State{State: &v1.RunState{Id: request.Id, Number: 1, State: state, Sequence: 1}}}); err != nil {
		return err
	}
	select {
	case m.reads <- struct{}{}:
	default:
	}
	select {
	case <-m.finish:
		return stream.Send(&v1.RunEvent{Sequence: 2, AtMs: time.Now().UnixMilli(), Event: &v1.RunEvent_Outcome{Outcome: &v1.Outcome{Status: "canceled"}}})
	case <-stream.Context().Done():
		return stream.Context().Err()
	}
}

func TestForegroundCancelDeliversBeforeExitAndAwaitStillWaitsForTerminal(t *testing.T) {
	for _, kind := range []string{"serving", "job"} {
		t.Run(kind, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			must(t, os.MkdirAll(layout.Machine, 0700))
			_, problem = machines.NewHost(layout.Machine, "", nil).Owner()
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			certSource := httptest.NewTLSServer(http.NotFoundHandler())
			cert := certSource.TLS.Certificates[0]
			certSource.Close()
			listener, err := net.Listen("tcp", "127.0.0.1:0") //cozy:allow test machine transport
			must(t, err)
			machine := &foregroundCancelMachine{entered: make(chan struct{}), ack: make(chan struct{}), finish: make(chan struct{}), reads: make(chan struct{}, 16)}
			server := grpc.NewServer(grpc.Creds(credentials.NewTLS(&tls.Config{MinVersion: tls.VersionTLS12, Certificates: []tls.Certificate{cert}})))
			v1.RegisterMachineServer(server, machine)
			go server.Serve(listener)
			defer server.Stop()
			ep := &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: listener.Addr().String(), WorkerID: "fixture-worker", WorkerBootID: "fixture-boot", WorkspaceID: "fixture",
				CertificatePEM: string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: cert.Certificate[0]}))}
			// Start the persistent owner before recording held queue intent, so its
			// startup recovery cannot win dispatch before this explicit cancel.
			startDaemonProcess(t, root)
			queued, _, problem := store.SubmitWithEvent(records.Request{ID: "req-" + kind + "-queued", IdemKey: kind + "-queued", Package: "proof/foreground", Entrypoint: "main", Kind: kind,
				Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true}, map[string]any{"machine_endpoint": ep})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(queued.ID, ep.Name()))
			if code, out := runCozy(t, root, "run", "cancel", queued.ID, "--json"); code != 0 || !strings.Contains(out, `"status":"canceled"`) {
				state, _ := store.RequestRow(queued.ID)
				link, _ := store.MachineExecution(queued.ID)
				t.Logf("queued state=%s receipt=%d controls=%d", state.State, len(link.Receipt), machine.calls.Load())
				t.Fatalf("unsent cancellation: %d %s", code, out)
			}
			if machine.calls.Load() != 0 {
				t.Fatal("locally canceled unsent work reached the remote machine")
			}
			row, _, problem := store.SubmitWithEvent(records.Request{ID: "req-foreground-" + kind, IdemKey: kind, Package: "proof/foreground", Entrypoint: "main", Kind: kind,
				Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true}, map[string]any{"machine_endpoint": ep})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(row.ID, ep.Name()))
			fatal(t, store.AppendEvent(row.ID, records.RunV1Sent, 0, map[string]any{"machine": ep.Name()}))
			fatal(t, store.AcceptRunV1(row.ID, ep.Name(), &v1.RunState{Id: row.ID, Number: 1, State: "running", Sequence: 1, Attempt: 1}))
			if code, out := runCozy(t, root, "run", "show", row.ID, "--json"); code != 0 {
				t.Fatalf("observation: %d %s", code, out)
			}
			if machine.calls.Load() != 0 {
				t.Fatal("an ordinary observation originated cancellation")
			}
			start := func(args ...string) (<-chan error, *bytes.Buffer, *bytes.Buffer) {
				t.Helper()
				ctx, cancel := context.WithCancel(t.Context())
				t.Cleanup(cancel)
				cmd := exec.CommandContext(ctx, cozyBin, args...)
				cmd.Env = childEnv(t, root)
				out, diagnostic := new(bytes.Buffer), new(bytes.Buffer)
				cmd.Stdout, cmd.Stderr = out, diagnostic
				must(t, cmd.Start())
				done := make(chan error, 1)
				go func() { done <- cmd.Wait() }()
				return done, out, diagnostic
			}
			done, out, diagnostic := start("run", "cancel", row.ID, "--json")
			select {
			case <-machine.entered:
			case err := <-done:
				t.Fatalf("cancel exited before remote delivery: %v %s %s", err, out, diagnostic)
			case <-time.After(10 * time.Second):
				t.Fatal("cancel never reached its recorded endpoint")
			}
			select {
			case err := <-done:
				t.Fatalf("cancel exited before acknowledgement: %v %s %s", err, out, diagnostic)
			case <-time.After(150 * time.Millisecond):
			}
			close(machine.ack)
			select {
			case err := <-done:
				if err != nil || !machine.applied.Load() || !strings.Contains(out.String(), `"status":"canceling"`) {
					t.Fatalf("cancel did not return after acknowledgement: %v %s %s", err, out, diagnostic)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("unawaited cancel waited for the terminal outcome")
			}
			for len(machine.reads) > 0 {
				<-machine.reads
			}
			done, out, diagnostic = start("run", "cancel", row.ID, "--await", "--json")
			select {
			case <-machine.reads:
			case err := <-done:
				t.Fatalf("await exited before observation: %v %s %s", err, out, diagnostic)
			case <-time.After(10 * time.Second):
				t.Fatal("await did not observe the canceled run")
			}
			select {
			case err := <-done:
				t.Fatalf("await exited before terminal: %v %s %s", err, out, diagnostic)
			case <-time.After(150 * time.Millisecond):
			}
			close(machine.finish)
			select {
			case err := <-done:
				if err != nil || !strings.Contains(out.String(), `"status":"canceled"`) {
					t.Fatalf("awaited cancel: %v %s %s", err, out, diagnostic)
				}
			case <-time.After(10 * time.Second):
				t.Fatal("await did not finish at the terminal")
			}
			actor, _, _, problem := store.CancelAttribution(row.ID)
			fatal(t, problem)
			if actor != "cozy run cancel" {
				t.Fatalf("cancellation lost its explicit actor: %q", actor)
			}
			machine.unavailable.Store(true)
			refused, _, problem := store.SubmitWithEvent(records.Request{ID: "req-" + kind + "-unavailable", IdemKey: kind + "-unavailable", Package: "proof/foreground", Entrypoint: "main", Kind: kind,
				Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true}, map[string]any{"machine_endpoint": ep})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(refused.ID, ep.Name()))
			fatal(t, store.AppendEvent(refused.ID, records.RunV1Sent, 0, map[string]any{"machine": ep.Name()}))
			if code, out := runCozy(t, root, "run", "cancel", refused.ID, "--json"); code != 0 || !strings.Contains(out, `"status":"canceling"`) {
				t.Fatalf("unavailable transport lost pending cancellation: %d %s", code, out)
			}
			retained, problem := store.RequestRow(refused.ID)
			fatal(t, problem)
			link, problem := store.MachineExecution(refused.ID)
			fatal(t, problem)
			actor, _, _, problem = store.CancelAttribution(refused.ID)
			fatal(t, problem)
			if retained.State != "canceling" || !link.CancelRequested || actor != "cozy run cancel" {
				t.Fatalf("transport refusal discarded explicit intent: %+v %+v %q", retained, link, actor)
			}
		})
	}
}
