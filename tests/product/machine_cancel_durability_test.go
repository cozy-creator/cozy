package producttest

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
)

type cancelGate struct {
	held    atomic.Bool
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func newCancelGate() *cancelGate {
	return &cancelGate{entered: make(chan struct{}, 1), release: make(chan struct{})}
}

func (g *cancelGate) wait(ctx context.Context) error {
	if !g.held.Load() {
		return nil
	}
	select {
	case g.entered <- struct{}{}:
	default:
	}
	select {
	case <-g.release:
		return nil
	case <-ctx.Done():
		return status.FromContextError(ctx.Err()).Err()
	}
}

func (g *cancelGate) open() { g.once.Do(func() { g.held.Store(false); close(g.release) }) }

type durableCancelMachine struct {
	*heldMachine
	readGate         *cancelGate
	stale, loseReply atomic.Bool
	controlAttempts  []struct {
		id         string
		generation uint64
	}
}

func (m *durableCancelMachine) GetMachineExecution(ctx context.Context, q *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	if err := m.readGate.wait(ctx); err != nil {
		return nil, err
	}
	return m.runtimeMachine.GetMachineExecution(ctx, q)
}

func (m *durableCancelMachine) ControlMachineExecution(ctx context.Context, q *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	m.controlAttempts = append(m.controlAttempts, struct {
		id         string
		generation uint64
	}{q.CommandId, q.ExpectedGeneration})
	if m.stale.Swap(false) {
		m.state.Generation++
		m.mu.Unlock()
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_generation_stale"))
		return nil, status.Error(codes.FailedPrecondition, "generation changed before acceptance")
	}
	m.mu.Unlock()
	result, err := m.heldMachine.ControlMachineExecution(ctx, q)
	if err == nil && m.loseReply.Swap(false) {
		return nil, status.Error(codes.Unavailable, "acknowledgement lost after commit")
	}
	return result, err
}

func durableCancelFixture(t *testing.T) (string, *records.Store, *durableCancelMachine, *cancelGate, string) {
	t.Helper()
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	m := &durableCancelMachine{heldMachine: &heldMachine{runtimeMachine: &runtimeMachine{triage: bundle}, commands: map[string]bool{}, canceled: make(chan struct{})}, readGate: newCancelGate()}
	connect := newCancelGate()
	pod := &fakePod{protocolInfo: func(ctx context.Context, _ *pb.ProtocolInfoRequest) (*pb.ProtocolInfoResult, error) {
		if err := connect.wait(ctx); err != nil {
			return nil, err
		}
		return &pb.ProtocolInfoResult{WireMinor: pb.WireMinor, MinimumWireMinor: pb.MinCompatibleWireMinor}, nil
	}}
	root := startRentedPod(t, h, pod, func(blocker string) machineExecutionPeer { m.blocker = blocker; return m }, "")
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("submit [%d]: %s", code, out)
	}
	store, problem := records.Open(home.Paths(root).DB)
	fatal(t, problem)
	t.Cleanup(store.Close)
	var id string
	waitFor(t, root, "recorded machine acceptance", func() bool {
		submitted := m.submitted()
		if submitted == nil {
			return false
		}
		id = submitted.Offer.RequestId
		link, _ := store.MachineExecution(id)
		return link != nil && len(link.Receipt) > 0
	})
	t.Cleanup(m.readGate.open)
	t.Cleanup(connect.open)
	return root, store, m, connect, id
}

func assertDurableCancel(t *testing.T, store *records.Store, id string) {
	t.Helper()
	link, problem := store.MachineExecution(id)
	fatal(t, problem)
	row, problem := store.RequestRow(id)
	fatal(t, problem)
	actor, _, _, problem := store.CancelAttribution(id)
	fatal(t, problem)
	if link == nil || row == nil {
		t.Fatal("cancel lost its local request or machine link")
	}
	if !link.CancelRequested || row.State != "canceling" || actor != "cozy run cancel" {
		t.Fatalf("cancel was not durably attributed before the machine replied: cancel=%v state=%s actor=%q", link.CancelRequested, row.State, actor)
	}
	events, problem := store.EventsAfter(id, 0, 1000)
	fatal(t, problem)
	count := 0
	for _, event := range events {
		if event.Type == "request.cancel_requested" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("cancellation attribution duplicated: %d", count)
	}
}

func TestMachineCancelRecordsIntentBeforeUnresponsiveRemote(t *testing.T) {
	for _, arm := range []string{"state_read_and_restart", "connection"} {
		t.Run(arm, func(t *testing.T) {
			root, store, m, connect, id := durableCancelFixture(t)
			gate := m.readGate
			if arm == "connection" {
				if code, out := runCozy(t, root, "down", "--json"); code != 0 {
					t.Fatalf("observer down [%d]: %s", code, out)
				}
				gate = connect
				gate.held.Store(true)
				startDaemonProcess(t, root)
			} else {
				gate.held.Store(true)
			}
			defer gate.open()
			type response struct {
				code int
				out  string
			}
			finished := make(chan response, 1)
			go func() { code, out := runCozy(t, root, "run", "cancel", id, "--json"); finished <- response{code, out} }()
			select {
			case result := <-finished:
				var body struct {
					Status string `json:"status"`
				}
				if result.code != 0 || json.Unmarshal([]byte(result.out), &body) != nil || body.Status != "canceling" {
					t.Fatalf("cancel intent acknowledgement [%d]: %s", result.code, result.out)
				}
			case <-time.After(5 * time.Second):
				assertDurableCancel(t, store, id)
				t.Fatal("cancel command waited for an unavailable machine after recording intent")
			}
			assertDurableCancel(t, store, id)
			select {
			case <-gate.entered:
			case <-time.After(5 * time.Second):
				t.Fatal("fixture did not reach the held remote boundary")
			}
			if code, out := runCozy(t, root, "run", "cancel", id, "--json"); code != 0 {
				t.Fatalf("repeat cancel [%d]: %s", code, out)
			}
			assertDurableCancel(t, store, id)
			if arm == "state_read_and_restart" {
				if code, out := runCozy(t, root, "down", "--json"); code != 0 {
					t.Fatalf("observer down [%d]: %s", code, out)
				}
				assertDurableCancel(t, store, id)
				startDaemonProcess(t, root)
				assertDurableCancel(t, store, id)
			}
			gate.open()
			select {
			case <-m.canceled:
			case <-time.After(20 * time.Second):
				t.Fatal("recorded cancellation did not reach the recovered machine")
			}
			waitFor(t, root, "authoritative canceled outcome", func() bool { r, _ := store.RequestRow(id); return r != nil && r.State == "canceled" })
			m.mu.Lock()
			commands := len(m.commands)
			m.mu.Unlock()
			if commands != 1 {
				t.Fatalf("recovery delivered %d effective commands", commands)
			}
		})
	}
}

func TestMachineCancelReconcilesGenerationAndLostReply(t *testing.T) {
	for _, arm := range []string{"stale_generation", "lost_reply"} {
		t.Run(arm, func(t *testing.T) {
			root, store, m, _, id := durableCancelFixture(t)
			if arm == "stale_generation" {
				m.stale.Store(true)
			} else {
				m.loseReply.Store(true)
			}
			if code, out := runCozy(t, root, "run", "cancel", id, "--json"); code != 0 {
				t.Fatalf("cancel admission [%d]: %s", code, out)
			}
			waitFor(t, root, "reconciled cancellation", func() bool {
				m.mu.Lock()
				attempts := len(m.controlAttempts)
				m.mu.Unlock()
				r, _ := store.RequestRow(id)
				return attempts >= 2 && r != nil && r.State == "canceled"
			})
			m.mu.Lock()
			attempts := append([]struct {
				id         string
				generation uint64
			}{}, m.controlAttempts...)
			commands := len(m.commands)
			m.mu.Unlock()
			if commands != 1 {
				t.Fatalf("cancel had %d effects across retry", commands)
			}
			if arm == "lost_reply" && attempts[0] != attempts[1] {
				t.Fatalf("lost reply changed command identity: %+v", attempts)
			}
			if arm == "stale_generation" && (attempts[0].id == attempts[1].id || attempts[1].generation <= attempts[0].generation) {
				t.Fatalf("stale generation was not refreshed: %+v", attempts)
			}
			actor, _, _, problem := store.CancelAttribution(id)
			fatal(t, problem)
			if actor != "cozy run cancel" {
				t.Fatalf("retry lost actor: %q", actor)
			}
		})
	}
}
