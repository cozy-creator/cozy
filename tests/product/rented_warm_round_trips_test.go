package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// A warm rental's submission and cancellation cost round trips, not repeated reads. On
// 2026-09-27 an H3 submission spent 63-118 s probing 81 captured default rungs that name
// 3 checkpoints, and a cancel took 8-10 s to reach Runtime behind the run's observation.

type machineControlPeer interface {
	ControlMachineExecution(context.Context, *pb.MachineExecutionControl) (*pb.MachineExecutionState, error)
}

// startRentedFixture attaches the fake pod as rental tessa for proof/h3@1.0.0, whose
// prepare facts carry lockedExtra, and starts the daemon with daemonEnv. Run 1 is the
// blocker the machine's GPU wait names; the test's own run is run 2.
func startRentedFixture(t *testing.T, h *ladderHub, machine func(blocker string) machineExecutionPeer, lockedExtra string, daemonEnv ...string) string {
	return startRentedPod(t, h, &fakePod{}, machine, lockedExtra, daemonEnv...)
}

func startRentedPod(t *testing.T, h *ladderHub, pod *fakePod, machine func(blocker string) machineExecutionPeer, lockedExtra string, daemonEnv ...string) string {
	t.Helper()
	publishWorkflowRelease(t, h)
	var detail hub.PackageReleaseDetail
	response, err := http.Get(h.server.URL + "/v1/packages/proof/h3/releases/1.0.0")
	must(t, err)
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	root := ladderRoot(t, h)
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	holder, _, problem := store.Submit(records.Request{ID: "req-gpu-holder", IdemKey: "gpu-holder", BodyDigest: childDigest("7"),
		Package: ladderPackage, Entrypoint: "generate", Payload: []byte(`{}`)})
	fatal(t, problem)
	_, problem = store.FailQueuedRequest(holder.ID, map[string]any{"error_type": "proof", "error": "held elsewhere"})
	fatal(t, problem)
	identity, problem := rental.PendingCreatorIdentity(layout, "rented-round-trips")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	pod.controlKey, pod.machine, pod.deviceCount = public, machine(holder.ID), 4
	pod.releases = map[string]*pb.DescribedRelease{ladderPackage: {Package: ladderPackage, Release: "1.0.0", PackageInterface: detail.PackageInterface}}
	pod.preparedPlacement = func(download []byte, pkg, release string) *pb.Placement {
		placement := modelBearingPlacement(t)(download, pkg, release)
		placement.PackageInterface = detail.PackageInterface
		return placement
	}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	h.mu.Lock()
	h.rentals[podRental] = map[string]any{"rental_id": podRental, "name": "tessa", "state": "ready",
		"requested_accelerator_model": "fake-4090", "accelerator_count": 4, "hourly_rate_usd_micros": 1,
		"worker_address": connection.Addr, "media_address": connection.Media.Addr}
	h.mu.Unlock()
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "tessa", SKU: "fake-x", State: "ready",
		AcceleratorModel: "fake-4090", AcceleratorCount: 4, HourlyRateUSDMicros: 1, Hub: h.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr,
		ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	store.Close()
	startDaemonProcess(t, root, daemonEnv...)
	return root
}

// heldMachine is runtimeMachine whose journal reads can be held open, as a slow machine
// holds an observation, and which accepts one cancel.
type heldMachine struct {
	*runtimeMachine
	hold     chan struct{}
	holding  atomic.Int32
	commands map[string]bool
	canceled chan struct{}
	once     sync.Once
}

func (m *heldMachine) ListMachineExecutionEvents(ctx context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	hold := m.hold
	m.mu.Unlock()
	if hold != nil {
		m.holding.Add(1)
		defer m.holding.Add(-1)
		select {
		case <-hold:
		case <-ctx.Done():
			return nil, status.FromContextError(ctx.Err()).Err()
		}
	}
	return m.runtimeMachine.ListMachineExecutionEvents(ctx, query)
}

func (m *heldMachine) ControlMachineExecution(_ context.Context, command *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if !m.commands[command.CommandId] && m.state.State != "canceled" {
		if command.Action != pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL || command.ExpectedGeneration != m.state.Generation {
			return nil, status.Error(codes.FailedPrecondition, "unexpected control")
		}
		m.state.Generation++
		m.state.State = "canceled"
		m.record("control", []byte(`{"action":"cancel","generation":1}`))
		m.state.Sequence++
		m.events = append(m.events, outcomeEvent(m.state.Sequence, "canceled", m.outcome()))
	}
	m.commands[command.CommandId] = true
	m.once.Do(func() { close(m.canceled) })
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// A cancel reaches Runtime without waiting for an observation of the same run: the
// observation only reads the journal and resumes where it stopped.
func TestRentedCancelDoesNotWaitBehindAnObservation(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &heldMachine{runtimeMachine: &runtimeMachine{triage: bundle}, commands: map[string]bool{}, canceled: make(chan struct{})}
	root := startRentedFixture(t, h, func(blocker string) machineExecutionPeer { machine.blocker = blocker; return machine }, "")
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	for deadline, list := time.Now().Add(20*time.Second), ""; !strings.Contains(list, "waiting for GPU"); _, list = runCozy(t, root, "run", "list") {
		if time.Now().After(deadline) {
			t.Fatalf("the daemon never observed the accepted run:\n%s", list)
		}
	}
	hold := make(chan struct{})
	release := sync.OnceFunc(func() {
		machine.mu.Lock()
		machine.hold = nil
		machine.mu.Unlock()
		close(hold)
	})
	defer release()
	machine.mu.Lock()
	machine.hold = hold
	machine.mu.Unlock()
	waitFor(t, root, "an observation held open by the machine", func() bool { return machine.holding.Load() > 0 })

	finished := make(chan string, 1)
	go func() {
		_, out := runCozy(t, root, "run", "cancel", "2", "--json")
		finished <- out
	}()
	select {
	case <-machine.canceled:
	case <-time.After(20 * time.Second):
		t.Fatalf("the cancel waited behind the held observation: %s", tail(filepath.Join(root, "daemon.log")))
	}
	release()
	select {
	case out := <-finished:
		var result struct {
			Status  string `json:"status"`
			Changed bool   `json:"changed"`
		}
		if json.Unmarshal([]byte(out), &result) != nil || result.Status != "canceled" || !result.Changed {
			t.Fatalf("cozy run cancel did not report the canceled run: %s", out)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("cozy run cancel did not return after Runtime canceled: %s", tail(filepath.Join(root, "daemon.log")))
	}
	machine.mu.Lock()
	sent := len(machine.commands)
	machine.mu.Unlock()
	if sent != 1 {
		t.Fatalf("Runtime received %d cancel commands; want one", sent)
	}
}

// acceptingMachines accepts every submission and keeps each one waiting for GPUs.
type acceptingMachines struct {
	mu     sync.Mutex
	states map[string]*pb.MachineExecutionState
}

func (m *acceptingMachines) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *acceptingMachines) SubmitMachineExecution(_ context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	id := submit.Offer.RequestId
	if m.states[id] == nil {
		m.states[id] = &pb.MachineExecutionState{RequestId: id, WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId,
			ExecutionWorkspaceId: "rented-workspace", Generation: 1, AttemptOrdinal: 1, State: "running"}
	}
	return &pb.MachineExecutionReceipt{RequestId: id, SubmissionId: submit.SubmissionId, CaptureDigest: submit.CaptureDigest,
		InvocationSpecDigest: submit.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: submit.Claim.WorkerId, WorkerBootId: submit.Claim.WorkerBootId, ExecutionWorkspaceId: "rented-workspace"}, nil
}

func (m *acceptingMachines) GetMachineExecution(_ context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if state := m.states[query.RequestId]; state != nil {
		return proto.Clone(state).(*pb.MachineExecutionState), nil
	}
	return nil, status.Error(codes.NotFound, "no such execution")
}

func (m *acceptingMachines) ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return &pb.MachineExecutionEventPage{}, nil
}

func (m *acceptingMachines) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return nil, status.Error(codes.FailedPrecondition, "still running")
}

func (m *acceptingMachines) accepted(id string) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.states[id] != nil
}
