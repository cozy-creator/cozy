package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
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
	t.Helper()
	publishWorkflowRelease(t, h)
	var detail hub.PackageReleaseDetail
	response, err := http.Get(h.server.URL + "/v1/packages/proof/h3/releases/1.0.0")
	must(t, err)
	must(t, json.NewDecoder(response.Body).Decode(&detail))
	response.Body.Close()
	facts := testPrepareFacts(ladderPackage, "1.0.0")
	inventory, err := json.Marshal(map[string]any{"format": "tensorhub.image_inventory/1",
		"profile": facts.ImageInventory.Profile, "python": facts.ImageInventory.Python,
		"distributions": []map[string]string{{"name": runtimeDistribution, "version": "0.18.41"}}})
	must(t, err)
	served := h.server.Config.Handler
	h.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/rentals/"+podRental+"/prepare-facts" {
			served.ServeHTTP(w, r)
			return
		}
		_ = json.NewEncoder(w).Encode(hub.PrepareFactsView{Application: "h3:app", ModelSlotPaths: []string{ladderSlot},
			ImageInventory: inventory, LockedRequirements: string(facts.LockedRequirements) + lockedExtra})
	})
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
	pod := &fakePod{controlKey: public, machine: machine(holder.ID), deviceCount: 4,
		preparedPlacement: func(download []byte, pkg, release string) *pb.Placement {
			placement := modelBearingPlacement(t)(download, pkg, release)
			placement.PackageInterface = detail.PackageInterface
			return placement
		}}
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

// Every rung of a captured default ladder is probed at the public origin, but a checkpoint
// is content-addressed: one capture asks the origin about each checkpoint once.
func TestRentedSubmissionProbesEachDefaultCheckpointOnce(t *testing.T) {
	h := newLadderHub(t)
	h.bind(hub.PackageBindingRow{Slot: ladderSlot, Model: ladderModel, Release: ladderRelease,
		Ladder: []hub.BindingRung{{GPU: "H100", Lane: "bf16-full"}, {GPU: "B200", Lane: "bf16-full"}, {GPU: "*", Lane: "bf16-full"}}, Revision: 3})
	header := "sha256:" + strings.Repeat("e", 64)
	var resolves, reads atomic.Int32
	origin := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodGet && r.URL.Path == "/v1/models/resolve" && r.URL.Query().Get("ref") == ladderModel+"@"+bf16Manifest:
			resolves.Add(1)
			_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: ladderModel, ManifestID: bf16Manifest, ManifestLength: 161, HeaderID: header})
		case r.Method == http.MethodPost && r.URL.Path == "/v1/models/"+ladderModel+"/checkpoints/"+bf16Manifest+"/reads" && r.Header.Get("Authorization") == "":
			reads.Add(1)
			_ = json.NewEncoder(w).Encode(map[string]any{"reads": []hub.Read{{ObjectID: header, Length: 64, URL: "https://" + r.Host + "/objects/header"}}})
		default:
			http.NotFound(w, r)
		}
	}))
	defer origin.Close()
	trust := filepath.Join(t.TempDir(), "public-origin.pem")
	must(t, os.WriteFile(trust, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: origin.Certificate().Raw}), 0o600))
	machine := &runtimeMachine{}
	root := startRentedFixture(t, h, func(blocker string) machineExecutionPeer { machine.blocker = blocker; return machine },
		"--extra-index-url "+origin.URL+"/v1/index/proof/simple/\n", "SSL_CERT_FILE="+trust)

	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	var capture pb.MachineExecutionCapture
	must(t, canonical.Unmarshal(machine.submitted().CaptureCanonicalBytes, &capture))
	var row *pb.MachineModelDefault
	for _, candidate := range capture.ModelDefaults {
		if candidate.Entrypoint == "generate" {
			row = candidate
		}
	}
	if row == nil || len(row.Rungs) != 3 || row.PublicOrigin != origin.URL || row.UnavailableCode != "" {
		t.Fatalf("the capture does not carry the three probed rungs: %+v", capture.ModelDefaults)
	}
	if resolves.Load() != 1 || reads.Load() != 1 {
		t.Fatalf("three rungs of one checkpoint cost %d resolves and %d header reads; want one each", resolves.Load(), reads.Load())
	}
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
		m.record("retention_released", []byte(`{"collected":false}`))
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
	machine := &heldMachine{runtimeMachine: &runtimeMachine{}, commands: map[string]bool{}, canceled: make(chan struct{})}
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
