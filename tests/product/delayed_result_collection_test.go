package producttest

import (
	"context"
	"encoding/base64"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type machineExecutionPeer interface {
	GetMachineExecution(context.Context, *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error)
	ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error)
	CollectMachineExecution(context.Context, *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error)
	AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error)
}

// finishedMachine has already run the job; custody of its result is established only
// once `ready`, the way a large result keeps collection busy after the run completes.
type finishedMachine struct {
	mu      sync.Mutex
	state   *pb.MachineExecutionState
	outcome *pb.AttemptOutcome
	ready   time.Time
}

func (m *finishedMachine) GetMachineExecution(context.Context, *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *finishedMachine) ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return &pb.MachineExecutionEventPage{}, nil
}

func (m *finishedMachine) CollectMachineExecution(context.Context, *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ready.IsZero() {
		m.ready = time.Now().Add(3 * time.Second)
	}
	return proto.Clone(m.outcome).(*pb.AttemptOutcome), nil
}

func (m *finishedMachine) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.ready.IsZero() || time.Now().Before(m.ready) {
		return nil, status.Error(codes.Unavailable, "result custody is still being established")
	}
	m.state.Collected = true
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// A run whose terminal is observed before its result is collected is followed through
// collection instead of refusing with the daemon's in-flight observation error.
func TestWatchFollowsDelayedResultCollection(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "delayed-collection")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	request, _, problem := store.Submit(records.Request{ID: "job-delayed-collection", IdemKey: "delayed-collection",
		Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "persistent-workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "persistent-workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	machine := &finishedMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "persistent-workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "delayed-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}

	pod := &fakePod{controlKey: public, machine: machine}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "collector")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	row := records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)

	code, out := runCozy(t, root, "run", "watch", request.ID, "--json")
	if code != 0 || strings.Contains(out, "result_collection_pending") || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("watch refused an in-flight collection [exit %d]: %s\n%s", code, out, tail(filepath.Join(root, "daemon.log")))
	}
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	if link == nil || !link.Collected {
		t.Fatal("the result was not collected")
	}
}
