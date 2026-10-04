package producttest

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/installkey"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

const reconciledPublication = "call-5e0c5e0c"

// reconcilingMachine is Runtime with one sent checkpoint publication whose machine
// authorization expired: it reports the publication unresolved and settles it only from the
// owner's forwarded finalization read, reporting the settlement as Runtime does.
type reconcilingMachine struct {
	mu       sync.Mutex
	state    *pb.MachineExecutionState
	events   []*pb.MachineExecutionEvent
	controls []*pb.MachineExecutionControl
}

func (m *reconcilingMachine) record(kind string, body map[string]any) {
	raw, _ := json.Marshal(body)
	m.events = append(m.events, &pb.MachineExecutionEvent{Sequence: uint64(len(m.events) + 1),
		AtMs: uint64(time.Now().UnixMilli()), AttemptOrdinal: 1, Kind: kind, BodyCanonicalBytes: raw})
	m.state.Sequence = uint64(len(m.events))
}

func (m *reconcilingMachine) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId,
		ExecutionWorkspaceId: "workspace"}, nil
}

func (m *reconcilingMachine) SubmitMachineExecution(context.Context, *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	return nil, status.Error(codes.FailedPrecondition, "already accepted")
}

func (m *reconcilingMachine) GetMachineExecution(context.Context, *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *reconcilingMachine) ListMachineExecutionEvents(_ context.Context, query *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	page := &pb.MachineExecutionEventPage{NextAfter: uint64(len(m.events)), HeadSequence: uint64(len(m.events))}
	for _, event := range m.events {
		if event.Sequence > query.After {
			page.Events = append(page.Events, proto.Clone(event).(*pb.MachineExecutionEvent))
		}
	}
	return page, nil
}

func (m *reconcilingMachine) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return nil, status.Error(codes.FailedPrecondition, "still running")
}

func (m *reconcilingMachine) ControlMachineExecution(_ context.Context, control *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.controls = append(m.controls, proto.Clone(control).(*pb.MachineExecutionControl))
	if control.Action != pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_RECONCILE_PUBLICATION || control.Publication.GetCallIndex() != 3 {
		return nil, status.Error(codes.FailedPrecondition, "not a publication reconciliation")
	}
	m.record("publication_settled", map[string]any{"call_index": 3, "publication": reconciledPublication,
		"destination": "alice/model", "committed": control.Publication.HttpStatus == http.StatusOK})
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *reconcilingMachine) reconciled() []*pb.MachineExecutionControl {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]*pb.MachineExecutionControl(nil), m.controls...)
}

// A rented conversion's checkpoint upload was sent, then the machine's publication
// authorization expired: Runtime keeps the obligation and cannot read whether it committed.
// The run shows it as awaiting its owner; the owner's credential on the run's Hub reads the
// finalization, and Runtime settles from exactly that answer. Driven through the real daemon
// and CLI with a stand-in pod and a stand-in Hub that refuses the machine's token.
func TestExpiredMachinePublicationIsSettledFromTheOwnersHubRead(t *testing.T) {
	committed := `{"operation":"` + reconciledPublication + `","result":{"checkpoint_id":"sha256:` + strings.Repeat("c", 64) +
		`"},"state":"completed","status_url":"/v1/models/alice/model/publications/` + reconciledPublication + `/finalization"}`
	for _, arm := range []struct {
		name   string
		status int
		body   string
	}{
		{"committed", http.StatusOK, committed},
		{"never finalized", http.StatusNotFound, `{"error":{"code":"publication.finalization_absent","message":"finalization has not been requested"}}`},
	} {
		t.Run(arm.name, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			identity, problem := installkey.Ensure(layout.Root)
			fatal(t, problem)
			public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
			must(t, err)
			request, _, problem := store.Submit(records.Request{ID: "job-owner-reconciliation", IdemKey: "owner-reconciliation",
				Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
				MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, podRental))
			capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
			submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
				CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
				Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
			fatal(t, store.RecordMachineSubmission(request.ID, submission))
			fatal(t, store.AcceptMachineExecution(request.ID, &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey,
				CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
				AcceptedAtMs: uint64(time.Now().UnixMilli()), WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}))
			machine := &reconcilingMachine{state: &pb.MachineExecutionState{RequestId: request.ID,
				WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "running"}}
			machine.record("running", map[string]any{})
			machine.record("publication_unresolved", map[string]any{"call_index": 3, "publication": reconciledPublication,
				"destination": "alice/model", "code": "publication.authority_refused"})

			// Hub answers the owner's own bearer and nothing else. Until `answer` flips it
			// is still finalizing out of reach (503), which settles nothing.
			var mu sync.Mutex
			var answered bool
			var reads []string
			hubStandIn := newFakeRentalHub(t, 0)
			hubStandIn.publishListing()
			hubStandIn.add(podRental, "collector")
			hubStandIn.set(podRental, "requested_accelerator_model", "fake-4090")
			fallback := hubStandIn.server.Config.Handler
			finalization := "/v1/models/alice/model/publications/" + reconciledPublication + "/finalization"
			hubStandIn.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != finalization {
					fallback.ServeHTTP(w, r)
					return
				}
				mu.Lock()
				defer mu.Unlock()
				reads = append(reads, r.Header.Get("Authorization"))
				w.Header().Set("Content-Type", "application/json")
				switch {
				case r.Header.Get("Authorization") != "Bearer rental-idle-test":
					w.WriteHeader(http.StatusUnauthorized)
					_, _ = w.Write([]byte(`{"error":{"code":"auth.machine_authorization_expired","message":"expired"}}`))
				case !answered:
					w.WriteHeader(http.StatusServiceUnavailable)
					_, _ = w.Write([]byte(`{"error":{"code":"hub.unavailable","message":"try again"}}`))
				default:
					w.WriteHeader(arm.status)
					_, _ = w.Write([]byte(arm.body))
				}
			})
			// The machine's own token is refused: only the owner can read this publication.
			machineRead, err := http.NewRequest(http.MethodGet, hubStandIn.server.URL+finalization, nil)
			must(t, err)
			machineRead.Header.Set("Authorization", "Bearer machine-token")
			response, err := http.DefaultClient.Do(machineRead)
			must(t, err)
			response.Body.Close()
			if response.StatusCode != http.StatusUnauthorized {
				t.Fatalf("the stand-in Hub answered the machine token: %d", response.StatusCode)
			}
			mu.Lock()
			reads = nil
			mu.Unlock()

			pod := &fakePod{controlKey: public, machine: machine}
			connection, certPath := startFakePod(t, root, pod)
			cert, err := os.ReadFile(certPath)
			must(t, err)
			fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu",
				AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hubStandIn.server.URL,
				Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
				ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token))
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hubStandIn.server.URL+
				"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
			startDaemonProcess(t, root)

			const surfaced = "awaiting owner reconciliation: machine authorization expired (publication " + reconciledPublication + ")"
			number := strconv.FormatInt(request.Number, 10)
			waitFor(t, root, "the run to show its awaiting publication", func() bool {
				_, list := runCozy(t, root, "run", "list")
				_, show := runCozy(t, root, "run", "show", number)
				return strings.Contains(list, surfaced) && strings.Contains(show, surfaced)
			})
			waitFor(t, root, "an owner read Hub could not answer yet", func() bool {
				mu.Lock()
				defer mu.Unlock()
				return len(reads) > 0
			})
			if controls := machine.reconciled(); len(controls) != 0 {
				t.Fatalf("a read that settled nothing was forwarded: %+v", controls)
			}
			mu.Lock()
			answered = true
			mu.Unlock()
			waitFor(t, root, "the forwarded owner read", func() bool { return len(machine.reconciled()) > 0 })
			control := machine.reconciled()[0]
			if control.Execution.GetRequestId() != request.ID || control.Publication.GetCallIndex() != 3 ||
				control.Publication.GetHttpStatus() != uint32(arm.status) || string(control.Publication.GetFinalization()) != arm.body {
				t.Fatalf("the machine was not handed Hub's exact answer: %+v", control)
			}
			mu.Lock()
			for _, bearer := range reads {
				if bearer != "Bearer rental-idle-test" {
					t.Errorf("the daemon read the finalization with %q, not the owner's credential", bearer)
				}
			}
			mu.Unlock()
			waitFor(t, root, "the settlement to clear the run", func() bool {
				_, list := runCozy(t, root, "run", "list")
				_, show := runCozy(t, root, "run", "show", number)
				return !strings.Contains(list, "awaiting owner reconciliation") && !strings.Contains(show, "awaiting owner reconciliation")
			})
		})
	}
}
