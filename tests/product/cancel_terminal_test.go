package producttest

import (
	"context"
	"database/sql"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

// releasingMachine is Runtime's cancel: unfinished work ends canceled, finished work stays
// as it ended, and either way what the execution retains is released.
type releasingMachine struct {
	*runtimeMachine
	cancels atomic.Int32
}

func (m *releasingMachine) ControlMachineExecution(_ context.Context, command *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if command.Action != pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_CANCEL {
		return nil, status.Error(codes.FailedPrecondition, "unexpected control")
	}
	if command.ExpectedGeneration == m.state.Generation {
		m.cancels.Add(1)
		m.state.Generation++
		m.record("control", []byte(`{"action":"cancel"}`))
		m.record("retention_released", []byte(`{"collected":false}`))
	}
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// Cozy before #931 announced a cancel of a sent submission canceled at once. Its closure
// resolves later in the background: whether the machine never took the run, or took it and
// finished it, the run stays canceled and nothing after its terminal reopens it.
func TestCanceledRunStaysCanceledWhileItsClosureResolves(t *testing.T) {
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	for _, accepted := range []bool{false, true} {
		t.Run(map[bool]string{false: "never accepted", true: "accepted and finished"}[accepted], func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := &releasingMachine{runtimeMachine: &runtimeMachine{triage: bundle}}
			root, layout := rentedLadderHome(t, h, &fakePod{machine: machine}, nil)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			request, _, problem := store.Submit(records.Request{ID: "req-canceled-early", IdemKey: "canceled-early", Package: ladderPackage,
				Entrypoint: "generate", Payload: []byte(`{}`), BodyDigest: childDigest("8"), MachineExecutionObserver: true,
				Rental: true, RequestedRental: podRental})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, podRental))
			submit := &pb.MachineExecutionSubmit{SubmissionId: request.IdemKey, ExpectedExecutionWorkspaceId: "rented-workspace",
				Offer: &pb.AttemptOffer{RequestId: request.ID}, PayloadCanonicalBytes: []byte(`{}`),
				ReleaseRoot: &pb.ReleaseRoot{Package: ladderPackage, Release: "1.0.0", Entrypoint: "generate"}}
			fatal(t, store.RecordMachineSubmission(request.ID, submit))
			store.Close()
			if accepted {
				// The machine accepted the submission and ran it to its end; the reply was lost.
				sent := proto.Clone(submit).(*pb.MachineExecutionSubmit)
				sent.Claim = &pb.Claim{WorkerId: podWorkerID, WorkerBootId: podBootID}
				_, err := machine.SubmitMachineExecution(context.Background(), sent)
				must(t, err)
				machine.finish()
			}
			db, err := sql.Open("sqlite", layout.DB)
			must(t, err)
			for _, statement := range []string{`UPDATE machine_executions SET cancel_requested=1 WHERE request_id=?`,
				`UPDATE requests SET state='canceled' WHERE id=?`,
				`INSERT INTO request_events(request_id,type,attempt,payload,at) VALUES(?,'run.canceled',0,'{"scope":"before_machine_acceptance"}',datetime())`} {
				_, err = db.Exec(statement, request.ID)
				must(t, err)
			}
			must(t, db.Close())

			startDaemonProcess(t, root)
			store, problem = records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			waitFor(t, root, "the closure to resolve", func() bool {
				link, _ := store.MachineExecution(request.ID)
				owed, problem := store.MachineExecutionOwesWork(request.ID)
				return link != nil && (link.SubmissionClosed || link.Collected) && problem == nil && !owed
			})
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			events, problem := store.EventsAfter(request.ID, 0, 1000)
			fatal(t, problem)
			var terminal bool
			var after, notes []string
			for _, event := range events {
				if terminal && (strings.HasPrefix(event.Type, "run.") || event.Type == "request.canceling") {
					after = append(after, event.Type)
				}
				if terminal && (event.Type == "machine.submission_closed" || event.Type == "machine.outcome") {
					notes = append(notes, event.Type)
				}
				terminal = terminal || event.Type == "run.canceled"
			}
			if row.State != "canceled" || len(after) > 0 || len(notes) == 0 {
				t.Fatalf("the canceled run is %s; after its terminal came %v (notes %v)", row.State, after, notes)
			}
			if accepted && machine.cancels.Load() != 1 {
				t.Fatalf("the accepted execution received %d cancels; want one", machine.cancels.Load())
			}
			if _, shown := runCozy(t, root, "run", "show", request.ID, "--json"); !strings.Contains(shown, `"status":"canceled"`) {
				t.Fatalf("run show no longer reports the cancel: %s", shown)
			}
		})
	}
}

// gatedAcceptance journals a submission as accepted and holds its reply until the client
// withdraws it, as a machine answering slowly does.
type gatedAcceptance struct {
	*heldMachine
	withdrawn atomic.Bool
}

func (m *gatedAcceptance) SubmitMachineExecution(ctx context.Context, submit *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	if _, err := m.heldMachine.SubmitMachineExecution(ctx, submit); err != nil {
		return nil, err
	}
	<-ctx.Done()
	m.withdrawn.Store(true)
	return nil, status.FromContextError(ctx.Err()).Err()
}

// The API reads a run as unaccepted, then acceptance commits before its cancel does. The
// cancel is recorded in the transaction that reads the acceptance, and reaches the machine.
func TestCancelRacingAcceptanceReachesTheMachine(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	machine := &gatedAcceptance{heldMachine: &heldMachine{runtimeMachine: &runtimeMachine{triage: bundle}, commands: map[string]bool{}, canceled: make(chan struct{})}}
	root := startRentedFixture(t, h, func(blocker string) machineExecutionPeer { machine.blocker = blocker; return machine }, "")
	if code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the machine to accept the submission", func() bool { return machine.submitted() != nil })
	id := machine.submitted().Offer.RequestId
	machine.mu.Lock()
	receipt, err := proto.MarshalOptions{Deterministic: true}.Marshal(machine.receipt)
	machine.mu.Unlock()
	must(t, err)

	// Acceptance is being recorded: the API's read of the run predates it, its cancel follows it.
	db, err := sql.Open("sqlite", filepath.Join(root, "creator.sqlite")+"?_pragma=busy_timeout(20000)")
	must(t, err)
	defer db.Close()
	recording, err := db.Begin()
	must(t, err)
	_, err = recording.Exec(`UPDATE machine_executions SET receipt=? WHERE request_id=?`, receipt, id)
	must(t, err)
	finished := make(chan string, 1)
	go func() {
		_, out := runCozy(t, root, "run", "cancel", id, "--json")
		finished <- out
	}()
	time.Sleep(3 * time.Second)
	must(t, recording.Commit())

	select {
	case <-machine.canceled:
	case <-time.After(20 * time.Second):
		t.Fatalf("the accepted run never received its cancel: %s", tail(filepath.Join(root, "daemon.log")))
	}
	if !machine.withdrawn.Load() {
		t.Fatal("the cancel read the run after its acceptance; the race was not reproduced")
	}
	select {
	case out := <-finished:
		if !strings.Contains(out, `"canceled"`) && !strings.Contains(out, `"canceling"`) {
			t.Fatalf("cozy run cancel did not acknowledge cancellation: %s", out)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("cozy run cancel did not return: %s", tail(filepath.Join(root, "daemon.log")))
	}
	waitFor(t, root, "authoritative cancellation after racing acceptance", func() bool {
		var state string
		return db.QueryRow(`SELECT state FROM requests WHERE id=?`, id).Scan(&state) == nil && state == "canceled"
	})
}

// The store half of that race: the intent survives for the observer to send.
func TestCancelAfterAcceptanceKeepsItsIntent(t *testing.T) {
	store, request, receipt := machineObserverFixture(t)
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	accepted, problem := store.RequestMachineCancellation(request.ID, "")
	fatal(t, problem)
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if !accepted || !link.CancelRequested || row.State != "canceling" {
		t.Fatalf("a cancel after acceptance was lost (accepted %v, intent kept %v, run %s)", accepted, link.CancelRequested, row.State)
	}
}
