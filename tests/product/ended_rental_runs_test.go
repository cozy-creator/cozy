package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// endpointRunOnRental records a run a daemon older than cozy.machine.v1 sent to a rental as an
// explicit endpoint (a foreground --rental run): accepted there, running, and still owed.
func endpointRunOnRental(t *testing.T, store *records.Store, label string, ep *machineendpoint.Endpoint, cancel bool) records.Request {
	t.Helper()
	request, _, problem := store.SubmitWithEvent(records.Request{ID: "rented-" + label, IdemKey: "rented-" + label, Package: "local/rented",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true},
		map[string]any{"machine_endpoint": ep})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, ep.Name()))
	capture, spec := []byte(`{"capture":"`+label+`"}`), []byte(`{"invocation":"`+label+`"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: ep.WorkspaceID, SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: 1000, WorkerId: ep.WorkerID, WorkerBootId: ep.WorkerBootID, ExecutionWorkspaceId: ep.WorkspaceID}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	state := &pb.MachineExecutionState{RequestId: request.ID, WorkerId: receipt.WorkerId, WorkerBootId: receipt.WorkerBootId,
		ExecutionWorkspaceId: receipt.ExecutionWorkspaceId, Generation: 1, AttemptOrdinal: 1, State: "running", Sequence: 1}
	fatal(t, store.ObserveMachineExecution(request.ID, state, &pb.MachineExecutionEventPage{NextAfter: 1, HeadSequence: 1,
		Events: []*pb.MachineExecutionEvent{{Sequence: 1, AttemptOrdinal: 1, AtMs: 1001, Kind: "running", BodyCanonicalBytes: []byte(`{}`)}}}))
	if cancel {
		_, problem := store.RequestMachineCancellation(request.ID, "")
		fatal(t, problem)
	}
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	return *row
}

// A run on a rental settles by itself once its rental is known to have ended: failed with a
// plain reason, or canceled when its owner had asked. Runs an older cozy left unsettled on
// rentals that ended (reached as explicit endpoints) settle when they are read, and
// `cozy run show` answers from the records without reaching the dead machine.
func TestARunWhoseRentalEndedSettlesWithoutReachingItsMachine(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	der, err := x509.CreateCertificate(rand.Reader, &x509.Certificate{SerialNumber: big.NewInt(1), NotAfter: time.Now().Add(time.Hour)},
		&x509.Certificate{SerialNumber: big.NewInt(1)}, public, private)
	must(t, err)
	leaf := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	// This computer's machine owner key, which signs a call to an endpoint that is no
	// recorded rental: without the fix, reading such a run dials its address.
	owner, err := x509.MarshalPKCS8PrivateKey(private)
	must(t, err)
	must(t, os.MkdirAll(filepath.Join(root, "machine"), 0o700))
	must(t, os.WriteFile(filepath.Join(root, "machine", "owner.pem"), pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: owner}), 0o600))
	// Each rental's address accepts and never answers, as a released pod's address can.
	dialed := map[string]*atomic.Int32{}
	endpoint := func(rental string) *machineendpoint.Endpoint {
		silent, err := net.Listen("tcp", "127.0.0.1:0")
		must(t, err)
		t.Cleanup(func() { silent.Close() })
		dialed[rental] = &atomic.Int32{}
		go func() {
			for {
				conn, err := silent.Accept()
				if err != nil {
					return
				}
				dialed[rental].Add(1)
				defer conn.Close()
			}
		}()
		return &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: silent.Addr().String(), WorkerID: "ra-" + rental,
			WorkerBootID: rental, CertificatePEM: leaf, WorkspaceID: rental}
	}

	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	// History an older cozy left: the Hub confirmed one rental's release and reported the
	// other failed, while their runs stayed unsettled.
	op, _, problem := store.BeginRentalOperation(records.RentalOperation{Key: "released-op", Hub: "https://hub.example", Reason: "rental", HourlyRateUSDMicros: 1},
		func(string) ([]byte, string, *exit.Error) { return []byte(`{}`), "proof", nil })
	fatal(t, problem)
	fatal(t, store.AdvanceRentalOperation(op.Key, "pr-released", "attached"))
	_, problem = store.ForgetRental("pr-released")
	fatal(t, problem)
	fatal(t, store.RecordRental(records.Rental{ID: "pr-failed", MachineName: "kirisame", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 1, State: "failed", Hub: "https://hub.example"}))
	released := endpointRunOnRental(t, store, "released", endpoint("pr-released"), false)
	canceling := endpointRunOnRental(t, store, "canceling", endpoint("pr-failed"), true)
	// A live rental: its run settles the moment the Hub reports the pod gone.
	fatal(t, store.RecordRental(records.Rental{ID: "pr-live", MachineName: "hinanawi", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 1, State: "ready", Hub: "https://hub.example"}))
	live := endpointRunOnRental(t, store, "live", endpoint("pr-live"), false)
	store.Close()
	for _, run := range []records.Request{released, canceling, live} {
		if run.State != "dispatching" && run.State != "canceling" {
			t.Fatalf("fixture %s is %s", run.ID, run.State)
		}
	}

	for _, want := range []struct {
		run    records.Request
		status string
	}{{released, "failed"}, {canceling, "canceled"}} {
		began := time.Now()
		code, out := runCozy(t, root, "run", "show", want.run.ID, "--json")
		took := time.Since(began)
		var report struct {
			Status string `json:"status"`
			Error  string `json:"error"`
			Events []struct {
				Type    string `json:"type"`
				Payload struct {
					Error string `json:"error"`
				} `json:"payload"`
			} `json:"events"`
		}
		if code != 0 || json.Unmarshal([]byte(lastJSONLine(out)), &report) != nil {
			t.Fatalf("run show %s [exit %d]\n%s", want.run.ID, code, out)
		}
		// A canceled run carries no error; its ending names the reason all the same.
		const reason = "its rental ended before it finished"
		ended := false
		for _, event := range report.Events {
			ended = ended || event.Type == "run."+want.status && event.Payload.Error == reason
		}
		if report.Status != want.status || !ended || want.status == "failed" && report.Error != reason {
			t.Fatalf("run %s on an ended rental shows %q, %q; want %q with the plain reason\n%s", want.run.ID, report.Status, report.Error, want.status, out)
		}
		if took > 5*time.Second {
			t.Fatalf("run show %s took %s", want.run.ID, took)
		}
	}
	// The rental still up is the daemon's to follow; the ended ones are never reached.
	for _, rental := range []string{"pr-released", "pr-failed"} {
		if n := dialed[rental].Load(); n != 0 {
			t.Fatalf("reading runs reached the machine of ended rental %s %d times", rental, n)
		}
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}

	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	fatal(t, store.ReconcileEndedMachineExecutions())
	if owed, problem := store.MachineExecutionOwesWork(live.ID); problem != nil || !owed {
		t.Fatalf("a run on a rental still up was settled: owed %v %v", owed, problem)
	}
	_, problem = store.ForgetRental("pr-live")
	fatal(t, problem)
	row, problem := store.RequestRow(live.ID)
	fatal(t, problem)
	lost, problem := store.MachineExecutionLost(live.ID)
	fatal(t, problem)
	if row.State != "failed" || !lost {
		t.Fatalf("a run whose rental just ended is %s, lost %v", row.State, lost)
	}
	for _, run := range []records.Request{released, canceling, live} {
		if owed, problem := store.MachineExecutionOwesWork(run.ID); problem != nil || owed {
			t.Fatalf("run %s of an ended rental still owes its machine: %v %v", run.ID, owed, problem)
		}
	}
}
