package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"encoding/pem"
	"math/big"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/exit"
	"github.com/cozy-creator/cozy/internal/machineendpoint"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// endpointRunOnRental records a run a daemon older than cozy.machine.v1 sent to a rental as an
// explicit endpoint (a foreground --rental run): accepted there, running, still owed, with an
// output export waiting on its result.
func endpointRunOnRental(t *testing.T, store *records.Store, root, label string, ep *machineendpoint.Endpoint, cancel bool) records.Request {
	t.Helper()
	request, _, problem := store.SubmitWithEvent(records.Request{ID: "rented-" + label, IdemKey: "rented-" + label, Package: "local/rented",
		Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true,
		OutputExport: &records.OutputExportIntent{Directory: filepath.Join(root, "out", label),
			Outputs: []records.OutputExportEntry{{OutputID: "image", MediaType: "image/webp"}}}},
		map[string]any{"machine_endpoint": ep})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, ep.Name()))
	fatal(t, store.AppendEvent(request.ID, records.RunV1Sent, 0, map[string]any{"machine": ep.Name()}))
	fatal(t, store.AcceptRunV1(request.ID, ep.Name(), &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
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
// rentals that ended (reached as explicit endpoints) settle when they are read, and `cozy run
// show` answers from the records. Their output exports, and that of a run abandoned with its
// export pending, settle too, so nothing of theirs stays in flight.
func TestARunWhoseRentalEndedSettlesWithoutReachingItsMachine(t *testing.T) {
	root := t.TempDir()
	t.Cleanup(func() { _, _ = runCozy(t, root, "down") })
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	der, err := x509.CreateCertificate(rand.Reader, &x509.Certificate{SerialNumber: big.NewInt(1), NotAfter: time.Now().Add(time.Hour)},
		&x509.Certificate{SerialNumber: big.NewInt(1)}, public, private)
	must(t, err)
	leaf := string(pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}))
	endpoint := func(rental string) *machineendpoint.Endpoint {
		return &machineendpoint.Endpoint{Format: machineendpoint.Format, Address: "127.0.0.1:1", WorkerID: "ra-" + rental,
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
	released := endpointRunOnRental(t, store, root, "released", endpoint("pr-released"), false)
	canceling := endpointRunOnRental(t, store, root, "canceling", endpoint("pr-failed"), true)
	// A run abandoned with its export pending: nothing will ever fill that export.
	fatal(t, store.RecordRental(records.Rental{ID: "pr-gone", MachineName: "shameimaru", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 1, State: "ready", Hub: "https://hub.example"}))
	abandoned := endpointRunOnRental(t, store, root, "abandoned", endpoint("pr-gone"), false)
	_, problem = store.AbandonMachineExecution(abandoned.ID, "cozy", "abandoned by its owner")
	fatal(t, problem)
	// A live rental: its run settles the moment the Hub reports the pod gone.
	fatal(t, store.RecordRental(records.Rental{ID: "pr-live", MachineName: "hinanawi", SKU: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1,
		HourlyRateUSDMicros: 1, State: "ready", Hub: "https://hub.example"}))
	live := endpointRunOnRental(t, store, root, "live", endpoint("pr-live"), false)
	store.Close()

	if code, out := runCozy(t, root, "up"); code != 0 {
		t.Fatalf("up [exit %d]\n%s", code, out)
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
	code, out := runCozy(t, root, "down", "--json")
	if code != 0 {
		t.Fatalf("down [exit %d]\n%s", code, out)
	}
	for _, run := range []records.Request{released, canceling, abandoned} {
		if strings.Contains(out, run.ID) {
			t.Fatalf("run %s of an ended rental is still in flight at `cozy down`:\n%s", run.ID, out)
		}
	}

	store, problem = records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
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
	for _, run := range []records.Request{released, canceling, abandoned, live} {
		if owed, problem := store.MachineExecutionOwesWork(run.ID); problem != nil || owed {
			t.Fatalf("run %s of an ended rental still owes its machine: %v %v", run.ID, owed, problem)
		}
		export, problem := store.OutputExportOf(run.ID)
		fatal(t, problem)
		if export == nil || export.State != "skipped" {
			t.Fatalf("run %s's output export is %+v, want skipped", run.ID, export)
		}
	}
	held, problem := store.Obligations()
	fatal(t, problem)
	for _, o := range held {
		if o.Kind == "output_export" {
			t.Fatalf("an output export of an ended run is still owed: %s", o)
		}
	}
}
