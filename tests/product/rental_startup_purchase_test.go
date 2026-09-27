package producttest

import (
	"database/sql"
	"encoding/base64"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A daemon restarted with queued --rental work serves before any purchase completes. Two
// requeued requests each buy a pod the Hub never readies: the API answers `cozy run list`
// and `cozy rental list`, each request says which rental it waits for, neither purchase
// waits on the other, and a run on an attached machine still completes. Before, startup
// resumed each queued request's placement inline and waited out its purchase.
func TestDaemonStartupNeverWaitsOnAQueuedPurchase(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()

	// The unrelated run: its attached machine has finished it and holds the result.
	identity, problem := rental.PendingCreatorIdentity(layout, "startup-purchase")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)
	done, _, problem := store.Submit(records.Request{ID: "job-attached", IdemKey: "job-attached",
		Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(done.ID, podRental))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: done.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: done.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(done.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: done.ID, SubmissionId: done.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(done.ID, receipt))
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: done.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	machine := &finishedMachine{
		state: &pb.MachineExecutionState{RequestId: done.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
		outcome: &pb.AttemptOutcome{RequestId: done.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "startup-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}
	connection, certPath := startFakePod(t, root, &fakePod{controlKey: public, machine: machine})
	cert, err := os.ReadFile(certPath)
	must(t, err)

	peer := newFakeRentalHub(t, 0)
	peer.publishListing()
	peer.add(podRental, "collector")
	peer.set(podRental, "requested_accelerator_model", "fake-4090")
	peer.packageReleases = map[string]any{"proof/source-producer@1": rentalReleaseFacts(), "proof/other-producer@1": rentalReleaseFacts()}
	peer.setSKUs(map[string]any{"name": "cpu", "accelerator_model": "CPU", "accelerator_count": 1,
		"price_usd_micros_per_hour": 100_000, "base_worker_profile": "python3.12-cpu-linux-x86"})
	var mu sync.Mutex
	var bought []string
	peer.rent = func(body map[string]any) map[string]any {
		mu.Lock()
		defer mu.Unlock()
		id := fmt.Sprintf("pr-queued-%d", len(bought)+1)
		bought = append(bought, id)
		// Accepted, and never readied: the Hub keeps saying `acquiring`.
		return map[string]any{"rental_id": id, "name": body["name"], "state": "acquiring",
			"requested_accelerator_model": "CPU", "accelerator_count": 1, "hourly_rate_usd_micros": 100_000}
	}
	purchases := func() []string {
		mu.Lock()
		defer mu.Unlock()
		return append([]string(nil), bought...)
	}
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu",
		AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100_000, Hub: peer.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
		ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))

	// Two jobs a prior daemon requeued, each insisting on a fresh machine of its own.
	queued := map[string]string{"job-queued-a": "proof/source-producer", "job-queued-b": "proof/other-producer"}
	for id, pkg := range queued {
		_, _, problem := store.Submit(records.Request{ID: id, IdemKey: id, BodyDigest: childDigest("7"),
			Package: pkg, Release: "1", Entrypoint: "convert", Kind: "job", Rental: true, RentNew: true,
			Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]"})
		fatal(t, problem)
	}
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	_, err = db.Exec(`UPDATE requests SET state='queued' WHERE id IN ('job-queued-a','job-queued-b')`)
	must(t, err)
	must(t, db.Close())
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+peer.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0o600))

	startDaemonProcess(t, root)
	waitFor(t, root, "both queued purchases to reach the Hub", func() bool { return len(purchases()) == 2 })
	for id := range queued {
		waitFor(t, root, id+" to say which rental it waits for", func() bool {
			return strings.HasPrefix(lastEventField(t, store, id, "request.parked", "reason"), "waiting for rental ")
		})
	}
	if code, out, errs := runCozyWithin(t, root, "run", "list", "--json"); code != 0 {
		t.Fatalf("run list failed beside queued purchases [exit %d]: %s%s", code, out, errs)
	}
	listed := listStuck(t, root)
	for _, id := range purchases() {
		if !listed.acquiring(id) {
			t.Fatalf("purchase %s is not listed as acquiring: %+v", id, listed.Rentals)
		}
	}
	code, out, errs := runCozyWithin(t, root, "run", "watch", done.ID, "--json")
	if code != 0 || !strings.Contains(out, `"status":"completed"`) {
		t.Fatalf("the run on the attached machine did not complete beside queued purchases [exit %d]: %s%s", code, out, errs)
	}
}
