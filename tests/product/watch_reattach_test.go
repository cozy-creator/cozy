package producttest

import (
	"bytes"
	"encoding/base64"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// A daemon restart (a binary upgrade by any session) does not end a follower: `cozy run
// watch` waits for the daemon to answer again, re-attaches to the same run, says so once,
// and still receives the result.
func TestWatchReattachesAcrossADaemonRestart(t *testing.T) {
	root, err := os.MkdirTemp(scratchBase, "watch-reattach-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "watch-reattach")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	request, _, problem := store.Submit(records.Request{ID: "job-watch-reattach", IdemKey: "watch-reattach",
		Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(request.ID, submission))
	receipt := &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey, CaptureDigest: submission.CaptureDigest,
		InvocationSpecDigest: submission.Offer.InvocationSpecDigest, AcceptedAtMs: uint64(time.Now().UnixMilli()),
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}
	fatal(t, store.AcceptMachineExecution(request.ID, receipt))
	specDigest, err := canonical.Spell(receipt.InvocationSpecDigest)
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	machine := &finishedMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "running"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: receipt.InvocationSpecDigest,
			OutcomeId: "reattach-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}
	connection, certPath := startFakePod(t, root, &fakePod{controlKey: public, machine: machine})
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "watched")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "watched", State: "ready", SKU: "cpu",
		AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL,
		Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
		ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	first := startDaemonProcess(t, root)

	watcher := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run", "watch", request.ID, "--json")
	watcher.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	watcher.Stdout, watcher.Stderr = &stdout, &stderr
	must(t, watcher.Start())
	watched := make(chan error, 1)
	go func() { watched <- watcher.Wait() }()
	t.Cleanup(func() { _ = watcher.Process.Kill() })
	waitFor(t, root, "the daemon to observe the running execution", func() bool {
		link, problem := store.MachineExecution(request.ID)
		return problem == nil && link != nil && len(link.ObservedState) > 0
	})
	time.Sleep(time.Second) // the watcher opens its stream

	must(t, first.cmd.Process.Signal(syscall.SIGTERM))
	<-first.exited
	machine.mu.Lock()
	machine.state.State = "succeeded"
	machine.mu.Unlock()
	select {
	case err := <-watched:
		t.Fatalf("the watcher exited while the daemon was down: %v\n%s\n%s", err, stdout.String(), stderr.String())
	case <-time.After(2 * time.Second):
	}
	startDaemonProcess(t, root)

	select {
	case err := <-watched:
		if err != nil || !strings.Contains(stdout.String(), `"status":"completed"`) {
			t.Fatalf("the watcher did not receive the result [%v]:\n%s\n%s", err, stdout.String(), stderr.String())
		}
	case <-time.After(2 * time.Minute):
		t.Fatalf("the watcher never finished:\n%s\n%s\n%s", stdout.String(), stderr.String(), tail(filepath.Join(root, "daemon.log")))
	}
	if strings.Count(stderr.String(), "daemon restarted; reattached") != 1 {
		t.Fatalf("the re-attachment was not said exactly once:\n%s", stderr.String())
	}
}
