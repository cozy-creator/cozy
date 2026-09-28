package producttest

import (
	"context"
	"database/sql"
	"encoding/base64"
	"os"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

type machineSubmitPeer interface {
	GetMachineExecutionWorkspace(context.Context, *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error)
	SubmitMachineExecution(context.Context, *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error)
}

// acceptingMachine accepts one submission into its current workspace.
type acceptingMachine struct {
	finishedMachine
	submitted *pb.MachineExecutionSubmit
}

func (m *acceptingMachine) GetMachineExecutionWorkspace(context.Context, *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "current-workspace"}, nil
}

func (m *acceptingMachine) SubmitMachineExecution(_ context.Context, submission *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.submitted = proto.Clone(submission).(*pb.MachineExecutionSubmit)
	return &pb.MachineExecutionReceipt{RequestId: submission.Offer.RequestId, SubmissionId: submission.SubmissionId,
		CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
		AcceptedAtMs: uint64(time.Now().UnixMilli()), WorkerId: podWorkerID, WorkerBootId: podBootID,
		ExecutionWorkspaceId: submission.ExpectedExecutionWorkspaceId}, nil
}

// A submission recorded before workspace fencing carries no workspace identity. It is
// submitted into the machine's current workspace instead of being refused forever.
func TestUnfencedRecordedSubmissionUsesTheCurrentWorkspace(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "unfenced-submission")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	raw := []byte(`{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[]}}]}`)
	installed := &pb.InstalledPackage{InstallationId: "unfenced-install", Package: "local/example", Release: "1.0.0", PackageInterface: raw}
	capture, captureDigest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	request, _, problem := store.Submit(records.Request{ID: "job-unfenced", IdemKey: "unfenced", Package: installed.Package,
		Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
		MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	spec := []byte(`{"invocation":"immutable"}`)
	old := &pb.MachineExecutionSubmit{SubmissionId: request.IdemKey, CaptureCanonicalBytes: capture, CaptureDigest: captureDigest,
		Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	recorded, err := proto.MarshalOptions{Deterministic: true}.Marshal(old)
	must(t, err)
	db, err := sql.Open("sqlite", layout.DB)
	must(t, err)
	_, err = db.Exec(`UPDATE machine_executions SET submission=? WHERE request_id=?`, recorded, request.ID)
	must(t, err)
	must(t, db.Close())

	specDigest, err := canonical.Spell(canonical.Digest(spec))
	must(t, err)
	body, digest, err := canonical.Identity(&pb.AttemptOutcomeBody{RequestId: request.ID, AttemptOrdinal: 1,
		InvocationSpecDigest: specDigest, Status: pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED})
	must(t, err)
	machine := &acceptingMachine{finishedMachine: finishedMachine{
		state: &pb.MachineExecutionState{RequestId: request.ID, WorkerId: podWorkerID, WorkerBootId: podBootID,
			ExecutionWorkspaceId: "current-workspace", Generation: 1, AttemptOrdinal: 1, State: "succeeded"},
		outcome: &pb.AttemptOutcome{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecDigest: canonical.Digest(spec),
			OutcomeId: "unfenced-outcome", OutcomeDigest: digest, OutcomeCanonicalBytes: body},
	}}
	pod := &fakePod{controlKey: public, machine: machine}
	connection, certPath := startFakePod(t, root, pod)
	cert, err := os.ReadFile(certPath)
	must(t, err)
	hub := newFakeRentalHub(t, 0)
	hub.publishListing()
	hub.add(podRental, "unfenced")
	hub.set(podRental, "requested_accelerator_model", "fake-4090")
	row := records.Rental{ID: podRental, MachineName: "unfenced", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hub.server.URL, Address: connection.Addr,
		MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), connection.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hub.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)

	waitFor(t, root, "the unfenced submission's acceptance", func() bool {
		link, problem := store.MachineExecution(request.ID)
		return problem == nil && link != nil && len(link.Receipt) > 0
	})
	machine.mu.Lock()
	defer machine.mu.Unlock()
	if machine.submitted == nil || machine.submitted.ExpectedExecutionWorkspaceId != "current-workspace" {
		t.Fatalf("the submission did not default to the current workspace: %+v", machine.submitted)
	}
}
