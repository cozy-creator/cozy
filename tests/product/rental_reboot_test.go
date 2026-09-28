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

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// rebootingMachine holds one running execution in its workspace until the pod restarts
// without it, as a Runtime whose journal did not survive the restart answers.
type rebootingMachine struct {
	mu        sync.Mutex
	workspace string
	state     *pb.MachineExecutionState
}

func (m *rebootingMachine) GetMachineExecution(ctx context.Context, query *pb.MachineExecutionQuery) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if query.ExpectedExecutionWorkspaceId != m.workspace {
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_workspace_changed"))
		return nil, status.Error(codes.FailedPrecondition, "execution workspace was replaced or does not match")
	}
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

func (m *rebootingMachine) ListMachineExecutionEvents(context.Context, *pb.MachineExecutionEventsQuery) (*pb.MachineExecutionEventPage, error) {
	return &pb.MachineExecutionEventPage{}, nil
}

func (m *rebootingMachine) CollectMachineExecution(context.Context, *pb.MachineExecutionCollect) (*pb.AttemptOutcome, error) {
	return nil, status.Error(codes.FailedPrecondition, "still running")
}

func (m *rebootingMachine) AcknowledgeMachineExecutionCollection(context.Context, *pb.MachineExecutionCollectionAck) (*pb.MachineExecutionState, error) {
	return nil, status.Error(codes.FailedPrecondition, "still running")
}

// A pod that restarts on a new boot, certificate and address stays this host's rental:
// the daemon re-attaches it from the Hub's re-attested readiness, the execution its old
// boot lost fails alone, and a new installation is served on the new boot.
func TestRebootedRentalIsReattachedAndServesNewWork(t *testing.T) {
	root := t.TempDir()
	layout, problem := home.Open(root)
	fatal(t, problem)
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	identity, problem := rental.PendingCreatorIdentity(layout, "reboot")
	fatal(t, problem)
	public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
	must(t, err)

	const old = "job-on-the-old-boot"
	request, _, problem := store.Submit(records.Request{ID: old, IdemKey: old, Package: "local/example", Entrypoint: "main",
		Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
	submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace-boot-1", SubmissionId: old,
		CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture),
		Offer: &pb.AttemptOffer{RequestId: old, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
	fatal(t, store.RecordMachineSubmission(old, submission))
	fatal(t, store.AcceptMachineExecution(old, &pb.MachineExecutionReceipt{RequestId: old, SubmissionId: old,
		CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
		AcceptedAtMs: uint64(time.Now().UnixMilli()), WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace-boot-1"}))
	machine := &rebootingMachine{workspace: "workspace-boot-1", state: &pb.MachineExecutionState{RequestId: old,
		WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace-boot-1", Generation: 1, AttemptOrdinal: 1, State: "running"}}
	pod := &fakePod{controlKey: public, machine: machine}
	first, firstCert := startFakePod(t, root, pod)
	cert, err := os.ReadFile(firstCert)
	must(t, err)

	peer := newFakeRentalHub(t, 0)
	peer.publishListing()
	peer.add(podRental, "rebooted")
	attest := func(address, mediaAddress, boot string, certPEM []byte) {
		for key, value := range map[string]any{"requested_accelerator_model": "fake-4090", "worker_address": address,
			"media_address": mediaAddress, "cert_pem": string(certPEM), "worker_id": podWorkerID, "worker_boot_id": boot,
			"creator_public_key": identity.PublicKey(), "media_token_sha256": []string{secret.HashHex(first.Media.Token)}} {
			peer.set(podRental, key, value)
		}
	}
	attest(first.Addr, first.Media.Addr, podBootID, cert)
	release := rentalReleaseFacts()
	release.PackageInterface = []byte(`{"application":"proof:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[]}`)
	peer.packageReleases = map[string]any{"proof/restart@1": release}
	row := records.Rental{ID: podRental, MachineName: "rebooted", State: "ready", SKU: "cpu", AcceleratorModel: "fake-4090",
		AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: peer.server.URL, Address: first.Addr,
		MediaAddress: first.Media.Addr, ExpectedWorkerID: podWorkerID, ExpectedWorkerBootID: podBootID}
	fatal(t, rental.Attach(layout, store, row, string(cert), first.Media.Token, identity))
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+peer.server.URL+
		"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	startDaemonProcess(t, root)
	waitFor(t, root, "the old boot's execution is observed", func() bool {
		link, problem := store.MachineExecution(old)
		return problem == nil && link != nil && len(link.ObservedState) > 0
	})

	// The container restarts: a new boot, certificate and port mapping, the workspace gone.
	const boot = "boot-pod-2"
	pod.rebooted.Store(ptr(boot))
	machine.mu.Lock()
	machine.workspace = "workspace-boot-2"
	machine.mu.Unlock()
	second, secondCert := startFakePod(t, root, pod)
	cert, err = os.ReadFile(secondCert)
	must(t, err)
	attest(second.Addr, second.Media.Addr, boot, cert)

	deadline := time.Now().Add(90 * time.Second)
	for {
		current, problem := store.RentalRow(podRental)
		fatal(t, problem)
		lost, problem := store.RequestRow(old)
		fatal(t, problem)
		if current != nil && current.ExpectedWorkerBootID == boot && current.Address == second.Addr && lost.State == "failed" {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("the rebooted rental was not re-attached or its lost execution not settled: %+v %s\n%s",
				current, lost.State, tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(100 * time.Millisecond)
	}
	if code, out := runCozy(t, root, "run", "show", old, "--json"); !strings.Contains(out, "machine_execution.state_lost") || !strings.Contains(out, "restarted") {
		t.Fatalf("the lost execution does not say why [%d]: %s", code, out)
	}
	if code, out := runCozy(t, root, "package", "install", "proof/restart", "--rental=rebooted", "--version=1", "--json"); code != 0 {
		t.Fatalf("install on the rebooted rental: %d %s", code, out)
	}
	waitFor(t, root, "the installation is prepared on the new boot", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.prepares) == 1 && pod.prepares[0].Claim.GetWorkerBootId() == boot
	})
	if current, problem := store.RentalRow(podRental); problem != nil || current == nil || current.State != "ready" {
		t.Fatalf("the rebooted rental did not stay: %+v %v", current, problem)
	}
}

func ptr[T any](value T) *T { return &value }
