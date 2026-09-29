package producttest

import (
	"bytes"
	"context"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type closureAdmissionMachine struct {
	*terminalMachines
	mode                     string
	probes, sent, reconciled atomic.Int32
}

func (m *closureAdmissionMachine) CloseMachineSubmission(ctx context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if strings.HasPrefix(q.RequestId, "closure-probe-") {
		m.probes.Add(1)
	} else {
		m.reconciled.Add(1)
	}
	if m.mode == "missing route" || m.mode == "frozen refusal" {
		return nil, status.Error(codes.Unimplemented, "this machine does not serve CloseMachineSubmission")
	}
	answer, err := m.terminalMachines.CloseMachineSubmission(ctx, q)
	if err == nil && m.mode == "wrong identity" {
		answer.RequestId = "another-request"
	}
	return answer, err
}
func (m *closureAdmissionMachine) SubmitMachineExecution(ctx context.Context, q *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.sent.Add(1)
	if m.mode == "frozen refusal" {
		return nil, status.Error(codes.InvalidArgument, "reference model is missing its required image")
	}
	return m.terminalMachines.SubmitMachineExecution(ctx, q)
}
func newClosureAdmissionMachine(mode string) *closureAdmissionMachine {
	return &closureAdmissionMachine{mode: mode, terminalMachines: newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})}
}

// Ordinary CLI admission traverses the authenticated machine transport. Unsupported
// capabilities and invalid closure proofs fail before any real offer is frozen/sent.
func TestNewRunRequiresClosureBeforeTransmission(t *testing.T) {
	for _, mode := range []string{"missing capability", "missing route", "wrong identity", "supported"} {
		t.Run(mode, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := newClosureAdmissionMachine(mode)
			advertised := mode != "missing capability"
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4, submissionClose: &advertised}, nil)
			code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "closure-admission")
			if code != 0 && !strings.Contains(out, "no submission was sent") {
				t.Fatalf("submit local request: %d %s", code, out)
			}
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			request, problem := store.RequestByIdempotencyKey("closure-admission")
			fatal(t, problem)
			want := "failed"
			if mode == "supported" {
				want = "succeeded"
			}
			waitFor(t, root, "closure admission result", func() bool { row, _ := store.RequestRow(request.ID); return row != nil && row.State == want })
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if mode == "supported" {
				if len(link.Receipt) == 0 || machine.sent.Load() == 0 || machine.probes.Load() != 1 {
					t.Fatalf("supported machine did not accept exactly one cached probe: sent=%d probes=%d", machine.sent.Load(), machine.probes.Load())
				}
			} else {
				if len(link.Submission) != 0 || len(link.Receipt) != 0 || machine.sent.Load() != 0 {
					t.Fatal("unsupported machine received or froze real work")
				}
				_, shown := runCozy(t, root, "run", "show", request.ID, "--json")
				if !strings.Contains(shown, "no submission was sent") {
					t.Fatalf("missing actionable refusal: %s", shown)
				}
				if mode == "missing capability" && machine.probes.Load() != 0 {
					t.Fatal("advertised absence still attempted closure")
				}
			}
		})
	}
}

// A frozen offer on a retired peer keeps its identity and unresolved intent. No
// work RPC is retried until the operator upgrades and reconnects the machine.
func TestFrozenSubmissionOnRetiredPeerPreservesUncertainAcceptance(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newClosureAdmissionMachine("frozen refusal")
	advertised := false
	var request records.Request
	var frozenBytes []byte
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, submissionClose: &advertised}, func(_ home.Layout, store *records.Store) {
		raw := []byte(`{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[]}}]}`)
		installed := &pb.InstalledPackage{InstallationId: "frozen-install", Package: "local/example", Release: "1.0.0", PackageInterface: raw}
		capture, digest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
		must(t, err)
		req, _, p := store.Submit(records.Request{ID: "frozen-request", IdemKey: "frozen-refusal", Package: installed.Package, Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
		fatal(t, p)
		request = req
		fatal(t, store.LinkMachineExecution(request.ID, podRental))
		invocation := []byte(`{"invocation":"immutable"}`)
		frozen := &pb.MachineExecutionSubmit{SubmissionId: request.IdemKey, ExpectedExecutionWorkspaceId: "rented-workspace", CaptureCanonicalBytes: capture, CaptureDigest: digest, Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: invocation, InvocationSpecDigest: canonical.Digest(invocation)}}
		fatal(t, store.RecordMachineSubmission(request.ID, frozen))
		link, p := store.MachineExecution(request.ID)
		fatal(t, p)
		frozenBytes = bytes.Clone(link.Submission)
	})
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "refusal and unavailable reconciliation visible", func() bool {
		_, shown := runCozy(t, root, "run", "show", request.ID, "--json")
		return strings.Contains(shown, "acceptance remains unresolved") && strings.Contains(shown, "durable submission closure")
	})
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	owed, problem := store.MachineExecutionOwesWork(request.ID)
	fatal(t, problem)
	if !bytes.Equal(link.Submission, frozenBytes) || len(link.Receipt) != 0 || link.SubmissionClosed || !owed || records.Settled(row.State) {
		t.Fatal("missing closure settled or changed unknown acceptance")
	}
	if machine.probes.Load() != 0 || machine.sent.Load() != 0 || machine.reconciled.Load() != 0 {
		t.Fatal("retired peer received real work or unsupported closure calls")
	}
	var retained pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(link.Submission, &retained))
	if retained.ExpectedExecutionWorkspaceId != "rented-workspace" {
		t.Fatal("reconciliation rebound frozen workspace")
	}
}
