package producttest

import (
	"bytes"
	"context"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/proto"
)

type closureAdmissionMachine struct {
	*terminalMachines
	mode                     atomic.Value
	probes, sent, reconciled atomic.Int32
}

func (m *closureAdmissionMachine) CloseMachineSubmission(ctx context.Context, q *pb.MachineSubmissionClose) (*pb.MachineSubmissionClosure, error) {
	if strings.HasPrefix(q.RequestId, "closure-probe-") {
		m.probes.Add(1)
	} else {
		m.reconciled.Add(1)
	}
	switch m.mode.Load().(string) {
	case "missing route", "frozen refusal":
		return nil, status.Error(codes.Unimplemented, "this machine does not serve CloseMachineSubmission")
	case "runtime missing":
		return nil, status.Error(codes.FailedPrecondition, pb.CapabilityUnavailableCode+": Runtime cannot close submissions")
	case "journal error":
		return nil, status.Error(codes.FailedPrecondition, "journal write failed: disk full")
	case "workspace changes during probe":
		if m.probes.Load() == 1 {
			_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_workspace_changed"))
			return nil, status.Error(codes.FailedPrecondition, "workspace replaced during probe")
		}
	case "close replaced", "refusal then close replaced":
		if !strings.HasPrefix(q.RequestId, "closure-probe-") {
			_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", "execution_workspace_changed"))
			return nil, status.Error(codes.FailedPrecondition, "workspace replaced")
		}
	}
	answer, err := m.terminalMachines.CloseMachineSubmission(ctx, q)
	if err == nil && m.mode.Load().(string) == "wrong identity" {
		answer.RequestId = "another-request"
	}
	return answer, err
}
func (m *closureAdmissionMachine) SubmitMachineExecution(ctx context.Context, q *pb.MachineExecutionSubmit) (*pb.MachineExecutionReceipt, error) {
	m.sent.Add(1)
	if m.mode.Load().(string) == "frozen refusal" || m.mode.Load().(string) == "refusal then close replaced" {
		return nil, status.Error(codes.InvalidArgument, "reference model is missing its required image")
	}
	if m.mode.Load().(string) == "submit replaced" || m.mode.Load().(string) == "workspace required" {
		code := "execution_workspace_changed"
		if m.mode.Load().(string) == "workspace required" {
			code = "execution_workspace_required"
		}
		_ = grpc.SetTrailer(ctx, metadata.Pairs("cozy-error-code", code))
		return nil, status.Error(codes.FailedPrecondition, code)
	}
	return m.terminalMachines.SubmitMachineExecution(ctx, q)
}
func newClosureAdmissionMachine(mode string) *closureAdmissionMachine {
	machine := &closureAdmissionMachine{terminalMachines: newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})}
	machine.mode.Store(mode)
	return machine
}

// Ordinary CLI admission traverses the authenticated machine transport. Unsupported
// capabilities and invalid closure proofs fail before any real offer is frozen/sent.
func TestNewRunRequiresClosureBeforeTransmission(t *testing.T) {
	for _, mode := range []string{"missing capability", "missing route", "runtime missing", "journal error", "workspace changes during probe", "wrong identity", "supported"} {
		t.Run(mode, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := newClosureAdmissionMachine(mode)
			advertised := mode != "missing capability"
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4, submissionClose: &advertised}, nil)
			code, out := runCozy(t, root, "run", ladderPackage+"/generate", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "closure-admission")
			if code != 0 && !strings.Contains(out, "no submission was sent") && !strings.Contains(out, "disk full") {
				t.Fatalf("submit local request: %d %s", code, out)
			}
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			request, problem := store.RequestByIdempotencyKey("closure-admission")
			fatal(t, problem)
			want := "failed"
			if mode == "supported" || mode == "workspace changes during probe" {
				want = "succeeded"
			}
			waitFor(t, root, "closure admission result", func() bool { row, _ := store.RequestRow(request.ID); return row != nil && row.State == want })
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			if mode == "supported" || mode == "workspace changes during probe" {
				if len(link.Receipt) == 0 || machine.sent.Load() == 0 || machine.probes.Load() < 1 {
					t.Fatalf("supported machine did not accept exactly one cached probe: sent=%d probes=%d", machine.sent.Load(), machine.probes.Load())
				}
			} else {
				if len(link.Submission) != 0 || len(link.Receipt) != 0 || machine.sent.Load() != 0 {
					t.Fatal("unsupported machine received or froze real work")
				}
				_, shown := runCozy(t, root, "run", "show", request.ID, "--json")
				if mode == "journal error" {
					if !strings.Contains(shown, "disk full") || strings.Contains(shown, "machine.submission_closure_required") {
						t.Fatalf("journal error was relabeled as missing capability: %s", shown)
					}
				} else if !strings.Contains(shown, "no submission was sent") {
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
		request, frozenBytes = frozenSubmissionRecord(t, store, false)
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

func frozenSubmissionRecord(t *testing.T, store *records.Store, cancel bool, configure ...func(*pb.MachineExecutionSubmit)) (records.Request, []byte) {
	t.Helper()
	raw := []byte(`{"application":"proof:app","format":"cozy.package.interface/1","entrypoints":[],"jobs":[{"name":"main","models":[],"publishes":false,"weights_outputs":[],"request":{"fields":[]},"result":{"fields":[]}}]}`)
	installed := &pb.InstalledPackage{InstallationId: "frozen-install", Package: "local/example", Release: "1.0.0", PackageInterface: raw}
	capture, digest, err := canonical.Identity(&pb.MachineExecutionCapture{RootInstallationId: installed.InstallationId, InstalledPackages: []*pb.InstalledPackage{installed}})
	must(t, err)
	req, _, p := store.Submit(records.Request{ID: "frozen-request", IdemKey: "frozen-refusal", Package: installed.Package, Release: installed.Release, Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"), MachineExecutionObserver: true, LocalInstallationID: installed.InstallationId})
	fatal(t, p)
	request := req
	fatal(t, store.LinkMachineExecution(request.ID, podRental))
	invocation := []byte(`{"invocation":"immutable"}`)
	frozen := &pb.MachineExecutionSubmit{SubmissionId: request.IdemKey, ExpectedExecutionWorkspaceId: "rented-workspace", CaptureCanonicalBytes: capture, CaptureDigest: digest, PayloadCanonicalBytes: []byte(`{}`), Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: invocation, InvocationSpecDigest: canonical.Digest(invocation)}}
	for _, apply := range configure {
		apply(frozen)
	}
	fatal(t, store.RecordMachineSubmission(request.ID, frozen))
	link, p := store.MachineExecution(request.ID)
	fatal(t, p)
	if cancel {
		_, p = store.RequestMachineCancellation(request.ID, "")
		fatal(t, p)
	}
	return request, bytes.Clone(link.Submission)
}

func TestFrozenSubmissionWorkspaceLossNeedsAuthoritativeEvidence(t *testing.T) {
	for _, mode := range []string{"submit replaced", "close replaced", "refusal then close replaced", "workspace required"} {
		t.Run(mode, func(t *testing.T) {
			h := newLadderHub(t)
			h.bind(goodLadder())
			machine := newClosureAdmissionMachine(mode)
			var request records.Request
			var frozen []byte
			cancel := mode == "close replaced"
			root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, func(_ home.Layout, store *records.Store) {
				request, frozen = frozenSubmissionRecord(t, store, cancel)
			})
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			if mode == "workspace required" {
				waitFor(t, root, "required identity remains unresolved", func() bool {
					_, shown := runCozy(t, root, "run", "show", request.ID, "--json")
					return strings.Contains(shown, "prior acceptance remains unresolved")
				})
			} else {
				waitFor(t, root, "replaced execution workspace settles", func() bool {
					lost, _ := store.MachineExecutionLost(request.ID)
					return lost
				})
			}
			link, problem := store.MachineExecution(request.ID)
			fatal(t, problem)
			row, problem := store.RequestRow(request.ID)
			fatal(t, problem)
			owed, problem := store.MachineExecutionOwesWork(request.ID)
			fatal(t, problem)
			if !bytes.Equal(frozen, link.Submission) || len(link.Receipt) != 0 || link.SubmissionClosed {
				t.Fatal("workspace evidence rewrote frozen identity or fabricated closure")
			}
			if mode == "workspace required" {
				if !owed || records.Settled(row.State) {
					t.Fatal("required identity was treated as loss")
				}
			} else {
				want := "failed"
				if cancel {
					want = "canceled"
				}
				if owed || row.State != want {
					t.Fatalf("lost workspace is %s, owed=%v", row.State, owed)
				}
			}
			if cancel && machine.sent.Load() != 0 {
				t.Fatal("pending cancel resubmitted work")
			}
		})
	}
}

func TestUnsupportedClosurePreservesCancelUntilExplicitReconnect(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	machine := newClosureAdmissionMachine("runtime missing")
	var request records.Request
	var frozen []byte
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine}, func(_ home.Layout, store *records.Store) {
		request, frozen = frozenSubmissionRecord(t, store, true)
	})
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	defer store.Close()
	waitFor(t, root, "unsupported cancellation is visible", func() bool {
		_, shown := runCozy(t, root, "run", "show", request.ID, "--json")
		return strings.Contains(shown, "acceptance remains unresolved") && strings.Contains(shown, "cozy down")
	})
	link, problem := store.MachineExecution(request.ID)
	fatal(t, problem)
	row, problem := store.RequestRow(request.ID)
	fatal(t, problem)
	if row.State != "canceling" || !link.CancelRequested || link.SubmissionClosed || !bytes.Equal(frozen, link.Submission) || machine.sent.Load() != 0 {
		t.Fatal("unsupported closure altered unknown acceptance or cancel intent")
	}
	// Static capability failure has no work/control retry loop while this daemon
	// remains up. Observe longer than its ordinary first retry interval.
	time.Sleep(1500 * time.Millisecond)
	if machine.reconciled.Load() != 1 {
		t.Fatalf("unsupported closure retried without a peer change: %d", machine.reconciled.Load())
	}
	if code, out := runCozy(t, root, "down"); code != 0 {
		t.Fatalf("detach: %d %s", code, out)
	}
	if machine.reconciled.Load() != 1 {
		t.Fatalf("unsupported closure retried %d times", machine.reconciled.Load())
	}
	// The old in-flight RPC commits while the client is absent. Reconnecting after
	// upgrading must recover its receipt, never claim that the key was unaccepted.
	var late pb.MachineExecutionSubmit
	must(t, proto.Unmarshal(frozen, &late))
	late.Claim = &pb.Claim{WorkerId: podWorkerID, WorkerBootId: podBootID}
	_, err := machine.terminalMachines.SubmitMachineExecution(context.Background(), &late)
	must(t, err)
	machine.mode.Store("supported")
	if code, out := runCozy(t, root, "up"); code != 0 {
		t.Fatalf("reconnect: %d %s", code, out)
	}
	waitFor(t, root, "late acceptance receipt is retained", func() bool {
		link, _ := store.MachineExecution(request.ID)
		return link != nil && len(link.Receipt) > 0
	})
	link, problem = store.MachineExecution(request.ID)
	fatal(t, problem)
	if link.SubmissionClosed || !link.CancelRequested || machine.sent.Load() != 0 || len(machine.submitted()) != 1 {
		t.Fatal("reconnect duplicated accepted work or fabricated cancellation proof")
	}
}
