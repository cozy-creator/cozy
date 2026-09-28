package producttest

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
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
	"google.golang.org/protobuf/proto"
)

// memoMachine is Runtime asking its record owner, on a local miss, for a memoized
// operation's completed result (memo.lookup) and taking the owner's answer.
type memoMachine struct {
	reconcilingMachine
	memo bool
}

func (m *memoMachine) GetMachineExecutionWorkspace(_ context.Context, query *pb.MachineExecutionWorkspaceQuery) (*pb.MachineExecutionWorkspace, error) {
	return &pb.MachineExecutionWorkspace{WorkerId: query.Claim.WorkerId, WorkerBootId: query.Claim.WorkerBootId,
		ExecutionWorkspaceId: "workspace"}, nil
}

func (m *memoMachine) ControlMachineExecution(_ context.Context, control *pb.MachineExecutionControl) (*pb.MachineExecutionState, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.controls = append(m.controls, proto.Clone(control).(*pb.MachineExecutionControl))
	return proto.Clone(m.state).(*pb.MachineExecutionState), nil
}

// An upload another run completed is this owner's own memo: a machine that asks before
// computing the same upload is answered with that exact recorded result while the Hub
// still serves its checkpoint, and with a miss once it does not. The result stays on this
// machine; the Hub is only asked whether the checkpoint exists. A machine that does not
// report memo_lookup is never answered.
func TestAMachineMemoLookupIsAnsweredFromTheOwnersRecords(t *testing.T) {
	manifest := []byte(`{"format":"cozy.manifest/proof"}`)
	sum := sha256.Sum256(manifest)
	key := "sha256:" + strings.Repeat("a", 64)
	result := map[string]any{"destination": "alice/model", "checkpoint": "ck-earlier",
		"manifest":    map[string]any{"digest": "sha256:" + hex.EncodeToString(sum[:]), "length": len(manifest)},
		"publication": "call-earlier", "observation": "sha256:" + strings.Repeat("b", 64)}
	recorded, err := json.Marshal(result)
	must(t, err)
	for _, arm := range []struct {
		name     string
		served   bool
		answered []byte
	}{
		{"hit", true, recorded},
		{"checkpoint gone", false, nil},
	} {
		t.Run(arm.name, func(t *testing.T) {
			root := t.TempDir()
			layout, problem := home.Open(root)
			fatal(t, problem)
			store, problem := records.Open(layout.DB)
			fatal(t, problem)
			defer store.Close()
			identity, problem := rental.PendingCreatorIdentity(layout, "owner-memo")
			fatal(t, problem)
			public, err := base64.RawURLEncoding.DecodeString(identity.PublicKey())
			must(t, err)
			// An earlier run on another machine completed the upload and reported it.
			earlier, _, problem := store.Submit(records.Request{ID: "job-earlier-upload", IdemKey: "earlier-upload",
				Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("2")})
			fatal(t, problem)
			fatal(t, store.AppendEvent(earlier.ID, "machine.memo.record", 1, map[string]any{"call_index": 0,
				"operation": "upload_huggingface", "computation_digest": key, "result": result}))
			fatal(t, store.SettleRequest(earlier.ID, "succeeded"))

			request, _, problem := store.Submit(records.Request{ID: "job-owner-memo", IdemKey: "owner-memo",
				Package: "local/example", Entrypoint: "main", Kind: "job", Payload: []byte(`{}`), BodyDigest: childDigest("1"),
				MachineExecutionObserver: true})
			fatal(t, problem)
			fatal(t, store.LinkMachineExecution(request.ID, podRental))
			capture, spec := []byte(`{"capture":"immutable"}`), []byte(`{"invocation":"immutable"}`)
			submission := &pb.MachineExecutionSubmit{ExpectedExecutionWorkspaceId: "workspace", SubmissionId: request.IdemKey,
				CaptureCanonicalBytes: capture, CaptureDigest: canonical.Digest(capture), OwnerMemo: true,
				Offer: &pb.AttemptOffer{RequestId: request.ID, AttemptOrdinal: 1, InvocationSpecCanonicalBytes: spec, InvocationSpecDigest: canonical.Digest(spec)}}
			fatal(t, store.RecordMachineSubmission(request.ID, submission))
			fatal(t, store.AcceptMachineExecution(request.ID, &pb.MachineExecutionReceipt{RequestId: request.ID, SubmissionId: request.IdemKey,
				CaptureDigest: submission.CaptureDigest, InvocationSpecDigest: submission.Offer.InvocationSpecDigest,
				AcceptedAtMs: uint64(time.Now().UnixMilli()), WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace"}))
			machine := &memoMachine{memo: true, reconcilingMachine: reconcilingMachine{state: &pb.MachineExecutionState{RequestId: request.ID,
				WorkerId: podWorkerID, WorkerBootId: podBootID, ExecutionWorkspaceId: "workspace", Generation: 1, AttemptOrdinal: 1, State: "running"}}}
			machine.record("running", map[string]any{})
			machine.record("memo.lookup", map[string]any{"call_index": 0, "operation": "upload_huggingface", "computation_digest": key})

			var mu sync.Mutex
			var reads []string
			hubStandIn := newFakeRentalHub(t, 0)
			hubStandIn.publishListing()
			hubStandIn.add(podRental, "collector")
			hubStandIn.set(podRental, "requested_accelerator_model", "fake-4090")
			fallback := hubStandIn.server.Config.Handler
			hubStandIn.server.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.URL.Path != "/v1/models/alice/model/checkpoints/ck-earlier" {
					fallback.ServeHTTP(w, r)
					return
				}
				mu.Lock()
				reads = append(reads, r.Header.Get("Authorization"))
				mu.Unlock()
				if !arm.served {
					w.WriteHeader(http.StatusNotFound)
					_, _ = w.Write([]byte(`{"error":{"code":"checkpoint.not_found","message":"no such checkpoint"}}`))
					return
				}
				_, _ = w.Write(manifest)
			})

			pod := &fakePod{controlKey: public, machine: machine}
			connection, certPath := startFakePod(t, root, pod)
			cert, err := os.ReadFile(certPath)
			must(t, err)
			fatal(t, rental.Attach(layout, store, records.Rental{ID: podRental, MachineName: "collector", State: "ready", SKU: "cpu",
				AcceleratorModel: "fake-4090", AcceleratorCount: 1, HourlyRateUSDMicros: 100000, Hub: hubStandIn.server.URL,
				Address: connection.Addr, MediaAddress: connection.Media.Addr, ExpectedWorkerID: podWorkerID,
				ExpectedWorkerBootID: podBootID}, string(cert), connection.Media.Token, identity))
			must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+hubStandIn.server.URL+
				"\ntensorhub_token: rental-idle-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
			startDaemonProcess(t, root)

			waitFor(t, root, "the owner's answer", func() bool { return len(machine.reconciled()) > 0 })
			answer := machine.reconciled()[0]
			digest, _ := hex.DecodeString(strings.Repeat("a", 64))
			if answer.Action != pb.MachineExecutionAction_MACHINE_EXECUTION_ACTION_ANSWER_MEMO || answer.Execution.GetRequestId() != request.ID ||
				answer.Memo.GetLookupSequence() != 2 || string(answer.Memo.GetComputationDigest()) != string(digest) ||
				string(answer.Memo.GetResultCanonicalBytes()) != string(arm.answered) {
				t.Fatalf("the machine was not answered with the recorded result: %+v", answer)
			}
			mu.Lock()
			defer mu.Unlock()
			if len(reads) == 0 || reads[0] != "Bearer rental-idle-test" {
				t.Fatalf("the checkpoint was not confirmed with the owner's credential: %v", reads)
			}
		})
	}
}
