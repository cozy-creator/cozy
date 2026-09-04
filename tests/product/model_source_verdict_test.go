package producttest

import (
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
)

// The run-205 selection, reduced to the two members that matter: one that verifies and one
// that fails permanently in the first seconds. The digest is the transfer's own selection.
const (
	verdictSelection = "sha256:" + "22222222222222222222222222222222" + "22222222222222222222222222222222"
	verdictGoodSHA   = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	verdictBadSHA    = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
	verdictGoodBytes = int64(110_331_470_098)
	verdictBadBytes  = int64(100_000_000_000)
	verdictCode      = "source_fetch_refused"
	verdictDetail    = "the origin answered 401 for b.safetensors and will not answer another way"
)

// TestAStaleRevisionStatusKeepsItsVerdict is run 205 in miniature, on the real wire.
//
// `job-4d4cfe11a6c0a5f0e5436a31` moved all 210.3 GB of its 44 shards and then sat at
// "44 of 48 verified" for 2h55m. The four outstanding members had failed permanently in
// the first seconds; every one of those FAILED frames answered a capability revision the
// row had already been carried past, and every one was discarded whole — no fail, no wake,
// no log. The row read state=accepted with an empty safe_code and safe_detail.
//
// The arm drives exactly that ordering: the row is carried to revision 4 by an echo, and
// then the verdict for revision 1 arrives.
func TestAStaleRevisionStatusKeepsItsVerdict(t *testing.T) {
	const requestID = "job-stale-revision-verdict"
	pod := &standInPod{}
	selection, err := canonical.Raw(verdictSelection)
	must(t, err)

	// The session is claimed by attachStandInRental below, BEFORE the transfer row exists.
	// A status frame for a transfer this owner has never heard of is correctly ignored, so
	// the pod holds its frames until the row is recorded — which is the real ordering too:
	// a pod answers a member it was asked for.
	ready := make(chan struct{})
	var once sync.Once
	pod.onSession = func(send func(*pb.WorkerFrame) error, ownerEpoch, controlEpoch uint64, bootID string) {
		select {
		case <-ready:
		case <-time.After(30 * time.Second):
			return
		}
		once.Do(func() {
			status := func(member, sha string, length int64, state pb.ModelSourceFileState,
				revision, transferred uint64, code, detail string,
			) {
				_ = send(&pb.WorkerFrame{Msg: &pb.WorkerFrame_ModelSourceFileStatus{
					ModelSourceFileStatus: &pb.ModelSourceFileStatus{
						RecordOwnerEpoch: ownerEpoch, ControlStreamEpoch: controlEpoch,
						WorkerBootId: bootID, OperationId: requestID,
						SourceSelectionDigest: selection, Member: member,
						ObjectId: "sha256:" + sha, Length: uint64(length),
						CapabilityRevision: revision, State: state,
						TransferredBytes: transferred, SafeCode: code, SafeDetail: detail,
					}}})
			}
			// 44 of 48: this one is done and stays done.
			status("a.safetensors", verdictGoodSHA, verdictGoodBytes,
				pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_VERIFIED, 1,
				uint64(verdictGoodBytes), "", "")
			// The remaining member is acknowledged, and then the row is carried past the
			// revision the pod is still answering — cl-134's restatement loop did this
			// ~3,000 times per member, but one bump is all the drop ever needed.
			status("b.safetensors", verdictBadSHA, verdictBadBytes,
				pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED, 1, 0, "", "")
			status("b.safetensors", verdictBadSHA, verdictBadBytes,
				pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_ACCEPTED, 4, 0, "", "")
			// THE FRAME THAT WAS THROWN AWAY. It answers revision 1 and it is the pod's
			// last word on these bytes.
			status("b.safetensors", verdictBadSHA, verdictBadBytes,
				pb.ModelSourceFileState_MODEL_SOURCE_FILE_STATE_FAILED, 1, 0,
				verdictCode, verdictDetail)
		})
	}

	o, instance := attachStandInRental(t, "stale-revision-verdict", pod)
	submitVerdictTransfer(t, o, requestID, "rental-stale-revision-verdict")
	close(ready)

	// The pod's session is already up; the frames ride it as soon as it is claimed. If the
	// transfer was submitted after the claim, the arm re-drives the lane on the next
	// stream, so wait on the durable answer rather than on the wire.
	deadline := time.Now().Add(30 * time.Second)
	for {
		transfer, problem := o.store.ModelTransferOf(requestID)
		fatal(t, problem)
		if transfer != nil && transfer.State == "failed" {
			if transfer.ErrorCode != verdictCode || transfer.SafeError != verdictDetail {
				t.Fatalf("the transfer failed with %q: %q, want the pod's own verdict %q: %q",
					transfer.ErrorCode, transfer.SafeError, verdictCode, verdictDetail)
			}
			break
		}
		if time.Now().After(deadline) {
			statuses, _ := o.store.ModelTransferSourceStatuses(requestID)
			state := "<absent>"
			if transfer != nil {
				state = transfer.State
			}
			t.Fatalf("the FAILED that answered a superseded capability revision was "+
				"discarded: transfer state %s, rows %+v\n%s", state, statuses, tail(o.root+"/orchestrator.log"))
		}
		time.Sleep(25 * time.Millisecond)
	}

	// The row is what every reader looks at, so the verdict has to be ON it — and only the
	// verdict. The counts stay what the row held: the stale frame decides nothing about
	// bytes or revisions.
	statuses, problem := o.store.ModelTransferSourceStatuses(requestID)
	fatal(t, problem)
	var failed *records.ModelTransferSourceStatus
	for i := range statuses {
		if statuses[i].Member == "b.safetensors" {
			failed = &statuses[i]
		}
	}
	if failed == nil {
		t.Fatalf("no row for the failed member: %+v", statuses)
	}
	if failed.State != "failed" {
		t.Fatalf("the failed member's row reads %q, want failed", failed.State)
	}
	if failed.SafeCode != verdictCode || failed.SafeDetail != verdictDetail {
		t.Fatalf("the row carries %q: %q, want the pod's %q: %q",
			failed.SafeCode, failed.SafeDetail, verdictCode, verdictDetail)
	}
	if failed.CapabilityRevision != 4 {
		t.Fatalf("the stale frame carried the row back to revision %d; a superseded status "+
			"is ignored for its counts", failed.CapabilityRevision)
	}
	if !strings.Contains(instance, "ins-") {
		t.Fatalf("the stand-in rental attached as %q", instance)
	}
}

func submitVerdictTransfer(t *testing.T, o *owner, requestID, rentalID string) {
	t.Helper()
	intent := &records.ModelTransferIntent{
		Kind: "model-upload", Destination: "paul/minimax-h3",
		Source:          "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40),
		SourceSelection: verdictSelection,
		SourceFiles: []records.ModelTransferSourceFile{
			{Member: "a.safetensors", SHA256: verdictGoodSHA, Length: verdictGoodBytes},
			{Member: "b.safetensors", SHA256: verdictBadSHA, Length: verdictBadBytes},
		},
		Outputs: []records.ModelTransferOutput{{Name: "full"}},
	}
	_, created, problem := o.store.Submit(records.Request{
		ID: requestID, IdemKey: requestID, BodyDigest: "sha256:" + strings.Repeat("c", 64),
		Package: "paul/minimax-h3-tools", Entrypoint: "four-lane", State: "queued",
		Kind: "job", Payload: []byte("{}"), Outputs: "[]", WeightsOutputs: "[]",
		Worker: rentalID, Rental: true, ModelTransfer: intent,
	})
	fatal(t, problem)
	if !created {
		t.Fatal("the transfer request was not created; the arm proves nothing")
	}
}
