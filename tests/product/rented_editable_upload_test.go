package producttest

import (
	"encoding/json"
	"strings"
	"sync"
	"testing"

	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// editablePod is a pod taking an editable package's captured files. `refuse` answers each
// upload call's header, in order, with a status; nil receives the file.
type editablePod struct {
	*fakePod
	mu       sync.Mutex
	calls    int
	refuse   []error
	block    chan struct{} // closed once an upload holds its stream open without acknowledging
	released chan struct{} // closed once that held upload's stream ends
	prepared int
}

func newEditablePod(machine machineExecutionPeer, refuse ...error) *editablePod {
	p := &editablePod{fakePod: &fakePod{machine: machine}, refuse: refuse}
	p.localUpload = p.upload
	return p
}

func (p *editablePod) upload(stream grpc.BidiStreamingServer[pb.LocalPackageUploadFrame, pb.LocalPackageFileStatus]) error {
	frame, err := stream.Recv()
	if err != nil {
		return err
	}
	header := frame.GetHeader()
	if header == nil || header.File == nil {
		return status.Error(codes.InvalidArgument, "header required")
	}
	if err := p.verifyClaim(header.Claim, false); err != nil {
		return err
	}
	p.mu.Lock()
	call, block := p.calls, p.block
	p.calls++
	p.mu.Unlock()
	if call < len(p.refuse) && p.refuse[call] != nil {
		return p.refuse[call]
	}
	if block != nil {
		close(block)
		<-stream.Context().Done()
		close(p.released)
		return stream.Context().Err()
	}
	state := &pb.LocalPackageFileStatus{OperationId: header.OperationId, Digest: header.File.Digest, Filename: header.File.Filename,
		Length: header.File.Length, State: pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_RECEIVING}
	if err := stream.Send(state); err != nil {
		return err
	}
	for state.ReceivedBytes < header.File.Length {
		frame, err := stream.Recv()
		if err != nil {
			return err
		}
		state.ReceivedBytes += uint64(len(frame.GetChunk().GetData()))
		if state.ReceivedBytes == header.File.Length {
			state.State = pb.LocalPackageFileState_LOCAL_PACKAGE_FILE_STATE_VERIFIED
		}
		if err := stream.Send(state); err != nil {
			return err
		}
	}
	return nil
}

// rentedEditable installs local/upload-proof, whose `main` job the tests run on the pod,
// attaches the pod as tessa and starts the daemon.
func rentedEditable(t *testing.T, pod *editablePod) (string, *records.Store) {
	t.Helper()
	var surface []byte
	pod.localPrepare = func(call *pb.PrepareLocalPackageCall, stream grpc.ServerStreamingServer[pb.PrepareEvent]) error {
		if err := pod.verifyClaim(call.Claim, false); err != nil {
			return err
		}
		pod.mu.Lock()
		pod.prepared++
		pod.mu.Unlock()
		selected := call.LocalPackageSet.Package
		return stream.Send(&pb.PrepareEvent{Stage: pb.PrepareStage_PREPARE_STAGE_PREPARED, InstalledPackage: &pb.InstalledPackage{
			InstallationId: selected.InstallationId, Package: selected.Package, Release: selected.Release, PackageInterface: surface}})
	}
	h := newLadderHub(t)
	h.bind(goodLadder())
	root, layout := rentedLadderMachine(t, h, pod.fakePod, func(layout home.Layout, store *records.Store) {
		surface = installLocalPackage(t, layout, store, "upload-proof", "upload_proof", `import msgspec
from cozy_runtime.author import App, Context, invocable


class Result(msgspec.Struct):
    value: int


@invocable(memoize=False)
async def main(ctx: Context, *, steps: int) -> Result:
    return Result(steps)


app = App()
app.job(main)
`)
	})
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	return root, store
}

// unblock lets the next upload through.
func (p *editablePod) unblock() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.block = nil
}

// closed is whether signal has closed.
func closed(signal chan struct{}) func() bool {
	return func() bool {
		select {
		case <-signal:
			return true
		default:
			return false
		}
	}
}

// counts is how many upload calls and preparations the pod took.
func (p *editablePod) counts() (int, int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.calls, p.prepared
}

// An editable package's upload to a rental that the transport interrupts is retried: the
// run stays queued and reaches Runtime once.
func TestRentedEditableUploadRetriesATransportLoss(t *testing.T) {
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := newEditablePod(machines, status.Error(codes.Unavailable, "control transport interrupted"))
	root, store := rentedEditable(t, pod)
	code, out := runCozy(t, root, "run", "local/upload-proof/main", "steps=1", "--rental=tessa", "--await", "--json", "--idempotency-key", "retry")
	row, problem := store.RequestByIdempotencyKey("retry")
	fatal(t, problem)
	if code != 0 || row.State != "succeeded" || len(machines.submitted()) != 1 {
		t.Fatalf("the interrupted upload did not reach Runtime once [exit %d, %d submitted]: %s", code, len(machines.submitted()), out)
	}
	if calls, prepared := pod.counts(); calls < 2 || prepared != 1 {
		t.Fatalf("%d upload call(s) and %d preparation(s); want a retried upload and one preparation", calls, prepared)
	}
}

// A pod that refuses an editable package's upload fails the run with that refusal; the
// upload is not retried and nothing is prepared or submitted.
func TestRentedEditableUploadRefusalFailsTheRun(t *testing.T) {
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := newEditablePod(machines, status.Error(codes.InvalidArgument, "invalid local package upload header: filename refused"))
	root, _ := rentedEditable(t, pod)
	runCozy(t, root, "run", "local/upload-proof/main", "steps=1", "--rental=tessa", "--json")
	var page struct {
		Invocations []struct {
			Status    string `json:"status"`
			ErrorType string `json:"error_type"`
			Error     string `json:"error"`
		}
	}
	waitFor(t, root, "the refused upload to fail its run", func() bool {
		_, list := runCozy(t, root, "run", "list", "--json", "--full")
		return json.Unmarshal([]byte(list), &page) == nil && len(page.Invocations) == 1 && page.Invocations[0].Status == "failed"
	})
	if run := page.Invocations[0]; run.ErrorType != "local_package_upload_refused" || !strings.Contains(run.Error, "filename refused") {
		t.Fatalf("the run does not carry the pod's refusal: %+v", run)
	}
	if calls, prepared := pod.counts(); calls != 1 || prepared != 0 || len(machines.submitted()) != 0 {
		t.Fatalf("a refused upload was retried or went on: %d call(s), %d preparation(s), %d submission(s)", calls, prepared, len(machines.submitted()))
	}
}

// Canceling a rented run whose editable package is still uploading ends the upload's
// stream at once; the canceled run is never prepared or submitted.
func TestRentedEditableUploadCancelEndsTheUpload(t *testing.T) {
	machines := newTerminalMachines(func(map[string]any) *pb.AttemptOutcomeBody {
		return outcome(pb.OutcomeStatus_OUTCOME_STATUS_SUCCEEDED, "", nil)
	})
	pod := newEditablePod(machines)
	pod.block, pod.released = make(chan struct{}), make(chan struct{})
	root, store := rentedEditable(t, pod)
	if code, out := runCozy(t, root, "run", "local/upload-proof/main", "steps=1", "--rental=tessa", "--json", "--idempotency-key", "cancel"); code != 0 {
		t.Fatalf("the rented run was refused [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the editable package's upload", closed(pod.block))
	if code, out := runCozy(t, root, "run", "cancel", "1", "--json"); code != 0 {
		t.Fatalf("cozy run cancel failed [exit %d]: %s", code, out)
	}
	waitFor(t, root, "the canceled run's upload to end", closed(pod.released))
	row, problem := store.RequestByIdempotencyKey("cancel")
	fatal(t, problem)
	if _, prepared := pod.counts(); row.State != "canceled" || prepared != 0 || len(machines.submitted()) != 0 {
		t.Fatalf("the canceled run went on: %s, %d preparation(s), %d submission(s)", row.State, prepared, len(machines.submitted()))
	}
}
