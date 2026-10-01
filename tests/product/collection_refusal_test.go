package producttest

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/records"
	pb "github.com/cozy-creator/cozy/protocol/cozy/worker/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// releasedOutputMachine finishes its run with an output whose hold its Runtime already
// released, as Runtime 0.18.84 did for every returned product after the first (runs 1665-1670):
// every read of the output's bytes is refused, and no retry can change that.
type releasedOutputMachine struct {
	*finishingMachine
	reads atomic.Int64
}

func (m *releasedOutputMachine) ReadByteTreeObject(*pb.NativeByteReadCall, grpc.ServerStreamingServer[pb.NativeByteReadChunk]) error {
	m.reads.Add(1)
	return status.Error(codes.FailedPrecondition, "ordinary artifact has no exact received retention")
}

// A finished run whose output its machine refuses to hand over ends collection with that
// refusal on record, shown by `cozy run show`. The daemon does not ask again: it neither loops
// on the refusal nor keeps its machine busy for it; only the owner's `cozy run watch` asks again.
func TestARefusedOutputEndsItsCollection(t *testing.T) {
	h := newLadderHub(t)
	h.bind(goodLadder())
	h.workflow = []byte(strings.Replace(string(workflowInterface), `"result":{"fields":[]}}],"format"`,
		`"result":{"fields":[{"name":"image","type":{"asset":"image"},"asset_bound":{"max_bytes":1048576,"media_types":["image/png"]}}]}}],"format"`, 1))
	bundle, err := os.ReadFile(filepath.Join("testdata", "execution_evidence", "runtime-bundle.json"))
	must(t, err)
	finishing := &finishingMachine{runtimeMachine: &runtimeMachine{triage: bundle}, blob: []byte("\x89PNG released before collection")}
	machine := &releasedOutputMachine{finishingMachine: finishing}
	root, layout := rentedLadderMachine(t, h, &fakePod{machine: machine, deviceCount: 4}, nil)
	input := filepath.Join(t.TempDir(), "input.json")
	must(t, os.WriteFile(input, []byte(`{"steps":1}`), 0o600))
	out := filepath.Join(t.TempDir(), "out")
	command := exec.Command("/usr/bin/nice", "-n", "19", cozyBin, "run", ladderPackage+"/generate", "--input", input,
		"--rental=tessa", "--await", "--json", "--out", out, "--idempotency-key", "refused")
	command.Env = childEnv(t, root)
	var stdout, stderr bytes.Buffer
	command.Stdout, command.Stderr = &stdout, &stderr
	must(t, command.Start())
	exited := make(chan struct{})
	go func() { _ = command.Wait(); close(exited) }()
	t.Cleanup(func() { _ = command.Process.Kill(); <-exited })
	waitFor(t, root, "the Runtime submission", func() bool { return machine.submitted() != nil })
	store, problem := records.Open(layout.DB)
	fatal(t, problem)
	t.Cleanup(func() { store.Close() })
	var request *records.Request
	waitFor(t, root, "the run on its machine", func() bool {
		request, _ = store.RequestByIdempotencyKey("refused")
		link, _ := store.MachineExecution(request.ID)
		return link != nil && link.RemoteCursor > 0
	})
	finishing.arm()

	// The refusal goes on record at once; a daemon that asks the machine again and again
	// instead (it asked every few seconds, forever) fails here.
	var code, message string
	eventually(t, root, "the refusal on record, or a retry loop", func() bool {
		code, message, problem = store.MachineCollectionRefusal(request.ID)
		fatal(t, problem)
		return code != "" || machine.reads.Load() > 4
	})
	if code != "machine_execution.refused" || !strings.Contains(message, "no exact received retention") {
		t.Fatalf("after %d refused reads the run records %q %q: %s", machine.reads.Load(), code, message, tail(filepath.Join(root, "daemon.log")))
	}
	<-exited
	if command.ProcessState.ExitCode() == 0 || !strings.Contains(stdout.String(), `"code":"run.outputs_undelivered"`) ||
		!strings.Contains(stdout.String(), "no exact received retention") {
		t.Fatalf("--await did not end naming the refused output:\n%s\n%s", stdout.String(), stderr.String())
	}
	asked := machine.reads.Load()
	time.Sleep(12 * time.Second) // more than twice the observer's longest retry interval
	if again := machine.reads.Load(); again != asked {
		t.Fatalf("the daemon asked for the refused bytes %d more times after recording the refusal", again-asked)
	}
	if code, shown := runCozy(t, root, "run", "show", request.ID); code != 0 ||
		!strings.Contains(shown, "collection pending: machine_execution.refused") || !strings.Contains(shown, "no exact received retention") {
		t.Fatalf("run show does not name the refused collection [exit %d]\n%s", code, shown)
	}
}
