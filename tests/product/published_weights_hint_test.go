package producttest

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/records"
	v1 "github.com/cozy-creator/cozy/protocol/cozy/machine/v1"
)

// A run whose machine published its weights output to the run's destination names the new
// checkpoint and the publish command, as `cozy model quantize --await` says it does.
func TestARunThatPublishedItsWeightsNamesTheCheckpointAndThePublishCommand(t *testing.T) {
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: http://127.0.0.1:1\ntensorhub_token: unreachable\n"), 0o600))
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	request, _, problem := store.Submit(records.Request{ID: "quantize-run", IdemKey: "quantize-run",
		Package: "proof/quantize", Entrypoint: "fp8", Kind: "job", Payload: []byte(`{}`),
		BodyDigest: childDigest("1"), MachineExecutionObserver: true,
		ModelTransfer: &records.ModelTransferIntent{Kind: "model-upload", Destination: "proof/quantized",
			Outputs: []records.ModelTransferOutput{{Name: "fp8"}}}})
	fatal(t, problem)
	fatal(t, store.LinkMachineExecution(request.ID, "local"))
	fatal(t, store.AcceptRunV1(request.ID, &v1.RunState{Id: request.ID, Number: 1, State: "running", Attempt: 1}))
	checkpoint := "sha256:" + strings.Repeat("67", 32)
	fatal(t, store.RecordRunOutcomeV1(request.ID, records.RunEndV1{Outcome: &v1.Outcome{Status: "succeeded",
		Result:  []byte(`{"manifest":{"digest":"` + checkpoint + `","length":164},"output_slot":"fp8"}`),
		Outputs: []*v1.Product{{Output: "fp8", Rev: 1, Length: 164, Digest: checkpoint, MediaType: "application/vnd.cozy.weights"}}}}))
	store.Close()

	_ = startDaemonProcess(t, root)
	code, out := runCozy(t, root, "run", "watch", request.ID)
	t.Logf("cozy run watch:\n%s", out)
	if code != 0 || !strings.Contains(out, checkpoint) ||
		!strings.Contains(out, "cozy model publish proof/quantized --release <label> --lane fp8="+checkpoint) {
		t.Fatalf("the run does not name its checkpoint and the publish command [exit %d]\n%s", code, out)
	}
}
