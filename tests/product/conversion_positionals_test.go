package producttest

import (
	"bytes"
	"net/http"
	"path/filepath"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// A conversion job (one model input, weights outputs) reads
// `cozy run <job> <input> <org/model>`: the same request as model.<param>= plus --publish-to.
func TestConversionJobReadsInputAndDestinationPositionals(t *testing.T) {
	root, _, _, digest, _ := runModelCatalog(t, func(_ *http.ServeMux, detail *hub.PackageReleaseDetail) {
		iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"quantize.models.source"}],"name":"quantize","publishes":false,"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"}]}]}`)
		contract, problem := launch.DecodePackageInterface(iface)
		fatal(t, problem)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
		detail.Release.PackageInterfaceLength = int64(len(iface))
	})
	code, out := runCozy(t, root, "run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output", "proof/extra", "--rental-only")
	if code == 0 || !strings.Contains(out, "cozy run proof/quantize/quantize <source> [<org/model>] steps=<int>") {
		t.Fatalf("a third positional did not refuse with the conversion spelling: %d %s", code, out)
	}
	for _, args := range [][]string{
		{"run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output", "--publish-to", "proof/other"},
		{"run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "model.source=proof/source@1.0.0/bf16"},
	} {
		if code, out := runCozy(t, root, append(args, "--rental-only", "--json")...); code == 0 {
			t.Fatalf("ambiguous conversion spelling was accepted: %v %s", args, out)
		}
	}
	startDaemonProcess(t, root)
	code, out = runCozy(t, root, "run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output",
		"steps=7", "--rental-only", "--json", "--idempotency-key", "conversion-positionals")
	if code != 0 {
		t.Fatalf("conversion positionals did not queue: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("conversion-positionals")
	fatal(t, problem)
	if row == nil || !bytes.Equal(row.Payload, []byte(`{"steps":7}`)) || len(row.Models) != 1 ||
		row.Models[0].Slot != "source" || row.Models[0].Manifest != digest ||
		row.ModelTransfer == nil || row.ModelTransfer.Destination != "proof/output" ||
		len(row.ModelTransfer.Outputs) != 1 || row.ModelTransfer.Outputs[0].Name != "fp8" {
		t.Fatalf("positionals did not become the input binding and destination: %+v", row)
	}
}
