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

// A conversion (model inputs and weights outputs) reads `cozy run <fn> <input> <org/model>`:
// the same request as model.<first param>= plus --upload-to; later inputs stay model.<param>=.
func TestConversionReadsInputAndDestinationPositionals(t *testing.T) {
	root, _, _, digest, _ := runModelCatalog(t, func(_ *http.ServeMux, detail *hub.PackageReleaseDetail) {
		iface := []byte(`{"application":"q:app","entrypoints":[{"accelerator":false,"invocable":{"capabilities":["weights"],"context":"ctx","defaults":{},"enum_members":{},"export":"quantize","memoize":false,"module":"q","parameters":["steps"],"type_names":{"request":"quantizeRequest"}},"models":[{"class":"Model","component_use":{},"encoded_leaves":"refuse","path":"quantize.models.source","weights":true},{"class":"Model","component_use":{},"encoded_leaves":"refuse","path":"quantize.models.base","weights":true}],"name":"quantize","request":{"fields":[{"name":"steps","type":"int"}]},"result":{"input":"model"},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"}]}],"format":"cozy.package.interface/1"}`)
		contract, problem := launch.DecodePackageInterface(iface)
		fatal(t, problem)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
		detail.Release.PackageInterfaceLength = int64(len(iface))
	})
	code, out := runCozy(t, root, "run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output", "proof/extra", "--rental-only")
	if code == 0 || !strings.Contains(out, "cozy run proof/quantize/quantize <source> [<org/model>] model.base=<ref> steps=<int>") {
		t.Fatalf("a third positional did not refuse with the conversion spelling: %d %s", code, out)
	}
	for _, args := range [][]string{
		{"run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output", "--upload-to", "proof/other"},
		{"run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "model.source=proof/source@1.0.0/bf16"},
	} {
		if code, out := runCozy(t, root, append(args, "--rental-only", "--json")...); code == 0 {
			t.Fatalf("ambiguous conversion spelling was accepted: %v %s", args, out)
		}
	}
	startCalciferProcess(t, root)
	code, out = runCozy(t, root, "run", "proof/quantize/quantize", "proof/source@1.0.0/bf16", "proof/output",
		"steps=7", "model.base=proof/source@1.0.0/bf16", "--rental-only", "--json", "--idempotency-key", "conversion-positionals")
	if code != 0 {
		t.Fatalf("conversion positionals did not queue: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("conversion-positionals")
	fatal(t, problem)
	if row == nil || !bytes.Equal(row.Payload, []byte(`{"steps":7}`)) || len(row.Models) != 2 ||
		row.Models[0].Manifest != digest || row.Models[1].Manifest != digest || row.Models[0].Slot == row.Models[1].Slot ||
		row.ModelTransfer == nil || row.ModelTransfer.Destination != "proof/output" ||
		len(row.ModelTransfer.Outputs) != 1 || row.ModelTransfer.Outputs[0].Name != "fp8" {
		t.Fatalf("positionals did not become the input binding and destination: %+v", row)
	}
}
