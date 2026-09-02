package producttest

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/orchestrator"
)

// TestServingRequestDeclaresNoModelInput is the live defect of 2026-09-01: `cozy run
// paul/anima/generate` on this host failed PROTOCOL `grant_binding_mismatch` because the
// serving spec declared `model:generate.models.model` (the resolver had filled the
// manifest length) while the local grant carried only the payload — and had the grant
// agreed, the worker would have refused `grant_model_serving_refused` instead: a serving
// spec never carries a Model input. The fake worker applies both of the worker's rules to
// every offer; a bound model on a serving request must dispatch and settle through them.
func TestServingRequestDeclaresNoModelInput(t *testing.T) {
	o := hostOwner(t, "model-inputs")
	spec := fakeSpec("model-inputs", "0", "--arm", "output", "--cozy-home", o.root)
	instance, _, e := o.c.EnsureWorker(spec)
	fatal(t, e)
	planID := planIDOf(t, spec)
	fatal(t, o.c.EnsurePlacementReady(instance, planID))

	sub := submission(planID, "fake/model-inputs", "model-inputs-1", map[string]any{"prompt": "fox"})
	sub.Models = []orchestrator.ModelRef{{
		Package: "fake/model-inputs", Slot: "generate.models.model", Model: "paul/anima",
		Release: "1.0.0", Lane: "bf16", Manifest: "sha256:" + strings.Repeat("ab", 32),
		ManifestLength: 1045,
	}}
	requestID, attempt, e := o.c.Submit(sub)
	fatal(t, e)
	if _, e := o.c.Await(requestID, attempt, 60*time.Second); e != nil {
		t.Fatalf("a serving request with a bound model did not settle: %s", briefly(e))
	}
	for _, refusal := range []string{"grant_binding_mismatch", "grant_model_serving_refused"} {
		if n := countEvents(o, refusal); n != 0 {
			t.Errorf("the worker refused %s %d time(s)", refusal, n)
		}
	}
	row, e := o.store.RequestRow(requestID)
	fatal(t, e)
	if row == nil || len(row.Models) != 1 || row.Models[0].ManifestLength != 1045 {
		data, _ := json.Marshal(row)
		t.Errorf("the request row lost its bound model: %s", data)
	}
}
