package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
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

// H3's producer has two scalar Model inputs. Their order is the ordinal within
// each parameter, so both are zero regardless of the payload's position.
func TestJobScalarModelInputsHaveZeroOrder(t *testing.T) {
	public, private, err := ed25519.GenerateKey(rand.Reader)
	must(t, err)
	pod := &fakePod{controlKey: public, serve: true, jobReady: true}
	connection, _ := startFakePod(t, t.TempDir(), pod)
	o := hostOwner(t, "job-model-order", rentalWiring(connection, private))
	pkg := "cozy/h3-package"
	_, _, problem := o.c.Submit(orchestrator.Submission{
		IdemKey: "job-model-order", Package: pkg, Entrypoint: "four-lane",
		PlanID: "sha256:" + strings.Repeat("35", 32), Release: "1.0.7",
		Kind: "job", Org: "paul", Payload: []byte(`{"steps":4}`), Outputs: []string{"model"},
		Worker: podRental, Rental: true, RentalRequired: true,
		Models: []orchestrator.ModelRef{
			{Package: pkg, Slot: "shared", Model: "source/shared", Manifest: "sha256:" + strings.Repeat("2", 64), ManifestLength: 164},
			{Package: pkg, Slot: "dits", Model: "source/dits", Manifest: "sha256:" + strings.Repeat("1", 64), ManifestLength: 164},
		},
	})
	fatal(t, problem)
	waitUntil(t, "job offer with two scalar Models", func() bool {
		pod.mu.Lock()
		defer pod.mu.Unlock()
		return len(pod.offers) > 0
	})
	pod.mu.Lock()
	raw := append([]byte(nil), pod.offers[0].InvocationSpecCanonicalBytes...)
	pod.mu.Unlock()
	var spec struct {
		Inputs []struct {
			ID    string `json:"input_id"`
			Order uint32 `json:"order"`
		} `json:"inputs"`
	}
	must(t, json.Unmarshal(raw, &spec))
	models := 0
	for _, input := range spec.Inputs {
		if strings.HasPrefix(input.ID, "model:") {
			models++
			if input.Order != 0 {
				t.Errorf("scalar %s has order %d, want 0", input.ID, input.Order)
			}
		}
	}
	if models != 2 {
		t.Fatalf("offer carried %d Model inputs, want 2", models)
	}
}
