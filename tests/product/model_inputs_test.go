package producttest

import (
	"crypto/ed25519"
	"crypto/rand"
	"encoding/json"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/orchestrator"
)

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
