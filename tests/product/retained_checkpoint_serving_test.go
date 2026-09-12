package producttest

import (
	"net/http"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// Exercise CLI -> daemon admission using the existing metadata-only catalog.
// Its acquisition route refuses all rentals; this proves no model execution.
func TestRunServingAcceptsRetainedCheckpointWithoutRelease(t *testing.T) {
	root, _, _, digest, manifest := runModelCatalog(t, func(_ *http.ServeMux, detail *hub.PackageReleaseDetail) {
		iface := []byte(`{"application":"q:app","entrypoints":[{"models":[{"class":"Source","component_use":{},"path":"generate.models.model"}],"name":"generate","request":{"fields":[{"name":"prompt","type":"str"}]},"result":{"fields":[]}}],"format":"cozy.package.interface/1","jobs":[]}`)
		contract, problem := launch.DecodePackageInterface(iface)
		fatal(t, problem)
		detail.PackageInterface = iface
		detail.Release.PackageInterfaceDigest = contract.Digest
		detail.Release.PackageInterfaceLength = int64(len(iface))
	})
	daemon := startDaemonProcess(t, root)
	code, out := runCozy(t, root, "run", "proof/quantize/generate", "prompt=Compare the candidate",
		"model.model=proof/source#"+digest, "--rental-only", "--json", "--idempotency-key", "retained-serving")
	if code != 0 {
		t.Fatalf("exact candidate was rejected before serving admission: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("retained-serving")
	fatal(t, problem)
	if row == nil || row.IsJob() || len(row.Models) != 1 {
		t.Fatalf("candidate did not create one ordinary serving request: %+v", row)
	}
	model := row.Models[0]
	if !model.HubCheckpoint || !model.Downloadable() || model.Manifest != digest ||
		model.ManifestLength != int64(len(manifest)) || model.Release != "" || model.Lane != "" {
		t.Fatalf("serving admission changed the exact candidate: %+v", model)
	}
	for _, invalid := range []struct {
		name string
		edit func(*records.ModelRef)
	}{
		{"missing-hub-custody", func(m *records.ModelRef) { m.HubCheckpoint = false }},
		{"lane-without-release", func(m *records.ModelRef) { m.Lane = "bf16" }},
		{"empty-manifest", func(m *records.ModelRef) { m.Manifest = "" }},
		{"malformed-manifest", func(m *records.ModelRef) { m.Manifest = "sha256:invalid" }},
		{"missing-manifest-length", func(m *records.ModelRef) { m.ManifestLength = 0 }},
	} {
		t.Run(invalid.name, func(t *testing.T) {
			bad := model
			invalid.edit(&bad)
			response := daemon.call(t, "POST", "/v1/requests", map[string]any{
				"package": row.Package, "release": row.Release, "function": row.Entrypoint,
				"rental": true, "input": map[string]any{"prompt": "Compare the candidate"},
				"models": []records.ModelRef{bad},
			}, "Idempotency-Key", invalid.name)
			want := "rental.model_selection_mismatch"
			if invalid.name == "malformed-manifest" {
				want = "rental.model_manifest_invalid"
			}
			if response.Status < 400 || response.code() != want {
				t.Fatalf("malformed selection did not receive %s: %s", want, response.brief())
			}
			recorded, problem := store.RequestByIdempotencyKey(invalid.name)
			fatal(t, problem)
			if recorded != nil {
				t.Fatal("refusal created a request")
			}
		})
	}
	code, out = runCozy(t, root, "run", "proof/quantize/generate", "prompt=Compare the baseline",
		"model.model=proof/source@1.0.0/bf16", "--rental-only", "--json", "--idempotency-key", "released-serving")
	if code != 0 {
		t.Fatalf("released baseline stopped working: %d %s", code, out)
	}
}
