package producttest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/rental"
	"github.com/cozy-creator/cozy/internal/secret"
)

// Tensorhub, TensorFS and older or newer Creators are peers that evolve independently.
// An additive field, an unsorted list or one unusable row must not refuse the document,
// and a GPU product without a host-RAM figure is still rentable: placement fits GPUs only.
func TestHubAnswersWithAdditiveFieldsAreRead(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/models/proof/model/publications/op-1/finalize", func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusAccepted)
		_, _ = w.Write([]byte(`{"operation":"op-1","state":"queued","status_url":"/x","queued_at":"2026-09-26T00:00:00Z"}`))
	})
	mux.HandleFunc("GET /v1/models/proof/model/publications/op-1/finalization", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"operation":"op-1","state":"completed","attempts":2,"result":{"publish_id":"pub-1",` +
			`"checkpoint_id":"ckpt-1","manifest":{"sha256":"` + strings.Repeat("a", 64) + `","length":9,"media_type":"x"},` +
			`"objects":3,"bytes":27,"state":"committed","duplicate":false,"retained_until":"later"}}`))
	})
	mux.HandleFunc("GET /v1/packages/proof/demo/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"release":{"release":"1.0.0","package_interface_digest":"sha256:x","package_interface_length":2,` +
			`"provenance":{"builder":"future"}},"package_interface":{},"requirements":["torch>=2","","numpy","torch>=2"],` +
			`"requires_python":">=3.12","python_version":"3.12.12","signatures":[]}`))
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`[
			{"name":"cpu","accelerator_model":"CPU","widths":[{"accelerator_count":1,"price_usd_micros_per_hour":1000}],"region":"future"},
			{"name":"unpriced","accelerator_model":"NVIDIA H100","compute_capability":"9.0","vram_gb":80,"minimum_ram_per_gpu_gb":100,
			 "widths":[{"accelerator_count":1,"price_usd_micros_per_hour":0}]},
			{"name":"h100","accelerator_model":"NVIDIA H100","compute_capability":"9.0","vram_gb":80,
			 "widths":[{"accelerator_count":1,"price_usd_micros_per_hour":3000000},{"accelerator_count":0,"price_usd_micros_per_hour":1},
			           {"accelerator_count":2,"price_usd_micros_per_hour":6000000}]},
			{"name":"cpu","accelerator_model":"CPU","widths":[{"accelerator_count":1,"price_usd_micros_per_hour":1}]}
		]`))
	})
	server := httptest.NewServer(mux)
	defer server.Close()
	client := hub.New(config.Config{HubURL: server.URL, HubToken: secret.New("proof")}, "peer-tolerance")

	checkpoint, problem := client.FinalizePublication(t.Context(), hub.Ref{Org: "proof", Name: "model"}, "op-1",
		hub.FinalizePublicationRequest{ManifestID: "sha256:" + strings.Repeat("a", 64), ManifestLength: 9}, "finalize")
	fatal(t, problem)
	if checkpoint.CheckpointID != "ckpt-1" || checkpoint.Manifest.Length != 9 {
		t.Fatalf("finalization with additive fields was misread: %+v", checkpoint)
	}

	detail, problem := client.PackageRelease(t.Context(), hub.Ref{Org: "proof", Name: "demo"}, "1.0.0")
	fatal(t, problem)
	requirements, problem := detail.Requirements()
	fatal(t, problem)
	if !slices.Equal(requirements, []string{"numpy", "torch>=2"}) {
		t.Fatalf("requirements were not normalized: %q", requirements)
	}

	skus, problem := client.RentalSKUs(t.Context())
	fatal(t, problem)
	var names []string
	for _, sku := range skus {
		names = append(names, fmt.Sprintf("%s/%d/%d", sku.Name, sku.AcceleratorCount, sku.PriceUSDMicrosPerHour))
	}
	if !slices.Equal(names, []string{"cpu/1/1000", "h100/1/3000000", "h100/2/6000000"}) {
		t.Fatalf("catalog kept an unusable row or dropped a usable one: %v", names)
	}
}

// A paid rental intent persisted by another Creator build still resumes: replay sends the
// persisted bytes unchanged, so only the fields this build reads must be present.
func TestPersistedRentalIntentFromAnotherBuildResumes(t *testing.T) {
	raw := []byte(`{"sku":"h200","name":"twine","accelerator_count":2,"media_token_sha256":"` + strings.Repeat("ab", 32) +
		`","creator_public_key":"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA","placement_hint":{"region":"eu"}}`)
	request, problem := hub.ParseRentalRequestBytes(raw)
	fatal(t, problem)
	if request.Name != "twine" || request.SKU != "h200" || request.AcceleratorCount != 2 {
		t.Fatalf("persisted intent misread: %+v", request)
	}
	if _, problem := hub.ParseRentalRequestBytes([]byte(`{"name":"twine"}`)); problem == nil {
		t.Fatal("an intent naming no SKU or width was accepted")
	}
}

// A newer image inventory document keeps preparing packages on rentals.
func TestImageInventoryFromNewerHubIsRead(t *testing.T) {
	inventory, err := rental.ImageInventory(json.RawMessage(`{"format":"tensorhub.image_inventory/2","profile":"python3.12-cpu-linux-x86",` +
		`"python":"3.12.12","cuda":{"runtime":"12.8"},"distributions":[{"name":"numpy","version":"2.1.0","source":"image"},{"name":"","version":"1"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if inventory.Profile != "python3.12-cpu-linux-x86" || len(inventory.Distributions) != 1 ||
		inventory.Distributions[0].Distribution != "numpy" {
		t.Fatalf("inventory misread: %+v", inventory)
	}
}

// A package install plan from a newer hub may carry additive fields, a repeated identical
// wheel row and a file kind this build does not install; the package still installs.
func TestPackageInstallReadsAdditivePlan(t *testing.T) {
	plan, wheel := reportingRelease(t)
	raw, err := json.Marshal(plan)
	must(t, err)
	var document map[string]any
	must(t, json.Unmarshal(raw, &document))
	rows := document["downloads"].([]any)
	project := rows[0].(map[string]any)
	project["signature"] = "future"
	document["downloads"] = append(rows, project, map[string]any{"kind": "source_archive", "path": "demo.tar.gz"})
	document["generated_at"] = "2026-09-26T00:00:00Z"

	root := t.TempDir()
	mux := http.NewServeMux()
	mux.HandleFunc("POST /v1/packages/proof/install-reporting/download", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(document)
	})
	mux.HandleFunc("GET /v1/index/proof/simple/install-reporting/", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = fmt.Fprintf(w, `<a href="/project.whl/%s#sha256=%s">%s</a>`, plan.Downloads[0].Path,
			strings.TrimPrefix(plan.Downloads[0].Digest, "sha256:"), plan.Downloads[0].Path)
	})
	mux.HandleFunc("GET /project.whl/{name}", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(wheel) })
	server := httptest.NewServer(mux)
	defer server.Close()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\n"), 0600))
	code, stdout, stderr := runCozyStreams(t, root, "package", "install", "proof/install-reporting", "--version=1.0.1", "--json", "--full")
	if code != 0 || !strings.Contains(stdout, `"status":"installed"`) {
		t.Fatalf("additive install plan was refused: %d %s %s", code, stdout, stderr)
	}
}
