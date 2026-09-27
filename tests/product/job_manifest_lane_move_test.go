package producttest

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
)

// A job grants the exact checkpoint it selected. When the release lane has moved on by
// the time its root is read, the selected checkpoint's own bytes are read instead.
func TestJobReadsTheSelectedCheckpointAfterItsLaneMoves(t *testing.T) {
	selected := []byte(`{"fixture":"selected root"}`)
	digest, err := canonical.Spell(canonical.Digest(selected))
	must(t, err)
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"quantize.models.dits"},{"class":"Source","component_use":{},"path":"quantize.models.shared"}],"name":"quantize","publishes":false,"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"}]}]}`)
	contract, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/quantize", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "quantize"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/quantize/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
	mux.HandleFunc("GET /v1/models/proof/source", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "proof", Name: "source"}, Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"},
			Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: digest, Bytes: 1_000, Components: []string{"model"}}}}}})
	})
	mux.HandleFunc("GET /v1/models/proof/source/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"fixture":"a newer root"}`))
	})
	mux.HandleFunc("GET /v1/models/proof/source/checkpoints/{id}", func(w http.ResponseWriter, r *http.Request) {
		if r.PathValue("id") != digest {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(selected)
	})
	mux.HandleFunc("/", http.NotFound)
	server := httptest.NewServer(mux)
	defer server.Close()
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: proof\n"), 0600))
	request, _, out := submitRun(t, root, "lane-moved", "run", "proof/quantize/quantize", "steps=7",
		"model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--rental-only", "--json")
	if request == nil || len(request.Models) != 2 {
		t.Fatalf("a moved lane refused the selected checkpoint: %s", out)
	}
	for _, model := range request.Models {
		if model.Manifest != digest || model.ManifestLength != int64(len(selected)) {
			t.Fatalf("the job did not grant the selected checkpoint: %+v", model)
		}
	}
}
