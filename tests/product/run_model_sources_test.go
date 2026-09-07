package producttest

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/cozy-creator/cozy/internal/canonical"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/launch"
	"github.com/cozy-creator/cozy/internal/records"
)

// This catalog supplies declared metadata only. It refuses paid creation; no
// successful source conversion, worker receipt, or byte custody is simulated.
func runModelCatalog(t *testing.T) (string, *sync.Mutex, *[][]byte, string, []byte) {
	t.Helper()
	manifest := []byte(`{"fixture":"small root, not model closure bytes"}`)
	digest, err := canonical.Spell(canonical.Digest(manifest))
	must(t, err)
	iface := []byte(`{"application":"q:app","entrypoints":[],"format":"cozy.package.interface/1","jobs":[{"models":[{"class":"Source","component_use":{},"path":"quantize.models.dits"},{"class":"Source","component_use":{},"path":"quantize.models.shared"}],"name":"quantize","publishes":false,"request":{"fields":[{"name":"steps","type":"int"}]},"result":{"fields":[]},"weights_outputs":[{"max_bytes":1048576,"mime_type":"application/vnd.cozy.model-manifest","output_id":"fp8"}]}]}`)
	contract, problem := launch.DecodePackageInterface(iface)
	fatal(t, problem)
	var detail hub.PackageReleaseDetail
	detail.PackageInterface = iface
	detail.Release.Release = "1.0.0"
	detail.Release.PackageInterfaceDigest = contract.Digest
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	var mu sync.Mutex
	var posts [][]byte
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/quantize", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "quantize"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/quantize/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
	mux.HandleFunc("GET /v1/models/proof/source", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "proof", Name: "source"}, Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: digest, Bytes: 210_000_000_000}}}}})
	})
	mux.HandleFunc("GET /v1/models/proof/source/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(manifest) })
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("ref") != "proof/source@"+digest || r.URL.Query().Get("lane") != "" {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"manifest.not_found","message":"not retained"}}`))
			return
		}
		_ = json.NewEncoder(w).Encode(hub.ModelResolution{Model: "proof/source", ManifestID: digest,
			ManifestLength: int64(len(manifest)), Bytes: 210_000_000_000, Objects: 100,
			Components: []string{"model"}, ComponentBytes: map[string]int64{"model": 100}})
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", PriceUSDMicrosPerHour: 100_000, BaseWorkerProfile: "python3.12-cpu-linux-x86"}})
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`[]`)) })
	mux.HandleFunc("POST /v1/rentals", func(w http.ResponseWriter, r *http.Request) {
		raw, err := io.ReadAll(r.Body)
		if err != nil {
			t.Error(err)
		}
		mu.Lock()
		posts = append(posts, raw)
		mu.Unlock()
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte(`{"error":{"code":"proof.no_paid_create","message":"captured without renting"}}`))
	})
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	root := t.TempDir()
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: model-run-test\nrentals:\n  max_hourly_spend_usd: 20\n  idle_release_s: 0\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	return root, &mu, &posts, digest, manifest
}

func TestRunForeignModelInputsRefuseBeforeAcquisition(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	source := "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40)
	base := []string{"run", "proof/quantize/quantize", "steps=4", "model.dits=" + source, "model.shared=" + source, "--publish-to", "proof/output", "--rental-only", "--dry-run", "--json"}
	for _, test := range []struct {
		name string
		args []string
		want string
	}{
		{"missing profile", base, "model_producer.source_profile_absent"},
		{"extra profile", append(append([]string{}, base...), "--source-profile", "dits=x/1", "--source-profile", "shared=y/1", "--source-profile", "other=z/1"), "model_producer.source_profile_unknown_slot"},
		{"duplicate profile", append(append([]string{}, base...), "--source-profile", "dits=x/1", "--source-profile", "dits=y/1"), "names slot dits twice"},
		{"two sources", append(append([]string{}, base[:4]...), append([]string{"model.shared=civitai://123"}, base[5:]...)...), "model_source.multiple_sources_unsupported"},
		{"mixed inputs", append(append([]string{}, base[:4]...), append([]string{"model.shared=proof/source@1.0.0/bf16"}, base[5:]...)...), "model_source.mixed_inputs_unsupported"},
		{"unknown model slot", append(append(append([]string{}, base[:5]...), "model.other="+source), base[5:]...), "model_slot_unknown"},
	} {
		t.Run(test.name, func(t *testing.T) {
			code, out := runCozy(t, root, test.args...)
			if code == 0 || !strings.Contains(out, test.want) {
				t.Fatalf("exit=%d: %s; want %s", code, out, test.want)
			}
		})
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatalf("preflight started a daemon: %v", err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 0 {
		t.Fatal("invalid model source reached paid acquisition")
	}
	for _, verb := range []string{"upload", "download"} {
		code, out := runCozy(t, root, "model", verb, source, "proof/output", "--producer", "proof/quantize@v1/quantize", "--json")
		if code == 0 || !strings.Contains(out, "unknown flag --producer") {
			t.Fatalf("retired producer flag accepted: %d %s", code, out)
		}
	}
}

func TestRunRetainedCheckpointPinsFactsWithoutRelease(t *testing.T) {
	root, mu, posts, digest, manifest := runModelCatalog(t)
	args := []string{"run", "proof/quantize/quantize", "steps=7",
		"model.dits=proof/source#" + digest, "model.shared=proof/source#" + digest,
		"--publish-to", "proof/output", "--rental-only", "--json", "--idempotency-key", "retained-model-job"}
	code, out := runCozy(t, root, append(append([]string{}, args...), "--dry-run")...)
	if code != 0 {
		t.Fatalf("digest-only preflight refused: %d %s", code, out)
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("preflight started owner")
	}
	startDaemonProcess(t, root)
	code, out = runCozy(t, root, args...)
	if code != 0 {
		t.Fatalf("digest-only job did not queue: %d %s", code, out)
	}
	st, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer st.Close()
	row, problem := st.RequestByIdempotencyKey("retained-model-job")
	fatal(t, problem)
	if row == nil || len(row.Models) != 2 {
		t.Fatal("retained inputs missing from request")
	}
	for _, model := range row.Models {
		if !model.HubCheckpoint || !model.Downloadable() || model.Manifest != digest || model.ManifestLength != int64(len(manifest)) || model.Release != "" || model.Lane != "" || model.ComponentBytes["model"] != 100 {
			t.Fatal("job lost Hub facts or invented release metadata")
		}
	}
	waitUntil(t, "retained checkpoint identity reaches sizing boundary", func() bool { mu.Lock(); defer mu.Unlock(); return len(*posts) > 0 })
	mu.Lock()
	body := append([]byte(nil), (*posts)[0]...)
	mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if len(request.ServingModels) != 1 || request.ServingModels[0].Manifest != digest || request.ServingModels[0].Release != "" || request.ServingModels[0].Lane != "" {
		t.Fatal("rental declaration lost exact release-less checkpoint")
	}
}

func TestManualRentalAcceptsRetainedCheckpointIdentity(t *testing.T) {
	root, mu, posts, digest, _ := runModelCatalog(t)
	startDaemonProcess(t, root)
	args := []string{"rental", "new", "cpu", "--idempotency-key", "retained-manual-rental",
		"--model", "proof/source#" + digest, "--model", "proof/source#" + digest, "--json"}
	code, out := runCozy(t, root, args...)
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("retained checkpoint did not reach isolated sizing boundary: %d %s", code, out)
	}
	mu.Lock()
	if len(*posts) != 1 {
		mu.Unlock()
		t.Fatal("expected one isolated rental request")
	}
	body := append([]byte(nil), (*posts)[0]...)
	mu.Unlock()
	request, problem := hub.ParseRentalRequestBytes(body)
	fatal(t, problem)
	if len(request.ServingModels) != 1 || request.ServingModels[0] != (hub.ServingModel{Model: "proof/source", Manifest: digest}) {
		t.Fatal("manual rental invented release metadata or duplicated checkpoint")
	}
	code, out = runCozy(t, root, "rental", "new", "cpu", "--idempotency-key", "retained-manual-rental", "--json")
	if code == 0 || !strings.Contains(out, "proof.no_paid_create") {
		t.Fatalf("pinned retry failed: %d %s", code, out)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(*posts) != 2 || !bytes.Equal((*posts)[1], body) {
		t.Fatal("retry changed the frozen checkpoint sizing identity")
	}
}

func TestRunPublishedModelJobKeepsPayloadAndDeclaresRentalClosure(t *testing.T) {
	root, mu, posts, digest, manifest := runModelCatalog(t)
	input := filepath.Join(root, "quantize.json")
	must(t, os.WriteFile(input, []byte(`{"steps":7}`), 0600))
	args := []string{"run", "proof/quantize/quantize", "model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--in", input, "--publish-to", "proof/output", "--rental-only", "--json", "--full", "--idempotency-key", "published-model-job"}
	code, out := runCozy(t, root, append(append([]string{}, args...), "--dry-run")...)
	if code != 0 || !strings.Contains(out, `"steps":7`) {
		t.Fatalf("dry run lost typed payload: %d %s", code, out)
	}
	if _, err := os.Stat(filepath.Join(root, "daemon.lock")); !os.IsNotExist(err) {
		t.Fatal("dry run started a daemon")
	}
	startDaemonProcess(t, root)
	code, out = runCozy(t, root, args...)
	if code != 0 {
		t.Fatalf("ordinary modeled job did not queue: %d %s", code, out)
	}
	store, problem := records.Open(filepath.Join(root, "creator.sqlite"))
	fatal(t, problem)
	defer store.Close()
	row, problem := store.RequestByIdempotencyKey("published-model-job")
	fatal(t, problem)
	if row == nil || row.Kind != "job" || !bytes.Equal(row.Payload, []byte(`{"steps":7}`)) || len(row.Models) != 2 || row.ModelTransfer == nil || row.ModelTransfer.HasAcquisition() {
		t.Fatalf("ordinary request changed: %+v", row)
	}
	for _, model := range row.Models {
		if (model.Slot != "dits" && model.Slot != "shared") || model.Manifest != digest || model.ManifestLength != int64(len(manifest)) || model.BindingPath != "quantize.models."+model.Slot {
			t.Fatalf("job grant lost exact root identity: %+v", model)
		}
	}
	deadline := time.Now().Add(15 * time.Second)
	var paid []byte
	for len(paid) == 0 {
		mu.Lock()
		if len(*posts) > 0 {
			paid = append([]byte(nil), (*posts)[0]...)
		}
		mu.Unlock()
		if time.Now().After(deadline) {
			t.Fatalf("no rental declaration: %s", tail(filepath.Join(root, "daemon.log")))
		}
		time.Sleep(10 * time.Millisecond)
	}
	request, problem := hub.ParseRentalRequestBytes(paid)
	fatal(t, problem)
	if request.PlannedSourceBytes != 0 || len(request.ServingModels) != 1 || request.ServingModels[0].Manifest != digest || bytes.Contains(paid, []byte("210000000000")) {
		t.Fatalf("rental did not reuse exact catalog workload: %s", paid)
	}
	code, out = runCozy(t, root, args...)
	if code != 0 {
		t.Fatalf("same job replay failed: %d %s", code, out)
	}
	replayed, problem := store.RequestByIdempotencyKey("published-model-job")
	fatal(t, problem)
	if replayed.ID != row.ID || replayed.BodyDigest != row.BodyDigest {
		t.Fatal("replay changed request identity")
	}
}
