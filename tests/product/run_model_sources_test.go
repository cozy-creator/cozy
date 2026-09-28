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
func runModelCatalog(t *testing.T, configure ...func(*http.ServeMux, *hub.PackageReleaseDetail)) (string, *sync.Mutex, *[][]byte, string, []byte) {
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
	detail.Release.PackageInterfaceDigest = assessmentDigest(contract.Raw)
	detail.Release.PackageInterfaceLength = int64(len(iface))
	detail.ExecutionRequirements = []string{"cozy-runtime>=0.2.25"}
	var mu sync.Mutex
	var posts [][]byte
	mux := http.NewServeMux()
	mux.HandleFunc("GET /v1/packages/proof/quantize", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.PackageCard{Package: hub.Resource{Org: "proof", Name: "quantize"}, Releases: []hub.ReleaseSummary{{Release: "1.0.0"}}})
	})
	mux.HandleFunc("GET /v1/packages/proof/quantize/releases/1.0.0", func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(detail) })
	mux.HandleFunc("POST /v1/packages/proof/quantize/download", func(w http.ResponseWriter, r *http.Request) {
		current, problem := launch.DecodePackageInterface(detail.PackageInterface)
		if problem != nil {
			t.Error(problem)
			return
		}
		_ = json.NewEncoder(w).Encode(declaredInstallPlan("quantize", r.URL.Query().Get("release"), current))
	})
	mux.HandleFunc("GET /v1/models/proof/source", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "proof", Name: "source"}, Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "1.0.0"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16", ManifestID: digest, Bytes: 210_000_000_000}}}}})
	})
	mux.HandleFunc("GET /v1/models/proof/source/releases/1.0.0/lanes/bf16/manifest", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(manifest) })
	mux.HandleFunc("GET /v1/models/resolve", func(w http.ResponseWriter, r *http.Request) {
		ref, lane := r.URL.Query().Get("ref"), r.URL.Query().Get("lane")
		released := ref == "proof/source@1.0.0@"+digest && lane == "bf16"
		if !released && (ref != "proof/source@"+digest || lane != "") {
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte(`{"error":{"code":"manifest.not_found","message":"not retained"}}`))
			return
		}
		resolution := hub.ModelResolution{Model: "proof/source", ManifestID: digest,
			ManifestLength: int64(len(manifest)), Bytes: 210_000_000_000, Objects: 100,
			Components: []string{"model"}, ComponentBytes: map[string]int64{"model": 100}}
		if released {
			resolution.Release, resolution.Lane = "1.0.0", "bf16"
		}
		_ = json.NewEncoder(w).Encode(resolution)
	})
	mux.HandleFunc("GET /v1/rental-skus", func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(hub.RentalProducts([]hub.RentalSKU{{Name: "cpu", AcceleratorModel: "CPU", AcceleratorCount: 1, PriceUSDMicrosPerHour: 100_000, BaseWorkerProfile: "python3.12-cpu-linux-x86"}}))
	})
	mux.HandleFunc("GET /v1/rentals", func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write([]byte(`{"rentals":[]}`)) })
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
	for _, apply := range configure {
		apply(mux, &detail)
	}
	server := httptest.NewServer(mux)
	t.Cleanup(server.Close)
	// Runs start the daemon, whose children may still be leaving when the test ends;
	// the root is reaped with the daemon rather than by the strict TempDir check.
	root, err := os.MkdirTemp(scratchBase, "model-catalog-")
	must(t, err)
	t.Cleanup(func() { _ = removeAllForce(root) })
	must(t, os.WriteFile(filepath.Join(root, config.FileName), []byte("tensorhub_url: "+server.URL+"\ntensorhub_token: model-run-test\ndaemon:\n  idle_shutdown_s: 0\n"), 0600))
	return root, &mu, &posts, digest, manifest
}

// declaredInstallPlan is a release's exact install plan with no dependencies. Rental
// placement reads it to find published callees' model defaults; its lock names none.
// It carries wheel facts only; no wheel byte is served.
func declaredInstallPlan(distribution, release string, iface *launch.PackageInterface) hub.PackageDownloadPlan {
	exact := func(raw []byte) hub.ExactDocument {
		return hub.ExactDocument{CanonicalBytes: raw, Digest: mustSpell(raw), Length: int64(len(raw))}
	}
	wheel := []byte(distribution + " wheel")
	return hub.PackageDownloadPlan{Release: release,
		PackageConfig:    exact([]byte("[application]\nobject = \"" + iface.Application + "\"\n")),
		PackageInterface: exact(iface.Raw),
		Pyproject:        exact([]byte("[project]\nname = \"" + distribution + "\"\nversion = \"" + release + "\"\nrequires-python = \">=3.12\"\ndependencies = []\n")),
		UVLock:           exact([]byte("version = 1\nrequires-python = \">=3.12\"\n\n[[package]]\nname = \"" + distribution + "\"\nversion = \"" + release + "\"\nsource = { editable = \".\" }\n")),
		Downloads: []hub.PackageInstallDownload{{Kind: "project_wheel", Path: distribution + "-" + release + "-py3-none-any.whl",
			Distribution: distribution, Version: release, Digest: mustSpell(wheel), Length: int64(len(wheel)),
			Tags: []string{"py3-none-any"}, ImportRoots: []string{distribution}}}}
}

func TestPinnedCheckpointTransferKeepsReleaseAndLaneConstraints(t *testing.T) {
	root, _, _, digest, _ := runModelCatalog(t)
	source := "proof/source@1.0.0/bf16#" + digest
	request, _, out := submitRun(t, root, "pinned-download", "model", "download", source, "local/pinned", "--json", "--full")
	if request == nil || request.ModelTransfer == nil || request.ModelTransfer.SourceSelection != digest ||
		request.ModelTransfer.InputLane != "bf16" || !strings.Contains(request.ModelTransfer.Source, "1.0.0") {
		t.Fatalf("the accepted transfer discarded the explicit release/lane/digest constraints: %+v %s", request, out)
	}
	code, out := runCozy(t, root, "model", "download", source, "local/pinned", "--lane", "fp8", "--json")
	if code == 0 || !strings.Contains(out, "disagree") {
		t.Fatal("conflicting explicit lane was accepted")
	}
	code, out = runCozy(t, root, "model", "download", "proof/source@1.0.0/bf16#sha256:"+strings.Repeat("f", 64), "local/pinned", "--json")
	if code == 0 || !strings.Contains(out, "manifest.not_found") {
		t.Fatal("checkpoint outside the pinned release/lane was accepted")
	}
}

func TestRunForeignModelInputsRefuseBeforeAcquisition(t *testing.T) {
	root, mu, posts, _, _ := runModelCatalog(t)
	source := "hf://MiniMaxAI/MiniMax-H3@" + strings.Repeat("4", 40)
	base := []string{"run", "proof/quantize/quantize", "steps=4", "model.dits=" + source, "model.shared=" + source, "--upload-to", "proof/output", "--rental-only", "--json"}
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
	startDaemonProcess(t, root)
	if request, _, out := submitRun(t, root, "checkpoint-download", "model", "download", "proof/source#"+digest, "local/checkpoint-proof", "--json"); request == nil {
		t.Fatalf("checkpoint download was refused: %s", out)
	}
	args := []string{"run", "proof/quantize/quantize", "steps=7",
		"model.dits=proof/source#" + digest, "model.shared=proof/source#" + digest,
		"--upload-to", "proof/output", "--rental-only", "--json", "--idempotency-key", "retained-model-job"}
	code, out := runCozy(t, root, args...)
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
	args := []string{"run", "proof/quantize/quantize", "model.dits=proof/source@1.0.0/bf16", "model.shared=proof/source@1.0.0/bf16", "--input", input, "--upload-to", "proof/output", "--rental-only", "--json", "--full", "--idempotency-key", "published-model-job"}
	startDaemonProcess(t, root)
	code, out := runCozy(t, root, args...)
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

func TestCatalogCheckpointReferenceRoundTripsThroughRun(t *testing.T) {
	root, _, _, digest, _ := runModelCatalog(t)
	code, out := runCozy(t, root, "model", "info", "proof/source", "--json")
	var document struct {
		Releases []struct {
			Lanes []struct {
				Ref string `json:"checkpoint_ref"`
			} `json:"lanes"`
		} `json:"releases"`
	}
	if code != 0 || json.Unmarshal([]byte(out), &document) != nil || len(document.Releases) != 1 || len(document.Releases[0].Lanes) != 1 {
		t.Fatalf("catalog info unavailable: %d %s", code, out)
	}
	ref := document.Releases[0].Lanes[0].Ref
	if ref != "proof/source#"+digest {
		t.Fatalf("catalog emitted unsupported checkpoint syntax: %s", ref)
	}
	request, _, out := submitRun(t, root, "catalog-checkpoint-ref", "run", "proof/quantize/quantize", "steps=7", "--model.dits="+ref, "--model.shared="+ref, "--rental-only", "--json")
	if request == nil || len(request.Models) != 2 || !request.Models[0].HubCheckpoint || request.Models[0].Manifest != digest {
		t.Fatalf("copied catalog ref did not resolve checkpoint: %+v %s", request, out)
	}
}
