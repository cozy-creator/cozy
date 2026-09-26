package producttest

import (
	"bytes"
	"encoding/json"
	"github.com/alecthomas/kong"
	"github.com/cozy-creator/cozy/internal/api"
	"github.com/cozy-creator/cozy/internal/cli"
	"github.com/cozy-creator/cozy/internal/config"
	"github.com/cozy-creator/cozy/internal/daemon"
	"github.com/cozy-creator/cozy/internal/home"
	"github.com/cozy-creator/cozy/internal/hub"
	"github.com/cozy-creator/cozy/internal/orchestrator"
	"github.com/cozy-creator/cozy/internal/output"
	"github.com/cozy-creator/cozy/internal/records"
	"github.com/cozy-creator/cozy/internal/rental"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestRentalModelSelectorFreezesLatestMatchingLane(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	peer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/models/paul/minimax-h3" {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL)
			http.Error(w, "unexpected", 500)
			return
		}
		lane := hub.ModelLaneSummary{Lane: "fp8-pruned", ManifestID: digest, Bytes: 123, Components: []string{"transformer"}, ComponentBytes: map[string]int64{"transformer": 123}}
		json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "paul", Name: "minimax-h3"}, Releases: []hub.ModelReleaseSummary{
			{ReleaseSummary: hub.ReleaseSummary{Release: "1.0", CutAt: "2026-01-01T00:00:00Z"}, Lanes: []hub.ModelLaneSummary{lane}},
			{ReleaseSummary: hub.ReleaseSummary{Release: "2.0", CutAt: "2026-02-01T00:00:00Z"}, Lanes: []hub.ModelLaneSummary{lane}},
			{ReleaseSummary: hub.ReleaseSummary{Release: "3.0", CutAt: "2026-03-01T00:00:00Z", Yanked: true}, Lanes: []hub.ModelLaneSummary{lane}},
			{ReleaseSummary: hub.ReleaseSummary{Release: "4.0", CutAt: "2026-04-01T00:00:00Z"}, Lanes: []hub.ModelLaneSummary{{Lane: "bf16"}}},
		}})
	}))
	defer peer.Close()
	for _, test := range []struct{ ref, release string }{{"paul/minimax-h3#fp8-pruned", "2.0"}, {"paul/minimax-h3@1.0/fp8-pruned", "1.0"}} {
		var grammar cli.CLI
		var out, diagnostic bytes.Buffer
		parser, err := kong.New(&grammar, kong.Writers(&out, &diagnostic))
		if err != nil {
			t.Fatal(err)
		}
		parsed, err := parser.Parse([]string{"model", "download", test.ref, "--rental=kirukiru", "--dry-run"})
		if err != nil {
			t.Fatal(err)
		}
		err = parsed.Run(&cli.Runtime{Cfg: config.Config{Home: t.TempDir(), HubURL: peer.URL}, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: true}})
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(out.String(), test.release) || !strings.Contains(out.String(), digest) {
			t.Fatalf("selection not frozen: %s", out.String())
		}
	}
}
func TestStandaloneModelDownloadDocumentHasNoBinding(t *testing.T) {
	refs := orchestrator.DownloadModelRefs([]orchestrator.ModelRef{{Model: "paul/minimax-h3", CatalogRepository: "paul/minimax-h3", Release: "2.0", Lane: "fp8-pruned", Manifest: "sha256:" + strings.Repeat("a", 64)}})
	if len(refs) != 1 || refs[0].Package != "" || refs[0].Slot != "" {
		t.Fatalf("standalone selection lost or bound: %v", refs)
	}
	raw, problem := rental.DownloadSet(nil, refs)
	if problem != nil {
		t.Fatal(problem)
	}
	if !strings.Contains(string(raw), `"packages":[]`) {
		t.Fatal(string(raw))
	}
}
func TestModelLaneAndDigestSyntaxAreDistinct(t *testing.T) {
	digest := "sha256:" + strings.Repeat("a", 64)
	for _, test := range []struct{ ref, lane, manifest string }{{"org/model#fp8-pruned", "fp8-pruned", ""}, {"org/model#" + digest, "", digest}, {"org/model@1.0/fp8#" + digest, "fp8", digest}} {
		_, _, lane, manifest, problem := hub.ParseModelRef(test.ref)
		if problem != nil || lane != test.lane || manifest != test.manifest {
			t.Fatalf("%s: %s %s %v", test.ref, lane, manifest, problem)
		}
	}
	for _, ref := range []string{"org/model#sha256:nope", "org/model@1.0/fp8#bf16", "org/model#"} {
		if _, _, _, _, problem := hub.ParseModelRef(ref); problem == nil {
			t.Fatalf("accepted %s", ref)
		}
	}
}

func TestRentalModelDownloadQueuesExactSelectionWhileBooting(t *testing.T) {
	digest := "sha256:" + strings.Repeat("b", 64)
	hubPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "GET" || r.URL.Path != "/v1/models/paul/minimax-h3" {
			t.Errorf("unexpected Hub request: %s %s", r.Method, r.URL)
			http.Error(w, "unexpected", 500)
			return
		}
		json.NewEncoder(w).Encode(hub.ModelCard{Model: hub.Resource{Org: "paul", Name: "minimax-h3"}, Releases: []hub.ModelReleaseSummary{{ReleaseSummary: hub.ReleaseSummary{Release: "2.0", CutAt: "2026-02-01T00:00:00Z"}, Lanes: []hub.ModelLaneSummary{{Lane: "fp8-pruned", ManifestID: digest, Bytes: 123, ComponentBytes: map[string]int64{"transformer": 123}}}}}})
	}))
	defer hubPeer.Close()
	var accepted records.RentalInstallSelection
	localPeer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == "GET" && r.URL.Path == "/" {
			return
		}
		if r.Method != "POST" || r.URL.Path != "/v1/local/rentals/rental-proof/prepare" || r.Header.Get("Authorization") == "" {
			t.Errorf("unexpected local request: %s %s", r.Method, r.URL)
			http.Error(w, "unexpected", 500)
			return
		}
		if err := json.NewDecoder(r.Body).Decode(&accepted); err != nil {
			t.Error(err)
		}
		w.WriteHeader(http.StatusAccepted)
		json.NewEncoder(w).Encode(records.RentalInstall{ID: "install-proof", RentalID: "rental-proof", State: "queued", Selection: accepted})
	}))
	defer localPeer.Close()
	layout, problem := home.Open(t.TempDir())
	if problem != nil {
		t.Fatal(problem)
	}
	store, problem := records.Open(layout.DB)
	if problem != nil {
		t.Fatal(problem)
	}
	defer store.Close()
	if problem = store.RecordRental(records.Rental{ID: "rental-proof", MachineName: "kirukiru", State: "booting", AcceleratorModel: "CPU", AcceleratorCount: 1, HourlyRateUSDMicros: 1, Hub: hubPeer.URL}); problem != nil {
		t.Fatal(problem)
	}
	held, problem := daemon.Hold(layout, strings.TrimPrefix(localPeer.URL, "http://"), "")
	if problem != nil {
		t.Fatal(problem)
	}
	defer held.Release()
	if _, problem = api.Mint(layout); problem != nil {
		t.Fatal(problem)
	}
	var grammar cli.CLI
	var out, diagnostic bytes.Buffer
	parser, err := kong.New(&grammar, kong.Writers(&out, &diagnostic))
	if err != nil {
		t.Fatal(err)
	}
	parsed, err := parser.Parse([]string{"model", "download", "paul/minimax-h3#fp8-pruned", "--rental=kirukiru"})
	if err != nil {
		t.Fatal(err)
	}
	if err = parsed.Run(&cli.Runtime{Cfg: config.Config{Home: layout.Root, HubURL: hubPeer.URL}, Out: &out, Err: &diagnostic, Mode: output.Mode{JSON: true}}); err != nil {
		t.Fatal(err)
	}
	if accepted.Package != "" || accepted.Release != "" || len(accepted.Models) != 1 {
		t.Fatalf("not a standalone selection: %+v", accepted)
	}
	model := accepted.Models[0]
	if model.Package != "" || model.Slot != "" || model.Model != "paul/minimax-h3" || model.Release != "2.0" || model.Lane != "fp8-pruned" || model.Manifest != digest || model.CatalogRepository != model.Model || model.Bytes != 123 || model.ComponentBytes["transformer"] != 123 {
		t.Fatalf("lost model identity or grant facts: %+v", model)
	}
	if !strings.Contains(out.String(), "queued") || !strings.Contains(out.String(), "install-proof") {
		t.Fatalf("missing queue receipt: %s", out.String())
	}
}
